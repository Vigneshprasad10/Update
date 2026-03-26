"""
Jira Tool Integration via Atlassian Python API + MCP.

This module provides LangGraph-compatible tools for:
  - Searching Jira for known bugs and related issues
  - Creating enriched support tickets
  - Fetching issue details and comments

In production, these wrap the Atlassian MCP server endpoints.
For local dev / demo, a mock layer is provided via JIRA_MOCK=true.
"""

import json
import os
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from langchain_core.tools import tool
from atlassian import Jira

from config.settings import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Jira Client Factory
# ─────────────────────────────────────────────

def get_jira_client() -> Optional[Jira]:
    """Return a configured Jira client, or None if credentials are defaults."""
    if settings.jira.api_token in ("your-jira-api-token", "", None):
        return None
    return Jira(
        url=settings.jira.url,
        username=settings.jira.email,
        password=settings.jira.api_token,
        cloud=True,
    )


# ─────────────────────────────────────────────
# Mock Data (for demo / testing)
# ─────────────────────────────────────────────

MOCK_BUGS = [
    {
        "id": "BUG-2341",
        "summary": "Authentication token expiry causes 401 on long sessions",
        "status": "In Progress",
        "priority": "High",
        "assignee": "Sarah Chen",
        "created": "2024-01-15",
        "labels": ["authentication", "session", "token"],
        "fix_version": "v2.4.1",
        "description": "Users experience sudden 401 errors after ~2 hours of inactivity. Root cause traced to JWT token refresh not being triggered correctly.",
        "comments_count": 8,
    },
    {
        "id": "BUG-2298",
        "summary": "Database connection pool exhaustion under high load",
        "status": "Resolved",
        "priority": "Critical",
        "assignee": "Marcus Williams",
        "created": "2024-01-08",
        "labels": ["database", "performance", "connection-pool"],
        "fix_version": "v2.3.9",
        "description": "Under sustained load >500 RPS, DB connections are not released properly causing timeout errors.",
        "comments_count": 14,
    },
    {
        "id": "BUG-2367",
        "summary": "Null pointer exception in payment processing module",
        "status": "Open",
        "priority": "Critical",
        "assignee": "Unassigned",
        "created": "2024-01-20",
        "labels": ["payment", "npe", "critical"],
        "fix_version": None,
        "description": "NPE thrown when processing refunds with missing customer_id field.",
        "comments_count": 3,
    },
]

MOCK_RELATED = [
    {
        "id": "SUPPORT-1892",
        "summary": "Customer reports session drops every 2 hours",
        "status": "Closed",
        "type": "Support",
        "resolution": "Linked to BUG-2341",
    },
    {
        "id": "SUPPORT-1756",
        "summary": "Intermittent 401 errors on API calls",
        "status": "Closed",
        "type": "Support",
        "resolution": "Workaround: manually refresh token",
    },
]


# ─────────────────────────────────────────────
# LangChain Tools
# ─────────────────────────────────────────────

@tool
def search_jira_bugs(keywords: str, product_version: str = "") -> str:
    """
    Search Jira for known bugs and issues matching the given keywords.

    Args:
        keywords: Space-separated keywords extracted from the support ticket.
        product_version: Optional version string to narrow search scope.

    Returns:
        JSON string with list of matching bugs and related issues.
    """
    client = get_jira_client()

    if client is None:
        # Demo mode — return curated mock data
        logger.info("[JIRA] Running in mock mode")
        kw_lower = keywords.lower()
        matched = [
            bug for bug in MOCK_BUGS
            if any(kw in bug["summary"].lower() or kw in " ".join(bug["labels"])
                   for kw in kw_lower.split())
        ]
        if not matched:
            matched = MOCK_BUGS[:2]  # Return first two as fallback

        return json.dumps({
            "source": "jira",
            "mode": "demo",
            "bugs_found": len(matched),
            "bugs": matched,
            "related_issues": MOCK_RELATED,
        }, indent=2)

    # ── Live Jira Query via JQL ──────────────────
    try:
        kw_list = [k.strip() for k in keywords.split() if len(k.strip()) > 2]
        text_clause = " OR ".join([f'text ~ "{kw}"' for kw in kw_list[:5]])
        version_clause = f'AND affectedVersion = "{product_version}"' if product_version else ""

        jql = (
            f'project = {settings.jira.project_key} '
            f'AND issuetype in (Bug) '
            f'AND ({text_clause}) '
            f'{version_clause} '
            f'ORDER BY created DESC'
        )

        results = client.jql(jql, limit=10)
        bugs = []
        for issue in results.get("issues", []):
            fields = issue.get("fields", {})
            bugs.append({
                "id": issue.get("key"),
                "summary": fields.get("summary"),
                "status": fields.get("status", {}).get("name"),
                "priority": fields.get("priority", {}).get("name"),
                "assignee": (fields.get("assignee") or {}).get("displayName"),
                "created": fields.get("created", "")[:10],
                "labels": fields.get("labels", []),
                "fix_version": [v.get("name") for v in fields.get("fixVersions", [])],
            })

        # Search for similar support tickets
        related_jql = (
            f'project = {settings.jira.project_key} '
            f'AND issuetype in (Support, "Service Request") '
            f'AND ({text_clause}) '
            f'AND status in (Closed, Resolved) '
            f'ORDER BY resolved DESC'
        )
        related_results = client.jql(related_jql, limit=5)
        related = []
        for issue in related_results.get("issues", []):
            fields = issue.get("fields", {})
            related.append({
                "id": issue.get("key"),
                "summary": fields.get("summary"),
                "status": fields.get("status", {}).get("name"),
                "resolution": (fields.get("resolution") or {}).get("name"),
            })

        return json.dumps({
            "source": "jira",
            "mode": "live",
            "bugs_found": len(bugs),
            "bugs": bugs,
            "related_issues": related,
        }, indent=2)

    except Exception as exc:
        logger.error(f"[JIRA] Search failed: {exc}")
        return json.dumps({"source": "jira", "error": str(exc), "bugs": [], "related_issues": []})


@tool
def create_jira_ticket(
    summary: str,
    description: str,
    priority: str = "Medium",
    labels: list[str] = None,
    linked_bugs: list[str] = None,
) -> str:
    """
    Create a new Jira support ticket enriched with AI-gathered context.

    Args:
        summary: Ticket title/summary.
        description: Full ticket description (supports Jira markdown).
        priority: One of: Critical, High, Medium, Low.
        labels: List of label strings to attach.
        linked_bugs: List of bug IDs to link (e.g. ["BUG-2341"]).

    Returns:
        JSON with created ticket ID and URL.
    """
    client = get_jira_client()

    if client is None:
        # Demo mode
        ticket_id = f"SUPPORT-{2000 + hash(summary) % 999}"
        return json.dumps({
            "source": "jira",
            "mode": "demo",
            "ticket_id": ticket_id,
            "url": f"{settings.jira.url}/browse/{ticket_id}",
            "status": "created",
        })

    try:
        def sanitize_label(label: str) -> str:
            import re
            label = label.strip().replace(" ", "_")
            label = re.sub(r"[^a-zA-Z0-9_\-]", "", label)
            return label[:50]
 
        clean_labels = [sanitize_label(l) for l in (labels or []) if l.strip()]
        clean_labels = [l for l in clean_labels if l]
 
        priority_map = {
            "critical": "Highest",
            "high": "High", 
            "medium": "Medium",
            "low": "Low"
        }
        mapped_priority = priority_map.get(priority.lower() if priority else "medium", "Medium")
 
        issue_data = {
            "project": {"key": settings.jira.project_key},
            "issuetype": {"name": "Task"},
            "summary": summary,
            "description": description,
            "priority": {"name": mapped_priority},
            "labels": clean_labels,
        }

        created = client.issue_create(fields=issue_data)
        ticket_id = created.get("key")

        # Link related bugs
        if linked_bugs and ticket_id:
            for bug_id in linked_bugs[:3]:
                try:
                    client.create_issue_link(
                        link_type="is caused by",
                        inward_issue=ticket_id,
                        outward_issue=bug_id,
                    )
                except Exception:
                    pass  # Non-fatal

        return json.dumps({
            "source": "jira",
            "mode": "live",
            "ticket_id": ticket_id,
            "url": f"{settings.jira.url}/browse/{ticket_id}",
            "status": "created",
        })

    except Exception as exc:
        logger.error(f"[JIRA] Ticket creation failed: {exc}")
        return json.dumps({"source": "jira", "error": str(exc)})


@tool
def get_jira_issue_details(issue_id: str) -> str:
    """
    Fetch full details and comments for a specific Jira issue.

    Args:
        issue_id: Jira issue key (e.g. "BUG-2341").

    Returns:
        JSON with full issue details, comments, and change log.
    """
    client = get_jira_client()

    if client is None:
        # Mock detail response
        mock = next((b for b in MOCK_BUGS if b["id"] == issue_id), MOCK_BUGS[0])
        mock["comments"] = [
            {"author": "Dev Team", "body": "Confirmed and reproduced in v2.4.0", "date": "2024-01-16"},
            {"author": "QA Team", "body": "Fix verified in v2.4.1-rc1", "date": "2024-01-19"},
        ]
        return json.dumps({"source": "jira", "mode": "demo", "issue": mock})

    try:
        issue = client.issue(issue_id)
        fields = issue.get("fields", {})
        comments_raw = client.issue_get_comments(issue_id)
        comments = [
            {
                "author": c.get("author", {}).get("displayName"),
                "body": c.get("body", "")[:500],
                "date": c.get("created", "")[:10],
            }
            for c in comments_raw.get("comments", [])[-5:]  # Last 5 comments
        ]

        return json.dumps({
            "source": "jira",
            "mode": "live",
            "issue": {
                "id": issue_id,
                "summary": fields.get("summary"),
                "description": (fields.get("description") or "")[:1000],
                "status": fields.get("status", {}).get("name"),
                "priority": fields.get("priority", {}).get("name"),
                "assignee": (fields.get("assignee") or {}).get("displayName"),
                "fix_versions": [v.get("name") for v in fields.get("fixVersions", [])],
                "comments": comments,
            },
        })
    except Exception as exc:
        return json.dumps({"source": "jira", "error": str(exc)})


# Exported tools list
JIRA_TOOLS = [search_jira_bugs, create_jira_ticket, get_jira_issue_details]
