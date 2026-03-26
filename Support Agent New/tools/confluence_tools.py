"""
Confluence Tool Integration via Atlassian Python API + MCP.

Provides LangGraph-compatible tools for:
  - Searching Confluence knowledge base
  - Finding runbooks and SOPs
  - Fetching release notes
  - Creating/updating documentation from resolved issues
"""

import json
import logging
from typing import Optional

from langchain_core.tools import tool
from atlassian import Confluence

from config.settings import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Client Factory
# ─────────────────────────────────────────────

def get_confluence_client() -> Optional[Confluence]:
    if settings.confluence.api_token in ("your-confluence-api-token", "", None):
        return None
    return Confluence(
        url=settings.confluence.url,
        username=settings.confluence.email,
        password=settings.confluence.api_token,
        cloud=True,
    )


# ─────────────────────────────────────────────
# Mock Knowledge Base
# ─────────────────────────────────────────────

MOCK_KB_ARTICLES = [
    {
        "id": "KB-1001",
        "title": "Troubleshooting Authentication & Session Issues",
        "space": "SUPPORT",
        "url": "https://your-org.atlassian.net/wiki/spaces/SUPPORT/pages/KB-1001",
        "excerpt": "This article covers common authentication failures including 401 errors, token expiry, and session management issues. Step-by-step diagnostic guide included.",
        "last_updated": "2024-01-18",
        "labels": ["authentication", "session", "401", "token"],
        "views": 342,
    },
    {
        "id": "KB-0892",
        "title": "Database Connection Pool - Configuration & Troubleshooting",
        "space": "ENGINEERING",
        "url": "https://your-org.atlassian.net/wiki/spaces/ENG/pages/KB-0892",
        "excerpt": "Guide for diagnosing and resolving database connection pool exhaustion. Includes monitoring queries, tuning parameters, and emergency procedures.",
        "last_updated": "2024-01-10",
        "labels": ["database", "connection-pool", "performance"],
        "views": 187,
    },
    {
        "id": "KB-1045",
        "title": "API Rate Limiting & Error Code Reference",
        "space": "SUPPORT",
        "url": "https://your-org.atlassian.net/wiki/spaces/SUPPORT/pages/KB-1045",
        "excerpt": "Complete reference for all API error codes (4xx, 5xx), their causes, and resolution steps. Includes customer-facing messaging templates.",
        "last_updated": "2024-01-22",
        "labels": ["api", "errors", "rate-limiting"],
        "views": 521,
    },
]

MOCK_RUNBOOKS = [
    {
        "id": "RB-0041",
        "title": "Runbook: P1 Authentication Service Outage",
        "space": "OPS",
        "url": "https://your-org.atlassian.net/wiki/spaces/OPS/pages/RB-0041",
        "excerpt": "Step-by-step runbook for handling P1 authentication failures. Includes escalation matrix, rollback procedures, and communication templates.",
        "severity": "P1",
        "owner": "Platform Team",
    },
    {
        "id": "RB-0055",
        "title": "Runbook: Database Performance Degradation",
        "space": "OPS",
        "url": "https://your-org.atlassian.net/wiki/spaces/OPS/pages/RB-0055",
        "excerpt": "Diagnostic runbook for DB slowness, timeouts, and connection pool issues. Includes CloudWatch dashboards and query optimization steps.",
        "severity": "P2",
        "owner": "Database Team",
    },
]

MOCK_RELEASE_NOTES = [
    {
        "version": "v2.4.1",
        "date": "2024-01-20",
        "url": "https://your-org.atlassian.net/wiki/spaces/PRODUCT/pages/release-v241",
        "highlights": [
            "Fixed: JWT token refresh on long-lived sessions (BUG-2341)",
            "Improved: Connection pool sizing algorithm",
            "Security: Updated TLS certificate chain",
        ],
        "breaking_changes": [],
    },
    {
        "version": "v2.4.0",
        "date": "2024-01-08",
        "url": "https://your-org.atlassian.net/wiki/spaces/PRODUCT/pages/release-v240",
        "highlights": [
            "New: Real-time webhook delivery system",
            "Improved: Payment processing throughput +40%",
            "Fixed: Null handling in customer profile API",
        ],
        "breaking_changes": ["Removed deprecated /v1/auth/legacy endpoint"],
    },
]


# ─────────────────────────────────────────────
# LangChain Tools
# ─────────────────────────────────────────────

@tool
def search_confluence_kb(keywords: str, page_type: str = "all") -> str:
    """
    Search Confluence for knowledge base articles, runbooks, and documentation.

    Args:
        keywords: Search terms related to the support issue.
        page_type: Filter type — "kb", "runbook", "release_notes", or "all".

    Returns:
        JSON with matched Confluence pages including excerpts and URLs.
    """
    client = get_confluence_client()

    if client is None:
        # Demo mode
        logger.info("[CONFLUENCE] Running in mock mode")
        kw_lower = keywords.lower()

        def keyword_match(item: dict) -> bool:
            searchable = (
                item.get("title", "").lower()
                + " "
                + item.get("excerpt", "").lower()
                + " "
                + " ".join(item.get("labels", []))
            )
            return any(kw in searchable for kw in kw_lower.split())

        matched_kb = [a for a in MOCK_KB_ARTICLES if keyword_match(a)] or MOCK_KB_ARTICLES[:1]
        matched_runbooks = [r for r in MOCK_RUNBOOKS if keyword_match(r)] or []
        matched_releases = MOCK_RELEASE_NOTES[:1]

        result = {"source": "confluence", "mode": "demo"}
        if page_type in ("all", "kb"):
            result["knowledge_base_articles"] = matched_kb
        if page_type in ("all", "runbook"):
            result["runbooks"] = matched_runbooks
        if page_type in ("all", "release_notes"):
            result["release_notes"] = matched_releases

        return json.dumps(result, indent=2)

    # ── Live Confluence Search ────────────────────
    try:
        cql = f'space = "{settings.confluence.space_key}" AND text ~ "studio login issue" ORDER BY lastModified DESC'
        results = client.cql(cql, limit=10)

        pages = []
        for page in results.get("results", []):
            content = page.get("content", {})
            pages.append({
                "id": content.get("id"),
                "title": content.get("title"),
                "space": content.get("space", {}).get("key"),
                "url": settings.confluence.url + content.get("_links", {}).get("webui", ""),
                "excerpt": page.get("excerpt", "")[:400],
                "last_updated": content.get("history", {}).get("lastUpdated", {}).get("when", "")[:10],
            })

        return json.dumps({
            "source": "confluence",
            "mode": "live",
            "results_count": len(pages),
            "pages": pages,
        }, indent=2)

    except Exception as exc:
        logger.error(f"[CONFLUENCE] Search failed: {exc}")
        return json.dumps({"source": "confluence", "error": str(exc), "pages": []})


@tool
def get_confluence_page(page_id: str) -> str:
    """
    Fetch the full content of a specific Confluence page by ID.

    Args:
        page_id: Confluence page ID (e.g., "KB-1001" or numeric ID).

    Returns:
        JSON with page title, full body text, and metadata.
    """
    client = get_confluence_client()

    if client is None:
        # Return mock content for demo
        all_pages = MOCK_KB_ARTICLES + MOCK_RUNBOOKS
        page = next((p for p in all_pages if p["id"] == page_id), MOCK_KB_ARTICLES[0])
        page["full_content"] = (
            page.get("excerpt", "")
            + "\n\n## Diagnostic Steps\n"
            + "1. Check application logs for error patterns\n"
            + "2. Verify service health via monitoring dashboard\n"
            + "3. Review recent deployments in the deployment tracker\n"
            + "4. Escalate to L2 if unresolved after 30 minutes"
        )
        return json.dumps({"source": "confluence", "mode": "demo", "page": page})

    try:
        page = client.get_page_by_id(page_id, expand="body.view,history,metadata.labels")
        body = page.get("body", {}).get("view", {}).get("value", "")

        # Strip HTML tags roughly for LLM consumption
        import re
        clean_body = re.sub(r"<[^>]+>", " ", body)
        clean_body = re.sub(r"\s+", " ", clean_body).strip()[:3000]

        labels = [lb.get("name") for lb in page.get("metadata", {}).get("labels", {}).get("results", [])]

        return json.dumps({
            "source": "confluence",
            "mode": "live",
            "page": {
                "id": page_id,
                "title": page.get("title"),
                "content": clean_body,
                "labels": labels,
                "last_updated": page.get("history", {}).get("lastUpdated", {}).get("when", "")[:10],
                "url": settings.confluence.url + page.get("_links", {}).get("webui", ""),
            },
        })
    except Exception as exc:
        return json.dumps({"source": "confluence", "error": str(exc)})


@tool
def search_release_notes(version: str = "", keywords: str = "") -> str:
    """
    Search Confluence release notes for a given product version or keywords.

    Args:
        version: Product version string (e.g., "v2.4.1"). Optional.
        keywords: Keywords to match in release notes content. Optional.

    Returns:
        JSON with matched release notes entries and change details.
    """
    client = get_confluence_client()

    if client is None:
        # Filter mock release notes
        matched = MOCK_RELEASE_NOTES
        if version:
            matched = [r for r in matched if version in r.get("version", "")]
        if keywords:
            kw_lower = keywords.lower()
            matched = [
                r for r in matched
                if any(kw in " ".join(r.get("highlights", [])).lower() for kw in kw_lower.split())
            ] or matched

        return json.dumps({
            "source": "confluence",
            "mode": "demo",
            "release_notes": matched[:3],
        }, indent=2)

    try:
        filters = []
        if version:
            filters.append(f'title ~ "{version}"')
        if keywords:
            filters.append(f'text ~ "{keywords}"')

        cql = (
            f'space = "{settings.confluence.space_key}" '
            f'AND type = page '
            f'AND label = "release-notes" '
            + ("AND " + " AND ".join(filters) if filters else "")
            + " ORDER BY lastModified DESC"
        )

        results = client.cql(cql, limit=5)
        notes = []
        for r in results.get("results", []):
            content = r.get("content", {})
            notes.append({
                "title": content.get("title"),
                "url": settings.confluence.url + content.get("_links", {}).get("webui", ""),
                "excerpt": r.get("excerpt", "")[:300],
            })

        return json.dumps({"source": "confluence", "mode": "live", "release_notes": notes})

    except Exception as exc:
        return json.dumps({"source": "confluence", "error": str(exc)})


# Exported tools
CONFLUENCE_TOOLS = [search_confluence_kb, get_confluence_page, search_release_notes]
