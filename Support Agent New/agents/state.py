"""
State definition for the Support AI Agent using LangGraph.

This module defines the complete state graph that flows through
all agent nodes during a support ticket analysis.

KEY DESIGN: Parallel node fan-out/fan-in pattern.
──────────────────────────────────────────────────
When multiple nodes run in parallel (query_jira, query_confluence,
query_github), LangGraph requires that every key they write to has
a *reducer* — otherwise writing the same key from two parallel
branches raises InvalidUpdateError.

Solution used here:
  • Keys written ONLY by parallel nodes (jira_context, confluence_context,
    github_context) are typed as Optional and each node returns ONLY its
    own key — no {**state, ...} spreading — so there is never a conflict.
  • Keys that must accumulate across parallel branches (steps_completed,
    errors) use operator.add as their reducer via Annotated.
  • messages uses the built-in add_messages reducer.
  • All read-only input keys (ticket_summary, ticket_description, …) are
    never returned by parallel nodes, so they remain conflict-free.
"""

import operator
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


# ─────────────────────────────────────────────
# Reducer helpers
# ─────────────────────────────────────────────

def _keep_last(a, b):
    """Reducer that always keeps the most recent (right-hand) value."""
    return b


def _merge_dict(a, b):
    """Reducer that merges two dicts, with b taking precedence."""
    if a is None:
        return b
    if b is None:
        return a
    return {**a, **b}


# ─────────────────────────────────────────────
# Sub-state types for each integration
# ─────────────────────────────────────────────

class JiraContext(TypedDict, total=False):
    """Context gathered from Jira."""
    known_bugs: list[dict]          # Matching open/closed bugs
    related_issues: list[dict]      # Related tickets by keyword
    ticket_id: Optional[str]        # Newly created ticket ID
    affected_versions: list[str]    # Affected product versions
    status: str                     # Tool execution status


class ConfluenceContext(TypedDict, total=False):
    """Context gathered from Confluence."""
    relevant_pages: list[dict]      # Matching KB articles
    runbooks: list[dict]            # Matching runbooks/SOPs
    release_notes: list[dict]       # Relevant release notes
    status: str


class GitHubContext(TypedDict, total=False):
    """Context gathered from GitHub."""
    recent_commits: list[dict]      # Recent commits in affected area
    open_prs: list[dict]            # Open PRs touching related code
    recent_releases: list[dict]     # Recent release tags
    blame_suspects: list[dict]      # Commits likely related to issue
    status: str


class Resolution(TypedDict, total=False):
    """AI-generated resolution suggestions."""
    root_cause_hypothesis: str      # Best guess at root cause
    immediate_actions: list[str]    # Steps to resolve now
    workaround: Optional[str]       # Temporary workaround if available
    escalation_needed: bool         # Whether L2/L3 escalation is needed
    confidence_score: float         # 0.0–1.0 confidence in analysis
    similar_resolved_cases: list[str]


# ─────────────────────────────────────────────
# Main Agent State
# ─────────────────────────────────────────────

class SupportAgentState(TypedDict, total=False):
    """
    Complete state for the Support AI Agent graph.

    Reducer annotations explain how LangGraph merges concurrent writes:
      - Annotated[X, operator.add]  → lists are concatenated
      - Annotated[X, _keep_last]    → last writer wins (scalar fields)
      - Annotated[X, _merge_dict]   → dicts are shallow-merged
      - Annotated[list, add_messages] → LangGraph message deduplication
      - Plain type (no Annotated)   → field is NEVER written by parallel
                                      nodes, so no reducer needed
    """

    # ── Input (written once by the caller, never by parallel nodes) ────
    ticket_summary: str
    ticket_description: str
    customer_id: Optional[str]
    product_version: Optional[str]
    environment: Optional[str]
    priority: str
    tags: list[str]

    # ── Conversation ── add_messages deduplicates by message id ────────
    messages: Annotated[list, add_messages]

    # ── Extracted Intelligence (written only by extract_intent) ────────
    keywords: list[str]
    error_codes: list[str]
    affected_components: list[str]

    # ── Tool Results ── each written by exactly ONE parallel node ───────
    # _merge_dict lets create_ticket_node later add ticket_id to
    # jira_context without conflicting with query_jira_node's write.
    jira_context: Annotated[Optional[JiraContext], _merge_dict]
    confluence_context: Annotated[Optional[ConfluenceContext], _merge_dict]
    github_context: Annotated[Optional[GitHubContext], _merge_dict]

    # ── Agent Control ── accumulate safely across parallel branches ─────
    current_step: Annotated[str, _keep_last]
    steps_completed: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    iteration_count: int

    # ── Output (written only by synthesize / create_ticket nodes) ───────
    enriched_ticket: Optional[dict]
    resolution: Optional[Resolution]
    final_summary: Optional[str]
