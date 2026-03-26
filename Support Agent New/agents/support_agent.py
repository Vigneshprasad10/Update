"""
Support AI Agent — LangGraph Orchestration Engine.

This module implements the multi-agent graph using LangGraph with:
  - Google Gemini as the LLM backbone
  - Specialized nodes for each integration (Jira, Confluence, GitHub)
  - A synthesis node that produces the final enriched ticket + resolution

Graph Flow:
  START
    └─► extract_intent        (Gemini: parse ticket, extract keywords)
    └─► [parallel dispatch]
          ├─► query_jira      (search bugs, related tickets)
          ├─► query_confluence (search KB, runbooks, release notes)
          └─► query_github    (search commits, PRs, releases)
    └─► synthesize_context    (Gemini: merge all context)
    └─► generate_resolution   (Gemini: suggest resolution)
    └─► create_jira_ticket    (create enriched ticket)
    └─► END
"""

import json
import logging
import os
from typing import Any, Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from agents.state import SupportAgentState, JiraContext, ConfluenceContext, GitHubContext, Resolution
from tools.jira_tools import search_jira_bugs, create_jira_ticket, get_jira_issue_details
from tools.confluence_tools import search_confluence_kb, get_confluence_page, search_release_notes
from tools.github_tools import search_recent_commits, get_recent_pull_requests, get_recent_releases
from config.settings import settings
from pdf_generator import generate_pdf_report

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# LLM Initialization
# ─────────────────────────────────────────────

def get_llm() -> ChatGoogleGenerativeAI:
    """Initialize and return the Gemini LLM."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key="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",
        temperature=0.0,
        max_tokens=8192,
    )


# ─────────────────────────────────────────────
# Node: Extract Intent
# ─────────────────────────────────────────────

def extract_intent_node(state: SupportAgentState) -> SupportAgentState:
    """
    Use Gemini to extract structured intent from the raw support ticket.

    Extracts:
      - Primary keywords for searching
      - Detected error codes / exception types
      - Likely affected components/services
      - Normalized priority
    """
    logger.info("[NODE] extract_intent — analyzing support ticket")
    llm = get_llm()

    system_prompt = """You are a senior support engineer at a software company.
Your task is to analyze a customer support ticket and extract structured information
to help search for relevant context across Jira, Confluence, and GitHub.

Respond with ONLY valid JSON matching this exact schema:
{
  "keywords": ["list", "of", "5-10", "search", "keywords"],
  "error_codes": ["any", "error", "codes", "or", "exception", "class", "names"],
  "affected_components": ["likely", "subsystems", "or", "modules"],
  "priority_assessment": "critical|high|medium|low",
  "technical_summary": "One sentence technical summary for searching"
}"""

    user_message = f"""Analyze this support ticket:

SUMMARY: {state.get('ticket_summary', '')}
DESCRIPTION: {state.get('ticket_description', '')}
REPORTED VERSION: {state.get('product_version', 'Unknown')}
ENVIRONMENT: {state.get('environment', 'Unknown')}
PRIORITY: {state.get('priority', 'Medium')}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ])

    try:
        # Parse the JSON response
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(content)
        return {
            "keywords": parsed.get("keywords", []),
            "error_codes": parsed.get("error_codes", []),
            "affected_components": parsed.get("affected_components", []),
            "priority": parsed.get("priority_assessment", state.get("priority", "medium")),
            "current_step": "extract_intent",
            "steps_completed": ["extract_intent"],
            "messages": [AIMessage(content=f"Intent extracted. Keywords: {parsed.get('keywords', [])}")]
        }
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"[NODE] extract_intent — JSON parse error: {e}, using fallback")
        # Fallback: use raw ticket text as keywords
        raw_words = (state.get("ticket_summary", "") + " " + state.get("ticket_description", "")).split()
        keywords = list(set(w.lower() for w in raw_words if len(w) > 4))[:8]
        return {
            "keywords": keywords,
            "error_codes": [],
            "affected_components": [],
            "current_step": "extract_intent",
            "steps_completed": ["extract_intent"],
        }


# ─────────────────────────────────────────────
# Node: Query Jira
# ─────────────────────────────────────────────

def query_jira_node(state: SupportAgentState) -> dict:
    """Search Jira for known bugs and related support tickets.

    IMPORTANT: Returns ONLY the keys this node owns.
    Parallel nodes must never spread {**state, ...} — doing so causes
    InvalidUpdateError when multiple branches try to write shared keys
    like ticket_summary simultaneously.
    """
    logger.info("[NODE] query_jira — searching for known issues")

    keywords = " ".join(state.get("keywords", []) + state.get("error_codes", []))
    product_version = state.get("product_version", "")

    try:
        raw_result = search_jira_bugs.invoke({
            "keywords": keywords
            # "product_version": product_version,
        })
        data = json.loads(raw_result)

        jira_context: JiraContext = {
            "known_bugs": data.get("bugs", []),
            "related_issues": data.get("related_issues", []),
            "affected_versions": [
                v for bug in data.get("bugs", [])
                for v in (bug.get("fix_version") or [] if isinstance(bug.get("fix_version"), list)
                          else [bug.get("fix_version")] if bug.get("fix_version") else [])
            ],
            "status": "success" if "error" not in data else "error",
        }

        # Return ONLY the keys this node writes — no {**state} spread
        return {
            "jira_context": jira_context,
            "current_step": "query_jira",
            "steps_completed": ["query_jira"],          # operator.add appends this
            "messages": [AIMessage(content=f"Found {len(jira_context['known_bugs'])} known bugs in Jira.")],
        }
    except Exception as exc:
        logger.error(f"[NODE] query_jira error: {exc}")
        return {
            "jira_context": {"known_bugs": [], "related_issues": [], "status": "error"},
            "errors": [f"Jira query failed: {exc}"],    # operator.add appends this
            "steps_completed": ["query_jira"],
        }


# ─────────────────────────────────────────────
# Node: Query Confluence
# ─────────────────────────────────────────────

def query_confluence_node(state: SupportAgentState) -> dict:
    """Search Confluence for KB articles, runbooks, and release notes.

    Returns ONLY confluence_context and accumulator keys — never {**state}.
    """
    logger.info("[NODE] query_confluence — searching documentation")
    

    keywords = " ".join(state.get("keywords", []))
    #version = ''
    # version = state.get("product_version", "")
    logger.info("keywords: %s", keywords)
    try:
        kb_raw = search_confluence_kb.invoke({"keywords": keywords, "page_type": "all"})
        kb_data = json.loads(kb_raw)

        release_raw = search_release_notes.invoke({"keywords": keywords})
        release_data = json.loads(release_raw)

        confluence_context: ConfluenceContext = {
            "relevant_pages": kb_data.get("knowledge_base_articles", kb_data.get("pages", [])),
            "runbooks": kb_data.get("runbooks", []),
            "release_notes": release_data.get("release_notes", []),
            "status": "success",
        }

        # Return ONLY the keys this node writes — no {**state} spread
        return {
            "confluence_context": confluence_context,
            "current_step": "query_confluence",
            "steps_completed": ["query_confluence"],
            "messages": [AIMessage(content=(
                f"Found {len(confluence_context['relevant_pages'])} KB articles, "
                f"{len(confluence_context['runbooks'])} runbooks."
            ))],
        }
    except Exception as exc:
        logger.error(f"[NODE] query_confluence error: {exc}")
        return {
            "confluence_context": {"relevant_pages": [], "runbooks": [], "release_notes": [], "status": "error"},
            "errors": [f"Confluence query failed: {exc}"],
            "steps_completed": ["query_confluence"],
        }


# ─────────────────────────────────────────────
# Node: Query GitHub
# ─────────────────────────────────────────────

def query_github_node(state: SupportAgentState) -> dict:
    """Search GitHub for recent commits, PRs, and releases related to the issue.

    Returns ONLY github_context and accumulator keys — never {**state}.
    """
    logger.info("[NODE] query_github — searching code changes")

    components = " ".join(state.get("affected_components", []) + state.get("keywords", [])[:3])
    keywords = " ".join(state.get("keywords", []))
    search_terms = components if components.strip() else keywords
    try:
        commits_raw = search_recent_commits.invoke({
            "affected_files_or_keywords": search_terms,
        })
        commits_data = json.loads(commits_raw)

        # prs_raw = get_recent_pull_requests.invoke({
        #     "state": "all",
        #     "keywords": keywords,
        #     "days_back": settings.github.lookback_days,
        # })
        # prs_data = json.loads(prs_raw)

        try:
            releases_raw = get_recent_releases.invoke({"limit": 3})
            releases_data = json.loads(releases_raw)
            releases = releases_data.get("releases", [])
        except Exception as e:
            logger.warning(f"[GITHUB] Releases fetch failed: {e}")
            releases = []

        github_context: GitHubContext = {
            "recent_commits": commits_data.get("commits", []),
            #"open_prs": prs_data.get("pull_requests", []),
            "recent_releases": releases,
            "blame_suspects": [
                c for c in commits_data.get("commits", [])
                if any(kw in (c.get("message") or "").lower() for kw in state.get("keywords", []))
            ],
            "status": "success",
        }

        # Return ONLY the keys this node writes — no {**state} spread
        return {
            "github_context": github_context,
            "current_step": "query_github",
            "steps_completed": ["query_github"],
            "messages": [AIMessage(content=(
                f"Found {len(github_context['recent_commits'])} relevant commits, "
                #f"{len(github_context['open_prs'])} PRs in the last {settings.github.lookback_days} days."
            ))],
        }
    except Exception as exc:
        logger.error(f"[NODE] query_github error: {exc}")
        return {
            "github_context": {"recent_commits": [], "open_prs": [], "recent_releases": [], "blame_suspects": [], "status": "error"},
            "errors": [f"GitHub query failed: {exc}"],
            "steps_completed": ["query_github"],
        }


# ─────────────────────────────────────────────
# Node: Synthesize & Generate Resolution
# ─────────────────────────────────────────────

def synthesize_and_resolve_node(state: SupportAgentState) -> SupportAgentState:
    """
    Use Gemini to synthesize all gathered context and generate
    a root-cause hypothesis and resolution recommendations.
    """
    logger.info("[NODE] synthesize — generating resolution with Gemini")
    llm = get_llm()

    jira_ctx = state.get("jira_context", {})
    confluence_ctx = state.get("confluence_context", {})
    github_ctx = state.get("github_context", {})

    # Build context summary for Gemini
    context_block = f"""
=== SUPPORT TICKET ===
Summary: {state.get('ticket_summary')}
Description: {state.get('ticket_description')}
Version: {state.get('product_version', 'Unknown')}
Environment: {state.get('environment', 'Unknown')}
Priority: {state.get('priority', 'Medium')}

=== JIRA FINDINGS ===
Known Bugs ({len(jira_ctx.get('known_bugs', []))} found):
{json.dumps(jira_ctx.get('known_bugs', [])[:3], indent=2)}

Related Past Issues:
{json.dumps(jira_ctx.get('related_issues', [])[:2], indent=2)}

=== CONFLUENCE FINDINGS ===
Relevant KB Articles ({len(confluence_ctx.get('relevant_pages', []))} found):
{json.dumps(confluence_ctx.get('relevant_pages', [])[:2], indent=2)}

Available Runbooks:
{json.dumps(confluence_ctx.get('runbooks', [])[:2], indent=2)}

Recent Release Notes:
{json.dumps(confluence_ctx.get('release_notes', [])[:1], indent=2)}

=== GITHUB FINDINGS ===
Suspect Commits:
{json.dumps(github_ctx.get('blame_suspects', github_ctx.get('recent_commits', []))[:3], indent=2)}

Recent PRs in Affected Area:
{json.dumps(github_ctx.get('open_prs', [])[:2], indent=2)}

Recent Releases:
{json.dumps(github_ctx.get('recent_releases', [])[:2], indent=2)}
"""

    system_prompt = """You are a senior support engineer with deep knowledge of the product.
Given a support ticket and all the context gathered from Jira, Confluence, and GitHub,
produce a comprehensive analysis and resolution plan.

CRITICAL RULES TO PREVENT HALLUCINATION:
1. ONLY reference bugs, tickets, commits, or articles that explicitly appear in the context below.
2. NEVER invent or guess ticket IDs, commit SHAs, bug numbers, or article titles.
3. If the context does not contain enough information, say so honestly.
4. Every claim in root_cause_hypothesis MUST cite a specific source from the context.
5. Only set escalation_needed to true if confidence is below 0.7. If confidence is 0.7 or above, set escalation_needed to false unless there is a very specific critical reason.
6. linked_bug_ids must ONLY contain IDs from the JIRA FINDINGS section.
7. similar_resolved_cases must ONLY contain cases from the JIRA FINDINGS section.

Respond with ONLY valid JSON matching this exact schema:
{
  "root_cause_hypothesis": "Your best assessment of what caused this issue",
  "immediate_actions": [
    "Step 1: Specific actionable step",
    "Step 2: Another step",
    "Step 3: ..."
  ],
  "workaround": "Temporary workaround if available, or null",
  "escalation_needed": true or false,
  "escalation_reason": "Why escalation is needed (if applicable)",
  "confidence_score": 0.0 to 1.0,
  "similar_resolved_cases": ["TICKET-ID: Brief description of resolution"],
  "linked_bug_ids": ["BUG-XXXX"],
  "recommended_kb_articles": ["KB article title and URL"],
  "sources_used":{
    "jira": ["list of JIRA IDs referenced"],
    "confluence": ["list of article titles referenced"],
    "github": ["list of commit SHAs referenced"]
  },
  "executive_summary": "2-3 sentence summary citing only facts from the context above"
}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Analyze this support context and provide resolution:\n{context_block}"),
    ])

    try:
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(content)

        resolution: Resolution = {
            "root_cause_hypothesis": parsed.get("root_cause_hypothesis", ""),
            "immediate_actions": parsed.get("immediate_actions", []),
            "workaround": parsed.get("workaround"),
            "escalation_needed": parsed.get("escalation_needed", False),
            "confidence_score": float(parsed.get("confidence_score", 0.5)),
            "similar_resolved_cases": parsed.get("similar_resolved_cases", []),
        }

        # Build enriched ticket payload
        enriched_ticket = {
            "original_summary": state.get("ticket_summary"),
            "ai_enriched_description": build_enriched_description(state, parsed),
            "priority": state.get("priority", "Medium"),
            "labels": state.get("keywords", [])[:5],
            "linked_bugs": parsed.get("linked_bug_ids", []),
            "confidence": resolution["confidence_score"],
        }

        # Build human-readable summary
        final_summary = f"""🎯 AI ANALYSIS COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{parsed.get('executive_summary', '')}

🔍 ROOT CAUSE: {parsed.get('root_cause_hypothesis', 'Under investigation')}

📋 IMMEDIATE ACTIONS:
{chr(10).join(f"  {i+1}. {a}" for i, a in enumerate(parsed.get('immediate_actions', [])))}

🔗 LINKED BUGS: {', '.join(parsed.get('linked_bug_ids', [])) or 'None found'}
⚡ ESCALATION: {'YES — ' + parsed.get('escalation_reason', '') if parsed.get('escalation_needed') else 'Not required'}
🎲 CONFIDENCE: {int(resolution['confidence_score'] * 100)}%
"""

        return {
            "resolution": resolution,
            "enriched_ticket": enriched_ticket,
            "final_summary": final_summary,
            "current_step": "synthesize",
            "steps_completed": ["synthesize"],
            "messages": [AIMessage(content=final_summary)]
        }

    except Exception as exc:
        logger.error(f"[NODE] synthesize error: {exc}")
        return {
            "errors": [f"Synthesis failed: {exc}"],
            "steps_completed": ["synthesize"],
        }


def build_enriched_description(state: SupportAgentState, analysis: dict) -> str:
    """Build a rich Jira-formatted description from all gathered context."""
    jira_ctx = state.get("jira_context", {})
    confluence_ctx = state.get("confluence_context", {})
    github_ctx = state.get("github_context", {})

    bugs = jira_ctx.get("known_bugs", [])
    articles = confluence_ctx.get("relevant_pages", [])
    commits = github_ctx.get("blame_suspects", github_ctx.get("recent_commits", []))[:2]

    lines = [
        f"*Customer Report:*\n{state.get('ticket_description', '')}",
        "\n---",
        "*🤖 AI-Enriched Context (generated automatically):*",
        f"\n*Root Cause Hypothesis:* {analysis.get('root_cause_hypothesis', 'Under investigation')}",
        "\n*Known Related Bugs:*",
    ]

    for bug in bugs[:3]:
        lines.append(f"  • [{bug.get('id')}] {bug.get('summary')} — Status: {bug.get('status')}")

    lines.append("\n*Relevant Documentation:*")
    for article in articles[:2]:
        lines.append(f"  • [{article.get('title')}|{article.get('url', '#')}]")

    lines.append("\n*Recent Suspect Code Changes:*")
    for commit in commits[:2]:
        lines.append(f"  • `{commit.get('sha')}` by @{commit.get('author')}: {commit.get('message', '')[:80]}")

    lines.append(f"\n*AI Confidence Score:* {int(analysis.get('confidence_score', 0) * 100)}%")
    lines.append(f"\n*Environment:* {state.get('environment', 'Unknown')} | *Version:* {state.get('product_version', 'Unknown')}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Node: Create Enriched Ticket
# ─────────────────────────────────────────────

def create_ticket_node(state: SupportAgentState) -> SupportAgentState:
    """Create the final enriched Jira ticket with all gathered context."""
    logger.info("[NODE] create_ticket — creating enriched Jira ticket")

    enriched = state.get("enriched_ticket") or {}
    resolution = state.get("resolution") or {}

    confidence = resolution.get("confidence_score", 1.0)
    if confidence < 0.70:
        return {
            "current_step": "create_ticket",
            "steps_completed": ["create_ticket"],
            "messages": [AIMessage(content=(
                f"⚠️ Ticket creation skipped - confidence too low ({int(confidence * 100)}%). "
                "Please review manually before creating a ticket."
            ))],
        }

    try:
        priority_map = {
            "critical": "Highest",
            "high": "High",
            "medium": "Medium",
            "low": "Low"
        }
        raw_priority = enriched.get("priority", "medium")
        mapped_priority = priority_map.get(raw_priority.lower() if raw_priority else "medium", "Medium")

        result_raw = create_jira_ticket.invoke({
            "summary": f"[AI-Enriched] {enriched.get('original_summary', state.get('ticket_summary', 'Support Issue'))}",
            "description": enriched.get("ai_enriched_description", state.get("ticket_description", "")),
            "priority": mapped_priority,
            "labels": enriched.get("labels", []),
            "linked_bugs": enriched.get("linked_bugs", []),
            "issuetype": {"name": "Bug"},
        })
        result = json.loads(result_raw)

        jira_context_update = {"ticket_id": result.get("ticket_id")}

        # Update state with ticket ID before generating PDF (PDF will be generated on approval/rejection)
        state_with_ticket = {
            **state,
            "jira_context": {
                **(state.get("jira_context") or {}),
                "ticket_id": result.get("ticket_id")
            }
        }
        # PDF is now generated by the server's approve/reject endpoints, not by the agent
        # This ensures the PDF always has the correct decision (approval/rejection) context

        return {
            "jira_context": jira_context_update,   # _merge_dict adds ticket_id to existing jira_context
            "current_step": "create_ticket",
            "steps_completed": ["create_ticket"],
            "messages": [
                AIMessage(content=f"✅ Created enriched ticket: {result.get('ticket_id')} — {result.get('url', '')}")
            ]
        }
    except Exception as exc:
        logger.error(f"[NODE] create_ticket error: {exc}")
        return {
            "errors": [f"Ticket creation failed: {exc}"],
            "steps_completed": ["create_ticket"],
        }


# ─────────────────────────────────────────────
# Graph Construction
# ─────────────────────────────────────────────

def build_support_agent_graph() -> StateGraph:
    """
    Build and compile the LangGraph support agent workflow.

    Returns a compiled StateGraph ready to invoke.
    """
    graph = StateGraph(SupportAgentState)

    # Register all nodes
    graph.add_node("extract_intent", extract_intent_node)
    graph.add_node("query_jira", query_jira_node)
    graph.add_node("query_confluence", query_confluence_node)
    graph.add_node("query_github", query_github_node)
    graph.add_node("synthesize_and_resolve", synthesize_and_resolve_node)
    graph.add_node("create_ticket", create_ticket_node)

    # Define the workflow edges
    graph.add_edge(START, "extract_intent")
    graph.add_edge("extract_intent", "query_jira")
    graph.add_edge("extract_intent", "query_confluence")
    graph.add_edge("extract_intent", "query_github")

    # All three query nodes feed into synthesis
    graph.add_edge("query_jira", "synthesize_and_resolve")
    graph.add_edge("query_confluence", "synthesize_and_resolve")
    graph.add_edge("query_github", "synthesize_and_resolve")

    graph.add_edge("synthesize_and_resolve", "create_ticket")
    graph.add_edge("create_ticket", END)

    return graph.compile()


# Singleton compiled graph
support_agent = build_support_agent_graph()


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def run_support_agent(
    ticket_summary: str,
    ticket_description: str,
    priority: str = "high",
    product_version: str = "",
    environment: str = "production",
    customer_id: str = "",
    tags: list[str] = None,
) -> SupportAgentState:
    """
    Run the full support agent workflow for a given ticket.

    Args:
        ticket_summary: One-line issue description.
        ticket_description: Full customer-provided description.
        priority: critical / high / medium / low.
        product_version: e.g. "v2.4.0".
        environment: dev / staging / production.
        customer_id: Customer identifier.
        tags: Initial tags/labels.

    Returns:
        Final SupportAgentState with all context and resolution.
    """
    initial_state: SupportAgentState = {
        "ticket_summary": ticket_summary,
        "ticket_description": ticket_description,
        "priority": priority,
        "product_version": product_version,
        "environment": environment,
        "customer_id": customer_id,
        "tags": tags or [],
        "messages": [HumanMessage(content=ticket_summary)],
        "steps_completed": [],
        "errors": [],
        "iteration_count": 0,
    }

    logger.info(f"[AGENT] Starting support agent for: {ticket_summary[:60]}...")
    final_state = support_agent.invoke(initial_state)
    logger.info(f"[AGENT] Completed. Steps: {final_state.get('steps_completed', [])}")
    return final_state
