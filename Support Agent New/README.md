# 🤖 Support AI Agent
### Intelligent Ticket Enrichment System
*Powered by LangGraph · Google Gemini · Atlassian MCP · GitHub*

---

## Overview

This system implements an **agentic AI workflow** specifically designed for product support teams. When a customer reports an issue, instead of engineers manually searching across Jira, Confluence, and GitHub — an AI agent does it automatically in parallel, synthesizes the context using Gemini, and produces an enriched ticket with resolution recommendations.

```
Customer Report
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                 LangGraph Agent Workflow                 │
│                                                         │
│  ┌──────────────┐                                       │
│  │ Extract      │  Gemini parses ticket, extracts        │
│  │ Intent       │  keywords, error codes, components     │
│  └──────┬───────┘                                       │
│         │ (parallel dispatch)                           │
│    ┌────┴────┬──────────┐                               │
│    ▼         ▼          ▼                               │
│  Jira    Confluence   GitHub                            │
│  Search  Search       Search                            │
│  (bugs)  (KB+runbooks)(commits+PRs)                    │
│    └────┬────┴──────────┘                               │
│         ▼                                               │
│  ┌──────────────┐                                       │
│  │ Synthesize   │  Gemini merges all context,           │
│  │ & Resolve    │  generates root cause + actions       │
│  └──────┬───────┘                                       │
│         ▼                                               │
│  ┌──────────────┐                                       │
│  │ Create       │  Enriched Jira ticket created         │
│  │ Ticket       │  with all gathered context            │
│  └──────────────┘                                       │
└─────────────────────────────────────────────────────────┘
      │
      ▼
Enriched Ticket + Resolution Recommendations
```

---

## Architecture

```
support-agent/
├── agents/
│   ├── state.py              # LangGraph state definitions (TypedDict)
│   └── support_agent.py      # Main graph: nodes + edges + agent runner
│
├── tools/
│   ├── jira_tools.py         # Jira search + ticket creation (Atlassian API)
│   ├── confluence_tools.py   # Confluence KB + runbook search
│   ├── github_tools.py       # Commit + PR + release analysis
│   └── mcp_integration.py    # Atlassian MCP server client + setup guide
│
├── config/
│   └── settings.py           # Pydantic settings (env var management)
│
├── server.py                 # FastAPI REST + WebSocket server
├── run_agent.py              # CLI runner with Rich terminal UI
├── dashboard.html            # Interactive web dashboard (demo)
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd support-agent
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required credentials:

| Service | What You Need | Where to Get It |
|---------|---------------|-----------------|
| Google Gemini | `GEMINI_API_KEY` | [ai.google.dev](https://ai.google.dev) |
| Jira | `JIRA_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN` | [Atlassian API Tokens](https://id.atlassian.com/manage-profile/security/api-tokens) |
| Confluence | Same credentials as Jira | Same API token |
| GitHub | `GITHUB_TOKEN` | [GitHub Settings > Tokens](https://github.com/settings/tokens) |

> **Demo Mode:** If credentials are not set, the agent runs with built-in mock data showing realistic examples.

### 3. Run via CLI

```bash
# Demo mode (uses built-in scenario)
python run_agent.py

# With your own ticket
python run_agent.py \
  --summary "API returning 401 errors after 2 hours" \
  --description "Customers report sudden 401 errors..." \
  --version v2.4.0 \
  --priority critical

# Interactive mode


# JSON output (for integrations)
python run_agent.py --summary "..." --json
```

### 4. Run as API Server

```bash
python server.py
# Server starts on http://localhost:8000

# Submit a ticket
curl -X POST http://localhost:8000/api/tickets/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_summary": "401 errors after 2 hours",
    "ticket_description": "Multiple customers report...",
    "priority": "critical",
    "product_version": "v2.4.0"
  }'

# Poll status
curl http://localhost:8000/api/tickets/{job_id}

# API docs
open http://localhost:8000/docs
```

### 5. Open the Dashboard

Open `dashboard.html` in your browser for an interactive demo that visualizes the agent workflow in real-time.

---

## LangGraph Workflow Detail

### State Flow

```python
SupportAgentState
├── ticket_summary        # Input
├── ticket_description    # Input
├── keywords              # Extracted by extract_intent_node
├── error_codes           # Extracted by extract_intent_node
├── affected_components   # Extracted by extract_intent_node
├── jira_context          # Populated by query_jira_node
│   ├── known_bugs        
│   └── related_issues    
├── confluence_context    # Populated by query_confluence_node
│   ├── relevant_pages    
│   └── runbooks          
├── github_context        # Populated by query_github_node
│   ├── recent_commits    
│   └── open_prs          
├── resolution            # Generated by synthesize_and_resolve_node
│   ├── root_cause_hypothesis
│   ├── immediate_actions 
│   └── confidence_score  
└── enriched_ticket       # Created by create_ticket_node
```

### Graph Edges

```
START → extract_intent
extract_intent → query_jira        (parallel)
extract_intent → query_confluence  (parallel)
extract_intent → query_github      (parallel)
query_jira → synthesize_and_resolve
query_confluence → synthesize_and_resolve
query_github → synthesize_and_resolve
synthesize_and_resolve → create_ticket
create_ticket → END
```

---

## Atlassian MCP Integration

The system supports two integration modes:

### Mode 1: Direct Atlassian Python API (default)
Uses `atlassian-python-api` library to call Jira/Confluence REST APIs directly.

### Mode 2: Atlassian MCP Server
Uses the [MCP server](https://github.com/sooperset/mcp-atlassian) for a standardized LLM-to-Atlassian interface.

**Setup MCP Server:**
```bash
# Install
npx @sooperset/mcp-atlassian

# Or Docker
docker run -p 3000:3000 \
  -e ATLASSIAN_URL=https://your-org.atlassian.net \
  -e ATLASSIAN_EMAIL=you@company.com \
  -e ATLASSIAN_API_TOKEN=your-token \
  sooperset/mcp-atlassian

# Set in .env
MCP_SERVER_URL=http://localhost:3000
```

**Available MCP Tools:**
- `jira_search_issues` — JQL-based bug/ticket search
- `jira_create_issue` — Create new enriched tickets
- `jira_get_issue` — Fetch full issue details
- `confluence_search` — Full-text knowledge base search
- `confluence_get_page` — Fetch page content for LLM context

---

## Example Output

```
🎯 AI ANALYSIS COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The issue matches BUG-2341 (In Progress). JWT token refresh logic
has a race condition introduced in PR-891 that causes silent expiry.

🔍 ROOT CAUSE: TokenRefresher.refreshIfNeeded() skips refresh when 
   token is within 5 minutes of expiry due to off-by-one in 
   timestamp comparison (commit a3f7c2e, 2024-01-14)

📋 IMMEDIATE ACTIONS:
  1. Verify customer is on v2.4.0 (fix is in v2.4.1)
  2. Provide workaround: client-side refresh every 90 minutes
  3. Escalate to Platform Team if >10 customers affected
  4. Coordinate upgrade to v2.4.1 with CSM

🔗 LINKED BUGS: BUG-2341
⚡ ESCALATION: Not required (fix already available in v2.4.1)
🎲 CONFIDENCE: 92%
✅ Created: SUPPORT-2847
```

---

## Configuration Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google AI Studio API key | — |
| `GEMINI_MODEL` | Gemini model version | `gemini-1.5-pro` |
| `JIRA_URL` | Atlassian Cloud domain | — |
| `JIRA_PROJECT_KEY` | Default project for tickets | `SUPPORT` |
| `CONFLUENCE_SPACE_KEY` | Default search space | `SUPPORT` |
| `GITHUB_ORG` | GitHub organization name | — |
| `GITHUB_REPO` | Repository to analyze | — |
| `GITHUB_LOOKBACK_DAYS` | Days of history to search | `30` |
| `AGENT_MAX_ITERATIONS` | Safety limit on agent loops | `10` |
| `MCP_SERVER_URL` | Atlassian MCP server URL | `http://localhost:3000` |

---

## Extending the Agent

### Add a New Tool
```python
# tools/my_tool.py
from langchain_core.tools import tool

@tool
def check_monitoring_alerts(service: str) -> str:
    """Search monitoring alerts for a service."""
    # ... implementation
    return json.dumps(results)
```

### Add a New Node
```python
# agents/support_agent.py
def query_monitoring_node(state: SupportAgentState) -> SupportAgentState:
    result = check_monitoring_alerts.invoke({"service": "auth-service"})
    return {**state, "monitoring_context": json.loads(result)}

graph.add_node("query_monitoring", query_monitoring_node)
graph.add_edge("extract_intent", "query_monitoring")
graph.add_edge("query_monitoring", "synthesize_and_resolve")
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Agent Orchestration** | LangGraph 0.2+ |
| **LLM** | Google Gemini 1.5 Pro |
| **Jira/Confluence** | Atlassian MCP + atlassian-python-api |
| **GitHub** | PyGitHub |
| **API Server** | FastAPI + WebSockets |
| **Terminal UI** | Rich |
| **Configuration** | Pydantic Settings |
