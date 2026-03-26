"""
Atlassian MCP (Model Context Protocol) Server Integration.

This module provides integration with the Atlassian MCP server
(https://github.com/sooperset/mcp-atlassian) which exposes
Jira and Confluence as MCP tools for LLM agents.

The MCP approach provides:
  - Standardized tool interface for LLM agents
  - Automatic authentication handling
  - Built-in rate limiting and retries
  - Structured responses compatible with LangChain tools

Setup:
  1. Install: npx @sooperset/mcp-atlassian
  2. Configure: Set ATLASSIAN_URL, ATLASSIAN_EMAIL, ATLASSIAN_TOKEN env vars
  3. Run: The MCP server starts on localhost:3000 by default

Usage:
  This module wraps the MCP client to create LangChain-compatible tools.
"""

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# MCP Client (requires 'mcp' package)
# ─────────────────────────────────────────────

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:3000")


class AtlassianMCPClient:
    """
    Client for the Atlassian MCP server.
    
    Provides a Python interface to call Atlassian MCP tools:
      - jira_search_issues
      - jira_create_issue  
      - jira_get_issue
      - confluence_search
      - confluence_get_page
    """

    def __init__(self, server_url: str = MCP_SERVER_URL):
        self.server_url = server_url
        self._session = None
        self._available = False
        self._check_availability()

    def _check_availability(self):
        """Check if the MCP server is reachable."""
        try:
            import httpx
            resp = httpx.get(f"{self.server_url}/health", timeout=2)
            self._available = resp.status_code == 200
            if self._available:
                logger.info(f"[MCP] Atlassian MCP server available at {self.server_url}")
            else:
                logger.warning(f"[MCP] MCP server returned {resp.status_code}, falling back to direct API")
        except Exception as e:
            logger.info(f"[MCP] MCP server not available ({e}), using direct Atlassian API")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Call an MCP tool on the Atlassian MCP server.
        
        Args:
            tool_name: Name of the MCP tool (e.g., "jira_search_issues")
            arguments: Tool arguments as a dictionary
            
        Returns:
            Tool response as a dictionary
        """
        if not self._available:
            raise ConnectionError("MCP server is not available")

        import httpx
        response = httpx.post(
            f"{self.server_url}/tools/{tool_name}",
            json={"arguments": arguments},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def search_jira_issues(self, jql: str, max_results: int = 10) -> list[dict]:
        """Search Jira issues via MCP."""
        result = self.call_tool("jira_search_issues", {
            "jql": jql,
            "maxResults": max_results,
            "fields": ["summary", "status", "priority", "assignee", "labels", "fixVersions", "created"],
        })
        return result.get("issues", [])

    def create_jira_issue(self, project_key: str, summary: str, description: str,
                          issue_type: str = "Bug", priority: str = "Medium") -> dict:
        """Create a Jira issue via MCP."""
        return self.call_tool("jira_create_issue", {
            "projectKey": project_key,
            "summary": summary,
            "description": description,
            "issueType": issue_type,
            "priority": priority,
        })

    def search_confluence(self, query: str, space_key: str = None, limit: int = 10) -> list[dict]:
        """Search Confluence pages via MCP."""
        args = {"query": query, "limit": limit}
        if space_key:
            args["spaceKey"] = space_key
        result = self.call_tool("confluence_search", args)
        return result.get("results", [])

    def get_confluence_page(self, page_id: str) -> dict:
        """Get a Confluence page by ID via MCP."""
        return self.call_tool("confluence_get_page", {"pageId": page_id})


# ─────────────────────────────────────────────
# MCP Tool Wrappers for LangChain
# ─────────────────────────────────────────────

def create_mcp_jira_search_tool(client: AtlassianMCPClient):
    """Create a LangChain tool that uses MCP for Jira search."""
    from langchain_core.tools import tool

    @tool
    def mcp_jira_search(keywords: str, project_key: str = "SUPPORT") -> str:
        """
        Search Jira for issues matching keywords via Atlassian MCP server.
        
        Uses the MCP server protocol for standardized LLM-to-Atlassian communication.
        Falls back to direct API if MCP server is unavailable.
        """
        if not client.is_available:
            return json.dumps({"error": "MCP server unavailable", "fallback": "use direct API"})

        try:
            kw_list = keywords.split()[:5]
            text_clause = " OR ".join([f'text ~ "{kw}"' for kw in kw_list])
            jql = f'project = {project_key} AND ({text_clause}) AND issuetype = Bug ORDER BY created DESC'

            issues = client.search_jira_issues(jql, max_results=10)
            return json.dumps({
                "source": "jira_mcp",
                "issues": issues,
                "count": len(issues),
            }, indent=2)

        except Exception as exc:
            logger.error(f"[MCP] Jira search failed: {exc}")
            return json.dumps({"error": str(exc)})

    return mcp_jira_search


def create_mcp_confluence_search_tool(client: AtlassianMCPClient):
    """Create a LangChain tool that uses MCP for Confluence search."""
    from langchain_core.tools import tool

    @tool
    def mcp_confluence_search(query: str, space_key: str = "SUPPORT") -> str:
        """
        Search Confluence knowledge base via Atlassian MCP server.
        
        Leverages MCP protocol for consistent tool calling from LLM agents.
        """
        if not client.is_available:
            return json.dumps({"error": "MCP server unavailable", "fallback": "use direct API"})

        try:
            pages = client.search_confluence(query, space_key=space_key)
            return json.dumps({
                "source": "confluence_mcp",
                "pages": pages,
                "count": len(pages),
            }, indent=2)

        except Exception as exc:
            logger.error(f"[MCP] Confluence search failed: {exc}")
            return json.dumps({"error": str(exc)})

    return mcp_confluence_search


# ─────────────────────────────────────────────
# MCP Server Setup Instructions
# ─────────────────────────────────────────────

MCP_SETUP_GUIDE = """
╔═══════════════════════════════════════════════════════════════════════╗
║          ATLASSIAN MCP SERVER SETUP GUIDE                             ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  The Atlassian MCP server provides a standardized interface for       ║
║  LLM agents to interact with Jira and Confluence.                     ║
║                                                                       ║
║  INSTALLATION (choose one):                                           ║
║                                                                       ║
║  Option A — NPX (recommended):                                        ║
║    npx @sooperset/mcp-atlassian                                       ║
║                                                                       ║
║  Option B — Docker:                                                   ║
║    docker run -p 3000:3000 \\                                          ║
║      -e ATLASSIAN_URL=https://your-org.atlassian.net \\               ║
║      -e ATLASSIAN_EMAIL=you@company.com \\                            ║
║      -e ATLASSIAN_API_TOKEN=your-token \\                             ║
║      sooperset/mcp-atlassian                                          ║
║                                                                       ║
║  Option C — Claude Desktop Integration:                               ║
║    Add to ~/.claude/claude_desktop_config.json:                       ║
║    {                                                                  ║
║      "mcpServers": {                                                  ║
║        "atlassian": {                                                 ║
║          "command": "npx",                                            ║
║          "args": ["@sooperset/mcp-atlassian"],                        ║
║          "env": {                                                     ║
║            "ATLASSIAN_URL": "https://your-org.atlassian.net",         ║
║            "ATLASSIAN_EMAIL": "you@company.com",                      ║
║            "ATLASSIAN_API_TOKEN": "your-token"                        ║
║          }                                                            ║
║        }                                                              ║
║      }                                                                ║
║    }                                                                  ║
║                                                                       ║
║  AVAILABLE MCP TOOLS:                                                 ║
║    • jira_search_issues    — JQL-based issue search                   ║
║    • jira_create_issue     — Create new issues                        ║
║    • jira_get_issue        — Get issue details + comments             ║
║    • jira_update_issue     — Update existing issues                   ║
║    • confluence_search     — Full-text page search                    ║
║    • confluence_get_page   — Fetch page content                       ║
║    • confluence_create_page — Create new pages                        ║
║                                                                       ║
║  VERIFICATION:                                                        ║
║    curl http://localhost:3000/health                                  ║
║    curl http://localhost:3000/tools                                   ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""


# Singleton client instance
_mcp_client: Optional[AtlassianMCPClient] = None


def get_mcp_client() -> AtlassianMCPClient:
    """Get or create the singleton MCP client."""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = AtlassianMCPClient()
    return _mcp_client


if __name__ == "__main__":
    print(MCP_SETUP_GUIDE)
    client = get_mcp_client()
    print(f"MCP Server Available: {client.is_available}")
