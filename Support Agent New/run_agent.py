"""
CLI runner for the Support AI Agent.

Usage:
    python run_agent.py --summary "API returning 401 errors" \
                        --description "Customer reports..." \
                        --version v2.4.0 \
                        --priority high

Or interactively:
    python run_agent.py --interactive
"""

import argparse
import json
import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from rich import box

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.support_agent import run_support_agent

console = Console()


def print_banner():
    console.print(Panel.fit(
        "[bold cyan]🤖 Support AI Agent[/bold cyan]\n"
        "[dim]Powered by LangGraph + Google Gemini + Atlassian MCP[/dim]",
        border_style="cyan",
    ))


def print_results(state: dict):
    """Pretty-print the agent results using Rich."""

    # ── Resolution Panel ─────────────────────────────
    resolution = state.get("resolution") or {}
    confidence = int(resolution.get("confidence_score", 0) * 100)
    confidence_color = "green" if confidence >= 75 else "yellow" if confidence >= 50 else "red"

    console.print(Panel(
        f"[bold]{resolution.get('root_cause_hypothesis', 'Analysis incomplete')}[/bold]\n\n"
        f"[{confidence_color}]Confidence: {confidence}%[/{confidence_color}]\n"
        f"Escalation: {'⚠️ YES' if resolution.get('escalation_needed') else '✅ Not required'}",
        title="[bold green]🎯 Root Cause Analysis[/bold green]",
        border_style="green",
    ))

    # ── Actions Table ─────────────────────────────────
    actions = resolution.get("immediate_actions", [])
    if actions:
        table = Table(title="📋 Immediate Actions", box=box.ROUNDED, border_style="blue")
        table.add_column("#", style="bold cyan", width=4)
        table.add_column("Action", style="white")
        for i, action in enumerate(actions, 1):
            table.add_row(str(i), action)
        console.print(table)

    # ── Jira Context ─────────────────────────────────
    jira = state.get("jira_context") or {}
    bugs = jira.get("known_bugs", [])
    if bugs:
        table = Table(title="🐛 Known Jira Bugs", box=box.SIMPLE, border_style="red")
        table.add_column("ID", style="bold red", width=10)
        table.add_column("Summary")
        table.add_column("Status", width=12)
        table.add_column("Priority", width=10)
        for bug in bugs[:3]:
            status_color = "red" if bug.get("status") == "Open" else "green"
            table.add_row(
                bug.get("id", ""),
                bug.get("summary", "")[:60],
                f"[{status_color}]{bug.get('status', '')}[/{status_color}]",
                bug.get("priority", ""),
            )
        console.print(table)

    # ── Confluence Context ────────────────────────────
    confluence = state.get("confluence_context") or {}
    articles = confluence.get("relevant_pages", [])
    if articles:
        table = Table(title="📚 Confluence KB Articles", box=box.SIMPLE, border_style="yellow")
        table.add_column("Title")
        table.add_column("URL", style="dim")
        for article in articles[:3]:
            table.add_row(
                article.get("title", "")[:50],
                article.get("url", "")[:60],
            )
        console.print(table)

    # ── GitHub Context ────────────────────────────────
    github = state.get("github_context") or {}
    commits = github.get("blame_suspects", github.get("recent_commits", []))[:3]
    if commits:
        table = Table(title="🔧 Suspect GitHub Commits", box=box.SIMPLE, border_style="magenta")
        table.add_column("SHA", width=8, style="bold magenta")
        table.add_column("Author", width=15)
        table.add_column("Message")
        table.add_column("Date", width=12)
        for commit in commits:
            table.add_row(
                commit.get("sha", "")[:7],
                commit.get("author", ""),
                commit.get("message", "")[:50],
                commit.get("date", ""),
            )
        console.print(table)

    # ── Created Ticket ────────────────────────────────
    ticket_id = jira.get("ticket_id")
    if ticket_id:
        console.print(Panel(
            f"[bold green]✅ Enriched ticket created: [cyan]{ticket_id}[/cyan][/bold green]",
            border_style="green",
        ))

    # ── Errors ───────────────────────────────────────
    errors = state.get("errors", [])
    if errors:
        for err in errors:
            console.print(f"[yellow]⚠️  {err}[/yellow]")


def interactive_mode():
    """Run in interactive mode, prompting for ticket details."""
    print_banner()
    console.print("[dim]Enter support ticket details (Ctrl+C to exit)[/dim]\n")

    summary = console.input("[bold cyan]Ticket Summary:[/bold cyan] ").strip()
    if not summary:
        console.print("[red]Summary is required.[/red]")
        return

    console.print("[bold cyan]Description[/bold cyan] (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    description = "\n".join(lines).strip()

    version = console.input("[bold cyan]Product Version[/bold cyan] (e.g. v2.4.0): ").strip()
    environment = console.input("[bold cyan]Environment[/bold cyan] (production/staging/dev): ").strip() or "production"
    priority = console.input("[bold cyan]Priority[/bold cyan] (critical/high/medium/low): ").strip() or "high"

    run_agent(summary, description, version, environment, priority)


def run_agent(summary: str, description: str, version: str = "", environment: str = "production", priority: str = "high"):
    """Run the agent with the given parameters and display results."""
    console.print(f"\n[dim]Analyzing ticket: [bold]{summary[:60]}...[/bold][/dim]\n")

    steps_display = {
        "extract_intent": "🧠 Extracting intent & keywords...",
        "query_jira": "🐛 Searching Jira for known bugs...",
        "query_confluence": "📚 Searching Confluence documentation...",
        "query_github": "🔧 Analyzing recent code changes...",
        "synthesize": "⚡ Synthesizing context with Gemini...",
        "create_ticket": "📝 Creating enriched Jira ticket...",
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running AI agent...", total=None)

        final_state = run_support_agent(
            ticket_summary=summary,
            ticket_description=description,
            product_version=version,
            environment=environment,
            priority=priority,
        )

        progress.update(task, description="[green]✅ Analysis complete!")

    console.print()
    print_results(final_state)


def main():
    parser = argparse.ArgumentParser(description="Support AI Agent CLI")
    parser.add_argument("--summary", "-s", help="Ticket summary (one line)")
    parser.add_argument("--description", "-d", help="Full ticket description")
    parser.add_argument("--version", "-v", default="", help="Product version (e.g. v2.4.0)")
    parser.add_argument("--environment", "-e", default="production", help="Environment")
    parser.add_argument("--priority", "-p", default="high", help="Priority level")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--json", "-j", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.summary:
        description = args.description or args.summary
        if args.json:
            result = run_support_agent(
                ticket_summary=args.summary,
                ticket_description=description,
                product_version=args.version,
                environment=args.environment,
                priority=args.priority,
            )
            # Convert to serializable format
            output = {
                k: v for k, v in result.items()
                if k not in ("messages",)  # Skip LangChain message objects
            }
            print(json.dumps(output, indent=2, default=str))
        else:
            print_banner()
            run_agent(args.summary, description, args.version, args.environment, args.priority)
    else:
        interactive_mode()
        # Default demo
        # print_banner()
        # console.print("[dim]Running demo scenario...[/dim]\n")
        # run_agent(
        #     summary="Customers getting 401 Unauthorized after 2 hours of activity",
        #     description=(
        #         "Multiple enterprise customers reported that after about 2 hours of using the API, "
        #         "all requests start returning 401 Unauthorized errors. Restarting the client app or "
        #         "logging out and back in resolves the issue temporarily. Affecting production "
        #         "environment. Version v2.4.0. Error seen in logs: 'JWT token validation failed: "
        #         "token expired' even though users haven't been idle."
        #     ),
        #     version="v2.4.0",
        #     environment="production",
        #     priority="critical",
        # )


if __name__ == "__main__":
    main()
