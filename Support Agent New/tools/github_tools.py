"""
GitHub Tool Integration via PyGitHub.
 
Provides LangGraph-compatible tools for:
 - Searching recent commits that may have caused a regression
 - Finding open/recent PRs in affected areas
 - Fetching recent release/tag information
"""
 
import json
import logging
from datetime import datetime, timedelta
from typing import Optional
from langchain_core.tools import tool
from github import Github, GithubException
 
from config.settings import settings
 
logger = logging.getLogger(__name__)
 
# ─────────────────────────────────────────────
# Client Factory
# ─────────────────────────────────────────────
 
def get_github_client() -> Optional[Github]:
    if settings.github.token in ("ghp_your_github_token", "", None):
        logger.warning("[GITHUB] No valid token found — running in mock mode")
        return None
    return Github(settings.github.token)
 
# ─────────────────────────────────────────────
# GitHub Tools
# ─────────────────────────────────────────────
 
@tool
def search_recent_commits(
    affected_files_or_keywords: str,
    days_back: int = 30,
) -> str:
    """
    Search recent GitHub commits related to the issue.
 
    Args:
        affected_files_or_keywords: Keywords like "auth login JWT"
        days_back: Lookback period
 
    Returns:
        JSON string with relevant commits
    """
    gh = get_github_client()
 
    if gh is None:
        return json.dumps({
            "source": "github",
            "mode": "demo",
            "commits_found": 0,
            "commits": [],
            "message": "No GitHub token configured"
        }, indent=2)
 
    try:
        repo_name = f"{settings.github.org}/{settings.github.repo}"
        repo = gh.get_repo(repo_name)
 
        since = datetime.utcnow() - timedelta(days=days_back)
 
        logger.info(f"[GITHUB] Repo: {repo_name}")
        logger.info(f"[GITHUB] Keywords: {affected_files_or_keywords}")
        logger.info(f"[GITHUB] Looking back {days_back} days since {since.strftime('%Y-%m-%d')}")
 
        all_commits = []
        matched_commits = []
 
        # Step 1 — Get all commits in the lookback period
        try:
            for commit in repo.get_commits(since=since):
                try:
                    files = []
                    try:
                        files = [f.filename.lower() for f in commit.files]
                    except Exception:
                        pass
 
                    all_commits.append({
                        "sha": commit.sha[:7],
                        "message": commit.commit.message.split("\n")[0][:120],
                        "author": commit.commit.author.name if commit.commit.author else "unknown",
                        "date": commit.commit.author.date.strftime("%Y-%m-%d"),
                        "files_changed": files[:5],
                        "url": commit.html_url,
                        "message_lower": commit.commit.message.lower(),
                        "files_lower": files,
                    })
 
                    if len(all_commits) >= 20:
                        break
 
                except Exception as e:
                    logger.warning(f"[GITHUB] Commit parse error: {str(e)}")
                    continue
 
        except Exception as e:
            logger.error(f"[GITHUB] Error fetching commits: {e}")
 
        logger.info(f"[GITHUB] Total commits fetched: {len(all_commits)}")
 
        # Step 2 — Filter by keywords if provided
        if affected_files_or_keywords and all_commits:
            keywords = [k.lower() for k in affected_files_or_keywords.lower().split() if len(k) > 2]
            logger.info(f"[GITHUB] Filtering with keywords: {keywords}")
 
            for commit in all_commits:
                message = commit["message_lower"]
                files = commit["files_lower"]
                if any(kw in message or any(kw in f for f in files) for kw in keywords):
                    matched_commits.append(commit)
 
            logger.info(f"[GITHUB] Matched commits: {len(matched_commits)}")
 
            # If no keyword matches — return all commits anyway
            if not matched_commits:
                logger.info("[GITHUB] No keyword matches — returning all recent commits")
                matched_commits = all_commits[:10]
        else:
            matched_commits = all_commits[:10]
 
        # Clean up internal fields before returning
        clean_commits = []
        for c in matched_commits[:10]:
            clean_commits.append({
                "sha": c["sha"],
                "message": c["message"],
                "author": c["author"],
                "date": c["date"],
                "files_changed": c["files_changed"],
                "url": c["url"],
            })
 
        return json.dumps({
            "source": "github",
            "mode": "live",
            "commits_found": len(clean_commits),
            "commits": clean_commits,
        }, indent=2)
 
    except GithubException as exc:
        logger.error(f"[GITHUB] Commit fetch failed: {exc}")
        return json.dumps({
            "source": "github",
            "error": str(exc),
            "commits": []
        })
 
 
@tool
def get_recent_pull_requests(
    state: str = "all",
    keywords: str = "",
    days_back: int = 30,
) -> str:
    """
    Fetch recent pull requests from GitHub.
    """
    gh = get_github_client()
 
    if gh is None:
        return json.dumps({
            "source": "github",
            "mode": "demo",
            "prs_found": 0,
            "pull_requests": []
        }, indent=2)
 
    try:
        repo = gh.get_repo(f"{settings.github.org}/{settings.github.repo}")
        since = datetime.utcnow() - timedelta(days=days_back)
 
        prs = []
 
        for pr in repo.get_pulls(state="all", sort="updated", direction="desc"):
            if pr.updated_at < since:
                break
 
            if keywords:
                kw_lower = keywords.lower()
                pr_text = (pr.title + " " + (pr.body or "")).lower()
                if not any(kw in pr_text for kw in kw_lower.split()):
                    continue
 
            try:
                files = [f.filename for f in pr.get_files()[:5]]
            except Exception:
                files = []
 
            prs.append({
                "number": pr.number,
                "title": pr.title,
                "state": "merged" if pr.merged_at else pr.state,
                "author": pr.user.login,
                "url": pr.html_url,
                "files": files,
                "merged_at": pr.merged_at.strftime("%Y-%m-%d") if pr.merged_at else None,
                "created_at": pr.created_at.strftime("%Y-%m-%d"),
            })
 
            if len(prs) >= 10:
                break
 
        return json.dumps({
            "source": "github",
            "mode": "live",
            "prs_found": len(prs),
            "pull_requests": prs,
        }, indent=2)
 
    except GithubException as exc:
        return json.dumps({
            "source": "github",
            "error": str(exc),
            "pull_requests": []
        })
 
 
@tool
def get_recent_releases(limit: int = 5) -> str:
    """
    Fetch recent GitHub releases.
    """
    gh = get_github_client()
 
    if gh is None:
        return json.dumps({
            "source": "github",
            "mode": "demo",
            "releases": []
        }, indent=2)
 
    try:
        repo = gh.get_repo(f"{settings.github.org}/{settings.github.repo}")
        releases = []
 
        for release in repo.get_releases()[:limit]:
            releases.append({
                "tag": release.tag_name,
                "name": release.title,
                "date": release.published_at.strftime("%Y-%m-%d") if release.published_at else None,
                "url": release.html_url,
                "body_excerpt": (release.body or "")[:300],
                "prerelease": release.prerelease,
            })
 
        return json.dumps({
            "source": "github",
            "mode": "live",
            "releases": releases,
        }, indent=2)
 
    except GithubException as exc:
        return json.dumps({
            "source": "github",
            "error": str(exc),
            "releases": []
        })
 
 
# Export tools
GITHUB_TOOLS = [
    search_recent_commits,
    get_recent_pull_requests,
    get_recent_releases
]
 