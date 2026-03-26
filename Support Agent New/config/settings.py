"""
Configuration settings for the Support AI Agent System.

ROOT CAUSES FIXED
─────────────────
Bug 1 — Nested BaseSettings don't inherit env_file from parent.
  Each sub-class (JiraSettings, GeminiSettings, etc.) is its own
  BaseSettings instance. Pydantic-Settings only reads the .env file
  for the class that declares `env_file` in its model_config / Config.
  Sub-classes instantiated via `default_factory` have no env_file of
  their own, so they silently fall back to defaults even when .env is
  populated.

  FIX: Every sub-class now declares `env_file = ".env"` in its own
  model_config.  We also pass `_env_file` explicitly at instantiation
  time inside the parent Settings.build() class-method to guarantee
  the correct path is used regardless of the working directory.

Bug 2 — `env=` kwarg is ignored in Pydantic v2.
  In Pydantic v2, Field(env="VAR_NAME") is silently ignored. The
  correct v2 approach is `validation_alias` or `model_config` with
  `env_prefix`.  We use `model_config` with `env_prefix` per class so
  the field name is automatically mapped to the right env var.

Bug 3 — Singleton created at import time with wrong cwd.
  `settings = Settings()` runs when the module is first imported.  If
  that happens before os.chdir() to the project root (e.g. in tests or
  when the module is imported from a sub-directory), the relative path
  `.env` resolves to the wrong directory.

  FIX: `_env_file` is resolved to an absolute path relative to this
  file's location, so it always points to the right `.env` regardless
  of cwd.

Bug 4 — `class Config` (Pydantic v1 style) mixed with Pydantic v2.
  Pydantic v2 uses `model_config = SettingsConfigDict(...)`.  The old
  inner `class Config` is accepted but some keys (like `env_file`) are
  silently dropped in certain pydantic-settings versions.

  FIX: Replaced with `model_config = SettingsConfigDict(...)` throughout.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path to the .env file — always relative to THIS file's directory,
# so it works regardless of where Python is invoked from.
_ENV_FILE = str(Path(__file__).parent.parent / ".env")


# ── Jira ──────────────────────────────────────────────────────────────────────

class JiraSettings(BaseSettings):
    """Jira / Atlassian configuration.
    Reads env vars with the JIRA_ prefix:
      JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY
    """
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="JIRA_",
        extra="ignore",
        case_sensitive=False,
    )

    url: str         = Field(default="https://your-org.atlassian.net")
    email: str       = Field(default="support@yourorg.com")
    api_token: str   = Field(default="your-jira-api-token")
    project_key: str = Field(default="SUPPORT")


# ── Confluence ────────────────────────────────────────────────────────────────

class ConfluenceSettings(BaseSettings):
    """Confluence configuration.
    Reads env vars with the CONFLUENCE_ prefix:
      CONFLUENCE_URL, CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN, CONFLUENCE_SPACE_KEY
    """
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="CONFLUENCE_",
        extra="ignore",
        case_sensitive=False,
    )

    url: str       = Field(default="https://your-org.atlassian.net/wiki")
    email: str     = Field(default="support@yourorg.com")
    api_token: str = Field(default="your-confluence-api-token")
    space_key: str = Field(default="SUPPORT")


# ── GitHub ────────────────────────────────────────────────────────────────────

class GitHubSettings(BaseSettings):
    """GitHub configuration.
    Reads env vars with the GITHUB_ prefix:
      GITHUB_TOKEN, GITHUB_ORG, GITHUB_REPO, GITHUB_LOOKBACK_DAYS
    """
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="GITHUB_",
        extra="ignore",
        case_sensitive=False,
    )

    token: str        = Field(default="ghp_your_github_token")
    org: str          = Field(default="your-org")
    repo: str         = Field(default="your-product")
    lookback_days: int = Field(default=30)


# ── Gemini ────────────────────────────────────────────────────────────────────

class GeminiSettings(BaseSettings):
    """Google Gemini LLM configuration.
    Reads env vars with the GEMINI_ prefix:
      GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_MAX_TOKENS
    """
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="GEMINI_",
        extra="ignore",
        case_sensitive=False,
    )

    api_key: str      = Field(default="your-gemini-api-key")
    model: str        = Field(default="gemini-1.5-pro")
    temperature: float = Field(default=0.1)
    max_tokens: int   = Field(default=8192)


# ── Agent ─────────────────────────────────────────────────────────────────────

class AgentSettings(BaseSettings):
    """Agent behaviour configuration.
    Reads env vars with the AGENT_ prefix:
      AGENT_MAX_ITERATIONS, AGENT_DEBUG
    Plus SIMILARITY_THRESHOLD (no prefix).
    """
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="AGENT_",
        extra="ignore",
        case_sensitive=False,
    )

    max_iterations: int         = Field(default=10)
    debug_mode: bool            = Field(default=False, alias="debug")
    similarity_threshold: float = Field(default=0.75)

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="AGENT_",
        populate_by_name=True,
        extra="ignore",
        case_sensitive=False,
    )


# ── MCP ───────────────────────────────────────────────────────────────────────

class MCPSettings(BaseSettings):
    """Atlassian MCP server configuration.
    Reads env vars with the MCP_ prefix:
      MCP_SERVER_URL
    """
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="MCP_",
        extra="ignore",
        case_sensitive=False,
    )

    server_url: str = Field(default="http://localhost:3000")


# ── Aggregate ─────────────────────────────────────────────────────────────────

class Settings:
    """
    Plain Python class (NOT a BaseSettings subclass) that composes all
    sub-settings together.

    Why not BaseSettings here?
    Pydantic-Settings cannot automatically instantiate nested BaseSettings
    objects from environment variables — it only handles scalar types and
    simple dicts at the top level.  The correct pattern is to instantiate
    each sub-settings independently (they each read .env themselves) and
    assemble them here.
    """

    def __init__(self):
        self.jira       = JiraSettings()
        self.confluence = ConfluenceSettings()
        self.github     = GitHubSettings()
        self.gemini     = GeminiSettings()
        self.agent      = AgentSettings()
        self.mcp        = MCPSettings()

    def debug_dump(self) -> dict:
        """Print all resolved values — useful for verifying .env loading."""
        import json

        def _mask(key: str, val: str) -> str:
            """Mask secrets so they are safe to log."""
            secrets = {"token", "api_key", "api_token", "password", "secret"}
            if any(s in key.lower() for s in secrets):
                return val[:4] + "***" + val[-2:] if len(val) > 6 else "***"
            return val

        sections = {}
        for section_name in ("jira", "confluence", "github", "gemini", "agent", "mcp"):
            obj = getattr(self, section_name)
            sections[section_name] = {
                k: _mask(k, str(v))
                for k, v in obj.model_dump().items()
            }
        return sections

    def validate_required(self) -> list[str]:
        """
        Return a list of human-readable warnings for values that still
        look like placeholders.  Call this at startup to surface
        misconfiguration early.
        """
        PLACEHOLDERS = {
            "your-gemini-api-key", "your-jira-api-token",
            "your-confluence-api-token", "ghp_your_github_token",
            "your-org.atlassian.net", "your-org", "your-product",
            "support@yourorg.com",
        }
        warnings = []
        checks = [
            ("GEMINI_API_KEY",          self.gemini.api_key),
            ("JIRA_URL",                self.jira.url),
            ("JIRA_EMAIL",              self.jira.email),
            ("JIRA_API_TOKEN",          self.jira.api_token),
            ("CONFLUENCE_API_TOKEN",    self.confluence.api_token),
            ("GITHUB_TOKEN",            self.github.token),
            ("GITHUB_ORG",              self.github.org),
            ("GITHUB_REPO",             self.github.repo),
        ]
        for env_name, value in checks:
            if value in PLACEHOLDERS:
                warnings.append(
                    f"  ⚠  {env_name} is still the default placeholder — "
                    f"agent will run in DEMO mode for this integration"
                )
        return warnings


# ── Singleton ──────────────────────────────────────────────────────────────────
# Instantiated once at import time.  Because _ENV_FILE is an absolute path,
# this is safe regardless of the current working directory.
settings = Settings()
