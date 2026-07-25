"""Finance agent — enterprise markets + autonomous LangGraph.

See WORKFLOW.md for the locked build plan (F0–F10).
Implementation begins at F1 (Postgres + Redis).
"""

__all__ = ["__version__", "load_config"]
__version__ = "0.4.0-f4"

from projects.finance_agent.config import load_config  # noqa: E402
