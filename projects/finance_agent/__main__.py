"""Public entry: migrate + infra status."""

from __future__ import annotations

from projects.finance_agent.migrate import main

if __name__ == "__main__":
    raise SystemExit(main())
