"""CLI for finance API consumers (F3.5).

    python -m projects.finance_agent.consumers create --name demo --tier free
    python -m projects.finance_agent.consumers list
    python -m projects.finance_agent.consumers revoke --id 1
"""

from __future__ import annotations

import argparse
import json

from projects.finance_agent import auth
from projects.finance_agent.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finance API consumers")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="Create consumer + print API key once")
    p_create.add_argument("--name", required=True)
    p_create.add_argument("--tenant", default="default")
    p_create.add_argument("--tier", choices=sorted(auth.TIER_DEFAULTS), default=None)
    p_create.add_argument(
        "--scopes",
        default=None,
        help="Comma-separated scopes (default from tier)",
    )
    p_create.add_argument("--rpm", type=int, default=None)

    p_list = sub.add_parser("list", help="List consumers (no secrets)")
    p_list.add_argument("--tenant", default=None)
    p_list.add_argument("--all", action="store_true", help="Include revoked")

    p_revoke = sub.add_parser("revoke", help="Revoke a consumer by id")
    p_revoke.add_argument("--id", type=int, required=True)

    args = parser.parse_args(argv)
    cfg = load_config()

    if args.cmd == "create":
        scopes = (
            [s.strip() for s in args.scopes.split(",") if s.strip()]
            if args.scopes
            else None
        )
        result = auth.create_consumer(
            args.name,
            tenant_id=args.tenant,
            tier=args.tier,
            scopes=scopes,
            rate_limit_rpm=args.rpm,
            config=cfg,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.cmd == "list":
        rows = auth.list_consumers(
            tenant_id=args.tenant,
            include_revoked=args.all,
            config=cfg,
        )
        print(json.dumps(rows, indent=2))
        return 0

    if args.cmd == "revoke":
        ok = auth.revoke_consumer(args.id, config=cfg)
        print(json.dumps({"revoked": ok, "id": args.id}))
        return 0 if ok else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
