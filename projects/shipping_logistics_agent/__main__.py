"""CLI for prompt runs and graph output."""

from __future__ import annotations

import argparse
import json

from projects.shipping_logistics_agent.graph import (
    MERMAID,
    resume,
    run_prompt,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Shipping logistics multi-agent"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run prompt until completion/approval")
    run_parser.add_argument("prompt")
    run_parser.add_argument("--thread-id")

    approve_parser = sub.add_parser("approve", help="Resume pending write")
    approve_parser.add_argument("thread_id")
    approve_parser.add_argument(
        "--reject", action="store_true", help="Reject instead of approve"
    )
    approve_parser.add_argument("--reviewer", default="cli-reviewer")
    approve_parser.add_argument("--note", default="")

    sub.add_parser("graph", help="Print Mermaid graph")
    args = parser.parse_args(argv)

    if args.command == "run":
        result = run_prompt(args.prompt, thread_id=args.thread_id)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "approve":
        result = resume(
            args.thread_id,
            approve=not args.reject,
            reviewer=args.reviewer,
            note=args.note,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    print(MERMAID)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

