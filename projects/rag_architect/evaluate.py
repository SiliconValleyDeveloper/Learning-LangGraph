"""Offline eval across strategies on gold_qa.json."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from projects.rag_architect.config import EVAL_DIR, STRATEGIES
from projects.rag_architect.ingest import tokenize
from projects.rag_architect.service import ask, ingest_seed

GOLD_PATH = EVAL_DIR / "gold_qa.json"


def context_recall(context: str, must_include: list[str]) -> float:
    if not must_include:
        return 1.0
    lowered = context.lower()
    hits = sum(1 for key in must_include if key.lower() in lowered)
    return hits / len(must_include)


def faithfulness_proxy(answer: str, context: str) -> float:
    a = set(tokenize(answer))
    c = set(tokenize(context))
    if not a:
        return 0.0
    return len(a & c) / len(a)


def has_citation(answer: str) -> bool:
    return bool(re.search(r"\[\d+\]", answer))


def run_eval(
    *,
    strategies: list[str] | None = None,
    gold_path: Path | None = None,
) -> dict:
    from Learning.llm import require_ollama

    require_ollama()
    ingest_seed(rebuild=False)

    gold = json.loads((gold_path or GOLD_PATH).read_text(encoding="utf-8"))
    strategies = strategies or ["baseline", "hybrid", "crag"]
    for s in strategies:
        if s not in STRATEGIES:
            raise ValueError(f"Unknown strategy {s}")

    table: dict[str, dict[str, float]] = {}
    details: list[dict] = []

    for strategy in strategies:
        recalls: list[float] = []
        faiths: list[float] = []
        cites: list[float] = []
        for item in gold:
            result = ask(item["question"], strategy=strategy)  # type: ignore[arg-type]
            context = "\n".join(h.content for h in result.hits)
            recall = context_recall(context, item.get("must_include", []))
            faith = faithfulness_proxy(result.answer, context)
            cited = 1.0 if has_citation(result.answer) else 0.0
            recalls.append(recall)
            faiths.append(faith)
            cites.append(cited)
            details.append(
                {
                    "id": item["id"],
                    "strategy": strategy,
                    "recall": recall,
                    "faithfulness": faith,
                    "cited": bool(cited),
                    "grade": result.grade,
                    "answer": result.answer[:240],
                }
            )
        table[strategy] = {
            "context_recall": sum(recalls) / len(recalls),
            "faithfulness": sum(faiths) / len(faiths),
            "citation_rate": sum(cites) / len(cites),
        }

    return {"metrics": table, "details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rag_architect strategies")
    parser.add_argument(
        "--strategies",
        default="baseline,hybrid,crag",
        help="Comma-separated strategies",
    )
    args = parser.parse_args()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    report = run_eval(strategies=strategies)

    print(f"{'strategy':<12} {'recall':>8} {'faith':>8} {'cite':>8}")
    print("-" * 40)
    for name, metrics in report["metrics"].items():
        print(
            f"{name:<12} {metrics['context_recall']:>8.2f} "
            f"{metrics['faithfulness']:>8.2f} {metrics['citation_rate']:>8.2f}"
        )
    print("\nTip: compare baseline vs hybrid on ticket-ID questions.")


if __name__ == "__main__":
    main()
