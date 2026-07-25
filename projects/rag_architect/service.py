"""Public entrypoints: ingest seed KB and ask with a strategy."""

from __future__ import annotations

from projects.rag_architect.config import STRATEGIES, load_config
from projects.rag_architect.graph import get_graph
from projects.rag_architect.ingest import ingest_seed as _ingest_seed
from projects.rag_architect.models import AskResult, ChunkHit, Strategy


def ingest_seed(*, rebuild: bool = True) -> dict[str, int | str]:
    return _ingest_seed(rebuild=rebuild)


def ask(
    question: str,
    *,
    strategy: Strategy | str | None = None,
) -> AskResult:
    if not question.strip():
        raise ValueError("question is empty")

    cfg = load_config()
    chosen: Strategy = (strategy or cfg.default_strategy)  # type: ignore[assignment]
    if chosen not in STRATEGIES:
        raise ValueError(f"Unknown strategy {chosen!r}. Choose from {STRATEGIES}")

    # Ensure index exists before graph nodes run.
    ingest_seed(rebuild=False)

    result = get_graph().invoke(
        {
            "question": question.strip(),
            "strategy": chosen,
            "query": question.strip(),
            "hyde_passage": "",
            "hits": [],
            "grade": "",
            "retries": 0,
            "answer": "",
            "verified": False,
            "sources": [],
            "notes": [],
            "agent_plan": "",
        }
    )

    hits = result.get("hits") or []
    if hits and isinstance(hits[0], dict):
        # defensive: should be ChunkHit dataclasses
        hits = [
            ChunkHit(
                chunk_id=h.get("chunk_id", ""),
                source=h.get("source", ""),
                content=h.get("content", ""),
                score=float(h.get("score", 0)),
                metadata=h.get("metadata") or {},
            )
            for h in hits
        ]

    return AskResult(
        question=question.strip(),
        strategy=result.get("strategy") or chosen,
        answer=result.get("answer") or "",
        sources=list(result.get("sources") or []),
        grade=result.get("grade") or "",
        verified=bool(result.get("verified")),
        notes=list(result.get("notes") or []),
        hits=list(hits),
    )


def main() -> None:
    import argparse
    import json

    from Learning.llm import require_ollama

    parser = argparse.ArgumentParser(description="Contoso Ops RAG architect ask CLI")
    parser.add_argument("question", nargs="?", default="What is the P1 acknowledge time?")
    parser.add_argument(
        "--strategy",
        default=None,
        choices=list(STRATEGIES),
        help="Retrieval/generation strategy",
    )
    parser.add_argument("--rebuild", action="store_true", help="Rebuild indexes first")
    args = parser.parse_args()

    require_ollama()
    if args.rebuild:
        info = ingest_seed(rebuild=True)
        print("ingested:", info)
    else:
        ingest_seed(rebuild=False)

    out = ask(args.question, strategy=args.strategy)
    print(json.dumps(out.to_dict(), indent=2))


if __name__ == "__main__":
    main()
