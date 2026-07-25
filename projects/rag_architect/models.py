"""Shared types for the RAG architect project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Strategy = Literal["baseline", "hybrid", "hyde", "crag", "graph", "agentic"]


@dataclass
class ChunkHit:
    chunk_id: str
    source: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AskResult:
    question: str
    strategy: Strategy
    answer: str
    sources: list[str]
    grade: str
    verified: bool
    notes: list[str]
    hits: list[ChunkHit] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "strategy": self.strategy,
            "answer": self.answer,
            "sources": self.sources,
            "grade": self.grade,
            "verified": self.verified,
            "notes": self.notes,
            "hits": [
                {
                    "chunk_id": h.chunk_id,
                    "source": h.source,
                    "score": h.score,
                    "content": h.content[:200],
                }
                for h in self.hits
            ],
        }
