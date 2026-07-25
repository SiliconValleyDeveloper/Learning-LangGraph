"""
Phase 12 · Lesson 2 — Chunking strategies

Same corpus, three strategies:
  - fixed           — simple recursive windows
  - parent_child    — retrieve small, return parent for generation
  - sentence_window — index sentence, expand neighbors for context

Interview tip: pick strategy from failure mode
  - noisy retrieval → smaller children / sentence index
  - answers lack context → parent or window expansion
  - lost-in-the-middle → fewer, better-ranked chunks (not bigger prompts)

Run:
    python Learning/12_rag_architect/02_chunking_strategies.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import (
    load_documents,
    short_preview,
    split_fixed,
    split_parent_child,
    split_sentence_window,
)


def _show(name: str, chunks: list) -> None:
    print(f"\n{name}: {len(chunks)} chunks")
    for chunk in chunks[:3]:
        src = chunk.metadata.get("source")
        print(f"  [{src}] {short_preview(chunk.page_content)}")
        if chunk.metadata.get("parent_content"):
            print(f"    parent preview: {short_preview(chunk.metadata['parent_content'], 70)}")
        if chunk.metadata.get("window_content"):
            print(f"    window preview: {short_preview(chunk.metadata['window_content'], 70)}")


def main() -> None:
    docs = load_documents()
    print(f"Loaded {len(docs)} documents.\n")
    print("Why chunking matters: too big → noisy hits; too small → lost context.")

    fixed = split_fixed(docs)
    parent_child = split_parent_child(docs)
    sentence = split_sentence_window(docs)

    _show("fixed", fixed)
    _show("parent_child", parent_child)
    _show("sentence_window", sentence)

    print("\nWhen to use which:")
    print("  fixed           — baseline demos, uniform docs")
    print("  parent_child    — precise retrieval + rich generation context")
    print("  sentence_window — FAQ / definition lookups with local context")


if __name__ == "__main__":
    main()
