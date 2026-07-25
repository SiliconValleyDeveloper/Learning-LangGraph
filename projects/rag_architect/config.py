"""Runtime config for the RAG architect project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
_PKG = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env", override=False)

DATA_DIR = _PKG / "data"
EVAL_DIR = _PKG / "eval"
STRATEGIES = ("baseline", "hybrid", "hyde", "crag", "graph", "agentic")


@dataclass(frozen=True)
class RagArchitectConfig:
    embedding_model: str
    chat_model: str
    ollama_base_url: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    retrieve_candidates: int
    max_crag_retries: int
    default_strategy: str
    data_dir: Path


def load_config() -> RagArchitectConfig:
    strategy = os.getenv("RAG_ARCHITECT_STRATEGY", "hybrid").strip().lower()
    if strategy not in STRATEGIES:
        strategy = "hybrid"
    return RagArchitectConfig(
        embedding_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        chat_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "400")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "60")),
        top_k=int(os.getenv("RAG_TOP_K", "4")),
        retrieve_candidates=int(os.getenv("RAG_RETRIEVE_CANDIDATES", "8")),
        max_crag_retries=int(os.getenv("RAG_CRAG_RETRIES", "2")),
        default_strategy=strategy,
        data_dir=Path(os.getenv("RAG_ARCHITECT_DATA", str(DATA_DIR))),
    )
