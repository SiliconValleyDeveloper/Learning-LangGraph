"""Shared helpers for Phase 12 — RAG Architect lessons."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def load_documents() -> list[Document]:
    """Load enterprise markdown docs with source metadata."""
    documents: list[Document] = []
    for path in sorted(DATA_DIR.glob("*.md")):
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": path.name, "visibility": "internal"},
            )
        )
    if not documents:
        raise RuntimeError(f"No markdown documents found in {DATA_DIR}")
    return documents


def split_fixed(
    documents: list[Document] | None = None,
    *,
    chunk_size: int = 400,
    chunk_overlap: int = 60,
) -> list[Document]:
    """Fixed-size recursive character chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents or load_documents())
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', 'doc')}::fixed::{i}"
        chunk.metadata["strategy"] = "fixed"
    return chunks


def split_parent_child(
    documents: list[Document] | None = None,
    *,
    parent_size: int = 900,
    child_size: int = 220,
    child_overlap: int = 40,
) -> list[Document]:
    """Small children for retrieval; keep parent text for generation context."""
    docs = documents or load_documents()
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size, chunk_overlap=100, add_start_index=True
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size, chunk_overlap=child_overlap, add_start_index=True
    )
    out: list[Document] = []
    for doc in docs:
        parents = parent_splitter.split_documents([doc])
        for p_i, parent in enumerate(parents):
            parent_id = f"{doc.metadata.get('source', 'doc')}::parent::{p_i}"
            children = child_splitter.split_documents([parent])
            for c_i, child in enumerate(children):
                child.metadata = {
                    **child.metadata,
                    **doc.metadata,
                    "chunk_id": f"{parent_id}::child::{c_i}",
                    "parent_id": parent_id,
                    "parent_content": parent.page_content,
                    "strategy": "parent_child",
                }
                out.append(child)
    return out


def split_sentence_window(
    documents: list[Document] | None = None,
    *,
    window: int = 1,
) -> list[Document]:
    """Index one sentence; expand with neighboring sentences at read time."""
    docs = documents or load_documents()
    out: list[Document] = []
    for doc in docs:
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", doc.page_content)
            if s.strip()
        ]
        for i, sentence in enumerate(sentences):
            left = max(0, i - window)
            right = min(len(sentences), i + window + 1)
            window_text = " ".join(sentences[left:right])
            out.append(
                Document(
                    page_content=sentence,
                    metadata={
                        **doc.metadata,
                        "chunk_id": f"{doc.metadata.get('source', 'doc')}::sent::{i}",
                        "window_content": window_text,
                        "strategy": "sentence_window",
                    },
                )
            )
    return out


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class BM25Index:
    """Minimal Okapi BM25 over in-memory chunks (no extra dependency)."""

    def __init__(self, documents: list[Document], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._docs_tokens = [tokenize(d.page_content) for d in documents]
        self.N = len(documents)
        self.avgdl = (
            sum(len(toks) for toks in self._docs_tokens) / self.N if self.N else 0.0
        )
        df: Counter[str] = Counter()
        for toks in self._docs_tokens:
            df.update(set(toks))
        self.idf = {
            term: math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query: str, *, k: int = 5) -> list[tuple[Document, float]]:
        q_tokens = tokenize(query)
        if not q_tokens or not self.documents:
            return []
        scored: list[tuple[Document, float]] = []
        for doc, toks in zip(self.documents, self._docs_tokens, strict=True):
            if not toks:
                continue
            tf = Counter(toks)
            score = 0.0
            dl = len(toks)
            for term in q_tokens:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                score += idf * (freq * (self.k1 + 1) / denom)
            if score > 0:
                scored.append((doc, float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]


def rrf_fuse(
    ranked_lists: list[list[Document]],
    *,
    k: int = 5,
    rrf_k: int = 60,
) -> list[Document]:
    """Reciprocal Rank Fusion across multiple ranked document lists."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = str(doc.metadata.get("chunk_id") or id(doc))
            docs[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [docs[key] for key, _ in ordered[:k]]


@lru_cache(maxsize=1)
def build_vector_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore(embedding=get_embeddings())
    store.add_documents(split_fixed())
    return store


def refresh_vector_store() -> InMemoryVectorStore:
    build_vector_store.cache_clear()
    return build_vector_store()


def dense_search(query: str, *, k: int = 5) -> list[tuple[Document, float]]:
    return [
        (doc, float(score))
        for doc, score in build_vector_store().similarity_search_with_score(query, k=k)
    ]


@lru_cache(maxsize=1)
def build_bm25() -> BM25Index:
    return BM25Index(split_fixed())


def sparse_search(query: str, *, k: int = 5) -> list[tuple[Document, float]]:
    return build_bm25().search(query, k=k)


def hybrid_search(query: str, *, k: int = 5) -> list[Document]:
    dense_docs = [d for d, _ in dense_search(query, k=k * 2)]
    sparse_docs = [d for d, _ in sparse_search(query, k=k * 2)]
    return rrf_fuse([dense_docs, sparse_docs], k=k)


def format_context(documents: list[Document], *, use_parent: bool = False) -> str:
    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        text = document.page_content
        if use_parent and document.metadata.get("parent_content"):
            text = str(document.metadata["parent_content"])
        elif document.metadata.get("window_content"):
            text = str(document.metadata["window_content"])
        blocks.append(f"[{index}] Source: {source}\n{text}")
    return "\n\n".join(blocks)


def short_preview(text: str, n: int = 90) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= n else flat[: n - 3] + "..."
