"""Shared document, chunking, embedding, and retrieval helpers for Phase 11."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).resolve().parent / "data"
PRIVATE_DIR = DATA_DIR / "private"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
PRIVATE_EXAMPLE_NAMES = {"README.example.md"}


def _iter_markdown_files() -> list[tuple[Path, str]]:
    """Return (path, visibility) pairs for public and private markdown files."""
    files: list[tuple[Path, str]] = []
    for path in sorted(DATA_DIR.glob("*.md")):
        files.append((path, "public"))
    if PRIVATE_DIR.exists():
        for path in sorted(PRIVATE_DIR.glob("*.md")):
            if path.name in PRIVATE_EXAMPLE_NAMES:
                continue
            files.append((path, "private"))
    return files


def list_knowledge_files() -> list[dict[str, str | bool]]:
    """Describe the current knowledge-base files for the visual lab."""
    return [
        {
            "name": path.name,
            "path": str(path.relative_to(DATA_DIR)),
            "visibility": visibility,
            "private": visibility == "private",
        }
        for path, visibility in _iter_markdown_files()
    ]


def load_documents(*, include_private: bool = True) -> list[Document]:
    """Load the knowledge base with source and visibility metadata."""
    documents: list[Document] = []
    for path, visibility in _iter_markdown_files():
        if visibility == "private" and not include_private:
            continue
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={
                    "source": path.name,
                    "visibility": visibility,
                    "private": visibility == "private",
                    "relative_path": str(path.relative_to(DATA_DIR)),
                },
            )
        )
    if not documents:
        raise RuntimeError(f"No markdown documents found in {DATA_DIR}")
    return documents


def split_documents(
    documents: list[Document] | None = None,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[Document]:
    """Split documents while preserving their source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents or load_documents())


def get_embeddings() -> OllamaEmbeddings:
    """Create the local Ollama embedding client from environment settings."""
    return OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


@lru_cache(maxsize=1)
def build_vector_store() -> InMemoryVectorStore:
    """Embed the lesson chunks once and return an in-memory vector store."""
    store = InMemoryVectorStore(embedding=get_embeddings())
    store.add_documents(split_documents())
    return store


def refresh_vector_store() -> InMemoryVectorStore:
    """Clear the cached store so newly added files are indexed."""
    build_vector_store.cache_clear()
    return build_vector_store()


def retrieve(question: str, *, k: int = 3) -> list[Document]:
    """Return the top semantically similar chunks for a question."""
    return build_vector_store().similarity_search(question, k=k)


def retrieve_scored(
    question: str,
    *,
    k: int = 4,
    include_private: bool = True,
) -> list[tuple[Document, float]]:
    """Return scored chunks, optionally hiding private documents."""
    scored = build_vector_store().similarity_search_with_score(question, k=max(k * 3, k))
    filtered: list[tuple[Document, float]] = []
    for document, score in scored:
        is_private = bool(document.metadata.get("private"))
        if is_private and not include_private:
            continue
        filtered.append((document, float(score)))
        if len(filtered) >= k:
            break
    return filtered


def retrieve_filtered(
    question: str,
    *,
    k: int = 4,
    include_private: bool = True,
) -> list[Document]:
    """Retrieve chunks with an access filter for private knowledge."""
    return [document for document, _ in retrieve_scored(
        question, k=k, include_private=include_private
    )]


def format_context(documents: list[Document]) -> str:
    """Format chunks with stable source labels for a grounded prompt."""
    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        visibility = document.metadata.get("visibility", "public")
        label = f"{source} [{visibility}]"
        blocks.append(f"[{index}] Source: {label}\n{document.page_content}")
    return "\n\n".join(blocks)
