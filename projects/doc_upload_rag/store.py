"""Per-workspace document store with incremental chunk upserts."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[2]
UPLOAD_ROOT = ROOT / ".data" / "doc-uploads"
ALLOWED_SUFFIXES = {".md", ".txt", ".markdown"}
MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB

SAMPLE_DOCS = {
    "onboarding.md": """# Team Onboarding

## First week
- Set up Ollama and pull `qwen3:8b` plus `nomic-embed-text`.
- Clone the LangGraph learning repo and create a Python venv.
- Run the API on port 8000 and the Angular UI on port 4200.

## Support
Pager: oncall@example.com
Slack: #platform-help
""",
    "refund_policy.md": """# Refund Policy

Customers may request a full refund within 14 days of purchase if the product was not used in production.

Partial refunds after day 14 require manager approval.
Contact billing@example.com with the order id.
""",
}


def _embeddings() -> OllamaEmbeddings:
    import os

    return OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def _sanitize_filename(name: str) -> str:
    base = Path(name).name
    cleaned = re.sub(r"[^\w.\- ]+", "_", base).strip().replace(" ", "_")
    if not cleaned:
        raise ValueError("Invalid filename")
    suffix = Path(cleaned).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(f"Unsupported type {suffix or '(none)'}. Use: {', '.join(sorted(ALLOWED_SUFFIXES))}")
    return cleaned


@dataclass
class FileRecord:
    name: str
    bytes: int
    chunk_count: int
    chunk_ids: list[str] = field(default_factory=list)


class WorkspaceStore:
    """One user's upload space: files on disk + in-memory vectors."""

    def __init__(self, workspace_id: str) -> None:
        self.workspace_id = workspace_id
        self.root = UPLOAD_ROOT / workspace_id
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._vector = InMemoryVectorStore(embedding=_embeddings())
        self._files: dict[str, FileRecord] = {}
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
            add_start_index=True,
        )

    def list_files(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "name": record.name,
                    "bytes": record.bytes,
                    "chunk_count": record.chunk_count,
                }
                for record in sorted(self._files.values(), key=lambda item: item.name)
            ]

    def upsert_text(self, filename: str, text: str) -> dict:
        name = _sanitize_filename(filename)
        raw = text.encode("utf-8")
        if len(raw) > MAX_UPLOAD_BYTES:
            raise ValueError(f"File too large (max {MAX_UPLOAD_BYTES // 1024} KB)")
        if not text.strip():
            raise ValueError("File is empty")

        path = self.root / name
        document = Document(
            page_content=text,
            metadata={"source": name, "workspace_id": self.workspace_id},
        )
        chunks = self._splitter.split_documents([document])
        chunk_ids = [f"{name}::chunk::{index}" for index in range(len(chunks))]
        for chunk, chunk_id in zip(chunks, chunk_ids):
            chunk.metadata["chunk_id"] = chunk_id

        with self._lock:
            existing = self._files.get(name)
            if existing and existing.chunk_ids:
                self._vector.delete(ids=existing.chunk_ids)
            if chunks:
                self._vector.add_documents(chunks, ids=chunk_ids)
            path.write_text(text, encoding="utf-8")
            record = FileRecord(
                name=name,
                bytes=len(raw),
                chunk_count=len(chunks),
                chunk_ids=chunk_ids,
            )
            self._files[name] = record
            return {
                "name": record.name,
                "bytes": record.bytes,
                "chunk_count": record.chunk_count,
                "replaced": existing is not None,
            }

    def upsert_bytes(self, filename: str, content: bytes) -> dict:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Only UTF-8 text files (.md, .txt) are supported") from exc
        return self.upsert_text(filename, text)

    def delete_file(self, filename: str) -> None:
        name = _sanitize_filename(filename)
        with self._lock:
            record = self._files.pop(name, None)
            if record is None:
                raise KeyError(name)
            if record.chunk_ids:
                self._vector.delete(ids=record.chunk_ids)
            path = self.root / name
            if path.exists():
                path.unlink()

    def retrieve(self, question: str, *, k: int = 4) -> list[tuple[Document, float]]:
        with self._lock:
            if not self._files:
                return []
            return [
                (document, float(score))
                for document, score in self._vector.similarity_search_with_score(question, k=k)
            ]

    @property
    def document_count(self) -> int:
        with self._lock:
            return len(self._files)

    @property
    def chunk_count(self) -> int:
        with self._lock:
            return sum(record.chunk_count for record in self._files.values())


_registry: dict[str, WorkspaceStore] = {}
_registry_lock = Lock()


def create_workspace() -> WorkspaceStore:
    workspace_id = str(uuid.uuid4())
    store = WorkspaceStore(workspace_id)
    with _registry_lock:
        _registry[workspace_id] = store
    return store


def get_workspace(workspace_id: str) -> WorkspaceStore:
    with _registry_lock:
        store = _registry.get(workspace_id)
        if store is None:
            # Recreate empty in-memory workspace if API restarted (disk files remain).
            store = WorkspaceStore(workspace_id)
            _registry[workspace_id] = store
            _hydrate_from_disk(store)
        return store


def _hydrate_from_disk(store: WorkspaceStore) -> None:
    """Re-index files that survived an API restart."""
    for path in sorted(store.root.glob("*")):
        if path.suffix.lower() not in ALLOWED_SUFFIXES or not path.is_file():
            continue
        store.upsert_text(path.name, path.read_text(encoding="utf-8"))
