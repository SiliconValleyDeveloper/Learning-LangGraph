"""In-memory backend — works today without Postgres."""

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

from projects.advanced_chatbot.config import AdvancedChatConfig, load_config
from projects.advanced_chatbot.models import ChunkHit, DocumentRecord
from projects.advanced_chatbot.store.base import DocumentVectorStore

ALLOWED_SUFFIXES = {".md", ".txt", ".markdown"}


def _sanitize(name: str) -> str:
    base = Path(name).name
    cleaned = re.sub(r"[^\w.\- ]+", "_", base).strip().replace(" ", "_")
    if not cleaned:
        raise ValueError("Invalid filename")
    return cleaned


@dataclass
class _Workspace:
    vector: InMemoryVectorStore
    documents: dict[str, DocumentRecord] = field(default_factory=dict)
    chunk_ids_by_file: dict[str, list[str]] = field(default_factory=dict)


class MemoryDocumentStore(DocumentVectorStore):
    def __init__(self, config: AdvancedChatConfig | None = None) -> None:
        self.config = config or load_config()
        self._lock = Lock()
        self._workspaces: dict[str, _Workspace] = {}
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
            add_start_index=True,
        )
        self._embeddings = OllamaEmbeddings(
            model=self.config.embedding_model,
            base_url=self.config.ollama_base_url,
        )
        Path(self.config.upload_root).mkdir(parents=True, exist_ok=True)

    def _ws(self, workspace_id: str) -> _Workspace:
        if workspace_id not in self._workspaces:
            self._workspaces[workspace_id] = _Workspace(
                vector=InMemoryVectorStore(embedding=self._embeddings)
            )
        return self._workspaces[workspace_id]

    def list_documents(self, workspace_id: str) -> list[DocumentRecord]:
        with self._lock:
            return sorted(self._ws(workspace_id).documents.values(), key=lambda d: d.filename)

    def get_document(self, workspace_id: str, filename: str) -> DocumentRecord | None:
        name = _sanitize(filename)
        with self._lock:
            return self._ws(workspace_id).documents.get(name)

    def upsert_document(
        self,
        *,
        workspace_id: str,
        filename: str,
        content_text: str,
        mime_type: str = "text/plain",
        source_type: str = "text",
        metadata: dict | None = None,
    ) -> DocumentRecord:
        name = _sanitize(filename)
        if not content_text.strip():
            raise ValueError("Document content is empty")

        doc = Document(
            page_content=content_text,
            metadata={"source": name, "workspace_id": workspace_id},
        )
        chunks = self._splitter.split_documents([doc])
        chunk_ids = [f"{workspace_id}::{name}::chunk::{i}" for i in range(len(chunks))]
        for chunk, chunk_id in zip(chunks, chunk_ids):
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["document_filename"] = name

        disk = Path(self.config.upload_root) / workspace_id
        disk.mkdir(parents=True, exist_ok=True)
        (disk / name).write_text(content_text, encoding="utf-8")

        with self._lock:
            ws = self._ws(workspace_id)
            old_ids = ws.chunk_ids_by_file.get(name) or []
            if old_ids:
                ws.vector.delete(ids=old_ids)
            if chunks:
                ws.vector.add_documents(chunks, ids=chunk_ids)

            existing = ws.documents.get(name)
            record = DocumentRecord(
                id=existing.id if existing else str(uuid.uuid4()),
                workspace_id=workspace_id,
                filename=name,
                mime_type=mime_type,
                source_type=source_type,
                content_text=content_text,
                bytes=len(content_text.encode("utf-8")),
                chunk_count=len(chunks),
                metadata=metadata or {},
            )
            ws.documents[name] = record
            ws.chunk_ids_by_file[name] = chunk_ids
            return record

    def delete_document(self, workspace_id: str, filename: str) -> None:
        name = _sanitize(filename)
        with self._lock:
            ws = self._ws(workspace_id)
            if name not in ws.documents:
                raise KeyError(name)
            ids = ws.chunk_ids_by_file.pop(name, [])
            if ids:
                ws.vector.delete(ids=ids)
            ws.documents.pop(name, None)
        path = Path(self.config.upload_root) / workspace_id / name
        if path.exists():
            path.unlink()

    def retrieve(self, workspace_id: str, question: str, *, k: int = 5) -> list[ChunkHit]:
        with self._lock:
            ws = self._ws(workspace_id)
            if not ws.documents:
                return []
            scored = ws.vector.similarity_search_with_score(question, k=k)
        hits: list[ChunkHit] = []
        for document, score in scored:
            hits.append(
                ChunkHit(
                    chunk_id=str(document.metadata.get("chunk_id", "")),
                    document_id="",
                    filename=str(document.metadata.get("document_filename") or document.metadata.get("source", "unknown")),
                    content=document.page_content,
                    score=float(score),
                    metadata=dict(document.metadata),
                )
            )
        return hits

    def stats(self, workspace_id: str) -> dict:
        docs = self.list_documents(workspace_id)
        return {
            "backend": "memory",
            "document_count": len(docs),
            "chunk_count": sum(d.chunk_count for d in docs),
        }
