"""Abstract document + vector store (swap memory → pgvector later)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from projects.advanced_chatbot.models import ChunkHit, DocumentRecord


class DocumentVectorStore(ABC):
    """Store documents, upsert chunks, retrieve by similarity."""

    @abstractmethod
    def list_documents(self, workspace_id: str) -> list[DocumentRecord]:
        raise NotImplementedError

    @abstractmethod
    def get_document(self, workspace_id: str, filename: str) -> DocumentRecord | None:
        raise NotImplementedError

    @abstractmethod
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
        """Create or replace a document and re-index its chunks."""
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, workspace_id: str, filename: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, workspace_id: str, question: str, *, k: int = 5) -> list[ChunkHit]:
        raise NotImplementedError

    @abstractmethod
    def stats(self, workspace_id: str) -> dict:
        raise NotImplementedError
