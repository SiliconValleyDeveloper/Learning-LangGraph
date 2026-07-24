"""Postgres + pgvector backend — enable with VECTOR_BACKEND=pgvector."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from projects.advanced_chatbot.config import AdvancedChatConfig, load_config
from projects.advanced_chatbot.models import ChunkHit, DocumentRecord
from projects.advanced_chatbot.store.base import DocumentVectorStore


def _sanitize(name: str) -> str:
    base = Path(name).name
    cleaned = re.sub(r"[^\w.\- ]+", "_", base).strip().replace(" ", "_")
    if not cleaned:
        raise ValueError("Invalid filename")
    return cleaned


def _as_vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


class PgVectorDocumentStore(DocumentVectorStore):
    """Persistent store. Requires `psycopg` and a running Postgres with pgvector."""

    def __init__(self, config: AdvancedChatConfig | None = None) -> None:
        self.config = config or load_config()
        if not self.config.database_url:
            raise RuntimeError("DATABASE_URL is required for VECTOR_BACKEND=pgvector")
        try:
            import psycopg  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Install psycopg to use pgvector: pip install 'psycopg[binary]'"
            ) from exc
        self._embeddings = OllamaEmbeddings(
            model=self.config.embedding_model,
            base_url=self.config.ollama_base_url,
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
            add_start_index=True,
        )
        self._ensure_schema()

    def _connect(self):
        import psycopg

        return psycopg.connect(self.config.database_url)

    def _ensure_schema(self) -> None:
        dims = self.config.embed_dims
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    mime_type TEXT NOT NULL DEFAULT 'text/plain',
                    source_type TEXT NOT NULL DEFAULT 'text',
                    content_text TEXT NOT NULL,
                    bytes INT NOT NULL DEFAULT 0,
                    chunk_count INT NOT NULL DEFAULT 0,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (workspace_id, filename)
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    workspace_id TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({dims}),
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS chunks_workspace_idx
                ON chunks (workspace_id)
                """
            )
            # HNSW needs rows; create if missing (ignore if dims mismatch on rebuild)
            conn.execute(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_indexes WHERE indexname = 'chunks_embedding_hnsw'
                    ) THEN
                        CREATE INDEX chunks_embedding_hnsw
                        ON chunks USING hnsw (embedding vector_cosine_ops);
                    END IF;
                END $$;
                """
            )
            conn.commit()

    def list_documents(self, workspace_id: str) -> list[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, workspace_id, filename, mime_type, source_type,
                       content_text, bytes, chunk_count, metadata
                FROM documents
                WHERE workspace_id = %s
                ORDER BY filename
                """,
                (workspace_id,),
            ).fetchall()
        return [
            DocumentRecord(
                id=str(row[0]),
                workspace_id=row[1],
                filename=row[2],
                mime_type=row[3],
                source_type=row[4],
                content_text=row[5],
                bytes=row[6],
                chunk_count=row[7],
                metadata=row[8] or {},
            )
            for row in rows
        ]

    def get_document(self, workspace_id: str, filename: str) -> DocumentRecord | None:
        name = _sanitize(filename)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, workspace_id, filename, mime_type, source_type,
                       content_text, bytes, chunk_count, metadata
                FROM documents
                WHERE workspace_id = %s AND filename = %s
                """,
                (workspace_id, name),
            ).fetchone()
        if not row:
            return None
        return DocumentRecord(
            id=str(row[0]),
            workspace_id=row[1],
            filename=row[2],
            mime_type=row[3],
            source_type=row[4],
            content_text=row[5],
            bytes=row[6],
            chunk_count=row[7],
            metadata=row[8] or {},
        )

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

        chunks = self._splitter.split_text(content_text)
        vectors = self._embeddings.embed_documents(chunks) if chunks else []
        meta = metadata or {}
        existing = self.get_document(workspace_id, name)
        doc_id = existing.id if existing else str(uuid.uuid4())

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    id, workspace_id, filename, mime_type, source_type,
                    content_text, bytes, chunk_count, metadata, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, now()
                )
                ON CONFLICT (workspace_id, filename) DO UPDATE SET
                    mime_type = EXCLUDED.mime_type,
                    source_type = EXCLUDED.source_type,
                    content_text = EXCLUDED.content_text,
                    bytes = EXCLUDED.bytes,
                    chunk_count = EXCLUDED.chunk_count,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                """,
                (
                    doc_id,
                    workspace_id,
                    name,
                    mime_type,
                    source_type,
                    content_text,
                    len(content_text.encode("utf-8")),
                    len(chunks),
                    json.dumps(meta),
                ),
            )
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
            for index, (text, embedding) in enumerate(zip(chunks, vectors)):
                chunk_id = f"{workspace_id}::{name}::chunk::{index}"
                conn.execute(
                    """
                    INSERT INTO chunks (
                        id, document_id, workspace_id, chunk_index,
                        content, embedding, metadata
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s::vector, %s::jsonb
                    )
                    """,
                    (
                        chunk_id,
                        doc_id,
                        workspace_id,
                        index,
                        text,
                        _as_vector_literal(embedding),
                        json.dumps({"source": name, "start_index": index}),
                    ),
                )
            conn.commit()

        return DocumentRecord(
            id=doc_id,
            workspace_id=workspace_id,
            filename=name,
            mime_type=mime_type,
            source_type=source_type,
            content_text=content_text,
            bytes=len(content_text.encode("utf-8")),
            chunk_count=len(chunks),
            metadata=meta,
        )

    def delete_document(self, workspace_id: str, filename: str) -> None:
        name = _sanitize(filename)
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM documents WHERE workspace_id = %s AND filename = %s",
                (workspace_id, name),
            )
            if cur.rowcount == 0:
                raise KeyError(name)
            conn.commit()

    def retrieve(self, workspace_id: str, question: str, *, k: int = 5) -> list[ChunkHit]:
        query_vec = _as_vector_literal(self._embeddings.embed_query(question))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.document_id, d.filename, c.content,
                       1 - (c.embedding <=> %s::vector) AS score,
                       c.metadata
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.workspace_id = %s
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vec, workspace_id, query_vec, k),
            ).fetchall()
        return [
            ChunkHit(
                chunk_id=row[0],
                document_id=str(row[1]),
                filename=row[2],
                content=row[3],
                score=float(row[4] or 0.0),
                metadata=row[5] or {},
            )
            for row in rows
        ]

    def stats(self, workspace_id: str) -> dict:
        with self._connect() as conn:
            docs = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE workspace_id = %s",
                (workspace_id,),
            ).fetchone()[0]
            chunks = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE workspace_id = %s",
                (workspace_id,),
            ).fetchone()[0]
        return {
            "backend": "pgvector",
            "document_count": int(docs),
            "chunk_count": int(chunks),
        }
