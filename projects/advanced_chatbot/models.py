"""Shared types for documents and retrieval hits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentRecord:
    id: str
    workspace_id: str
    filename: str
    mime_type: str
    source_type: str  # text | ocr | mixed
    content_text: str
    bytes: int
    chunk_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkHit:
    chunk_id: str
    document_id: str
    filename: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
