"""Ingest helpers: text upload + optional DeepSeek OCR for images/PDFs."""

from __future__ import annotations

from pathlib import Path

from projects.advanced_chatbot.config import load_config
from projects.advanced_chatbot.models import DocumentRecord
from projects.advanced_chatbot.ocr import get_ocr_provider
from projects.advanced_chatbot.store import get_store

TEXT_SUFFIXES = {".md", ".txt", ".markdown"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".tif", ".tiff"}
PDF_SUFFIXES = {".pdf"}


def ingest_bytes(
    *,
    workspace_id: str,
    filename: str,
    content: bytes,
    mime_type: str | None = None,
) -> DocumentRecord:
    """Extract text (passthrough or OCR), then upsert into the active vector store."""
    suffix = Path(filename).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        text = content.decode("utf-8")
        source_type = "text"
        resolved_mime = mime_type or "text/plain"
    elif suffix in IMAGE_SUFFIXES or suffix in PDF_SUFFIXES:
        config = load_config()
        if config.ocr_provider in {"none", ""}:
            raise ValueError(
                "Image/PDF upload needs OCR. Set OCR_PROVIDER=ollama_vision "
                "(local Mac) or deepseek_http (GPU DeepSeek-OCR endpoint)."
            )
        ocr = get_ocr_provider(config)
        resolved_mime = mime_type or (
            "application/pdf" if suffix == ".pdf" else f"image/{suffix.lstrip('.')}"
        )
        if resolved_mime == "image/jpg":
            resolved_mime = "image/jpeg"
        text = ocr.extract_markdown(
            content=content, filename=filename, mime_type=resolved_mime
        )
        source_type = "ocr" if suffix in IMAGE_SUFFIXES else (
            "ocr" if "Page " not in text[:40] else "pdf_text"
        )
        # Prefer clearer labels
        if suffix in PDF_SUFFIXES:
            source_type = "pdf" if config.ocr_provider == "pdf_text" else "ocr"
        else:
            source_type = "ocr"
    else:
        raise ValueError(
            f"Unsupported type {suffix or '(none)'}. "
            f"Use text ({', '.join(sorted(TEXT_SUFFIXES))}) "
            f"or images/PDF with DeepSeek OCR enabled."
        )

    return get_store().upsert_document(
        workspace_id=workspace_id,
        filename=filename,
        content_text=text,
        mime_type=resolved_mime,
        source_type=source_type,
        metadata={"ingest": source_type},
    )
