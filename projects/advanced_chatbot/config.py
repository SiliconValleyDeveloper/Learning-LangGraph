"""Runtime config for the advanced chatbot."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env", override=False)


@dataclass(frozen=True)
class AdvancedChatConfig:
    """Environment-driven settings. Postgres is optional until you enable it."""

    vector_backend: str  # memory | pgvector
    database_url: str | None
    embedding_model: str
    embed_dims: int
    chat_model: str
    ollama_base_url: str
    ocr_provider: str  # none | ollama_vision | deepseek_http | pdf_text | auto
    deepseek_ocr_base_url: str | None
    deepseek_ocr_api_key: str | None
    deepseek_ocr_model: str
    ollama_vision_model: str
    upload_root: str


def load_config() -> AdvancedChatConfig:
    return AdvancedChatConfig(
        vector_backend=os.getenv("VECTOR_BACKEND", "memory").strip().lower(),
        database_url=os.getenv("DATABASE_URL") or None,
        embedding_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        embed_dims=int(os.getenv("EMBED_DIMS", "768")),
        chat_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ocr_provider=os.getenv("OCR_PROVIDER", "none").strip().lower(),
        deepseek_ocr_base_url=os.getenv("DEEPSEEK_OCR_BASE_URL") or None,
        deepseek_ocr_api_key=os.getenv("DEEPSEEK_OCR_API_KEY") or None,
        deepseek_ocr_model=os.getenv("DEEPSEEK_OCR_MODEL", "deepseek-ai/DeepSeek-OCR-2"),
        ollama_vision_model=os.getenv("OLLAMA_VISION_MODEL", "moondream"),
        upload_root=os.getenv("ADVANCED_UPLOAD_ROOT", ".data/advanced-uploads"),
    )
