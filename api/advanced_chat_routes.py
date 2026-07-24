"""HTTP API for the advanced chatbot (store/update + ask + OCR ingest)."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from projects.advanced_chatbot.config import load_config
from projects.advanced_chatbot.graph import ask
from projects.advanced_chatbot.service import ingest_bytes
from projects.advanced_chatbot.store import get_store

router = APIRouter(prefix="/api/advanced-chat", tags=["advanced-chat"])


class AskRequest(BaseModel):
    workspace_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=4000)
    web_search: bool = False


@router.get("/status")
def status() -> dict[str, Any]:
    config = load_config()
    return {
        "vector_backend": config.vector_backend,
        "ocr_provider": config.ocr_provider,
        "ollama_vision_model": config.ollama_vision_model,
        "embedding_model": config.embedding_model,
        "chat_model": config.chat_model,
        "database_configured": bool(config.database_url),
        "deepseek_ocr_configured": bool(config.deepseek_ocr_base_url),
        "phases": {
            "A_memory_rag": "ready",
            "B_pgvector": (
                "enabled"
                if config.vector_backend == "pgvector" and config.database_url
                else "ready when VECTOR_BACKEND=pgvector + DATABASE_URL"
            ),
            "C_deepseek_ocr": (
                "enabled"
                if config.ocr_provider
                in {"deepseek", "deepseek_http", "ollama", "ollama_vision", "tesseract", "local", "auto"}
                else "ready when OCR_PROVIDER=auto|tesseract|ollama_vision|deepseek_http"
            ),
            "D_deploy": (
                "enabled — API http://localhost:8001 via deploy/docker-compose.yml"
            ),
        },
    }


@router.post("/workspaces")
def create_workspace() -> dict[str, str]:
    return {"workspace_id": str(uuid.uuid4())}


@router.get("/workspaces/{workspace_id}")
def workspace_detail(workspace_id: str) -> dict[str, Any]:
    store = get_store()
    docs = store.list_documents(workspace_id)
    return {
        "workspace_id": workspace_id,
        "documents": [
            {
                "id": d.id,
                "name": d.filename,
                "bytes": d.bytes,
                "chunk_count": d.chunk_count,
                "source_type": d.source_type,
                "mime_type": d.mime_type,
            }
            for d in docs
        ],
        **store.stats(workspace_id),
    }


@router.post("/workspaces/{workspace_id}/upload")
async def upload(workspace_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read()
    try:
        record = await run_in_threadpool(
            ingest_bytes,
            workspace_id=workspace_id,
            filename=file.filename or "upload.txt",
            content=content,
            mime_type=file.content_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    store = get_store()
    return {
        "uploaded": {
            "id": record.id,
            "name": record.filename,
            "bytes": record.bytes,
            "chunk_count": record.chunk_count,
            "source_type": record.source_type,
        },
        "documents": [
            {
                "id": d.id,
                "name": d.filename,
                "bytes": d.bytes,
                "chunk_count": d.chunk_count,
                "source_type": d.source_type,
            }
            for d in store.list_documents(workspace_id)
        ],
        **store.stats(workspace_id),
    }


@router.delete("/workspaces/{workspace_id}/documents/{filename}")
def delete_document(workspace_id: str, filename: str) -> dict[str, Any]:
    store = get_store()
    try:
        store.delete_document(workspace_id, filename)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Not found: {filename}") from exc
    return {
        "deleted": filename,
        "documents": [
            {"name": d.filename, "chunk_count": d.chunk_count, "source_type": d.source_type}
            for d in store.list_documents(workspace_id)
        ],
        **store.stats(workspace_id),
    }


@router.post("/ask")
async def ask_docs(request: AskRequest) -> dict[str, Any]:
    try:
        return await run_in_threadpool(
            ask,
            request.workspace_id,
            request.question,
            web_search=request.web_search,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
