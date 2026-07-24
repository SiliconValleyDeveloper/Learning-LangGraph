"""HTTP routes for dynamic document upload + RAG Q&A."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from projects.doc_upload_rag.graph import ask
from projects.doc_upload_rag.store import SAMPLE_DOCS, create_workspace, get_workspace

router = APIRouter(prefix="/api/doc-rag", tags=["doc-rag"])


class AskRequest(BaseModel):
    workspace_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=4000)


@router.post("/workspaces")
def create_ws() -> dict[str, Any]:
    store = create_workspace()
    return {"workspace_id": store.workspace_id, "documents": []}


@router.get("/workspaces/{workspace_id}")
def workspace_detail(workspace_id: str) -> dict[str, Any]:
    store = get_workspace(workspace_id)
    return {
        "workspace_id": workspace_id,
        "documents": store.list_files(),
        "document_count": store.document_count,
        "chunk_count": store.chunk_count,
    }


@router.post("/workspaces/{workspace_id}/upload")
async def upload(workspace_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    store = get_workspace(workspace_id)
    content = await file.read()
    try:
        result = await run_in_threadpool(store.upsert_bytes, file.filename or "upload.txt", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "uploaded": result,
        "documents": store.list_files(),
        "document_count": store.document_count,
        "chunk_count": store.chunk_count,
    }


@router.post("/workspaces/{workspace_id}/seed")
async def seed(workspace_id: str) -> dict[str, Any]:
    """Load sample docs so you can try Q&A without preparing files."""
    store = get_workspace(workspace_id)

    def _seed() -> list[dict]:
        uploaded = []
        for name, text in SAMPLE_DOCS.items():
            uploaded.append(store.upsert_text(name, text))
        return uploaded

    uploaded = await run_in_threadpool(_seed)
    return {
        "uploaded": uploaded,
        "documents": store.list_files(),
        "document_count": store.document_count,
        "chunk_count": store.chunk_count,
    }


@router.delete("/workspaces/{workspace_id}/documents/{filename}")
def delete_document(workspace_id: str, filename: str) -> dict[str, Any]:
    store = get_workspace(workspace_id)
    try:
        store.delete_file(filename)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=f"Document not found: {filename}") from exc
    return {
        "deleted": filename,
        "documents": store.list_files(),
        "document_count": store.document_count,
        "chunk_count": store.chunk_count,
    }


@router.post("/ask")
async def ask_docs(request: AskRequest) -> dict[str, Any]:
    get_workspace(request.workspace_id)
    try:
        return await run_in_threadpool(ask, request.workspace_id, request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
