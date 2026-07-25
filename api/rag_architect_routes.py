"""HTTP routes for the RAG Architect enterprise KB lab."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from projects.rag_architect.config import STRATEGIES, load_config
from projects.rag_architect.evaluate import run_eval
from projects.rag_architect.service import ask, ingest_seed

router = APIRouter(prefix="/api/rag-architect", tags=["rag-architect"])


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    strategy: str | None = None


class EvalRequest(BaseModel):
    strategies: list[str] | None = None


@router.get("/status")
def status() -> dict[str, Any]:
    cfg = load_config()
    try:
        info = ingest_seed(rebuild=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "strategies": list(STRATEGIES),
        "default_strategy": cfg.default_strategy,
        "embedding_model": cfg.embedding_model,
        "chat_model": cfg.chat_model,
        "index": info,
        "phases": {"seed_kb": "ready", "eval": "ready"},
    }


@router.post("/rebuild")
async def rebuild() -> dict[str, Any]:
    try:
        info = await run_in_threadpool(lambda: ingest_seed(rebuild=True))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"rebuilt": True, "index": info}


@router.post("/ask")
async def ask_kb(request: AskRequest) -> dict[str, Any]:
    strategy = request.strategy
    if strategy is not None and strategy not in STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy {strategy!r}. Choose from {list(STRATEGIES)}",
        )

    def _ask() -> dict[str, Any]:
        return ask(request.question, strategy=strategy).to_dict()

    try:
        return await run_in_threadpool(_ask)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/eval")
async def evaluate(request: EvalRequest | None = None) -> dict[str, Any]:
    strategies = (request.strategies if request else None) or ["baseline", "hybrid", "crag"]
    for strategy in strategies:
        if strategy not in STRATEGIES:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown strategy {strategy!r}. Choose from {list(STRATEGIES)}",
            )

    def _eval() -> dict[str, Any]:
        return run_eval(strategies=strategies)

    try:
        return await run_in_threadpool(_eval)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
