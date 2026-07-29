"""Standalone JSON API for the shipping MCP/multi-agent project."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from projects.shipping_logistics_agent import db, repository
from projects.shipping_logistics_agent.graph import (
    MERMAID,
    graph_topology,
    resume,
    run_prompt,
)


class RunRequest(BaseModel):
    prompt: str = Field(min_length=2, max_length=4000)
    thread_id: str | None = Field(default=None, max_length=128)
    patches: dict[str, Any] = Field(default_factory=dict)
    base_prompt: str | None = Field(default=None, max_length=4000)
    history: list[dict[str, str]] = Field(default_factory=list)


class ApprovalDecision(BaseModel):
    thread_id: str = Field(min_length=1, max_length=128)
    approve: bool
    reviewer: str = Field(min_length=1, max_length=120)
    note: str = Field(default="", max_length=1000)


app = FastAPI(
    title="Shipping Logistics MCP + Multi-Agent API",
    version="1.0.0",
    description=(
        "Prompt → supervisor/operations/pricing/risk agents → human approval "
        "→ safe PostgreSQL write."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:8010"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

GRAPH_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Shipping Multi-Agent Graph</title>
  <style>
    body { margin: 0; padding: 28px; font: 15px system-ui; color: #18303a; background: #eef5f4; }
    h1 { margin: 0 0 6px; } p { color: #587079; }
    .graph { display: grid; gap: 14px; max-width: 1100px; margin-top: 28px; }
    .row { display: flex; align-items: center; justify-content: center; gap: 12px; flex-wrap: wrap; }
    .node { min-width: 130px; padding: 14px; text-align: center; border: 2px solid #157a78;
            border-radius: 12px; background: white; box-shadow: 0 5px 16px #17434a16; }
    .agent { background: #e7f5f3; } .human { border-color: #b46b12; background: #fff3df; }
    .write { border-color: #9a3f3f; background: #fbeaea; } .json { background: #e8eefb; border-color: #496bad; }
    .intent { background: #eef6ff; border-color: #3d6ea8; }
    .arrow { color: #547078; font-size: 22px; } .down { text-align: center; font-size: 24px; }
    .branch { padding: 6px 10px; border-radius: 999px; background: #dbe9e7; font-size: 12px; }
    code { display: block; margin-top: 28px; white-space: pre-wrap; padding: 16px;
           background: #10252d; color: #c8f1e8; border-radius: 10px; }
  </style>
</head>
<body>
  <h1>Shipping logistics multi-agent workflow</h1>
  <p>Hybrid intent router (rules + Qwen + history), RAG reads, DB facts, and HITL writes.</p>
  <main class="graph">
    <div class="row"><div class="node">START</div><span class="arrow">→</span>
      <div class="node intent">Intent router<br/><small>rules · Qwen · history</small></div></div>
    <div class="down">↓ chat · rag · db · write</div>
    <div class="row"><div class="node agent">Rewrite</div><span class="arrow">→</span>
      <div class="node agent">Retrieve</div><span class="arrow">→</span>
      <div class="node agent">Rerank</div><span class="arrow">→</span>
      <div class="node agent">Grade</div><span class="arrow">→</span>
      <div class="node agent">Generate</div><span class="arrow">→</span>
      <div class="node agent">Verify / Fix</div></div>
    <div class="down">↓ db facts lane &nbsp;|&nbsp; transactional write lane</div>
    <div class="row"><div class="node agent">Operations</div><span class="arrow">→</span>
      <div class="node agent">DB answer</div><span class="arrow">→</span>
      <div class="node json">JSON response</div></div>
    <div class="row"><div class="node agent">Operations</div><span class="arrow">→</span>
      <div class="node agent">Pricing</div><span class="arrow">→</span>
      <div class="node agent">Risk</div><span class="arrow">→</span>
      <div class="node">Approval request</div><span class="arrow">→</span>
      <div class="node human">HUMAN APPROVAL</div><span class="arrow">→</span>
      <div class="node write">Approved PostgreSQL write</div><span class="arrow">→</span>
      <div class="node json">JSON response</div><span class="arrow">→</span><div class="node">END</div></div>
  </main>
  <code>GET /api/shipping/graph — topology JSON
GET /api/shipping/graph/mermaid — Mermaid source
POST /api/shipping/run — prompt execution (optional history)
POST /api/shipping/approve — human decision</code>
</body>
</html>"""


@app.get("/api/shipping/health")
def health() -> dict[str, Any]:
    try:
        postgres_ok = db.ping()
        refs = repository.list_reference_data()
    except Exception as exc:  # noqa: BLE001
        return {"status": "degraded", "postgres_ok": False, "error": str(exc)}
    return {
        "status": "ok",
        "postgres_ok": postgres_ok,
        "project": "shipping-logistics-agent",
        "human_approval": True,
        "customers": len(refs["customers"]),
        "ports": len(refs["ports"]),
    }


@app.get("/api/shipping/graph")
def graph() -> dict[str, Any]:
    """Machine-readable graph topology plus Mermaid source."""
    return graph_topology()


@app.get(
    "/api/shipping/graph/mermaid",
    response_class=PlainTextResponse,
)
def graph_mermaid() -> str:
    """Mermaid graph for visual rendering in Mermaid Live or Markdown."""
    return MERMAID


@app.get("/api/shipping/graph/view", response_class=HTMLResponse)
def graph_view() -> str:
    """Self-contained visual graph; no frontend build or CDN required."""
    return GRAPH_HTML


@app.post("/api/shipping/run")
async def run(request: RunRequest) -> dict[str, Any]:
    try:
        prompt = request.prompt
        patches = dict(request.patches or {})
        if request.base_prompt and request.patches:
            # Structured continuation: keep the original request, apply field patches.
            prompt = request.base_prompt
        elif request.base_prompt and request.prompt.strip() != request.base_prompt.strip():
            prompt = (
                f"{request.base_prompt}\n"
                f"Additional information from the user: {request.prompt}"
            )
        return await run_in_threadpool(
            run_prompt,
            prompt,
            thread_id=request.thread_id,
            parameter_patches=patches or None,
            chat_history=list(request.history or [])[:8] or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/shipping/approve")
async def approve(request: ApprovalDecision) -> dict[str, Any]:
    try:
        return await run_in_threadpool(
            resume,
            request.thread_id,
            approve=request.approve,
            reviewer=request.reviewer,
            note=request.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/shipping/approvals/{thread_id}")
def approval(thread_id: str) -> dict[str, Any]:
    data = repository.get_approval(thread_id)
    if not data:
        raise HTTPException(status_code=404, detail="Approval not found")
    return data

