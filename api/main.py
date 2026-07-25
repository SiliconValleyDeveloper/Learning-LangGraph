"""HTTP API for the LangGraph Learning visual lab."""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
LEARNING = ROOT / "Learning"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LEARNING))

from Learning.concepts import catalog  # noqa: E402
from api.advanced_chat_routes import router as advanced_chat_router  # noqa: E402
from api.doc_rag_routes import router as doc_rag_router  # noqa: E402
from api.finance_routes import router as finance_router  # noqa: E402
from api.rag_architect_routes import router as rag_architect_router  # noqa: E402


class RunRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    concept_id: str = "tools"
    thread_id: str | None = None
    web_search: bool = False


class HitlRequest(BaseModel):
    thread_id: str
    approve: bool = True


class ToolEvent(BaseModel):
    name: str
    args: dict[str, Any]
    result: str


class ChatMessage(BaseModel):
    role: str
    content: str


class StateSnapshot(BaseModel):
    message_count: int
    last_message_type: str
    memory_enabled: bool = True


class TraceStep(BaseModel):
    sequence: int
    node: str
    summary: str
    state: StateSnapshot
    edge_from: str | None = None
    edge_to: str | None = None
    decision: str | None = None
    tool_names: list[str] = Field(default_factory=list)


class GraphAnatomy(BaseModel):
    nodes: list[str]
    node_count: int
    edges: list[str]
    edge_count: int
    conditional_edges: list[str]
    conditional_count: int
    tools: list[str]
    tool_count: int
    state_keys: list[str]
    topology_nodes: list[dict[str, str]] = Field(default_factory=list)
    topology_edges: list[dict[str, str]] = Field(default_factory=list)


class ExecutionStats(BaseModel):
    steps: int
    nodes_visited: int
    agent_runs: int
    tool_node_runs: int
    edges_traversed: int
    conditional_decisions: int
    tool_calls: int
    unique_tools: int
    loops: int
    state_messages: int
    path: list[str]


class RunResponse(BaseModel):
    reply: str
    thread_id: str
    concept_id: str
    concept_title: str
    messages: list[ChatMessage]
    tool_events: list[ToolEvent]
    trace: list[TraceStep]
    state: StateSnapshot
    graph: GraphAnatomy
    stats: ExecutionStats
    interrupted: bool = False
    pending: dict[str, Any] | None = None
    state_extra: dict[str, Any] = Field(default_factory=dict)


def _ollama_health() -> dict[str, str]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as response:
            if response.status != 200:
                raise OSError(f"Unexpected Ollama status {response.status}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {exc}") from exc
    return {"status": "ok", "model": model}


def _anatomy(concept_id: str) -> GraphAnatomy:
    concept = catalog.get_concept(concept_id)
    nodes = [n["id"] for n in concept.topology_nodes]
    edges = [
        f"{e['source']} → {e['target']}" + (f" ({e['label']})" if e.get("label") else "")
        for e in concept.topology_edges
    ]
    conditional = [
        f"{e['source']} → {e['target']}"
        for e in concept.topology_edges
        if e["kind"] == "conditional"
    ]
    return GraphAnatomy(
        nodes=nodes,
        node_count=len(nodes),
        edges=edges,
        edge_count=len(concept.topology_edges),
        conditional_edges=conditional,
        conditional_count=len(conditional),
        tools=concept.tools,
        tool_count=len(concept.tools),
        state_keys=concept.state_keys,
        topology_nodes=concept.topology_nodes,
        topology_edges=concept.topology_edges,
    )


def _stats(trace: list[dict[str, Any]], tool_events: list[dict[str, Any]], state_messages: int) -> ExecutionStats:
    path = [step["node"] for step in trace]
    agentish = {
        "agent",
        "chat",
        "greet",
        "classify",
        "draft",
        "supervisor",
        "prepare",
        "review",
        "remember_profile",
        "search_agent",
        "generate_queries",
        "extract_profile",
        "reflection",
        "generate",
    }
    toolish = {
        "tools",
        "billing",
        "tech",
        "general",
        "send",
        "researcher",
        "writer",
        "make_outline",
        "write_draft",
        "internet_search",
        "research_person",
        "web_search",
        "retrieve",
        "rewrite",
        "verify",
    }
    agent_runs = sum(1 for node in path if node in agentish)
    tool_node_runs = sum(1 for node in path if node in toolish)
    conditional = sum(1 for step in trace if step.get("decision"))
    loops = sum(1 for step in trace if (step.get("edge_to") in {"agent", "supervisor"}))
    return ExecutionStats(
        steps=len(trace),
        nodes_visited=len(set(path)),
        agent_runs=agent_runs,
        tool_node_runs=tool_node_runs,
        edges_traversed=len(trace),
        conditional_decisions=conditional,
        tool_calls=len(tool_events),
        unique_tools=len({event["name"] for event in tool_events}),
        loops=loops,
        state_messages=state_messages,
        path=path,
    )


def _to_response(raw: dict[str, Any]) -> RunResponse:
    concept_id = raw["concept_id"]
    trace_raw = raw.get("trace") or []
    tool_events = raw.get("tool_events") or []
    messages = raw.get("messages") or []
    state_messages = 0
    if trace_raw:
        state_messages = trace_raw[-1]["state"].get("message_count", 0)
    state_messages = max(state_messages, len(messages))
    last_type = "None"
    memory = True
    if trace_raw:
        last_type = trace_raw[-1]["state"].get("last_message_type", "None")
        memory = trace_raw[-1]["state"].get("memory_enabled", True)

    return RunResponse(
        reply=raw["reply"],
        thread_id=raw["thread_id"],
        concept_id=concept_id,
        concept_title=raw.get("concept_title", concept_id),
        messages=[ChatMessage(**m) for m in messages],
        tool_events=[ToolEvent(**e) for e in tool_events],
        trace=[TraceStep(**t) for t in trace_raw],
        state=StateSnapshot(
            message_count=state_messages,
            last_message_type=str(last_type),
            memory_enabled=bool(memory),
        ),
        graph=_anatomy(concept_id),
        stats=_stats(trace_raw, tool_events, state_messages),
        interrupted=bool(raw.get("interrupted")),
        pending=raw.get("pending"),
        state_extra=raw.get("state_extra") or {},
    )


def _run(request: RunRequest) -> RunResponse:
    try:
        concept = catalog.get_concept(request.concept_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown concept: {request.concept_id}") from exc

    if concept.needs_ollama:
        _ollama_health()

    thread_id = request.thread_id or str(uuid.uuid4())
    raw = catalog.run_concept(
        request.concept_id,
        request.message,
        thread_id,
        web_search=request.web_search,
    )
    return _to_response(raw)


def _hitl_resume(request: HitlRequest) -> RunResponse:
    raw = catalog.run_hitl_resume(request.thread_id, request.approve)
    raw["concept_id"] = "hitl"
    raw["concept_title"] = catalog.get_concept("hitl").title
    return _to_response(raw)


app = FastAPI(title="LangGraph Learning Lab API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(doc_rag_router)
app.include_router(advanced_chat_router)
app.include_router(rag_architect_router)
app.include_router(finance_router)


@app.get("/api/health")
def health() -> dict[str, Any]:
    status = {"status": "ok", "model": os.getenv("OLLAMA_MODEL", "qwen3:8b")}
    try:
        status.update(_ollama_health())
    except HTTPException:
        status["status"] = "degraded"
    return {
        **status,
        "concepts": catalog.list_concepts(),
        "graph": _anatomy("tools").model_dump(),
    }


@app.get("/api/concepts")
def concepts() -> list[dict[str, Any]]:
    return catalog.list_concepts()


@app.get("/api/concepts/{concept_id}")
def concept_detail(concept_id: str) -> dict[str, Any]:
    try:
        items = [c for c in catalog.list_concepts() if c["id"] == concept_id]
        if not items:
            raise KeyError(concept_id)
        return {**items[0], "graph": _anatomy(concept_id).model_dump()}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown concept: {concept_id}") from exc


@app.post("/api/run", response_model=RunResponse)
async def run(request: RunRequest) -> RunResponse:
    try:
        return await run_in_threadpool(_run, request)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/chat", response_model=RunResponse)
async def chat(request: RunRequest) -> RunResponse:
    """Backward-compatible alias for /api/run (defaults to tools)."""
    return await run(request)


@app.post("/api/hitl/resume", response_model=RunResponse)
async def hitl_resume(request: HitlRequest) -> RunResponse:
    try:
        return await run_in_threadpool(_hitl_resume, request)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
