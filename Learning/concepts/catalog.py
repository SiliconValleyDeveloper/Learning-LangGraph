"""
Learning concept catalog for the visual lab.

Each concept exposes:
  - metadata (title, teach points, sample prompts)
  - topology (nodes/edges for the SVG canvas)
  - build_graph() / run() used by the API
"""

from __future__ import annotations

import importlib.util
import sys
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

LEARNING = Path(__file__).resolve().parent.parent
ROOT = LEARNING.parent
sys.path.insert(0, str(LEARNING))


def _load(name: str, relative: str) -> ModuleType:
    path = LEARNING / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Concept:
    id: str
    title: str
    phase: str
    summary: str
    teach: list[str]
    needs_ollama: bool
    supports_chat: bool
    supports_hitl: bool
    sample_prompts: list[str]
    topology_nodes: list[dict[str, str]]
    topology_edges: list[dict[str, str]]
    tools: list[str] = field(default_factory=list)
    state_keys: list[str] = field(default_factory=list)


CONCEPTS: dict[str, Concept] = {
    "hello": Concept(
        id="hello",
        title="State · Nodes · Edges",
        phase="01_hello_graph",
        summary="The smallest LangGraph: START → node → END with shared state.",
        teach=[
            "State is a TypedDict shared by every node",
            "A node is a function that returns state updates",
            "Edges connect nodes into a runnable graph",
        ],
        needs_ollama=False,
        supports_chat=True,
        supports_hitl=False,
        sample_prompts=["Ankit", "LangGraph learner", "Say hello to Mira"],
        topology_nodes=[
            {"id": "__start__", "label": "START", "kind": "start"},
            {"id": "greet", "label": "greet", "kind": "agent"},
            {"id": "__end__", "label": "END", "kind": "end"},
        ],
        topology_edges=[
            {"id": "e1", "source": "__start__", "target": "greet", "kind": "normal", "label": ""},
            {"id": "e2", "source": "greet", "target": "__end__", "kind": "normal", "label": ""},
        ],
        state_keys=["name", "greeting"],
    ),
    "router": Concept(
        id="router",
        title="Conditional routing",
        phase="02_router",
        summary="One node decides the next branch with a conditional edge.",
        teach=[
            "Conditional edges pick the next node from state",
            "Routers can be rules or an LLM classifier",
            "Same idea powers tools_condition and multi-agent supervisors",
        ],
        needs_ollama=False,
        supports_chat=True,
        supports_hitl=False,
        sample_prompts=["I need a refund", "How do I install this?", "Just saying hi"],
        topology_nodes=[
            {"id": "__start__", "label": "START", "kind": "start"},
            {"id": "classify", "label": "classify", "kind": "agent"},
            {"id": "route", "label": "route?", "kind": "conditional"},
            {"id": "billing", "label": "billing", "kind": "tools"},
            {"id": "tech", "label": "tech", "kind": "tools"},
            {"id": "general", "label": "general", "kind": "tools"},
            {"id": "__end__", "label": "END", "kind": "end"},
        ],
        topology_edges=[
            {"id": "e1", "source": "__start__", "target": "classify", "kind": "normal", "label": ""},
            {"id": "e2", "source": "classify", "target": "route", "kind": "normal", "label": ""},
            {"id": "e3", "source": "route", "target": "billing", "kind": "conditional", "label": "billing"},
            {"id": "e4", "source": "route", "target": "tech", "kind": "conditional", "label": "tech"},
            {"id": "e5", "source": "route", "target": "general", "kind": "conditional", "label": "other"},
            {"id": "e6", "source": "billing", "target": "__end__", "kind": "normal", "label": ""},
            {"id": "e7", "source": "tech", "target": "__end__", "kind": "normal", "label": ""},
            {"id": "e8", "source": "general", "target": "__end__", "kind": "normal", "label": ""},
        ],
        state_keys=["message", "route", "reply"],
    ),
    "tools": Concept(
        id="tools",
        title="Tools · ReAct loop",
        phase="03_tools_agent",
        summary="Agent ↔ tools until the model answers without tool calls.",
        teach=[
            "bind_tools lets the LLM request tool calls",
            "ToolNode executes those calls",
            "tools_condition is the conditional edge that loops or ends",
        ],
        needs_ollama=True,
        supports_chat=True,
        supports_hitl=False,
        sample_prompts=[
            "What is (144 / 12) + 8?",
            "What time is it in Tokyo?",
            "Count the words: LangGraph makes agent loops visible",
        ],
        topology_nodes=[
            {"id": "__start__", "label": "START", "kind": "start"},
            {"id": "agent", "label": "agent", "kind": "agent"},
            {"id": "tools_condition", "label": "tools_condition", "kind": "conditional"},
            {"id": "tools", "label": "tools", "kind": "tools"},
            {"id": "__end__", "label": "END", "kind": "end"},
        ],
        topology_edges=[
            {"id": "e_start_agent", "source": "__start__", "target": "agent", "kind": "normal", "label": ""},
            {"id": "e_agent_cond", "source": "agent", "target": "tools_condition", "kind": "normal", "label": ""},
            {"id": "e_cond_tools", "source": "tools_condition", "target": "tools", "kind": "conditional", "label": "yes"},
            {"id": "e_cond_end", "source": "tools_condition", "target": "__end__", "kind": "conditional", "label": "no"},
            {"id": "e_tools_agent", "source": "tools", "target": "agent", "kind": "loop", "label": "loop"},
        ],
        tools=["calculator", "get_time", "word_count"],
        state_keys=["messages"],
    ),
    "memory": Concept(
        id="memory",
        title="Memory · Checkpoints",
        phase="04_memory",
        summary="MemorySaver + thread_id keep conversation state across turns.",
        teach=[
            "Without a checkpointer every invoke starts empty",
            "thread_id selects which checkpoint backpack to load",
            "Same thread remembers; different thread is a fresh chat",
        ],
        needs_ollama=True,
        supports_chat=True,
        supports_hitl=False,
        sample_prompts=[
            "My name is Ankit and I am learning LangGraph.",
            "What is my name?",
            "What am I learning?",
        ],
        topology_nodes=[
            {"id": "__start__", "label": "START", "kind": "start"},
            {"id": "chat", "label": "chat", "kind": "agent"},
            {"id": "checkpoint", "label": "MemorySaver", "kind": "conditional"},
            {"id": "__end__", "label": "END", "kind": "end"},
        ],
        topology_edges=[
            {"id": "e1", "source": "__start__", "target": "chat", "kind": "normal", "label": ""},
            {"id": "e2", "source": "chat", "target": "checkpoint", "kind": "normal", "label": "save"},
            {"id": "e3", "source": "checkpoint", "target": "__end__", "kind": "normal", "label": "thread_id"},
        ],
        state_keys=["messages"],
    ),
    "hitl": Concept(
        id="hitl",
        title="Human-in-the-loop",
        phase="05_hitl",
        summary="interrupt_before pauses the graph until a human approves.",
        teach=[
            "interrupt_before stops prior to a risky node",
            "A checkpointer is required so state survives the pause",
            "Resume with invoke(None) or edit state then continue",
        ],
        needs_ollama=False,
        supports_chat=True,
        supports_hitl=True,
        sample_prompts=[
            "Draft an email to alice@example.com about meeting at 3pm",
            "Email bob@example.com: ship the report tomorrow",
        ],
        topology_nodes=[
            {"id": "__start__", "label": "START", "kind": "start"},
            {"id": "draft", "label": "draft", "kind": "agent"},
            {"id": "approve", "label": "approve?", "kind": "conditional"},
            {"id": "send", "label": "send", "kind": "tools"},
            {"id": "__end__", "label": "END", "kind": "end"},
        ],
        topology_edges=[
            {"id": "e1", "source": "__start__", "target": "draft", "kind": "normal", "label": ""},
            {"id": "e2", "source": "draft", "target": "approve", "kind": "normal", "label": "interrupt"},
            {"id": "e3", "source": "approve", "target": "send", "kind": "conditional", "label": "yes"},
            {"id": "e4", "source": "send", "target": "__end__", "kind": "normal", "label": ""},
        ],
        state_keys=["to", "subject", "body", "status"],
    ),
    "multi_agent": Concept(
        id="multi_agent",
        title="Multi-agent supervisor",
        phase="06_multi_agent",
        summary="A supervisor routes work to specialist nodes, then finishes.",
        teach=[
            "Multi-agent = specialized nodes + a router supervisor",
            "Workers write into shared state",
            "Conditional edges from the supervisor choose the next worker",
        ],
        needs_ollama=True,
        supports_chat=True,
        supports_hitl=False,
        sample_prompts=[
            "Explain LangGraph memory + HITL in simple terms",
            "Summarize what conditional edges are for",
        ],
        topology_nodes=[
            {"id": "__start__", "label": "START", "kind": "start"},
            {"id": "supervisor", "label": "supervisor", "kind": "agent"},
            {"id": "route", "label": "route", "kind": "conditional"},
            {"id": "researcher", "label": "researcher", "kind": "tools"},
            {"id": "writer", "label": "writer", "kind": "tools"},
            {"id": "__end__", "label": "END", "kind": "end"},
        ],
        topology_edges=[
            {"id": "e1", "source": "__start__", "target": "supervisor", "kind": "normal", "label": ""},
            {"id": "e2", "source": "supervisor", "target": "route", "kind": "normal", "label": ""},
            {"id": "e3", "source": "route", "target": "researcher", "kind": "conditional", "label": "research"},
            {"id": "e4", "source": "route", "target": "writer", "kind": "conditional", "label": "write"},
            {"id": "e5", "source": "route", "target": "__end__", "kind": "conditional", "label": "done"},
            {"id": "e6", "source": "researcher", "target": "supervisor", "kind": "loop", "label": "back"},
            {"id": "e7", "source": "writer", "target": "supervisor", "kind": "loop", "label": "back"},
        ],
        state_keys=["task", "research_notes", "draft", "next_worker", "steps"],
    ),
    "production": Concept(
        id="production",
        title="Limits · Safe recovery",
        phase="07_production",
        summary="A bounded tool loop returns useful soft errors instead of crashing.",
        teach=[
            "recursion_limit stops runaway agent loops",
            "Tools can return an Error string as recoverable state",
            "The agent reads tool failures and explains the next safe action",
        ],
        needs_ollama=True,
        supports_chat=True,
        supports_hitl=False,
        sample_prompts=[
            "What is 10 divided by 0?",
            "Lookup the catalog key banana",
            "Lookup the catalog key langgraph",
        ],
        topology_nodes=[
            {"id": "__start__", "label": "START", "kind": "start"},
            {"id": "agent", "label": "bounded agent", "kind": "agent"},
            {"id": "tools_condition", "label": "tools_condition", "kind": "conditional"},
            {"id": "tools", "label": "safe tools", "kind": "tools"},
            {"id": "__end__", "label": "END", "kind": "end"},
        ],
        topology_edges=[
            {"id": "e1", "source": "__start__", "target": "agent", "kind": "normal", "label": ""},
            {"id": "e2", "source": "agent", "target": "tools_condition", "kind": "normal", "label": ""},
            {"id": "e3", "source": "tools_condition", "target": "tools", "kind": "conditional", "label": "tool call"},
            {"id": "e4", "source": "tools_condition", "target": "__end__", "kind": "conditional", "label": "answer"},
            {"id": "e5", "source": "tools", "target": "agent", "kind": "loop", "label": "recover"},
        ],
        tools=["safe_divide", "flaky_lookup"],
        state_keys=["messages", "recursion_limit"],
    ),
    "subgraph": Concept(
        id="subgraph",
        title="Reusable subgraphs",
        phase="08_advanced",
        summary="A parent graph delegates a focused workflow to a compiled child graph.",
        teach=[
            "A compiled child graph can be registered as a parent node",
            "Shared state lets parent and child collaborate without adapters",
            "Subgraphs keep orchestration readable and reusable",
        ],
        needs_ollama=False,
        supports_chat=True,
        supports_hitl=False,
        sample_prompts=[
            "durable agent workflows",
            "human approval in AI systems",
            "reusable LangGraph components",
        ],
        topology_nodes=[
            {"id": "__start__", "label": "START", "kind": "start"},
            {"id": "prepare", "label": "parent: prepare", "kind": "agent"},
            {"id": "make_outline", "label": "child: outline", "kind": "tools"},
            {"id": "write_draft", "label": "child: draft", "kind": "tools"},
            {"id": "review", "label": "parent: review", "kind": "agent"},
            {"id": "__end__", "label": "END", "kind": "end"},
        ],
        topology_edges=[
            {"id": "e1", "source": "__start__", "target": "prepare", "kind": "normal", "label": ""},
            {"id": "e2", "source": "prepare", "target": "make_outline", "kind": "normal", "label": "enter child"},
            {"id": "e3", "source": "make_outline", "target": "write_draft", "kind": "normal", "label": ""},
            {"id": "e4", "source": "write_draft", "target": "review", "kind": "normal", "label": "return parent"},
            {"id": "e5", "source": "review", "target": "__end__", "kind": "normal", "label": ""},
        ],
        state_keys=["topic", "outline", "draft", "review", "log"],
    ),
    "persistence": Concept(
        id="persistence",
        title="Durable SQLite memory",
        phase="08_advanced",
        summary="File-backed checkpoints preserve thread state across server restarts.",
        teach=[
            "SqliteSaver persists checkpoints outside Python memory",
            "thread_id reloads the matching durable checkpoint",
            "A database checkpointer enables restart-safe workflows",
        ],
        needs_ollama=False,
        supports_chat=True,
        supports_hitl=False,
        sample_prompts=[
            "My name is Ankit",
            "What is my name?",
            "Record another durable turn",
        ],
        topology_nodes=[
            {"id": "__start__", "label": "START", "kind": "start"},
            {"id": "load_checkpoint", "label": "SQLite: load", "kind": "conditional"},
            {"id": "remember_profile", "label": "remember profile", "kind": "agent"},
            {"id": "save_checkpoint", "label": "SQLite: save", "kind": "conditional"},
            {"id": "__end__", "label": "END", "kind": "end"},
        ],
        topology_edges=[
            {"id": "e1", "source": "__start__", "target": "load_checkpoint", "kind": "normal", "label": "thread_id"},
            {"id": "e2", "source": "load_checkpoint", "target": "remember_profile", "kind": "normal", "label": "restore"},
            {"id": "e3", "source": "remember_profile", "target": "save_checkpoint", "kind": "normal", "label": "commit"},
            {"id": "e4", "source": "save_checkpoint", "target": "__end__", "kind": "normal", "label": "durable"},
        ],
        state_keys=["message", "name", "turns", "reply"],
    ),
    "web_search": Concept(
        id="web_search",
        title="Internet search",
        phase="09_web_search",
        summary="A ReAct agent searches the live web and answers with source URLs.",
        teach=[
            "Fresh questions need a live search tool, not model memory",
            "Search results return through ToolMessage with titles, snippets, and URLs",
            "The final answer cites only sources returned by the tool",
        ],
        needs_ollama=True,
        supports_chat=True,
        supports_hitl=False,
        sample_prompts=[
            "Search the web for the latest LangGraph release",
            "What are today's major AI news stories?",
            "Find recent official Ollama announcements",
        ],
        topology_nodes=[
            {"id": "__start__", "label": "START", "kind": "start"},
            {"id": "search_agent", "label": "research agent", "kind": "agent"},
            {"id": "tools_condition", "label": "search needed?", "kind": "conditional"},
            {"id": "internet_search", "label": "internet search", "kind": "tools"},
            {"id": "__end__", "label": "END", "kind": "end"},
        ],
        topology_edges=[
            {"id": "e1", "source": "__start__", "target": "search_agent", "kind": "normal", "label": ""},
            {"id": "e2", "source": "search_agent", "target": "tools_condition", "kind": "normal", "label": ""},
            {"id": "e3", "source": "tools_condition", "target": "internet_search", "kind": "conditional", "label": "search"},
            {"id": "e4", "source": "tools_condition", "target": "__end__", "kind": "conditional", "label": "answer"},
            {"id": "e5", "source": "internet_search", "target": "search_agent", "kind": "loop", "label": "results"},
        ],
        tools=["internet_search"],
        state_keys=["messages", "search_results", "sources"],
    ),
    "person_finder": Concept(
        id="person_finder",
        title="Person Finder",
        phase="10_person_finder",
        summary=(
            "Research a person on the web, extract a structured profile, "
            "and reflect until required fields look complete."
        ),
        teach=[
            "Separate query generation, research, extraction, and reflection into nodes",
            "Targeted search queries beat one vague lookup for people research",
            "Reflection can loop back for missing schema fields before finishing",
        ],
        needs_ollama=True,
        supports_chat=True,
        supports_hitl=False,
        sample_prompts=[
            "Name: Harrison Chase\nCompany: LangChain\nRole: CEO",
            "Name: Andrej Karpathy\nNotes: Focus on public AI work",
            '{"name": "Ada Lovelace", "notes": "Historical public biography only"}',
        ],
        topology_nodes=[
            {"id": "__start__", "label": "START", "kind": "start"},
            {"id": "generate_queries", "label": "generate queries", "kind": "agent"},
            {"id": "research_person", "label": "web research", "kind": "tools"},
            {"id": "extract_profile", "label": "extract profile", "kind": "agent"},
            {"id": "reflection", "label": "reflect gaps?", "kind": "conditional"},
            {"id": "__end__", "label": "END", "kind": "end"},
        ],
        topology_edges=[
            {"id": "e1", "source": "__start__", "target": "generate_queries", "kind": "normal", "label": ""},
            {"id": "e2", "source": "generate_queries", "target": "research_person", "kind": "normal", "label": "queries"},
            {"id": "e3", "source": "research_person", "target": "extract_profile", "kind": "normal", "label": "notes"},
            {"id": "e4", "source": "extract_profile", "target": "reflection", "kind": "normal", "label": "profile"},
            {"id": "e5", "source": "reflection", "target": "research_person", "kind": "conditional", "label": "gaps"},
            {"id": "e6", "source": "reflection", "target": "__end__", "kind": "conditional", "label": "done"},
        ],
        tools=["web_search"],
        state_keys=[
            "person",
            "user_notes",
            "search_queries",
            "research_notes",
            "profile",
            "sources",
        ],
    ),
    "rag": Concept(
        id="rag",
        title="RAG · Retrieve and generate",
        phase="11_rag_llm_ecosystem",
        summary=(
            "Retrieve relevant chunks from a local knowledge base, then generate "
            "a grounded answer with source citations."
        ),
        teach=[
            "Embeddings retrieve text by semantic similarity, not exact keywords",
            "Retrieved documents travel through graph state into the generation node",
            "Grounded prompts require citations and admit when context is insufficient",
        ],
        needs_ollama=True,
        supports_chat=True,
        supports_hitl=False,
        sample_prompts=[
            "What are Acme Learning Labs core collaboration hours?",
            "Who is the manager for Ankit Rawat?",
            "What should I check first for a SEV-2 Ollama outage?",
            "Which models does the Product FAQ require for RAG?",
        ],
        topology_nodes=[
            {"id": "__start__", "label": "START", "kind": "start"},
            {"id": "retrieve", "label": "retrieve chunks", "kind": "tools"},
            {"id": "generate", "label": "grounded answer", "kind": "agent"},
            {"id": "__end__", "label": "END", "kind": "end"},
        ],
        topology_edges=[
            {"id": "e1", "source": "__start__", "target": "retrieve", "kind": "normal", "label": "question"},
            {"id": "e2", "source": "retrieve", "target": "generate", "kind": "normal", "label": "context"},
            {"id": "e3", "source": "generate", "target": "__end__", "kind": "normal", "label": "answer"},
        ],
        tools=["semantic_retrieval"],
        state_keys=["question", "documents", "sources", "answer"],
    ),
    "rag_complex": Concept(
        id="rag_complex",
        title="Complex RAG · rewrite · grade · retry",
        phase="11_rag_llm_ecosystem",
        summary=(
            "Production-style RAG loop: classify access, rewrite the query, retrieve "
            "with public/private filters, grade evidence, retry if needed, then "
            "generate and verify citations."
        ),
        teach=[
            "Query rewrite turns vague questions into better search queries",
            "Private vs public retrieval is an explicit graph decision",
            "A grader + conditional edge creates a retrieval retry loop",
            "Verify enforces citations before the answer is trusted",
        ],
        needs_ollama=True,
        supports_chat=True,
        supports_hitl=False,
        sample_prompts=[
            "What should I check first for a SEV-2 Ollama outage?",
            "Who is the manager for Ankit Rawat?",
            "Explain indexing path vs query path in our architecture notes",
            "How do I recover when retrieval returns empty context?",
        ],
        topology_nodes=[
            {"id": "__start__", "label": "START", "kind": "start"},
            {"id": "classify", "label": "classify access", "kind": "agent"},
            {"id": "rewrite", "label": "rewrite query", "kind": "agent"},
            {"id": "retrieve", "label": "retrieve", "kind": "tools"},
            {"id": "grade", "label": "grade evidence", "kind": "conditional"},
            {"id": "bump_retry", "label": "retry", "kind": "tools"},
            {"id": "generate", "label": "generate", "kind": "agent"},
            {"id": "verify", "label": "verify cites", "kind": "agent"},
            {"id": "__end__", "label": "END", "kind": "end"},
        ],
        topology_edges=[
            {"id": "e1", "source": "__start__", "target": "classify", "kind": "normal", "label": ""},
            {"id": "e2", "source": "classify", "target": "rewrite", "kind": "normal", "label": "access"},
            {"id": "e3", "source": "rewrite", "target": "retrieve", "kind": "normal", "label": "query"},
            {"id": "e4", "source": "retrieve", "target": "grade", "kind": "normal", "label": "chunks"},
            {"id": "e5", "source": "grade", "target": "generate", "kind": "conditional", "label": "pass"},
            {"id": "e6", "source": "grade", "target": "bump_retry", "kind": "conditional", "label": "fail"},
            {"id": "e7", "source": "bump_retry", "target": "rewrite", "kind": "loop", "label": "retry"},
            {"id": "e8", "source": "generate", "target": "verify", "kind": "normal", "label": "draft"},
            {"id": "e9", "source": "verify", "target": "__end__", "kind": "normal", "label": "ok"},
        ],
        tools=["semantic_retrieval", "access_filter", "evidence_grader"],
        state_keys=[
            "question",
            "rewritten_query",
            "needs_private",
            "documents",
            "grade",
            "retries",
            "answer",
            "verified",
        ],
    ),
    "doc_rag": Concept(
        id="doc_rag",
        title="Doc upload · ask",
        phase="projects/doc_upload_rag",
        summary=(
            "Upload your own .md/.txt files (or load samples). Chunks are embedded on "
            "upload with incremental upsert, then retrieve → generate answers only from "
            "your workspace."
        ),
        teach=[
            "This is a real project pattern: dynamic ingest, not a fixed lesson KB",
            "Same filename re-upload replaces only that file's chunks",
            "thread_id doubles as the upload workspace in the visual lab",
            "Open /chat/doc-rag for a dedicated upload UI, or upload here in the RAG panel",
        ],
        needs_ollama=True,
        supports_chat=True,
        supports_hitl=False,
        sample_prompts=[
            "What is the refund window?",
            "What Slack channel is used for support?",
            "What models should I pull for RAG?",
            "Who do I email for billing questions?",
        ],
        topology_nodes=[
            {"id": "__start__", "label": "START", "kind": "start"},
            {"id": "retrieve", "label": "retrieve uploads", "kind": "tools"},
            {"id": "generate", "label": "grounded answer", "kind": "agent"},
            {"id": "__end__", "label": "END", "kind": "end"},
        ],
        topology_edges=[
            {"id": "e1", "source": "__start__", "target": "retrieve", "kind": "normal", "label": "question"},
            {"id": "e2", "source": "retrieve", "target": "generate", "kind": "normal", "label": "context"},
            {"id": "e3", "source": "generate", "target": "__end__", "kind": "normal", "label": "answer"},
        ],
        tools=["semantic_retrieval", "doc_upload"],
        state_keys=["workspace_id", "question", "context", "sources", "answer"],
    ),
    "advanced_chatbot": Concept(
        id="advanced_chatbot",
        title="Advanced chat · OCR · pgvector",
        phase="projects/advanced_chatbot",
        summary=(
            "Understand the prompt first, then route: chat LLM, documents, web, or hybrid — "
            "OCR + pgvector, retrieve→rerank, grounded generate→verify."
        ),
        teach=[
            "Intent router: Hi/how are you → LLM only (ignores uploaded docs)",
            "Document questions → retrieve many chunks → rerank top-k → generate",
            "Toggle Search to force/enrich with live web results",
            "Deploy with deploy/docker-compose.yml (API on :8001)",
        ],
        needs_ollama=True,
        supports_chat=True,
        supports_hitl=False,
        sample_prompts=[
            "Hi, how are you?",
            "Summarize what this uploaded document is about",
            "Search the web: latest AWS certification exam tips",
            "How do I report a content error?",
        ],
        topology_nodes=[
            {"id": "__start__", "label": "START", "kind": "start"},
            {"id": "understand", "label": "understand intent", "kind": "conditional"},
            {"id": "chat_reply", "label": "chat (LLM)", "kind": "agent"},
            {"id": "rewrite", "label": "rewrite query", "kind": "agent"},
            {"id": "retrieve", "label": "retrieve docs", "kind": "tools"},
            {"id": "rerank", "label": "rerank chunks", "kind": "tools"},
            {"id": "web_search", "label": "web search", "kind": "tools"},
            {"id": "grade", "label": "grade evidence", "kind": "conditional"},
            {"id": "generate", "label": "generate", "kind": "agent"},
            {"id": "verify", "label": "verify", "kind": "conditional"},
            {"id": "fix", "label": "fix answer", "kind": "agent"},
            {"id": "__end__", "label": "END", "kind": "end"},
        ],
        topology_edges=[
            {"id": "e1", "source": "__start__", "target": "understand", "kind": "normal", "label": ""},
            {"id": "e2", "source": "understand", "target": "chat_reply", "kind": "conditional", "label": "chat"},
            {"id": "e3", "source": "understand", "target": "rewrite", "kind": "conditional", "label": "docs/web"},
            {"id": "e4", "source": "chat_reply", "target": "__end__", "kind": "normal", "label": ""},
            {"id": "e5", "source": "rewrite", "target": "retrieve", "kind": "conditional", "label": "docs"},
            {"id": "e6", "source": "rewrite", "target": "web_search", "kind": "conditional", "label": "web"},
            {"id": "e7", "source": "retrieve", "target": "rerank", "kind": "normal", "label": "candidates"},
            {"id": "e8", "source": "rerank", "target": "grade", "kind": "conditional", "label": "docs only"},
            {"id": "e9", "source": "rerank", "target": "web_search", "kind": "conditional", "label": "hybrid"},
            {"id": "e10", "source": "web_search", "target": "grade", "kind": "normal", "label": "hits"},
            {"id": "e11", "source": "grade", "target": "generate", "kind": "normal", "label": "pass/weak"},
            {"id": "e12", "source": "generate", "target": "verify", "kind": "normal", "label": "draft"},
            {"id": "e13", "source": "verify", "target": "__end__", "kind": "conditional", "label": "ok"},
            {"id": "e14", "source": "verify", "target": "fix", "kind": "conditional", "label": "retry"},
            {"id": "e15", "source": "fix", "target": "verify", "kind": "loop", "label": "rewrite"},
        ],
        tools=[
            "intent_router",
            "semantic_retrieval",
            "chunk_reranker",
            "internet_search",
            "evidence_grader",
            "doc_upsert",
        ],
        state_keys=[
            "workspace_id",
            "question",
            "intent",
            "route_reason",
            "use_web_search",
            "rewritten_query",
            "search_queries",
            "context",
            "web_context",
            "doc_score",
            "web_score",
            "evidence_grade",
            "rerank_backend",
            "sources",
            "web_results",
            "answer",
            "verified",
        ],
    ),
    "rag_architect": Concept(
        id="rag_architect",
        title="RAG Architect · strategies",
        phase="projects/rag_architect",
        summary=(
            "Enterprise Contoso Ops KB lab: compare baseline, hybrid, HyDE, CRAG, "
            "Graph RAG, and agentic strategies with citations and eval. "
            "Open /chat/rag-architect for the dedicated strategy UI."
        ),
        teach=[
            "Interview frame: knowledge → retrieval → validation layers",
            "Hybrid (dense + BM25 + RRF) recovers ticket IDs better than dense alone",
            "CRAG grades evidence and retries; Graph expands entity hops",
            "Use the dedicated page to switch strategies and run offline eval",
        ],
        needs_ollama=True,
        supports_chat=True,
        supports_hitl=False,
        sample_prompts=[
            "What ticket code is used for leave requests?",
            "What is the P1 acknowledge time?",
            "For a P1, what PagerDuty service do we page?",
            "How long does a prod-break-glass session last?",
        ],
        topology_nodes=[
            {"id": "__start__", "label": "START", "kind": "start"},
            {"id": "choose_strategy", "label": "choose strategy", "kind": "conditional"},
            {"id": "retrieve", "label": "retrieve", "kind": "tools"},
            {"id": "grade", "label": "grade", "kind": "conditional"},
            {"id": "rewrite", "label": "rewrite", "kind": "agent"},
            {"id": "generate", "label": "generate", "kind": "agent"},
            {"id": "verify", "label": "verify", "kind": "conditional"},
            {"id": "__end__", "label": "END", "kind": "end"},
        ],
        topology_edges=[
            {"id": "e1", "source": "__start__", "target": "choose_strategy", "kind": "normal", "label": ""},
            {"id": "e2", "source": "choose_strategy", "target": "retrieve", "kind": "normal", "label": "strategy"},
            {"id": "e3", "source": "retrieve", "target": "grade", "kind": "normal", "label": "hits"},
            {"id": "e4", "source": "grade", "target": "generate", "kind": "conditional", "label": "pass"},
            {"id": "e5", "source": "grade", "target": "rewrite", "kind": "conditional", "label": "CRAG fail"},
            {"id": "e6", "source": "rewrite", "target": "retrieve", "kind": "loop", "label": "retry"},
            {"id": "e7", "source": "generate", "target": "verify", "kind": "normal", "label": "draft"},
            {"id": "e8", "source": "verify", "target": "__end__", "kind": "normal", "label": "answer"},
        ],
        tools=["hybrid_retrieval", "hyde", "crag_grader", "graph_hops", "offline_eval"],
        state_keys=["question", "strategy", "hits", "grade", "answer", "notes", "verified"],
    ),
    "mcp_agent": Concept(
        id="mcp_agent",
        title="MCP · LangGraph · LLM",
        phase="13_mcp_langgraph",
        summary=(
            "Load tools from an MCP server (stdio demo market), convert them with "
            "langchain-mcp-adapters, then run the Phase-3 ReAct loop with Ollama."
        ),
        teach=[
            "MCP server exposes tools; adapters turn them into LangChain tools",
            "Same agent ⇄ ToolNode loop as Phase 3 — only the tool source changes",
            "stdio MCP matches how Cursor launches servers (command + args)",
            "Optional: point the client at Yahoo Finance MCP via npx",
        ],
        needs_ollama=True,
        supports_chat=True,
        supports_hitl=False,
        sample_prompts=[
            "What is the demo quote for TCS.NS?",
            "List supported tickers and give AAPL's price",
            "What is the demo USDINR rate?",
        ],
        topology_nodes=[
            {"id": "__start__", "label": "START", "kind": "start"},
            {"id": "agent", "label": "agent (LLM)", "kind": "agent"},
            {"id": "tools_condition", "label": "tools?", "kind": "conditional"},
            {"id": "tools", "label": "MCP tools", "kind": "tools"},
            {"id": "__end__", "label": "END", "kind": "end"},
        ],
        topology_edges=[
            {"id": "e1", "source": "__start__", "target": "agent", "kind": "normal", "label": ""},
            {"id": "e2", "source": "agent", "target": "tools_condition", "kind": "normal", "label": ""},
            {"id": "e3", "source": "tools_condition", "target": "tools", "kind": "conditional", "label": "yes"},
            {"id": "e4", "source": "tools_condition", "target": "__end__", "kind": "conditional", "label": "no"},
            {"id": "e5", "source": "tools", "target": "agent", "kind": "loop", "label": "loop"},
        ],
        tools=["list_tickers", "get_quote", "fx_rate"],
        state_keys=["messages"],
    ),
}


def list_concepts() -> list[dict[str, Any]]:
    return [
        {
            "id": c.id,
            "title": c.title,
            "phase": c.phase,
            "summary": c.summary,
            "teach": c.teach,
            "needs_ollama": c.needs_ollama,
            "supports_hitl": c.supports_hitl,
            "sample_prompts": c.sample_prompts,
            "tools": c.tools,
            "state_keys": c.state_keys,
            "topology_nodes": c.topology_nodes,
            "topology_edges": c.topology_edges,
            "node_count": len({n["id"] for n in c.topology_nodes}),
            "edge_count": len(c.topology_edges),
            "conditional_count": sum(1 for e in c.topology_edges if e["kind"] == "conditional"),
            "tool_count": len(c.tools),
        }
        for c in CONCEPTS.values()
    ]


def get_concept(concept_id: str) -> Concept:
    if concept_id not in CONCEPTS:
        raise KeyError(concept_id)
    return CONCEPTS[concept_id]


@lru_cache(maxsize=1)
def _tools_module() -> ModuleType:
    return _load("learn_tools", "03_tools_agent/02_react_agent.py")


@lru_cache(maxsize=1)
def _memory_module() -> ModuleType:
    return _load("learn_memory", "04_memory/01_checkpoint_memory.py")


@lru_cache(maxsize=1)
def _hitl_module() -> ModuleType:
    return _load("learn_hitl", "05_hitl/01_interrupt_before.py")


@lru_cache(maxsize=1)
def _multi_module() -> ModuleType:
    return _load("learn_multi", "06_multi_agent/01_supervisor.py")


@lru_cache(maxsize=1)
def _production_module() -> ModuleType:
    return _load("learn_production", "07_production/01_limits_errors.py")


@lru_cache(maxsize=1)
def _subgraph_module() -> ModuleType:
    return _load("learn_subgraph", "08_advanced/01_subgraph.py")


@lru_cache(maxsize=1)
def _persistence_module() -> ModuleType:
    return _load("learn_persistence", "08_advanced/02_sqlite_checkpoint.py")


@lru_cache(maxsize=1)
def _web_search_module() -> ModuleType:
    return _load("learn_web_search", "09_web_search/01_search_agent.py")


@lru_cache(maxsize=1)
def _person_finder_module() -> ModuleType:
    return _load("learn_person_finder", "10_person_finder/01_person_finder.py")


@lru_cache(maxsize=1)
def _rag_module() -> ModuleType:
    return _load("learn_rag", "11_rag_llm_ecosystem/05_rag_graph.py")


@lru_cache(maxsize=1)
def _rag_complex_module() -> ModuleType:
    return _load("learn_rag_complex", "11_rag_llm_ecosystem/06_complex_rag_graph.py")


@lru_cache(maxsize=1)
def _rag_helpers_module() -> ModuleType:
    return _load("learn_rag_helpers", "11_rag_llm_ecosystem/rag_helpers.py")


@lru_cache(maxsize=1)
def tools_graph():
    return _tools_module().build_graph(checkpointer=MemorySaver())


@lru_cache(maxsize=1)
def memory_graph():
    return _memory_module().build_graph()


@lru_cache(maxsize=1)
def hitl_graph():
    return _hitl_module().build_graph()


@lru_cache(maxsize=1)
def multi_graph():
    return _multi_module().build_graph()


@lru_cache(maxsize=1)
def production_graph():
    return _production_module().build_graph()


@lru_cache(maxsize=1)
def subgraph_graph():
    return _subgraph_module().build_graph()


@lru_cache(maxsize=1)
def persistence_resources():
    database = ROOT / ".data" / "lab-checkpoints.db"
    return _persistence_module().open_graph(database)


@lru_cache(maxsize=1)
def web_search_graph():
    return _web_search_module().build_graph(checkpointer=MemorySaver())


@lru_cache(maxsize=1)
def person_finder_graph():
    # No checkpointer: each research is independent. Reusing a thread_id with
    # append reducers (research_notes/sources) previously leaked old people.
    return _person_finder_module().build_graph(checkpointer=None)


@lru_cache(maxsize=1)
def rag_graph():
    return _rag_module().build_graph()


@lru_cache(maxsize=1)
def rag_complex_graph():
    return _rag_complex_module().build_graph()


def _content_text(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


def run_hello(message: str, thread_id: str) -> dict[str, Any]:
    name = message.strip() or "friend"
    greeting = f"Hello, {name}! This node only updated shared state."
    path = ["greet"]
    return {
        "reply": greeting,
        "thread_id": thread_id,
        "messages": [
            {"role": "user", "content": message},
            {"role": "assistant", "content": greeting},
        ],
        "tool_events": [],
        "interrupted": False,
        "pending": None,
        "trace": [
            {
                "sequence": 1,
                "node": "greet",
                "summary": f"Built greeting for {name!r}",
                "edge_from": "__start__",
                "edge_to": "__end__",
                "decision": None,
                "tool_names": [],
                "state": {
                    "message_count": 1,
                    "last_message_type": "greeting",
                    "memory_enabled": False,
                },
            }
        ],
        "path": path,
        "state_extra": {"name": name, "greeting": greeting},
    }


def run_router(message: str, thread_id: str) -> dict[str, Any]:
    text = message.lower()
    if any(word in text for word in ("refund", "invoice", "payment", "bill")):
        route, node, reply = "billing", "billing", "Billing desk: I can help with charges and refunds."
    elif any(word in text for word in ("install", "error", "bug", "crash", "api")):
        route, node, reply = "tech", "tech", "Tech desk: send logs and I will debug with you."
    else:
        route, node, reply = "general", "general", "General desk: happy to help with anything else."

    return {
        "reply": f"[{route}] {reply}",
        "thread_id": thread_id,
        "messages": [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"[{route}] {reply}"},
        ],
        "tool_events": [],
        "interrupted": False,
        "pending": None,
        "trace": [
            {
                "sequence": 1,
                "node": "classify",
                "summary": f"Classified as {route}",
                "edge_from": "__start__",
                "edge_to": "route",
                "decision": None,
                "tool_names": [],
                "state": {
                    "message_count": 1,
                    "last_message_type": "classify",
                    "memory_enabled": False,
                },
            },
            {
                "sequence": 2,
                "node": node,
                "summary": f"Routed to {node}",
                "edge_from": "route",
                "edge_to": node,
                "decision": f"route → {route}",
                "tool_names": [],
                "state": {
                    "message_count": 1,
                    "last_message_type": node,
                    "memory_enabled": False,
                },
            },
        ],
        "path": ["classify", node],
        "state_extra": {"route": route, "reply": reply},
    }


def run_tools(message: str, thread_id: str) -> dict[str, Any]:
    graph = tools_graph()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 12}
    previous = graph.get_state(config).values
    message_count = len(previous.get("messages", [])) + 1
    pending_calls: dict[str, tuple[str, dict[str, Any]]] = {}
    tool_events: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    previous_node = "__start__"

    for update in graph.stream(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        stream_mode="updates",
    ):
        for node, payload in update.items():
            new_messages = payload.get("messages", []) if payload else []
            message_count += len(new_messages)
            summary = f"{node} updated state"
            last_type = "unknown"
            decision = None
            tool_names: list[str] = []
            for msg in new_messages:
                last_type = msg.__class__.__name__
                if isinstance(msg, AIMessage):
                    calls = msg.tool_calls or []
                    tool_names = [call["name"] for call in calls]
                    for call in calls:
                        pending_calls[call["id"]] = (call["name"], call["args"])
                    if calls:
                        decision = "tools_condition → tools"
                        summary = f"Requested {', '.join(tool_names)}"
                    else:
                        decision = "tools_condition → __end__"
                        summary = "Produced final answer"
                elif isinstance(msg, ToolMessage):
                    name, args = pending_calls.get(
                        msg.tool_call_id, (msg.name or "tool", {})
                    )
                    result = _content_text(msg.content)
                    tool_events.append({"name": name, "args": args, "result": result})
                    tool_names.append(name)
                    summary = f"{name} returned {result}"

            edge_to = None
            if decision and decision.endswith("tools"):
                edge_to = "tools"
            elif decision and decision.endswith("__end__"):
                edge_to = "__end__"
            if node == "tools":
                edge_to = "agent"

            trace.append(
                {
                    "sequence": len(trace) + 1,
                    "node": node,
                    "summary": summary,
                    "edge_from": previous_node,
                    "edge_to": edge_to,
                    "decision": decision,
                    "tool_names": tool_names,
                    "state": {
                        "message_count": message_count,
                        "last_message_type": last_type,
                        "memory_enabled": True,
                    },
                }
            )
            previous_node = node

    result = graph.get_state(config).values
    all_messages = result["messages"]
    messages = []
    for msg in all_messages:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": _content_text(msg.content)})
        elif isinstance(msg, AIMessage) and msg.content:
            messages.append(
                {"role": "assistant", "content": _content_text(msg.content)}
            )
    reply = _content_text(all_messages[-1].content)
    return {
        "reply": reply,
        "thread_id": thread_id,
        "messages": messages,
        "tool_events": tool_events,
        "interrupted": False,
        "pending": None,
        "trace": trace,
        "path": [step["node"] for step in trace],
        "state_extra": {"message_count": len(all_messages)},
    }


def run_memory(message: str, thread_id: str) -> dict[str, Any]:
    graph = memory_graph()
    config = {"configurable": {"thread_id": thread_id}}
    before = len(graph.get_state(config).values.get("messages", []))
    result = graph.invoke({"messages": [HumanMessage(content=message)]}, config)
    all_messages = result["messages"]
    reply = _content_text(all_messages[-1].content)
    messages = []
    for msg in all_messages:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": _content_text(msg.content)})
        elif isinstance(msg, AIMessage) and msg.content:
            messages.append(
                {"role": "assistant", "content": _content_text(msg.content)}
            )
    return {
        "reply": reply,
        "thread_id": thread_id,
        "messages": messages,
        "tool_events": [],
        "interrupted": False,
        "pending": None,
        "trace": [
            {
                "sequence": 1,
                "node": "chat",
                "summary": f"Checkpointed thread {thread_id[:8]}… ({before + 2} msgs)",
                "edge_from": "__start__",
                "edge_to": "checkpoint",
                "decision": None,
                "tool_names": [],
                "state": {
                    "message_count": len(all_messages),
                    "last_message_type": "AIMessage",
                    "memory_enabled": True,
                },
            }
        ],
        "path": ["chat"],
        "state_extra": {
            "message_count": len(all_messages),
            "thread_id": thread_id,
            "remembered_turns": len(all_messages) // 2,
        },
    }


def _parse_email_prompt(message: str) -> dict[str, str]:
    text = message.strip()
    to = "alice@example.com"
    subject = "Hello from LangGraph"
    body = text
    lower = text.lower()
    if " to " in lower and "@" in text:
        # naive: "... to name@x.com ..."
        parts = text.replace(",", " ").split()
        for token in parts:
            if "@" in token:
                to = token.strip(".:;")
                break
    if "about " in lower:
        subject = text[lower.index("about ") + 6 :].strip().capitalize()[:80] or subject
    elif ":" in text:
        subject = text.split(":", 1)[-1].strip()[:80] or subject
    return {
        "to": to,
        "subject": subject or "Hello from LangGraph",
        "body": body,
        "status": "",
    }


def run_hitl_start(message: str, thread_id: str) -> dict[str, Any]:
    graph = hitl_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial = _parse_email_prompt(message)
    paused = graph.invoke(initial, config)
    snap = graph.get_state(config)
    pending = {
        "to": paused.get("to", initial["to"]),
        "subject": paused.get("subject", initial["subject"]),
        "body": paused.get("body", initial["body"]),
        "status": paused.get("status", "drafted"),
        "next": list(snap.next) if snap.next else ["send"],
    }
    reply = (
        "Paused before send (HITL).\n"
        f"To: {pending['to']}\n"
        f"Subject: {pending['subject']}\n\n"
        f"{pending['body']}\n\n"
        "Approve to run the send node, or reject to stop."
    )
    return {
        "reply": reply,
        "thread_id": thread_id,
        "messages": [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ],
        "tool_events": [],
        "interrupted": True,
        "pending": pending,
        "trace": [
            {
                "sequence": 1,
                "node": "draft",
                "summary": "Drafted email body",
                "edge_from": "__start__",
                "edge_to": "approve",
                "decision": "interrupt_before → send",
                "tool_names": [],
                "state": {
                    "message_count": 1,
                    "last_message_type": "draft",
                    "memory_enabled": True,
                },
            }
        ],
        "path": ["draft"],
        "state_extra": pending,
    }


def run_hitl_resume(thread_id: str, approve: bool) -> dict[str, Any]:
    graph = hitl_graph()
    config = {"configurable": {"thread_id": thread_id}}
    snap = graph.get_state(config)
    if not snap.next:
        return {
            "reply": "Nothing is waiting for approval on this thread.",
            "thread_id": thread_id,
            "messages": [],
            "tool_events": [],
            "interrupted": False,
            "pending": None,
            "trace": [],
            "path": [],
            "state_extra": {},
        }

    if not approve:
        return {
            "reply": "Rejected. Send node was not executed.",
            "thread_id": thread_id,
            "messages": [{"role": "assistant", "content": "Rejected. Send node was not executed."}],
            "tool_events": [],
            "interrupted": False,
            "pending": None,
            "trace": [
                {
                    "sequence": 1,
                    "node": "approve",
                    "summary": "Human rejected — stopped before send",
                    "edge_from": "draft",
                    "edge_to": None,
                    "decision": "reject",
                    "tool_names": [],
                    "state": {
                        "message_count": 1,
                        "last_message_type": "rejected",
                        "memory_enabled": True,
                    },
                }
            ],
            "path": ["draft"],
            "state_extra": {"status": "rejected"},
        }

    final = graph.invoke(None, config)
    reply = (
        f"Approved → send ran.\n"
        f"Status: {final.get('status')}\n"
        f"To: {final.get('to')}\n"
        f"Subject: {final.get('subject')}"
    )
    return {
        "reply": reply,
        "thread_id": thread_id,
        "messages": [{"role": "assistant", "content": reply}],
        "tool_events": [],
        "interrupted": False,
        "pending": None,
        "trace": [
            {
                "sequence": 1,
                "node": "send",
                "summary": "Human approved — email mock-sent",
                "edge_from": "approve",
                "edge_to": "__end__",
                "decision": "approve → send",
                "tool_names": [],
                "state": {
                    "message_count": 1,
                    "last_message_type": "sent",
                    "memory_enabled": True,
                },
            }
        ],
        "path": ["draft", "send"],
        "state_extra": {
            "status": final.get("status"),
            "to": final.get("to"),
            "subject": final.get("subject"),
        },
    }


def run_multi_agent(message: str, thread_id: str) -> dict[str, Any]:
    graph = multi_graph()
    trace: list[dict[str, Any]] = []
    previous = "__start__"
    final_state: dict[str, Any] = {}

    for update in graph.stream(
        {
            "task": message,
            "research_notes": "",
            "draft": "",
            "next_worker": "",
            "steps": 0,
        },
        config={"recursion_limit": 12},
        stream_mode="updates",
    ):
        for node, payload in update.items():
            payload = payload or {}
            final_state.update(payload)
            decision = None
            summary = f"{node} updated team state"
            if node == "supervisor":
                nxt = payload.get("next_worker", "")
                decision = f"route → {nxt}"
                summary = f"Supervisor chose {nxt}"
            elif node == "researcher":
                summary = "Researcher filled notes"
            elif node == "writer":
                summary = "Writer produced draft"
            edge_to = payload.get("next_worker") if node == "supervisor" else "supervisor"
            if edge_to == "done":
                edge_to = "__end__"
            trace.append(
                {
                    "sequence": len(trace) + 1,
                    "node": node,
                    "summary": summary,
                    "edge_from": previous,
                    "edge_to": edge_to,
                    "decision": decision,
                    "tool_names": [],
                    "state": {
                        "message_count": payload.get("steps", final_state.get("steps", 0)),
                        "last_message_type": node,
                        "memory_enabled": False,
                    },
                }
            )
            previous = node

    draft = final_state.get("draft") or "No draft produced."
    reply = draft
    return {
        "reply": reply,
        "thread_id": thread_id,
        "messages": [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ],
        "tool_events": [],
        "interrupted": False,
        "pending": None,
        "trace": trace,
        "path": [step["node"] for step in trace],
        "state_extra": {
            "steps": final_state.get("steps"),
            "next_worker": final_state.get("next_worker"),
            "has_notes": bool(final_state.get("research_notes")),
            "has_draft": bool(final_state.get("draft")),
        },
    }


def run_production(message: str, thread_id: str) -> dict[str, Any]:
    graph = production_graph()
    all_messages: list[Any] = [HumanMessage(content=message)]
    pending_calls: dict[str, tuple[str, dict[str, Any]]] = {}
    tool_events: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    previous_node = "__start__"

    for update in graph.stream(
        {"messages": all_messages},
        config={"recursion_limit": 8},
        stream_mode="updates",
    ):
        for node, payload in update.items():
            new_messages = payload.get("messages", []) if payload else []
            all_messages.extend(new_messages)
            summary = f"{node} updated state"
            decision = None
            tool_names: list[str] = []

            for msg in new_messages:
                if isinstance(msg, AIMessage):
                    calls = msg.tool_calls or []
                    tool_names = [call["name"] for call in calls]
                    for call in calls:
                        pending_calls[call["id"]] = (call["name"], call["args"])
                    if calls:
                        decision = "tools_condition → tools"
                        summary = f"Requested {', '.join(tool_names)}"
                    else:
                        decision = "tools_condition → __end__"
                        summary = "Recovered with a final answer"
                elif isinstance(msg, ToolMessage):
                    name, args = pending_calls.get(
                        msg.tool_call_id, (msg.name or "tool", {})
                    )
                    result = _content_text(msg.content)
                    tool_events.append({"name": name, "args": args, "result": result})
                    tool_names.append(name)
                    summary = (
                        f"{name} returned a recoverable error"
                        if result.startswith("Error:")
                        else f"{name} returned {result}"
                    )

            edge_to = "agent" if node == "tools" else None
            if decision == "tools_condition → tools":
                edge_to = "tools"
            elif decision == "tools_condition → __end__":
                edge_to = "__end__"
            trace.append(
                {
                    "sequence": len(trace) + 1,
                    "node": node,
                    "summary": summary,
                    "edge_from": previous_node,
                    "edge_to": edge_to,
                    "decision": decision,
                    "tool_names": tool_names,
                    "state": {
                        "message_count": len(all_messages),
                        "last_message_type": (
                            new_messages[-1].__class__.__name__
                            if new_messages
                            else node
                        ),
                        "memory_enabled": False,
                    },
                }
            )
            previous_node = node

    reply = next(
        (
            _content_text(msg.content)
            for msg in reversed(all_messages)
            if isinstance(msg, AIMessage) and msg.content
        ),
        "The bounded graph finished without a text response.",
    )
    return {
        "reply": reply,
        "thread_id": thread_id,
        "messages": [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ],
        "tool_events": tool_events,
        "interrupted": False,
        "pending": None,
        "trace": trace,
        "path": [step["node"] for step in trace],
        "state_extra": {
            "recursion_limit": 8,
            "soft_errors": sum(
                event["result"].startswith("Error:") for event in tool_events
            ),
        },
    }


def run_subgraph(message: str, thread_id: str) -> dict[str, Any]:
    module = _subgraph_module()
    result = subgraph_graph().invoke(module.initial_state(message))
    nodes = [
        ("prepare", "Parent normalized the topic", "make_outline"),
        ("make_outline", "Child graph created an outline", "write_draft"),
        ("write_draft", "Child graph produced the draft", "review"),
        ("review", result["review"], "__end__"),
    ]
    trace = [
        {
            "sequence": index,
            "node": node,
            "summary": summary,
            "edge_from": "__start__" if index == 1 else nodes[index - 2][0],
            "edge_to": edge_to,
            "decision": None,
            "tool_names": [],
            "state": {
                "message_count": index,
                "last_message_type": node,
                "memory_enabled": False,
            },
        }
        for index, (node, summary, edge_to) in enumerate(nodes, start=1)
    ]
    reply = f"{result['draft']}\n\n{result['review']}"
    return {
        "reply": reply,
        "thread_id": thread_id,
        "messages": [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ],
        "tool_events": [],
        "interrupted": False,
        "pending": None,
        "trace": trace,
        "path": [step["node"] for step in trace],
        "state_extra": {
            "outline": result["outline"],
            "composition_log": result["log"],
            "child_nodes": ["make_outline", "write_draft"],
        },
    }


def run_persistence(message: str, thread_id: str) -> dict[str, Any]:
    graph, _connection = persistence_resources()
    config = {"configurable": {"thread_id": thread_id}}
    previous = graph.get_state(config).values
    result = graph.invoke({"message": message}, config)
    turns = result.get("turns", 0)
    had_checkpoint = bool(previous)
    trace_nodes = [
        (
            "load_checkpoint",
            "Loaded existing SQLite checkpoint"
            if had_checkpoint
            else "No prior checkpoint; started a durable thread",
            "remember_profile",
        ),
        ("remember_profile", result["reply"], "save_checkpoint"),
        ("save_checkpoint", f"Committed durable turn {turns}", "__end__"),
    ]
    trace = [
        {
            "sequence": index,
            "node": node,
            "summary": summary,
            "edge_from": "__start__" if index == 1 else trace_nodes[index - 2][0],
            "edge_to": edge_to,
            "decision": None,
            "tool_names": [],
            "state": {
                "message_count": turns,
                "last_message_type": node,
                "memory_enabled": True,
            },
        }
        for index, (node, summary, edge_to) in enumerate(trace_nodes, start=1)
    ]
    return {
        "reply": result["reply"],
        "thread_id": thread_id,
        "messages": [
            {"role": "user", "content": message},
            {"role": "assistant", "content": result["reply"]},
        ],
        "tool_events": [],
        "interrupted": False,
        "pending": None,
        "trace": trace,
        "path": [step["node"] for step in trace],
        "state_extra": {
            "name": result.get("name", ""),
            "turns": turns,
            "checkpoint_loaded": had_checkpoint,
            "storage": ".data/lab-checkpoints.db",
        },
    }


def run_web_search(message: str, thread_id: str) -> dict[str, Any]:
    graph = web_search_graph()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 8}
    previous = graph.get_state(config).values
    message_count = len(previous.get("messages", [])) + 1
    pending_calls: dict[str, tuple[str, dict[str, Any]]] = {}
    tool_events: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    previous_node = "__start__"

    for update in graph.stream(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        stream_mode="updates",
    ):
        for node, payload in update.items():
            new_messages = payload.get("messages", []) if payload else []
            message_count += len(new_messages)
            summary = f"{node} updated search state"
            decision = None
            tool_names: list[str] = []

            for msg in new_messages:
                if isinstance(msg, AIMessage):
                    calls = msg.tool_calls or []
                    tool_names = [call["name"] for call in calls]
                    for call in calls:
                        pending_calls[call["id"]] = (call["name"], call["args"])
                    if calls:
                        decision = "tools_condition → internet_search"
                        query = calls[0].get("args", {}).get("query", message)
                        summary = f"Searching the internet for {query!r}"
                    else:
                        decision = "tools_condition → __end__"
                        summary = "Produced a sourced answer"
                elif isinstance(msg, ToolMessage):
                    name, args = pending_calls.get(
                        msg.tool_call_id, (msg.name or "internet_search", {})
                    )
                    result = _content_text(msg.content)
                    tool_events.append({"name": name, "args": args, "result": result})
                    tool_names.append(name)
                    result_count = result.count("\nURL:")
                    summary = (
                        "Internet search returned an error"
                        if result.startswith("Error:")
                        else f"Internet search returned {result_count} results"
                    )

            edge_to = "search_agent" if node == "internet_search" else None
            if decision == "tools_condition → internet_search":
                edge_to = "internet_search"
            elif decision == "tools_condition → __end__":
                edge_to = "__end__"
            trace.append(
                {
                    "sequence": len(trace) + 1,
                    "node": node,
                    "summary": summary,
                    "edge_from": previous_node,
                    "edge_to": edge_to,
                    "decision": decision,
                    "tool_names": tool_names,
                    "state": {
                        "message_count": message_count,
                        "last_message_type": (
                            new_messages[-1].__class__.__name__
                            if new_messages
                            else node
                        ),
                        "memory_enabled": True,
                    },
                }
            )
            previous_node = node

    result = graph.get_state(config).values
    all_messages = result.get("messages", [])
    reply = next(
        (
            _content_text(msg.content)
            for msg in reversed(all_messages)
            if isinstance(msg, AIMessage) and msg.content
        ),
        "The search graph finished without a text response.",
    )
    messages = []
    for msg in all_messages:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": _content_text(msg.content)})
        elif isinstance(msg, AIMessage) and msg.content:
            messages.append(
                {"role": "assistant", "content": _content_text(msg.content)}
            )
    source_count = sum(
        event["result"].count("\nURL:") for event in tool_events
    )
    return {
        "reply": reply,
        "thread_id": thread_id,
        "messages": messages,
        "tool_events": tool_events,
        "interrupted": False,
        "pending": None,
        "trace": trace,
        "path": [step["node"] for step in trace],
        "state_extra": {
            "source_count": source_count,
            "internet_enabled": True,
        },
    }


def run_person_finder(message: str, thread_id: str) -> dict[str, Any]:
    import json

    module = _person_finder_module()
    # Fresh graph, no checkpointer: prior runs must never leak into a new person.
    graph = module.build_graph(checkpointer=None)
    config = {"recursion_limit": 12}
    initial = module.initial_state_from_message(message)
    result: dict[str, Any] = dict(initial)
    previous_node = "__start__"
    trace: list[dict[str, Any]] = []
    tool_events: list[dict[str, Any]] = []
    message_count = 1
    latest_queries: list[str] = []

    for update in graph.stream(initial, config=config, stream_mode="updates"):
        for node, payload in update.items():
            payload = payload or {}
            # Manually merge append channels the same way the graph reducers do.
            if "research_notes" in payload:
                result["research_notes"] = list(result.get("research_notes") or []) + list(
                    payload.get("research_notes") or []
                )
            if "sources" in payload:
                result["sources"] = list(
                    dict.fromkeys(
                        list(result.get("sources") or []) + list(payload.get("sources") or [])
                    )
                )
            for key, value in payload.items():
                if key in {"research_notes", "sources"}:
                    continue
                result[key] = value

            summary = f"{node} updated person research state"
            decision = None
            tool_names: list[str] = []
            edge_to = None

            if node == "generate_queries":
                latest_queries = list(payload.get("search_queries") or [])
                summary = f"Generated {len(latest_queries)} search queries"
                edge_to = "research_person"
            elif node == "research_person":
                notes = payload.get("research_notes") or []
                sources = payload.get("sources") or []
                tool_names = ["web_search"]
                tool_events.append(
                    {
                        "name": "web_search",
                        "args": {"queries": latest_queries},
                        "result": (
                            f"{len(notes)} note block(s), {len(sources)} source URL(s)"
                        ),
                    }
                )
                summary = f"Researched the web ({len(sources)} sources)"
                edge_to = "extract_profile"
            elif node == "extract_profile":
                profile = payload.get("profile") or {}
                summary = f"Extracted profile with {len(profile)} fields"
                edge_to = "reflection"
            elif node == "reflection":
                ok = bool(payload.get("is_satisfactory"))
                steps = int(payload.get("reflection_steps_taken") or 0)
                if ok or steps > module.MAX_REFLECTION_STEPS:
                    decision = "reflection → __end__"
                    edge_to = "__end__"
                    summary = "Profile accepted"
                else:
                    decision = "reflection → research_person"
                    edge_to = "research_person"
                    latest_queries = list(payload.get("search_queries") or latest_queries)
                    summary = "Gaps found; scheduling follow-up research"

            message_count += 1
            trace.append(
                {
                    "sequence": len(trace) + 1,
                    "node": node,
                    "summary": summary,
                    "edge_from": previous_node,
                    "edge_to": edge_to,
                    "decision": decision,
                    "tool_names": tool_names,
                    "state": {
                        "message_count": message_count,
                        "last_message_type": node,
                        "memory_enabled": False,
                    },
                }
            )
            previous_node = node

    profile = result.get("profile") or {}
    reply = result.get("reply") or f"```json\n{json.dumps(profile, indent=2)}\n```"
    person = result.get("person") or initial.get("person") or {}
    return {
        "reply": reply,
        "thread_id": thread_id,
        "messages": [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ],
        "tool_events": tool_events,
        "interrupted": False,
        "pending": None,
        "trace": trace,
        "path": [step["node"] for step in trace],
        "state_extra": {
            "person": person,
            "profile": profile,
            "source_count": len(result.get("sources") or []),
            "is_satisfactory": bool(result.get("is_satisfactory")),
            "reflection_steps_taken": int(result.get("reflection_steps_taken") or 0),
            "internet_enabled": True,
        },
    }


def run_rag(message: str, thread_id: str) -> dict[str, Any]:
    helpers = _rag_helpers_module()
    helpers.refresh_vector_store()
    module = _rag_module()
    graph = rag_graph()
    initial = module.initial_state(message)
    result: dict[str, Any] = dict(initial)
    trace: list[dict[str, Any]] = []
    tool_events: list[dict[str, Any]] = []
    previous_node = "__start__"
    knowledge_files = helpers.list_knowledge_files()

    for update in graph.stream(initial, stream_mode="updates"):
        for node, payload in update.items():
            payload = payload or {}
            result.update(payload)
            sources = list(payload.get("sources") or [])

            if node == "retrieve":
                summary = f"Retrieved {len(payload.get('documents') or [])} chunks"
                edge_to = "generate"
                tool_names = ["semantic_retrieval"]
                tool_events.append(
                    {
                        "name": "semantic_retrieval",
                        "args": {"question": message, "k": 3},
                        "result": (
                            f"Retrieved {len(payload.get('documents') or [])} chunks "
                            f"from: {', '.join(sources) or 'no sources'}"
                        ),
                    }
                )
            else:
                summary = "Generated a grounded answer from retrieved context"
                edge_to = "__end__"
                tool_names = []

            trace.append(
                {
                    "sequence": len(trace) + 1,
                    "node": node,
                    "summary": summary,
                    "edge_from": previous_node,
                    "edge_to": edge_to,
                    "decision": None,
                    "tool_names": tool_names,
                    "state": {
                        "message_count": len(trace) + 2,
                        "last_message_type": node,
                        "memory_enabled": False,
                    },
                }
            )
            previous_node = node

    reply = str(result.get("answer") or "The RAG graph returned no answer.")
    sources = list(result.get("sources") or [])
    documents = list(result.get("documents") or [])
    retrieved_chunks = [
        {
            "rank": index,
            "source": str(document.metadata.get("source", "unknown")),
            "visibility": str(document.metadata.get("visibility", "public")),
            "private": bool(document.metadata.get("private")),
            "start_index": int(document.metadata.get("start_index", 0)),
            "content": document.page_content,
        }
        for index, document in enumerate(documents, start=1)
    ]
    public_count = sum(1 for item in knowledge_files if not item["private"])
    private_count = sum(1 for item in knowledge_files if item["private"])
    return {
        "reply": reply,
        "thread_id": thread_id,
        "messages": [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ],
        "tool_events": tool_events,
        "interrupted": False,
        "pending": None,
        "trace": trace,
        "path": [step["node"] for step in trace],
        "state_extra": {
            "question": message,
            "sources": sources,
            "source_count": len(sources),
            "retrieved_chunk_count": len(documents),
            "retrieved_chunks": retrieved_chunks,
            "knowledge_files": knowledge_files,
            "document_count": len(knowledge_files),
            "public_document_count": public_count,
            "private_document_count": private_count,
            "embedding_model": "nomic-embed-text",
            "chat_model": "qwen3:8b",
            "grounded": True,
            "rag_mode": "basic",
        },
    }


def run_doc_rag(message: str, thread_id: str) -> dict[str, Any]:
    """Ask over user-uploaded docs; workspace id == thread_id in the visual lab."""
    sys.path.insert(0, str(ROOT))
    from projects.doc_upload_rag.graph import build_ask_graph
    from projects.doc_upload_rag.store import SAMPLE_DOCS, get_workspace

    store = get_workspace(thread_id)
    seeded = False
    if store.document_count == 0:
        for name, text in SAMPLE_DOCS.items():
            store.upsert_text(name, text)
        seeded = True

    graph = build_ask_graph()
    initial = {
        "workspace_id": thread_id,
        "question": message,
        "context": "",
        "sources": [],
        "answer": "",
        "chunk_previews": [],
    }
    result: dict[str, Any] = dict(initial)
    trace: list[dict[str, Any]] = []
    tool_events: list[dict[str, Any]] = []
    previous_node = "__start__"
    knowledge_files = [
        {
            "name": item["name"],
            "path": item["name"],
            "visibility": "uploaded",
            "private": False,
        }
        for item in store.list_files()
    ]

    for update in graph.stream(initial, stream_mode="updates"):
        for node, payload in update.items():
            payload = payload or {}
            result.update(payload)
            sources = list(payload.get("sources") or result.get("sources") or [])
            if node == "retrieve":
                previews = list(payload.get("chunk_previews") or [])
                summary = f"Retrieved {len(previews)} uploaded chunks"
                edge_to = "generate"
                tool_names = ["semantic_retrieval"]
                tool_events.append(
                    {
                        "name": "semantic_retrieval",
                        "args": {"workspace_id": thread_id, "question": message, "k": 4},
                        "result": (
                            f"Retrieved {len(previews)} chunks from: "
                            f"{', '.join(sources) or 'no sources'}"
                            + (" (auto-seeded samples)" if seeded else "")
                        ),
                    }
                )
            else:
                summary = "Generated a grounded answer from uploaded documents"
                edge_to = "__end__"
                tool_names = []

            trace.append(
                {
                    "sequence": len(trace) + 1,
                    "node": node,
                    "summary": summary,
                    "edge_from": previous_node,
                    "edge_to": edge_to,
                    "decision": None,
                    "tool_names": tool_names,
                    "state": {
                        "message_count": len(trace) + 2,
                        "last_message_type": node,
                        "memory_enabled": False,
                    },
                }
            )
            previous_node = node

    reply = str(result.get("answer") or "No answer returned.")
    sources = list(result.get("sources") or [])
    previews = list(result.get("chunk_previews") or [])
    retrieved_chunks = [
        {
            "rank": index,
            "source": str(item.get("source", "unknown")),
            "visibility": "uploaded",
            "private": False,
            "start_index": 0,
            "content": str(item.get("preview", "")),
        }
        for index, item in enumerate(previews, start=1)
    ]
    return {
        "reply": reply,
        "thread_id": thread_id,
        "messages": [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ],
        "tool_events": tool_events,
        "interrupted": False,
        "pending": None,
        "trace": trace,
        "path": [step["node"] for step in trace],
        "state_extra": {
            "question": message,
            "sources": sources,
            "source_count": len(sources),
            "retrieved_chunk_count": len(retrieved_chunks),
            "retrieved_chunks": retrieved_chunks,
            "knowledge_files": knowledge_files,
            "document_count": store.document_count,
            "public_document_count": store.document_count,
            "private_document_count": 0,
            "chunk_count": store.chunk_count,
            "embedding_model": "nomic-embed-text",
            "chat_model": "qwen3:8b",
            "grounded": True,
            "rag_mode": "doc_upload",
            "workspace_id": thread_id,
            "auto_seeded": seeded,
        },
    }


def run_rag_complex(message: str, thread_id: str) -> dict[str, Any]:
    helpers = _rag_helpers_module()
    helpers.refresh_vector_store()
    module = _rag_complex_module()
    graph = rag_complex_graph()
    initial = module.initial_state(message)
    result: dict[str, Any] = dict(initial)
    trace: list[dict[str, Any]] = []
    tool_events: list[dict[str, Any]] = []
    previous_node = "__start__"
    knowledge_files = helpers.list_knowledge_files()

    edge_hints = {
        "classify": "rewrite",
        "rewrite": "retrieve",
        "retrieve": "grade",
        "grade": "generate|bump_retry",
        "bump_retry": "rewrite",
        "generate": "verify",
        "verify": "__end__",
    }

    for update in graph.stream(
        initial,
        stream_mode="updates",
        config={"recursion_limit": 24},
    ):
        for node, payload in update.items():
            payload = payload or {}
            result.update(payload)
            sources = list(result.get("sources") or payload.get("sources") or [])
            decision = None
            tool_names: list[str] = []
            edge_to = edge_hints.get(node)

            if node == "classify":
                summary = (
                    "Private access allowed"
                    if payload.get("needs_private")
                    else "Public docs only"
                )
                decision = str(payload.get("route") or summary)
            elif node == "rewrite":
                summary = f"Query → {payload.get('rewritten_query') or message}"
            elif node == "retrieve":
                summary = (
                    f"Retrieved {len(payload.get('documents') or [])} chunks "
                    f"(private={bool(result.get('needs_private'))})"
                )
                tool_names = ["semantic_retrieval", "access_filter"]
                tool_events.append(
                    {
                        "name": "semantic_retrieval",
                        "args": {
                            "query": result.get("rewritten_query") or message,
                            "include_private": bool(result.get("needs_private")),
                            "k": 4,
                        },
                        "result": (
                            f"{len(payload.get('documents') or [])} chunks from "
                            f"{', '.join(sources) or 'none'}"
                        ),
                    }
                )
            elif node == "grade":
                grade = str(payload.get("grade") or "fail")
                summary = f"Evidence grade: {grade}"
                decision = f"grade → {grade}"
                tool_names = ["evidence_grader"]
                retries = int(result.get("retries") or 0)
                edge_to = (
                    "generate"
                    if grade == "pass" or retries >= 2
                    else "bump_retry"
                )
            elif node == "bump_retry":
                summary = f"Retry retrieval #{payload.get('retries')}"
                decision = "retry loop"
            elif node == "generate":
                summary = "Drafted grounded answer"
            else:
                summary = (
                    "Verified citations"
                    if payload.get("verified")
                    else "Verification complete"
                )

            trace.append(
                {
                    "sequence": len(trace) + 1,
                    "node": node,
                    "summary": summary,
                    "edge_from": previous_node,
                    "edge_to": edge_to,
                    "decision": decision,
                    "tool_names": tool_names,
                    "state": {
                        "message_count": len(trace) + 2,
                        "last_message_type": node,
                        "memory_enabled": False,
                    },
                }
            )
            previous_node = node

    reply = str(result.get("answer") or "The complex RAG graph returned no answer.")
    sources = list(result.get("sources") or [])
    documents = list(result.get("documents") or [])
    retrieved_chunks = [
        {
            "rank": index,
            "source": str(document.metadata.get("source", "unknown")),
            "visibility": str(document.metadata.get("visibility", "public")),
            "private": bool(document.metadata.get("private")),
            "start_index": int(document.metadata.get("start_index", 0)),
            "content": document.page_content,
        }
        for index, document in enumerate(documents, start=1)
    ]
    public_count = sum(1 for item in knowledge_files if not item["private"])
    private_count = sum(1 for item in knowledge_files if item["private"])
    return {
        "reply": reply,
        "thread_id": thread_id,
        "messages": [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ],
        "tool_events": tool_events,
        "interrupted": False,
        "pending": None,
        "trace": trace,
        "path": [step["node"] for step in trace],
        "state_extra": {
            "question": message,
            "rewritten_query": result.get("rewritten_query"),
            "needs_private": bool(result.get("needs_private")),
            "grade": result.get("grade"),
            "retries": int(result.get("retries") or 0),
            "verified": bool(result.get("verified")),
            "notes": list(result.get("notes") or []),
            "sources": sources,
            "source_count": len(sources),
            "retrieved_chunk_count": len(documents),
            "retrieved_chunks": retrieved_chunks,
            "knowledge_files": knowledge_files,
            "document_count": len(knowledge_files),
            "public_document_count": public_count,
            "private_document_count": private_count,
            "embedding_model": "nomic-embed-text",
            "chat_model": "qwen3:8b",
            "grounded": True,
            "rag_mode": "complex",
        },
    }


def run_advanced_chatbot(
    message: str,
    thread_id: str,
    *,
    web_search: bool = False,
) -> dict[str, Any]:
    """Advanced chatbot: understand intent → chat | docs | web | hybrid."""
    sys.path.insert(0, str(ROOT))
    from projects.advanced_chatbot.graph import build_graph, looks_like_chat, reset_graph
    from projects.advanced_chatbot.store import get_store
    from projects.advanced_chatbot.web_search import web_result_summary

    reset_graph()
    store = get_store()
    seeded = False
    # Don't auto-seed docs for greetings — otherwise "Hi" would retrieve README nonsense.
    if (
        store.stats(thread_id)["document_count"] == 0
        and not web_search
        and not looks_like_chat(message)
    ):
        readme = (ROOT / "projects" / "advanced_chatbot" / "README.md").read_text(encoding="utf-8")
        store.upsert_document(
            workspace_id=thread_id,
            filename="advanced_chatbot_readme.md",
            content_text=readme,
            mime_type="text/markdown",
            source_type="text",
            metadata={"seed": True},
        )
        seeded = True

    graph = build_graph()
    initial = {
        "workspace_id": thread_id,
        "question": message,
        "use_web_search": web_search,
        "intent": "",
        "route_reason": "",
        "rewritten_query": "",
        "search_queries": [],
        "context": "",
        "web_context": "",
        "sources": [],
        "web_results": [],
        "chunk_candidates": [],
        "chunk_previews": [],
        "rerank_backend": "",
        "doc_score": 0.0,
        "web_score": 0.0,
        "evidence_grade": "fail",
        "answer": "",
        "verified": False,
        "fix_attempts": 0,
    }
    result: dict[str, Any] = dict(initial)
    trace: list[dict[str, Any]] = []
    tool_events: list[dict[str, Any]] = []
    previous_node = "__start__"
    knowledge_files = [
        {
            "name": item.filename,
            "path": item.filename,
            "visibility": item.source_type,
            "private": False,
        }
        for item in store.list_documents(thread_id)
    ]

    for update in graph.stream(initial, stream_mode="updates"):
        for node, payload in update.items():
            payload = payload or {}
            result.update(payload)
            sources = list(result.get("sources") or [])
            intent = str(result.get("intent") or "")
            tool_names: list[str] = []
            decision = None
            if node == "understand":
                summary = f"Intent={intent} · {str(result.get('route_reason') or '')[:90]}"
                tool_names = ["intent_router"]
                decision = "chat_reply" if intent == "chat" else "rewrite"
                tool_events.append(
                    {
                        "name": "intent_router",
                        "args": {"question": message, "search_toggle": web_search},
                        "result": f"{intent} — {result.get('route_reason') or ''}",
                    }
                )
            elif node == "chat_reply":
                summary = "Friendly LLM reply (no docs / no web)"
            elif node == "retrieve":
                candidates = list(result.get("chunk_candidates") or [])
                summary = f"Retrieved {len(candidates)} candidate chunks"
                tool_names = ["semantic_retrieval"]
                decision = "rerank"
                tool_events.append(
                    {
                        "name": "semantic_retrieval",
                        "args": {
                            "workspace_id": thread_id,
                            "query": result.get("rewritten_query") or message,
                        },
                        "result": (
                            f"{len(candidates)} candidates"
                            + (" · auto-seeded README" if seeded else "")
                        ),
                    }
                )
            elif node == "rerank":
                previews = list(result.get("chunk_previews") or [])
                backend = str(result.get("rerank_backend") or "lexical")
                summary = (
                    f"Reranked → top {len(previews)} · backend={backend} · "
                    f"doc_score={result.get('doc_score')}"
                )
                tool_names = ["chunk_reranker"]
                decision = (
                    "web_search"
                    if intent == "hybrid" or web_search
                    else "grade"
                )
                tool_events.append(
                    {
                        "name": "chunk_reranker",
                        "args": {
                            "backend": backend,
                            "kept": len(previews),
                        },
                        "result": (
                            f"top {len(previews)} · {', '.join(sources) or 'none'} · "
                            f"doc_score={result.get('doc_score')}"
                        ),
                    }
                )
            elif node == "web_search":
                hits = list(result.get("web_results") or [])
                summary = (
                    f"Web search · {web_result_summary(hits)} · "
                    f"web_score={result.get('web_score')}"
                )
                tool_names = ["internet_search"]
                tool_events.append(
                    {
                        "name": "internet_search",
                        "args": {
                            "queries": result.get("search_queries")
                            or [result.get("rewritten_query") or message]
                        },
                        "result": web_result_summary(hits),
                    }
                )
            elif node == "grade":
                summary = f"Evidence grade={result.get('evidence_grade')}"
                tool_names = ["evidence_grader"]
                decision = str(result.get("evidence_grade"))
            elif node == "rewrite":
                summary = f"Rewrote → {result.get('rewritten_query', '')[:80]}"
                decision = "web_search" if intent == "web" else "retrieve"
            elif node == "generate":
                summary = "Generated grounded answer"
            elif node == "fix":
                summary = "Rewrote after failed verification"
            else:
                summary = f"Verified={result.get('verified')}"
                decision = "ok" if result.get("verified") else "retry"

            if node == "understand":
                edge_to = "chat_reply" if intent == "chat" else "rewrite"
            elif node == "chat_reply":
                edge_to = "__end__"
            elif node == "verify":
                edge_to = (
                    "__end__"
                    if result.get("verified") or int(result.get("fix_attempts") or 0) >= 1
                    else "fix"
                )
            elif node == "retrieve":
                edge_to = "rerank"
            elif node == "rerank":
                edge_to = "web_search" if (intent == "hybrid" or web_search) else "grade"
            elif node == "web_search":
                edge_to = "grade"
            elif node == "grade":
                edge_to = "generate"
            elif node == "generate":
                edge_to = "verify"
            elif node == "fix":
                edge_to = "verify"
            elif node == "rewrite":
                edge_to = "web_search" if intent == "web" else "retrieve"
            else:
                edge_to = None

            trace.append(
                {
                    "sequence": len(trace) + 1,
                    "node": node,
                    "summary": summary,
                    "edge_from": previous_node,
                    "edge_to": edge_to,
                    "decision": decision,
                    "tool_names": tool_names,
                    "state": {
                        "message_count": len(trace) + 2,
                        "last_message_type": node,
                        "memory_enabled": False,
                    },
                }
            )
            previous_node = node

    reply = str(result.get("answer") or "No answer returned.")
    sources = list(result.get("sources") or [])
    web_results = list(result.get("web_results") or [])
    previews = list(result.get("chunk_previews") or [])
    intent = str(result.get("intent") or "")
    retrieved_chunks = [
        {
            "rank": index,
            "source": str(item.get("source", "unknown")),
            "visibility": "uploaded",
            "private": False,
            "start_index": 0,
            "content": str(item.get("preview", "")),
        }
        for index, item in enumerate(previews, start=1)
    ]
    stats = store.stats(thread_id)
    grounded = intent != "chat"
    return {
        "reply": reply,
        "thread_id": thread_id,
        "messages": [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ],
        "tool_events": tool_events,
        "interrupted": False,
        "pending": None,
        "trace": trace,
        "path": [step["node"] for step in trace],
        "state_extra": {
            "question": message,
            "intent": intent,
            "route_reason": result.get("route_reason") or "",
            "rewritten_query": result.get("rewritten_query"),
            "search_queries": result.get("search_queries") or [],
            "sources": sources,
            "source_count": len(sources),
            "web_results": web_results,
            "web_search_used": bool(web_results),
            "doc_score": result.get("doc_score"),
            "web_score": result.get("web_score"),
            "evidence_grade": result.get("evidence_grade"),
            "rerank_backend": result.get("rerank_backend") or "",
            "retrieved_chunk_count": len(retrieved_chunks),
            "retrieved_chunks": retrieved_chunks,
            "knowledge_files": knowledge_files,
            "document_count": stats["document_count"],
            "public_document_count": stats["document_count"],
            "private_document_count": 0,
            "chunk_count": stats["chunk_count"],
            "embedding_model": "nomic-embed-text",
            "chat_model": "qwen3:8b",
            "grounded": grounded,
            "verified": bool(result.get("verified")),
            "rag_mode": "advanced_chatbot",
            "vector_backend": stats.get("backend", "memory"),
            "workspace_id": thread_id,
            "auto_seeded": seeded,
        },
    }



def run_rag_architect(message: str, thread_id: str) -> dict[str, Any]:
    """Lab chip runner — default hybrid strategy. Prefer /chat/rag-architect UI."""
    sys.path.insert(0, str(ROOT))
    from projects.rag_architect.config import STRATEGIES
    from projects.rag_architect.service import ask, ingest_seed

    ingest_seed(rebuild=False)
    text = (message or "").strip()
    strategy = "hybrid"
    # Optional prefix: "crag: What is the P1 acknowledge time?"
    for name in STRATEGIES:
        prefix = f"{name}:"
        if text.lower().startswith(prefix):
            strategy = name
            text = text[len(prefix) :].strip()
            break

    result = ask(text or message, strategy=strategy)
    hits = [
        {
            "source": h.source,
            "score": round(h.score, 4),
            "preview": h.content[:180],
        }
        for h in result.hits
    ]
    tool_events = [
        {
            "name": "hybrid_retrieval",
            "args": {"strategy": result.strategy, "question": result.question},
            "result": (
                f"{len(result.hits)} hits from "
                f"{', '.join(result.sources) or 'no sources'} · grade={result.grade}"
            ),
        }
    ]
    trace = [
        {
            "sequence": 1,
            "node": "choose_strategy",
            "summary": f"strategy={result.strategy}",
            "edge_from": "__start__",
            "edge_to": "retrieve",
            "decision": result.strategy,
            "tool_names": [],
            "state": {
                "message_count": 2,
                "last_message_type": "choose_strategy",
                "memory_enabled": False,
            },
        },
        {
            "sequence": 2,
            "node": "retrieve",
            "summary": f"Retrieved {len(result.hits)} chunks",
            "edge_from": "choose_strategy",
            "edge_to": "grade",
            "decision": None,
            "tool_names": ["hybrid_retrieval"],
            "state": {
                "message_count": 2,
                "last_message_type": "retrieve",
                "memory_enabled": False,
            },
        },
        {
            "sequence": 3,
            "node": "generate",
            "summary": "Grounded answer with citations"
            if result.verified
            else "Answer generated (check verification)",
            "edge_from": "grade",
            "edge_to": "__end__",
            "decision": result.grade or None,
            "tool_names": [],
            "state": {
                "message_count": 2,
                "last_message_type": "generate",
                "memory_enabled": False,
            },
        },
    ]
    return {
        "reply": result.answer,
        "thread_id": thread_id,
        "messages": [
            {"role": "user", "content": message},
            {"role": "assistant", "content": result.answer},
        ],
        "tool_events": tool_events,
        "interrupted": False,
        "pending": None,
        "trace": trace,
        "path": [step["node"] for step in trace],
        "state_extra": {
            "question": result.question,
            "strategy": result.strategy,
            "sources": result.sources,
            "grade": result.grade,
            "verified": result.verified,
            "notes": result.notes,
            "chunk_previews": hits,
            "rag_mode": "rag_architect",
            "grounded": True,
            "open_dedicated_ui": "/chat/rag-architect",
        },
    }


def run_mcp_agent(message: str, thread_id: str) -> dict[str, Any]:
    """Phase 13: MCP demo tools → LangGraph ReAct → Ollama."""
    import asyncio

    from langchain_core.messages import HumanMessage

    sys.path.insert(0, str(ROOT))
    lesson = _load("learn_mcp_agent", "13_mcp_langgraph/01_mcp_tools_agent.py")

    async def _run() -> tuple[Any, list[str]]:
        tools = await lesson.load_mcp_tools()
        graph = lesson.build_graph(tools)
        tool_names = [t.name for t in tools]
        result = await graph.ainvoke({"messages": [HumanMessage(content=message)]})
        return result, tool_names

    result, tool_names = asyncio.run(_run())
    messages = list(result.get("messages") or [])
    final = messages[-1] if messages else None
    reply = ""
    if final is not None:
        reply = (
            final.content
            if isinstance(getattr(final, "content", None), str)
            else str(getattr(final, "content", final))
        )

    path_nodes: list[str] = ["agent"]
    tool_events: list[dict[str, Any]] = []
    for msg in messages:
        name = msg.__class__.__name__
        if name == "AIMessage" and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                tool_events.append(
                    {
                        "name": call.get("name") or "mcp_tool",
                        "args": call.get("args") or {},
                        "result": "(see ToolMessage)",
                    }
                )
            path_nodes.append("tools")
            path_nodes.append("agent")
        elif name == "ToolMessage":
            if tool_events:
                tool_events[-1]["result"] = str(getattr(msg, "content", ""))[:240]

    # Deduplicate consecutive agent/tools for a cleaner path label
    compact: list[str] = []
    for node in path_nodes:
        if not compact or compact[-1] != node:
            compact.append(node)

    trace: list[dict[str, Any]] = []
    previous = "__start__"
    for index, node in enumerate(compact, start=1):
        edge_to = compact[index] if index < len(compact) else "__end__"
        summary = (
            f"MCP tools available: {', '.join(tool_names)}"
            if node == "agent" and index == 1
            else ("Executed MCP tool call(s)" if node == "tools" else "LLM turn")
        )
        trace.append(
            {
                "sequence": index,
                "node": node,
                "summary": summary,
                "edge_from": previous,
                "edge_to": edge_to,
                "decision": "tools" if node == "agent" and edge_to == "tools" else None,
                "tool_names": tool_names if node == "tools" else [],
                "state": {
                    "message_count": len(messages),
                    "last_message_type": node,
                    "memory_enabled": False,
                },
            }
        )
        previous = node

    return {
        "reply": reply or "No answer returned.",
        "thread_id": thread_id,
        "messages": [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply or "No answer returned."},
        ],
        "tool_events": tool_events,
        "interrupted": False,
        "pending": None,
        "trace": trace,
        "path": [step["node"] for step in trace],
        "state_extra": {
            "mcp_tools": tool_names,
            "mcp_server": "demo_mcp_server.py",
            "rag_mode": "mcp_agent",
            "grounded": bool(tool_events),
        },
    }


RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "hello": run_hello,
    "router": run_router,
    "tools": run_tools,
    "memory": run_memory,
    "hitl": run_hitl_start,
    "multi_agent": run_multi_agent,
    "production": run_production,
    "subgraph": run_subgraph,
    "persistence": run_persistence,
    "web_search": run_web_search,
    "person_finder": run_person_finder,
    "rag": run_rag,
    "rag_complex": run_rag_complex,
    "doc_rag": run_doc_rag,
    "advanced_chatbot": run_advanced_chatbot,
    "rag_architect": run_rag_architect,
    "mcp_agent": run_mcp_agent,
}


def run_concept(
    concept_id: str,
    message: str,
    thread_id: str | None = None,
    *,
    web_search: bool = False,
) -> dict[str, Any]:
    concept = get_concept(concept_id)
    tid = thread_id or str(uuid.uuid4())
    if concept_id == "advanced_chatbot":
        result = run_advanced_chatbot(message, tid, web_search=web_search)
    elif concept_id == "web_search":
        result = run_web_search(message, tid)
    else:
        result = RUNNERS[concept_id](message, tid)
    result["concept_id"] = concept.id
    result["concept_title"] = concept.title
    return result
