# Learning — LangGraph concepts only

This folder is **learning only**. Keep future apps under something like
`projects/` at the repo root so lessons stay clean.

Start with the root [`README.md`](../README.md) for:

- why LangGraph (beginner view)
- how it helps in today’s AI work
- basic → advanced path
- ideas for complex real projects

## Path (basic → advanced)

| # | Folder | Concept | Visualize in UI? |
|---|--------|---------|------------------|
| 1 | `01_hello_graph/` | State, nodes, edges | Yes (`hello`) |
| 2 | `02_router/` | Conditional routing | Yes (`router`) |
| 3 | `03_tools_agent/` | Tools + ReAct loop | Yes (`tools`) |
| 4 | `04_memory/` | Checkpointer + `thread_id` | Yes (`memory`) |
| 5 | `05_hitl/` | Interrupt / approve / resume | Yes (`hitl`) |
| 6 | `06_multi_agent/` | Supervisor + workers | Yes (`multi_agent`) |
| 7 | `07_production/` | Limits, errors, streams | Yes (`production`) |
| 8 | `08_advanced/` | Subgraphs + durable SQLite checkpoints | Yes (`subgraph`, `persistence`) |
| 9 | `09_web_search/` | Live internet search + sourced answers | Yes (`web_search`) |
| 10 | `10_person_finder/` | Person research + extract + reflect | Yes (`person_finder`) + app |
| 11 | `11_rag_llm_ecosystem/` | LLMs, Lang stack, embeddings, RAG (+ complex loop) | Yes (`rag`, `rag_complex`) |
| 12 | `12_rag_architect/` | Hybrid, HyDE, CRAG, Graph RAG, eval, index tradeoffs | CLI + UI · [FLOW_AND_LEARNING.md](12_rag_architect/FLOW_AND_LEARNING.md) |
| 13 | `13_mcp_langgraph/` | MCP servers → LangGraph ReAct + Ollama | CLI · [FLOW_AND_LEARNING.md](13_mcp_langgraph/FLOW_AND_LEARNING.md) |

## Run a lesson (CLI)

From the **repo root**:

```bash
source .venv/bin/activate
python Learning/01_hello_graph/01_hello_state.py
python Learning/04_memory/01_checkpoint_memory.py
python Learning/05_hitl/01_interrupt_before.py
python Learning/06_multi_agent/01_supervisor.py
python Learning/07_production/01_limits_errors.py
python Learning/08_advanced/01_subgraph.py
python Learning/08_advanced/02_sqlite_checkpoint.py
python Learning/09_web_search/01_search_agent.py
python Learning/10_person_finder/01_person_finder.py
python Learning/11_rag_llm_ecosystem/01_llm_basics.py
python Learning/11_rag_llm_ecosystem/02_lang_ecosystem.py
python Learning/11_rag_llm_ecosystem/03_chunk_embed.py
python Learning/11_rag_llm_ecosystem/04_retrieve.py
python Learning/11_rag_llm_ecosystem/05_rag_graph.py
python Learning/11_rag_llm_ecosystem/06_complex_rag_graph.py

python Learning/12_rag_architect/01_enterprise_layers.py
python Learning/12_rag_architect/02_chunking_strategies.py
python Learning/12_rag_architect/03_hybrid_retrieval.py
python Learning/12_rag_architect/04_hyde_and_query_opt.py
python Learning/12_rag_architect/05_crag_self_rag.py
python Learning/12_rag_architect/06_graph_rag_light.py
python Learning/12_rag_architect/07_evaluate_rag.py
python Learning/12_rag_architect/08_index_tradeoffs.py

python Learning/13_mcp_langgraph/01_mcp_tools_agent.py
python Learning/13_mcp_langgraph/02_mcp_yahoo_optional.py
```

## Visual lab

```bash
uvicorn api.main:app --reload --port 8000
cd ui && npm start
# → http://localhost:4200/chat
# → http://localhost:4200/chat/mcp          (MCP · LangGraph · LLM)
# → http://localhost:4200/chat/person-finder
```

Open the lab and pick the **MCP · LangGraph · LLM** chip (or use the topbar **MCP Agent** link).

## Shared helpers

- `llm.py` — ChatOllama (`qwen3:8b` by default)
- `visualize.py` — ASCII + Mermaid diagram helpers
- `concepts/catalog.py` — topology + runners for the visual lab

Phases 11–12 also need the local embedding model:

```bash
ollama pull nomic-embed-text
```

Phase 12 project: `projects/rag_architect/` (hybrid / HyDE / CRAG / Graph RAG / eval).
