# Phase 12 — RAG Architect (Interview Masterclass)

Architect-level RAG patterns beyond Phase 11: hybrid retrieval, HyDE, CRAG,
lightweight Graph RAG, evaluation, and index tradeoffs.

Playlist: [RAG Interview Masterclass: Zero to Architect](https://www.youtube.com/playlist?list=PLNvQn5fLVQdiVo_EiAWvX0oj3fndP57Mp)

**Start here (full learning doc):** [FLOW_AND_LEARNING.md](./FLOW_AND_LEARNING.md)  
**Lesson checklist:** [LEARNING_PATH.md](./LEARNING_PATH.md)

## Prerequisites

```bash
source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

Complete Phase 11 first.

## Run in order

```bash
python Learning/12_rag_architect/01_enterprise_layers.py
python Learning/12_rag_architect/02_chunking_strategies.py
python Learning/12_rag_architect/03_hybrid_retrieval.py
python Learning/12_rag_architect/04_hyde_and_query_opt.py
python Learning/12_rag_architect/05_crag_self_rag.py
python Learning/12_rag_architect/06_graph_rag_light.py
python Learning/12_rag_architect/07_evaluate_rag.py
python Learning/12_rag_architect/08_index_tradeoffs.py
```

## Lessons

| File | Focus |
|------|--------|
| `01` | Enterprise three-layer framing |
| `02` | Fixed / parent-child / sentence-window chunking |
| `03` | Dense vs BM25 vs hybrid RRF |
| `04` | HyDE + multi-query rewrite |
| `05` | CRAG / Self-RAG LangGraph loop |
| `06` | Light Graph RAG vs vector-only |
| `07` | Offline eval (recall / faithfulness / citations) |
| `08` | HNSW vs IVF interview answers + toy IVF demo |

## Next

Ship the same ideas as a project: [`projects/rag_architect/`](../../projects/rag_architect/).
