# Phase 12 — RAG Architect Learning Path

Companion to the [RAG Interview Masterclass playlist](https://www.youtube.com/playlist?list=PLNvQn5fLVQdiVo_EiAWvX0oj3fndP57Mp).

**Full study guide (layers, examples, UI, interviews):** [FLOW_AND_LEARNING.md](./FLOW_AND_LEARNING.md)

Watch → run the matching lesson → practice the interview line.

| # | Lesson | Playlist / interview theme | One-line answer to rehearse |
|---|--------|----------------------------|-----------------------------|
| 01 | `01_enterprise_layers.py` | Why RAG is the enterprise standard | Knowledge → retrieval → validation layers |
| 02 | `02_chunking_strategies.py` | Parent-document vs sentence-window; lost-in-the-middle | Retrieve small, generate with expanded context; keep top-k short |
| 03 | `03_hybrid_retrieval.py` | Sparse vs dense vs hybrid | Hybrid (BM25 + dense + RRF) for IDs and meaning |
| 04 | `04_hyde_and_query_opt.py` | HyDE + pre-retrieval query opt | Hypothetical doc embedding closes vocab gap |
| 05 | `05_crag_self_rag.py` | Standard vs CRAG vs Self-RAG | Grade evidence; rewrite/correct before answering |
| 06 | `06_graph_rag_light.py` | Graph RAG vs vector RAG | Graphs for multi-hop relations; vectors for semantic recall |
| 07 | `07_evaluate_rag.py` | RAGAS-style evaluation | Measure retrieval and generation separately |
| 08 | `08_index_tradeoffs.py` | HNSW vs IVF; long-context myth | HNSW default mid-scale; IVF-PQ at huge scale; RAG still needed |

## How to study

```bash
source .venv/bin/activate
ollama pull qwen3:8b
ollama pull nomic-embed-text

python Learning/12_rag_architect/01_enterprise_layers.py
# … through 08
```

Then build / run the project: [`projects/rag_architect/`](../../projects/rag_architect/).

## Prerequisites

Finish [`Learning/11_rag_llm_ecosystem/`](../11_rag_llm_ecosystem/) first (chunk → embed → basic + complex RAG).
