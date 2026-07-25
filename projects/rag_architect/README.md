# RAG Architect — Enterprise Knowledge Base Lab

Separate project for the [RAG Interview Masterclass](https://www.youtube.com/playlist?list=PLNvQn5fLVQdiVo_EiAWvX0oj3fndP57Mp) path.

**Not** a replacement for [`advanced_chatbot`](../advanced_chatbot/) (product chat + OCR + pgvector).  
This lab compares **architect strategies** on a Contoso Ops enterprise KB.

**Start here for the full story:** [FLOW_AND_LEARNING.md](./FLOW_AND_LEARNING.md)

**Lessons first:** [Learning/12_rag_architect/LEARNING_PATH.md](../../Learning/12_rag_architect/LEARNING_PATH.md)

## Three-layer interview frame

1. **Knowledge** — handbook, playbooks, policies (+ optional entity graph)
2. **Retrieval** — dense / BM25 / hybrid RRF / HyDE / graph hops
3. **Validation** — grade (CRAG), citations, refuse-when-weak, offline eval

## Strategies

| Strategy | Pipeline |
|----------|----------|
| `baseline` | dense retrieve → generate → verify |
| `hybrid` | dense + BM25 → RRF → lexical rerank → generate → verify |
| `hyde` | HyDE passage → dense → generate → verify |
| `crag` | hybrid → grade → rewrite/retry → generate → verify |
| `graph` | hybrid + entity-graph hops → generate → verify |
| `agentic` | plan tool (`kb_search` / `graph` / `escalate`) → pipeline |

## Run (Angular UI)

```bash
# terminal 1
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000

# terminal 2
cd ui && npm start
```

Open **http://localhost:4200/chat/rag-architect** (also linked in the topbar as **RAG Architect**).

Lab chip: **RAG Architect · strategies** on `/chat` (prefix strategy in the message, e.g. `crag: What is the P1 acknowledge time?`).

## Run (CLI)

```bash
source .venv/bin/activate
ollama pull qwen3:8b
ollama pull nomic-embed-text

# ingest + ask
python -m projects.rag_architect --rebuild \
  "What ticket code is used for leave requests?" --strategy hybrid

python -m projects.rag_architect \
  "For a P1, what service do we page?" --strategy graph

# compare strategies on gold set
python -m projects.rag_architect.evaluate --strategies baseline,hybrid,crag
```

Default strategy: env `RAG_ARCHITECT_STRATEGY=hybrid`.

## Layout

```text
projects/rag_architect/
  ingest.py       # markdown → chunks → dense + BM25
  retrieve.py     # dense / sparse / hybrid / HyDE
  graph_rag.py    # light entity graph
  graph.py        # LangGraph strategies
  service.py      # ask / ingest CLI
  evaluate.py     # offline metrics
  data/           # Contoso Ops seed docs
  eval/gold_qa.json
```
