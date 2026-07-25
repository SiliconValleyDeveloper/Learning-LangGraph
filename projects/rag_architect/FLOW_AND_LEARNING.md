# RAG Architect — Flow & Learning

Enterprise KB lab aligned with the [RAG Interview Masterclass playlist](https://www.youtube.com/playlist?list=PLNvQn5fLVQdiVo_EiAWvX0oj3fndP57Mp).

**Phase 12 learning guide (layers, lessons, interview prep):**  
[`Learning/12_rag_architect/FLOW_AND_LEARNING.md`](../../Learning/12_rag_architect/FLOW_AND_LEARNING.md)

## 1. End-to-end flow

```text
Seed markdown (data/*.md)
        │
        ▼
  ingest: chunk → dense embeddings + BM25
        │
ask(question, strategy)
        │
        ▼
START → choose_strategy
          ├─ agentic → agent_plan (kb_search | graph | escalate)
          └─ others ──────────────┐
                                  ▼
                            retrieve (per strategy)
                                  ▼
                               grade
                     ┌────────────┴────────────┐
                     │ pass / non-CRAG         │ fail + CRAG retries
                     ▼                         ▼
                 generate                   rewrite → retrieve
                     │
                     ▼
                  verify → answer + citations (or refuse)
```

## 2. Node cheat sheet

| Node | Job |
|------|-----|
| `choose_strategy` | Lock `baseline\|hybrid\|hyde\|crag\|graph\|agentic` |
| `agent_plan` | Pick tool for agentic mode |
| `retrieve` | Dense / hybrid / HyDE / graph fusion |
| `grade` | LLM pass/fail for CRAG & agentic; non-empty check otherwise |
| `rewrite` | Multi-query rewrite when CRAG fails |
| `generate` | Grounded answer with `[n]` citations |
| `verify` | Citation + overlap check; one repair pass |

## 3. Why hybrid beats dense alone here

Contoso docs contain **ticket IDs** (`HR-LEAVE-24`, `ACC-PROD-WRITE`, `SEC-101`).  
Embeddings paraphrase well; BM25 nails exact tokens. RRF merges both rankings without heavy tuning.

## 4. Strategy map → playlist themes

| Strategy | Playlist theme |
|----------|----------------|
| baseline | Standard RAG |
| hybrid | Sparse vs dense vs hybrid |
| hyde | HyDE / zero-shot retrieval |
| crag | Corrective RAG / Self-RAG loops |
| graph | Graph RAG vs vector RAG |
| agentic | Agentic RAG (tool choice) |
| evaluate.py | RAGAS-style measurement |

## 5. How to start

### Angular UI (preferred)

```bash
# terminal 1
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000

# terminal 2
cd ui && npm start
```

Open **http://localhost:4200/chat/rag-architect** (topbar: **RAG Architect**).

APIs: `GET /api/rag-architect/status` · `POST /ask` · `POST /rebuild` · `POST /eval`.

### CLI

```bash
source .venv/bin/activate
python -m projects.rag_architect --rebuild "What is the P1 acknowledge time?" --strategy hybrid
python -m projects.rag_architect.evaluate
```

Lessons: `Learning/12_rag_architect/` (`01` → `08`).

## 6. What you learned (engineering)

1. Enterprise answers are a **system**, not “call embed API once.”
2. Chunking strategy is a product decision (parent-child / sentence window in Phase 12).
3. Retrieval quality and generation quality must be **measured separately**.
4. Refuse paths and citations are part of the architecture, not polish.
5. Graph hops help multi-hop policy questions; vectors help fuzzy wording.

## 7. Interview key points

### Design me a RAG pipeline

Use three layers:

1. **Knowledge** — sources, ACLs, chunking, indexes (dense + sparse + optional graph)
2. **Retrieval** — rewrite / HyDE / hybrid / rerank
3. **Validation** — grade, cite, refuse, evaluate, monitor cost/latency

### HNSW vs IVF

- **HNSW**: graph ANN, great recall/latency mid-scale, more RAM
- **IVF**: cluster lists + `nprobe`, cheaper at huge scale, often + PQ
- Pick with recall@k and p95 on *your* corpus

### CRAG vs Self-RAG vs standard

- **Standard**: retrieve → generate
- **CRAG**: retrieve → grade → correct/retry → generate
- **Self-RAG**: model decides when to retrieve / critique drafts  
  (our `crag` + `agentic` modes demonstrate the control-loop idea)

### Graph RAG when?

Multi-hop / relational questions (“who approves break-glass for P1?”).  
Keep vector search for semantic entry; expand via entities/edges.

### Do long-context models kill RAG?

No. Cost, freshness, ACLs, citations, and selective evidence still win.  
Long context is complementary, not a replacement.

### Cost levers

- Cache embeddings / query rewrites
- Smaller top-k after rerank
- Skip HyDE on ID lookups
- Grade only when risk is high (CRAG path)
- Cheaper embed model; stronger model only for final generate

## 8. Relation to other projects

| Project | Role |
|---------|------|
| `Learning/11_rag_llm_ecosystem` | Basics + complex rewrite/grade loop |
| `Learning/12_rag_architect` | Interview concept CLI lessons |
| `projects/doc_upload_rag` | Simple upload Q&A |
| `projects/advanced_chatbot` | Product chat + OCR + pgvector + web |
| `projects/rag_architect` | **This lab** — strategy comparison + eval |
