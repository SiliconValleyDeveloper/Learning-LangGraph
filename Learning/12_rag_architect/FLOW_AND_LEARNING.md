# Phase 12 — RAG Architect: How It Works & What You Learn

Study guide for the [RAG Interview Masterclass playlist](https://www.youtube.com/playlist?list=PLNvQn5fLVQdiVo_EiAWvX0oj3fndP57Mp) and this repo’s Phase 12 + Contoso Ops project.

**Use this doc to:** understand the three enterprise layers, run lessons in order, walk a real question through the pipeline, practice interview answers, then try the same ideas in the Angular UI.

| Resource | Path |
|----------|------|
| Lesson checklist | [LEARNING_PATH.md](./LEARNING_PATH.md) |
| CLI lessons | `Learning/12_rag_architect/01` … `08` |
| Project | [`projects/rag_architect/`](../../projects/rag_architect/) |
| Project flow | [`projects/rag_architect/FLOW_AND_LEARNING.md`](../../projects/rag_architect/FLOW_AND_LEARNING.md) |
| UI | http://localhost:4200/chat/rag-architect |

**Prerequisites:** finish [Phase 11 — RAG + Lang ecosystem](../11_rag_llm_ecosystem/README.md) (chunk → embed → basic + complex RAG).

---

## 0. In plain English — why RAG, and why these layers?

### The problem (without RAG)

A normal chatbot (ChatGPT-style) answers from **what it memorized during training**.

That fails at work when you ask:

- “What’s *our* leave ticket code?”
- “What’s the P1 acknowledge time in *our* playbook?”

The model may **guess**, sound confident, and still be **wrong** — because your company handbook was never in its training data (or is newer than the model).

### The simple idea (with RAG)

**RAG = “look it up, then answer.”**

1. Keep your real documents in a searchable library  
2. When someone asks a question, **find the relevant pages first**  
3. Give those pages to the AI and say: *answer only from this*  
4. Prefer **citations** (“according to handbook [1]”) or **refuse** if nothing useful was found  

Think of it like an employee who **opens the company wiki before answering**, instead of answering from memory.

### Why we split it into 3 layers

| Layer | Layman meaning | Why bother |
|-------|----------------|------------|
| **Knowledge** | The filing cabinet | Put policies/runbooks in one place, cut into searchable pieces, label what’s private |
| **Retrieval** | The search step | Find the right pages (meaning search + keyword search for codes like `HR-LEAVE-24`) |
| **Validation** | The double-check | Only answer if evidence is good; cite sources; say “I don’t know” when unsure |

Without this split, people build a demo that “kinda works” but can’t explain failures, improve search, or keep answers safe.

### Benefits (what you actually gain)

| Benefit | What it means day-to-day |
|---------|---------------------------|
| **Fewer hallucinations** | Answers stick to your docs instead of inventing policy |
| **Up-to-date knowledge** | Update a markdown/PDF — no need to retrain a giant model |
| **Private company data stays yours** | Docs can stay on your machine / VPC (we use local Ollama) |
| **Citations & trust** | Users can click/see *where* the answer came from |
| **Cheaper & faster to improve** | Fix chunking or search; don’t fine-tune the whole LLM |
| **Safer for enterprise** | Refuse weak answers; apply access rules; measure quality with eval |
| **Interview / architecture clarity** | You can design and defend a real system, not just call an API |

### What each “fancy” piece buys you

| Piece | Plain benefit |
|-------|----------------|
| **Chunking** | Search finds a useful paragraph, not a whole 50-page PDF |
| **Embeddings (dense search)** | Finds “outage response time” even if the doc says “P1 acknowledge” |
| **BM25 (keyword search)** | Finds exact ticket IDs / codes the meaning-search might miss |
| **Hybrid (both + RRF)** | Best of both worlds — meaning *and* exact codes |
| **HyDE** | Helps when the user asks vaguely and doesn’t use handbook wording |
| **CRAG / grade + retry** | If search was bad, try again instead of answering garbage |
| **Graph RAG** | Follows relationships (“P1 → which service → which role”) across docs |
| **Eval metrics** | Proof that a change made answers better, not just “feels better” |

### When you *don’t* need all of this

- Casual chat with no company facts → plain LLM is fine  
- One tiny FAQ you could paste into the prompt → simple RAG is enough  
- Full hybrid / CRAG / graph → when accuracy, IDs, and safety matter (support, ops, compliance)

---

## 1. Big picture — why this phase exists

Phase 11 taught **basic RAG** (`retrieve → generate`) and a **complex control loop** (rewrite → grade → retry).

Phase 12 teaches **architect / interview** skills:

- Design RAG as **three layers**, not “call an embedding API”
- Choose **chunking**, **hybrid retrieval**, **HyDE**, **CRAG**, **Graph RAG**
- **Evaluate** quality and talk about **indexes** (HNSW vs IVF) and cost

```text
┌──────────────────────────────────────────────────────────────┐
│                 Enterprise RAG (interview frame)             │
│                                                              │
│  [1] Knowledge     docs · chunks · dense · BM25 · graph·ACL  │
│           │                                                  │
│           ▼                                                  │
│  [2] Retrieval     rewrite · HyDE · hybrid · RRF · rerank    │
│           │                                                  │
│           ▼                                                  │
│  [3] Validation    grade · cite · refuse · verify · eval     │
└──────────────────────────────────────────────────────────────┘
```

**One-liner to memorize:**  
*An enterprise RAG pipeline retrieves internal knowledge, validates it with citations and confidence, and produces safe context-aware answers.*

---

## 2. The three layers (with Contoso examples)

### 2.1 Knowledge layer — where truth lives

| Piece | What it is | Contoso example |
|-------|------------|-----------------|
| Sources | Handbook, playbooks, policies, FAQs | `employee_handbook.md`, `incident_playbook.md`, … |
| Chunking | Split docs for retrieval | Fixed / parent-child / sentence-window |
| Dense index | Embedding vectors | `nomic-embed-text` over chunks |
| Sparse index | BM25 / keyword postings | Finds `HR-LEAVE-24` exactly |
| Entity graph | Optional relations | `P1 → PagerDuty → contoso-ops-p1` |
| ACLs | Visibility filters | `visibility=internal` / confidential |

**Learning:** if knowledge is messy or missing, no retrieval trick will save you.

### 2.2 Retrieval layer — how we find evidence

| Technique | When to use | Contoso example |
|-----------|-------------|-----------------|
| Dense search | Paraphrases / meaning | “How fast for customer outages?” → P1 playbook |
| BM25 | IDs, codes, rare tokens | `HR-LEAVE-24`, `ACC-PROD-WRITE` |
| Hybrid + RRF | Default enterprise | Fuse dense + sparse rankings |
| Query rewrite / multi-query | Vague user wording | Expand “leave ticket” → handbook phrases |
| HyDE | Vocab mismatch | Write a fake handbook paragraph, embed *that* |
| Graph hops | Multi-hop / relations | P1 + break-glass across two docs |
| **Rerank (CrossEncoder)** | Final ordering of a small candidate list | After hybrid/RRF, score `(query, chunk)` with `sentence-transformers` |

**Learning:** retrieval is rarely one vector call. Fuse, then rerank.

#### Why libraries for rerank (interview talking point)

| Approach | Pros | Cons | When |
|----------|------|------|------|
| Hand-rolled token overlap | No deps, instant | Misses paraphrases | Demos / fallback (`lexical`) |
| **`sentence-transformers` CrossEncoder** | Strong relevance, local, no per-call API fee | First download + CPU/GPU cost | Default production-style rerank |
| Paid rerank API (Cohere/Jina) | Easy, strong | Cost, network, vendor lock | Hosted products |
| LLM scores each chunk | Flexible | Slow/expensive | Last resort |

We use **`sentence-transformers`** because it wraps ready MS MARCO cross-encoders — the industry pattern of *retrieve many → rerank few* — without training or hosting our own ranking model.

### 2.3 Validation layer — how we stay safe

| Step | Job | Contoso example |
|------|-----|-----------------|
| Grade (CRAG) | Is evidence enough? | “Capital of France?” → fail → refuse |
| Generate | Answer only from context + `[n]` cites | “Leave ticket is HR-LEAVE-24 [1]” |
| Verify / repair | Citations + faithfulness | Fix draft if claims aren’t in chunks |
| Refuse | Better than hallucination | Weak evidence → “not enough grounded evidence…” |
| Offline eval | Measure before shipping | recall / faithfulness / citation rate |

**Learning:** generation without validation is a demo, not a product.

---

## 3. Worked example (all layers)

**Question:** *What ticket code is used for leave requests?*

```text
Knowledge
  handbook chunk already indexed:
  “…Ticket code for leave requests: HR-LEAVE-24”

Retrieval (hybrid)
  dense → leave-policy meaning
  BM25  → boosts HR-LEAVE-24
  RRF   → that chunk ranks #1

Validation
  generate → “The ticket code is HR-LEAVE-24 [1].”
  verify   → has citation + overlap → verified=true
```

**Try it**

```bash
# CLI lesson
python Learning/12_rag_architect/03_hybrid_retrieval.py

# Project
python -m projects.rag_architect \
  "What ticket code is used for leave requests?" --strategy hybrid

# UI
# http://localhost:4200/chat/rag-architect → Ask lab → hybrid → same question
```

---

## 4. Lessons — what each file teaches

Run from the **repo root** after `source .venv/bin/activate`.

| # | File | You learn | Interview line |
|---|------|-----------|----------------|
| 01 | `01_enterprise_layers.py` | 3-layer design frame | Knowledge → retrieval → validation |
| 02 | `02_chunking_strategies.py` | Fixed vs parent-child vs sentence-window | Retrieve small, generate with expanded context |
| 03 | `03_hybrid_retrieval.py` | Dense vs BM25 vs RRF hybrid | Hybrid for IDs **and** meaning |
| 04 | `04_hyde_and_query_opt.py` | HyDE + multi-query | Hypothetical doc closes vocab gap |
| 05 | `05_crag_self_rag.py` | Grade → rewrite loop in LangGraph | Don’t answer from bad evidence |
| 06 | `06_graph_rag_light.py` | Entity hops vs vector-only | Graph for multi-hop; vectors for semantics |
| 07 | `07_evaluate_rag.py` | Offline metrics | Measure retrieval and generation separately |
| 08 | `08_index_tradeoffs.py` | HNSW vs IVF + long-context myth | HNSW mid-scale; IVF-PQ at huge scale; RAG still needed |

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

Needs Ollama models: `qwen3:8b`, `nomic-embed-text` (lessons 01, 02, 08 work without LLM calls for the core demo).

---

## 5. Strategies in the project / UI

Same Contoso KB; switch **how** you retrieve and validate.

| Strategy | Pipeline | Maps to lesson |
|----------|----------|----------------|
| `baseline` | dense → generate → verify | Phase 11 simple RAG |
| `hybrid` | dense + BM25 → RRF → rerank → generate | `03` |
| `hyde` | HyDE passage → dense → generate | `04` |
| `crag` | hybrid → grade → rewrite/retry → generate | `05` |
| `graph` | hybrid + entity hops → generate | `06` |
| `agentic` | plan tool (`kb_search` / `graph` / `escalate`) | playlist “Agentic RAG” |

```text
START → choose_strategy
          ├─ agentic → agent_plan
          └─ others ─┐
                     ▼
               retrieve → grade
                            ├─ pass → generate → verify → END
                            └─ fail (CRAG) → rewrite → retrieve …
```

---

## 6. How to learn with the Angular UI

```bash
# terminal 1
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000

# terminal 2
cd ui && npm start
```

Open **http://localhost:4200/chat/rag-architect**

| Tab | What to do |
|-----|------------|
| **Architecture · layers + graph** | Click Knowledge / Retrieval / Validation; switch strategies to highlight the graph path; read Contoso examples |
| **Ask lab · strategies** | Ask the leave-ticket / P1 questions; compare `baseline` vs `hybrid`; use **Clear** to reset; **Run eval** for metrics |

Lab chip on `/chat`: **RAG Architect · strategies**  
(Optional prefix: `crag: What is the P1 acknowledge time?`)

---

## 7. Evaluation (how you know it improved)

Gold set: `Learning/12_rag_architect/eval/gold_qa.json` (and project copy under `projects/rag_architect/eval/`).

| Metric | Meaning |
|--------|---------|
| Context recall | Gold keywords appear in retrieved context |
| Faithfulness (proxy) | Answer tokens overlap context |
| Citation rate | Answer includes `[n]` style cites |

```bash
python Learning/12_rag_architect/07_evaluate_rag.py
python -m projects.rag_architect.evaluate --strategies baseline,hybrid,crag
```

**Learning:** never tune generation until retrieval metrics move. Fix retrieve first.

---

## 8. Interview key points

### Design me a RAG pipeline

1. **Knowledge** — sources, ACLs, chunking, dense + sparse (+ optional graph)  
2. **Retrieval** — rewrite / HyDE / hybrid / rerank  
3. **Validation** — grade, cite, refuse, evaluate, watch cost/latency  

### Sparse vs dense vs hybrid

- Dense = meaning / paraphrase  
- Sparse (BM25) = exact IDs and rare terms  
- Hybrid + RRF = default for enterprise KBs  

### HNSW vs IVF

- **HNSW** — graph ANN, strong recall/latency mid-scale, more RAM  
- **IVF** — cluster lists + `nprobe`, cheaper at huge scale (often + PQ)  
- Pick with **recall@k** and **p95** on *your* data  

### CRAG vs Self-RAG vs standard

- Standard: retrieve → generate  
- CRAG: retrieve → grade → correct/retry → generate  
- Self-RAG: model decides when to retrieve / critique  
  (our `crag` + `agentic` modes show the control-loop idea)

### Graph RAG when?

Multi-hop / relational questions (“who approves break-glass for P1?”).  
Keep vectors for semantic entry; expand via entities/edges.

### Do long-context models kill RAG?

No. Cost, freshness, ACLs, citations, and selective evidence still win.

### Cost levers

- Cache embeddings / rewrites  
- Smaller top-k after rerank  
- Skip HyDE on ID lookups  
- Grade only on the CRAG path when risk is high  
- Cheaper embed model; stronger model only for final generate  

---

## 9. How this fits the rest of the repo

| Path | Role |
|------|------|
| `Learning/11_rag_llm_ecosystem` | Basics + rewrite/grade/retry |
| `Learning/12_rag_architect` | **This phase** — architect concepts (CLI) |
| `projects/doc_upload_rag` | Simple upload Q&A |
| `projects/advanced_chatbot` | Product chat + OCR + pgvector + web |
| `projects/rag_architect` | Strategy lab + eval + UI |

`advanced_chatbot` = shipping-shaped app.  
`rag_architect` = interview / architecture lab.

---

## 10. Suggested study week

| Day | Do |
|-----|----|
| 1 | Read §1–3 here; run `01`, `02` |
| 2 | Run `03`, `04`; ask leave-ticket in UI with `baseline` vs `hybrid` |
| 3 | Run `05`, `06`; try `crag` + `graph` in Ask lab |
| 4 | Run `07`, `08`; `python -m projects.rag_architect.evaluate` |
| 5 | Rehearse §8 interview answers out loud; sketch the 3-layer diagram from memory |

---

## 11. Quick commands

```bash
source .venv/bin/activate
ollama pull qwen3:8b
ollama pull nomic-embed-text

# lessons
python Learning/12_rag_architect/01_enterprise_layers.py
# … through 08

# project ask
python -m projects.rag_architect --rebuild \
  "What is the P1 acknowledge time?" --strategy hybrid

# eval
python -m projects.rag_architect.evaluate

# UI
uvicorn api.main:app --reload --port 8000
# other terminal: cd ui && npm start
# → http://localhost:4200/chat/rag-architect
```
