# Advanced Chatbot — How It Works & What You Learned

This document explains the **end-to-end flow** of the advanced chatbot you built, and **what each part teaches** after going from LangGraph lessons → RAG → OCR → pgvector → deploy → web search.

Use it as a study guide: read the diagrams first, then walk a real question through each node. For interviews, jump to **[§10 Interview key points](#10-interview-key-points-how-to-talk-about-this)**.

---

## 1. Big picture

You built a **grounded assistant**, not a free-form chat bot.

```text
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  User / UI  │────▶│  FastAPI     │────▶│  LangGraph      │
│  Angular    │◀────│  api/        │◀────│  rewrite→…→fix │
└─────────────┘     └──────┬───────┘     └────────┬────────┘
                           │                      │
              ┌────────────┼────────────┐         │
              ▼            ▼            ▼         ▼
         Upload/OCR    pgvector     Web search   Ollama
         (Tesseract)   (Postgres)   (DDG/Tavily) (qwen3 + embeds)
```

**Core idea:** the LLM may only answer from **evidence** (uploaded docs and/or live web results). LangGraph controls the steps so you can see, test, and fix the pipeline.

| Layer | Role |
|-------|------|
| **Angular lab** (`ui/`) | Concept chips, upload, Search toggle, searching animation, graph SVG |
| **FastAPI** (`api/`) | `/api/run`, `/api/advanced-chat/*` |
| **LangGraph** (`projects/advanced_chatbot/graph.py`) | Stateful workflow nodes + edges |
| **Store** (`store/`) | Memory **or** Postgres + **pgvector** |
| **OCR** (`ocr.py`) | Images/PDFs → text before indexing |
| **Web search** (`web_search.py`) | Live internet hits with filters + retries |
| **Ollama** | Local chat (`qwen3:8b`) + embeddings (`nomic-embed-text`) |

---

## 2. Two paths: Index vs Ask

Real RAG systems always have **two different timelines**.

### Path A — Index (once per upload / update)

```text
File arrives
   │
   ├─ .md / .txt ──▶ read UTF-8 text
   │
   └─ .png / .pdf ──▶ OCR cascade
         PDF text extract
           → Tesseract (local, reliable on Mac)
           → Ollama vision (optional)
           → DeepSeek-OCR HTTP (optional GPU)
   │
   ▼
Chunk (≈500 chars, overlap)
   │
   ▼
Embed with nomic-embed-text (Ollama)
   │
   ▼
Upsert into vector store
   ├─ VECTOR_BACKEND=memory   → InMemoryVectorStore
   └─ VECTOR_BACKEND=pgvector → Postgres documents + chunks tables
```

**Important:** asking a question does **not** re-chunk the whole library. Only **new/changed files** are indexed (upsert by filename).

**Learning:** indexing is expensive; querying is cheap. Separate them.

### Path B — Ask (every question)

```text
User question (+ optional Search toggle)
   │
   ▼
understand ──▶ intent: chat | documents | web | hybrid
   │
   ├─ chat ──────▶ chat_reply (LLM only) ──▶ END
   │                 (even if docs are uploaded)
   │
   └─ docs/web/hybrid
         │
         ▼
      rewrite ──▶ better retrieval/search query (+ multi-query list)
         │
         ├─ web intent ──────────────▶ web_search ──┐
         └─ documents / hybrid ──────▶ retrieve ────┤
                │                                    │
                └─ hybrid / Search ON ─▶ web_search ┘
                                         │
                                         ▼
                                       grade
                                         │
                                         ▼
                                      generate
                                         │
                                         ▼
                                       verify
                                         │
                    fail once ──▶ fix ──┘
                                         │
                                         ▼
                                        END
```

**Key rule:** greetings like *“Hi, how are you?”* never answer from uploaded documents.

---

## 3. Ask flow — node by node

Shared state (`ChatState`) is a TypedDict. Every node returns **partial updates**; LangGraph merges them.

### 3.0 `understand` (intent router)

**Job:** read the user prompt first and pick a tool path — **before** touching docs or the web.

| Intent | When | Tools used |
|--------|------|------------|
| `chat` | Hi / how are you / thanks / who are you | LLM only (`chat_reply`) |
| `documents` | Asks about uploaded PDF/file contents | retrieve → generate |
| `web` | Needs live/current public facts | web_search → generate |
| `hybrid` | Needs docs **and** live web | retrieve + web_search |

**Critical product rule:** uploaded documents do **not** leak into greetings. *“Hi, how are you?”* never cites the PDF.

**Also:** *“Could you search on the internet?”* is a **capability** question → `chat` (“Yes, I can…”) — it does **not** run web search until they give a topic (or turn Search on with a real ask).

**Learning:** tool choice is a **routing** problem (Phase 2), not “always RAG everything.”

### 3.1 `rewrite`

**Job:** turn a messy human question into a precise search/retrieval query.

Example:

- User: *“hey how do I start those AWS practice things?”*
- Rewritten: *“start AWS Practice Question Set”*
- Also builds 1–3 `search_queries` for web recall (multi-query).

**Why:** better queries → better chunks and web hits.

**Learned from:** Phase 11 complex RAG (query rewrite) + Phase 2 routing mindset.

### 3.2 `retrieve`

**Job:** semantic search over **this workspace’s** documents.

Outputs:

- `context` — numbered excerpts with filenames  
- `sources` — file names  
- `chunk_previews` — for the UI  
- `doc_score` — blend of lexical overlap + vector score  

**Why score?** Weak doc hits should not force a confident answer.

**Learned from:** Phase 11 RAG (chunk → embed → retrieve) + evidence grading.

### 3.3 Conditional: web search or not?

```text
retrieve
   │
   ├─ use_web_search == true  → web_search → grade
   └─ else                    → grade
```

- **Search toggle ON** (UI globe): always run live internet search.  
- **Search OFF:** documents only (pure RAG).

**Learned from:** Phase 2 conditional edges + Phase 9 web search agent.

### 3.4 `web_search` (optional)

**Job:** fetch live results that are usable as evidence.

Reliability features:

1. Prefer **Tavily** if `TAVILY_API_KEY` is set; else **DuckDuckGo** (`ddgs`)
2. **Multi-query** search
3. **Retries** on empty/failed calls
4. Drop spam hosts / empty snippets / missing URLs
5. Dedupe by URL; prefer stronger hosts
6. Compute `web_score` (overlap with the question)

**UI:** ChatGPT/Claude-style pulse — *Planning → Scanning → Filtering → Grading → Writing*.

**Learned from:** Phase 9 (`internet_search` tool) + production hardening (filters, retries).

### 3.5 `grade`

**Job:** decide if evidence is good enough.

| Grade | Meaning |
|-------|---------|
| `pass` | Docs and/or web look relevant enough |
| `weak` | Something found, but thin — answer carefully |
| `fail` | No usable evidence — refuse instead of guessing |

**Learned from:** Phase 11 complex RAG grader + “don’t hallucinate” production rules.

### 3.6 `generate`

**Job:** write the answer **only from excerpts**.

Rules baked into the prompt:

- Prefer stronger evidence source (docs vs web by score)
- Cite `[filename.pdf]` and `[W1]`, `[W2]`
- No invented dates, scores, winners, or URLs
- If unsure, say you cannot verify

Uses Ollama `qwen3:8b` at temperature `0`.

**Learned from:** Phase 3 tools/ReAct grounding + Phase 11 grounded generate.

### 3.7 `verify` → maybe `fix`

**Job:** catch hallucinations after generation.

Checks:

- Refusal language is OK  
- Citations present when claiming facts  
- Answer tokens overlap evidence  
- Claimed URLs must appear in web results  

If verification **fails** once → `fix` rewrites cautiously → `verify` again → END.

**Learned from:** Phase 5 HITL (human gate idea) + Phase 7 production (limits / soft failures) — here the “gate” is automatic.

---

## 4. Upload / OCR flow (detail)

Entry: `POST /api/advanced-chat/workspaces/{id}/upload` → `service.ingest_bytes()`.

```text
                ingest_bytes()
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
       text         image         PDF
         │            │            │
      decode     OCR provider   try pypdf text
         │            │            │
         │            │            ├─ text found → use it
         │            │            └─ empty scan → render page → OCR
         │            │            │
         └────────────┴────────────┘
                      │
                      ▼
              upsert_document()
                 chunk + embed
                 replace old chunks
                 for same filename
```

`OCR_PROVIDER=auto` cascade:

1. PDF text (if PDF)  
2. **Tesseract** (installed locally / in Docker image)  
3. Ollama vision (`moondream`, etc.)  
4. DeepSeek-OCR HTTP (when you have a GPU endpoint)

**Learning:** OCR is part of **ingest**, not ask. Ask only sees text.

---

## 5. Storage: memory vs pgvector

Same interface: `DocumentVectorStore`.

| Backend | When | Persistence |
|---------|------|-------------|
| `memory` | Early demos | Lost on process restart |
| `pgvector` | Real project (`VECTOR_BACKEND=pgvector`) | Survives restarts |

Postgres tables (simplified):

- `documents` — file metadata + full text  
- `chunks` — chunk text + `embedding vector(768)`  

Upsert = **update document + delete old chunk rows + insert new embeddings**.

**Learning:** abstract the store early; swap backends without rewriting the graph.

---

## 6. How to start & deploy

Two common ways to run. Pick one.

### Option A — Local API (best for learning)

Ollama + Angular on the host; Postgres in Docker; FastAPI via uvicorn.

**1. Prerequisites**

```bash
# from repo root
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install 'psycopg[binary]' pytesseract pillow pypdf pypdfium2
brew install tesseract          # Mac OCR
ollama serve                    # keep running
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

**2. Start Postgres + DB UI**

```bash
docker compose -f deploy/docker-compose.yml up -d db db-ui adminer
```

| Service | URL |
|---------|-----|
| Postgres / pgvector | `localhost:5433` |
| pgweb (browse tables) | http://localhost:8082 |
| Adminer | http://localhost:8083 |

Adminer login: System `PostgreSQL`, Server `db`, User / Password / DB = `langgraph`.

**3. Configure `.env` (repo root)**

```env
VECTOR_BACKEND=pgvector
DATABASE_URL=postgresql://langgraph:langgraph@localhost:5433/langgraph
EMBED_DIMS=768
OCR_PROVIDER=auto
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:8b
OLLAMA_EMBED_MODEL=nomic-embed-text
```

**4. Start API + UI**

```bash
# terminal 1 — API
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000

# terminal 2 — Angular lab
cd ui && npm start
# → http://localhost:4200/chat
```

**5. Smoke check**

```bash
curl http://localhost:8000/api/advanced-chat/status
curl http://localhost:8000/api/health
```

Lab chip: **Advanced chat · OCR · pgvector**.

---

### Option B — Full Docker deploy (API + DB)

Everything except Ollama runs in Compose. Container API is on **:8001** so local uvicorn can stay on **:8000**.

**1. Keep Ollama on the host** (required)

```bash
ollama serve
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

**2. Build & start the stack**

```bash
# from repo root
docker compose -f deploy/docker-compose.yml up -d --build
```

| Service | URL |
|---------|-----|
| API (Docker) | http://localhost:8001 |
| Postgres | localhost:5433 |
| pgweb | http://localhost:8082 |
| Adminer | http://localhost:8083 |

**3. Verify**

```bash
curl http://localhost:8001/api/advanced-chat/status
curl http://localhost:8001/api/health
```

**4. Point the UI at the container API** (optional)

```ts
// ui/src/environments/environment.ts
apiUrl: 'http://localhost:8001'
```

Or leave `apiUrl` on `:8000` and use Option A’s local uvicorn while only the DB runs in Docker.

**5. Stop / clean**

```bash
docker compose -f deploy/docker-compose.yml down          # keep volumes (data safe)
docker compose -f deploy/docker-compose.yml down -v       # wipe DB volumes too
```

**Inside the API container**

- `DATABASE_URL` uses host `db` (Compose service name), port `5432`
- Ollama is reached via `host.docker.internal:11434`
- OCR uses Tesseract baked into the image (`OCR_PROVIDER=auto`)

More detail: [`deploy/README.md`](../../deploy/README.md).

---

### Typical day-to-day commands

```bash
# DB + UIs only (local uvicorn on :8000)
docker compose -f deploy/docker-compose.yml up -d db db-ui adminer

# Full stack (API :8001 + DB + UIs)
docker compose -f deploy/docker-compose.yml up -d --build

# Logs
docker compose -f deploy/docker-compose.yml logs -f api
docker compose -f deploy/docker-compose.yml logs -f db

# Peek at indexed docs from the CLI
docker exec langgraph-pgvector psql -U langgraph -d langgraph \
  -c "SELECT filename, chunk_count, source_type FROM documents ORDER BY updated_at DESC LIMIT 10;"
```

---

## 7. How a real example runs

### Example A — AWS practice PDF (docs only)

1. Upload PDF → OCR/text → 22 chunks in pgvector  
2. Ask: *“How do I start the practice set?”* (Search **off**)  
3. Flow: `rewrite → retrieve → grade → generate → verify`  
4. Answer cites `[AWS_….pdf]` from retrieved chunks  

### Example B — Current web fact (Search on)

1. Toggle **Search**  
2. Ask: *“What is DuckDuckGo known for?”*  
3. Flow: `rewrite → retrieve → web_search → grade → generate → verify`  
4. UI shows searching animation, then **Web sources** cards `[W1]…`  

### Example C — Bad / empty evidence

1. Ask something unrelated with no docs and Search off  
2. `grade=fail` → model refuses instead of inventing  

---

## 8. UI ↔ API map

| UI action | API |
|-----------|-----|
| Concept **Advanced chat · OCR · pgvector** | `POST /api/run` `concept_id=advanced_chatbot` |
| Upload file | `POST /api/advanced-chat/workspaces/{id}/upload` |
| Ask (lab) | `POST /api/run` with `web_search: true/false` |
| Ask (project API) | `POST /api/advanced-chat/ask` |
| Search toggle | sets `web_search` on the request |
| Status | `GET /api/advanced-chat/status` |

Workspace id in the lab ≈ `thread_id` for that chat session.

---

## 9. What you learned (curriculum → project)

### From `Learning/` lessons

| Lesson idea | Where it shows up now |
|-------------|------------------------|
| State + nodes + edges | `ChatState`, each graph node |
| Conditional routing | Search on/off; verify → fix/end |
| Tools / ReAct | `internet_search`, retrieval as tools |
| Memory / thread id | Workspace per conversation |
| HITL / approval mindset | Automatic verify gate (no blind trust) |
| Production limits | Soft refuse, retries, filters |
| Multi-agent / supervisor pattern | Grade then generate (roles in one graph) |
| Web search | Live search node + UI animation |
| RAG ecosystem | Chunk, embed, retrieve, cite |
| Complex RAG | Rewrite, grade, retry/fix |

### From building this project

1. **Grounding beats cleverness** — a short cited answer is better than a fluent guess.  
2. **Index ≠ query** — upload once; ask many times.  
3. **Pluggable storage** — memory for learning, pgvector for real.  
4. **OCR belongs in ingest** — ask path stays text-only.  
5. **Web search needs hygiene** — filters, multi-query, retries, citation checks.  
6. **Verify is a product feature** — catch hallucinations before the user sees them.  
7. **Deploy early** — Docker Compose makes Postgres + API reproducible.  
8. **UI teaches the system** — graph SVG + search animation make LangGraph visible.

### Mental model to keep

```text
Whiteboard the workflow
  → define shared state
  → one job per node
  → edges / conditionals
  → add memory / HITL / limits as needed
  → stream progress to the UI
```

That is the LangGraph way — and this project is that pattern end-to-end.

---

## 10. Interview key points (how to talk about this)

Use this as a **cheat sheet**. Say the short answer first, then one concrete example from this project.

### Elevator pitch (30 seconds)

> “I built a grounded RAG chatbot with **LangGraph**: upload docs (including OCR for PDFs/images), store embeddings in **Postgres + pgvector**, then answer only from retrieved evidence — with optional live web search, evidence grading, citation checks, and a verify/fix loop. The UI shows the graph path so the workflow isn’t a black box.”

### Why LangGraph (not a single prompt)?

| Interview ask | Strong answer |
|---------------|---------------|
| Why not one big prompt? | Multi-step control: rewrite → retrieve → (search) → grade → generate → verify. Each step is testable. |
| What is LangGraph? | A **stateful workflow** for LLM apps: shared state, nodes (functions), edges/conditionals. |
| Conditional edges? | Example: intent `chat` → `chat_reply`; `web` → search; verify fail once → `fix`. |
| State? | `ChatState` TypedDict; nodes return **partial updates**; graph merges them. |
| Tools / ReAct? | Intent router picks retrieval / `internet_search` / plain LLM; loops with limits. |

### RAG & vectors (core ML/systems)

| Topic | Say this |
|-------|----------|
| RAG | Retrieve relevant chunks → generate **only** from them; reduces hallucination. |
| Chunk + embed | Split text → `nomic-embed-text` → vectors; query embeds the same way. |
| Similarity search | Cosine/distance over embeddings; return top-k chunks for the question. |
| **pgvector** | Postgres extension storing `vector(768)`; same DB for metadata + embeddings; survives restarts. |
| Memory vs pgvector | Same store interface; memory for demos, pgvector for production persistence. |
| Upsert | Update doc + replace old chunks (delete + insert) so re-upload doesn’t duplicate. |
| Index ≠ query | Indexing is expensive (OCR/embed); asking reuses stored vectors. |

### Grounding, quality & safety

| Topic | Say this |
|-------|----------|
| Hallucination control | Grade evidence (`pass` / `weak` / `fail`); refuse when `fail`. |
| Verify node | Post-check citations / invented URLs; one **fix** retry. |
| Citations | Docs as `[filename]`; web as `[W1]`, `[W2]` with real URLs. |
| Web search hygiene | Multi-query, retries, spam filter, dedupe, `web_score`. |
| OCR placement | OCR at **ingest** only; ask path is text — keeps latency and graph simple. |

### Production / deploy talking points

| Topic | Say this |
|-------|----------|
| Deploy | Docker Compose: API + Postgres/pgvector (+ DB UIs); Ollama on host. |
| Config via env | `VECTOR_BACKEND`, `DATABASE_URL`, `OCR_PROVIDER` — swap backends without rewriting the graph. |
| Observability | Lab streams node path + searching animation; status endpoint for health. |
| What’s next | Auth, SSE streaming, checkpointer memory, eval set, GPU OCR. |

### Likely follow-ups (ready answers)

1. **“How do you prevent the model inventing facts?”**  
   Evidence gate + strict generate prompt + verify/fix + refuse on empty evidence.

2. **“Why Postgres/pgvector instead of a dedicated vector DB?”**  
   One system for docs + chunks + metadata; SQL + vectors; easy local Docker; enough for this scale; can migrate later if needed.

3. **“What happens when docs and web disagree?”**  
   Scores (`doc_score` / `web_score`); generate prefers stronger evidence and must cite sources.

4. **“How would you scale this?”**  
   Async workers for ingest, connection pooling, batch embeds, cache hot queries, move to managed Postgres, add eval harness for grounded accuracy.

5. **“Difference between LangChain and LangGraph?”**  
   LangChain = building blocks (models, retrievers, tools). LangGraph = **orchestration** of multi-step, branching, looping workflows with durable state.

### One whiteboard you can draw in an interview

```text
Ask → understand → chat? → LLM
                 └─ docs/web/hybrid → rewrite → retrieve/web → grade → generate → verify → answer

Upload → OCR? → chunk → embed → pgvector
```

**Closing line:** “I care that the answer is **grounded and inspectable**, not just fluent.”

---

## 11. File map (where to read code)

```text
projects/advanced_chatbot/
  README.md          # quick start + phases
  FLOW_AND_LEARNING.md  # this document
  config.py          # VECTOR_BACKEND, OCR_*, DATABASE_URL
  service.py         # ingest_bytes (text | OCR)
  ocr.py             # Tesseract / Ollama / DeepSeek providers
  store/             # memory.py | pgvector.py
  web_search.py      # reliable live search
  graph.py           # LangGraph ask pipeline

api/main.py                  # /api/run + concepts
api/advanced_chat_routes.py  # upload / ask / status
deploy/docker-compose.yml    # db + API + pgweb + Adminer
deploy/README.md             # start & deploy process
Learning/concepts/catalog.py # lab concept + runner
ui/.../chat/                 # Search toggle + animations
```

---

## 12. How to study this in one evening

1. Upload a small `.md` file; ask a question with Search **off**. Watch nodes: rewrite → retrieve → grade → generate → verify.  
2. Upload a screenshot/PDF; confirm `source_type` becomes `ocr`.  
3. Turn Search **on**; ask a public-web question; watch the animation and `[W#]` citations.  
4. Ask nonsense with Search off; confirm a **refusal**, not a story.  
5. Skim `graph.py` top-to-bottom once — you now know every node’s job.  
6. Skim §10 interview points and practice the 30-second pitch out loud.

---

## 13. What’s next (Phase E ideas)

When you continue:

- Auth + per-user workspaces  
- Streaming tokens (SSE) into the UI  
- Multi-turn memory (checkpointer) over the same thread  
- Stronger OCR (DeepSeek-OCR on GPU)  
- Evaluation set (golden Q&A) to measure grounded accuracy  

You already have the hard part: a **visible, testable, grounded LangGraph system**.
