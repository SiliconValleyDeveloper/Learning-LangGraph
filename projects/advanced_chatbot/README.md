# Advanced Chatbot

Production-oriented project next to `Learning/` and `projects/doc_upload_rag/`.

**Start here for the full story:** [FLOW_AND_LEARNING.md](./FLOW_AND_LEARNING.md) — end-to-end flow, node-by-node behavior, and what you learned.

**How to start & deploy:** [FLOW_AND_LEARNING.md §6](./FLOW_AND_LEARNING.md#6-how-to-start--deploy) · [deploy/README.md](../../deploy/README.md)

**Interview cheat sheet:** [FLOW_AND_LEARNING.md §10](./FLOW_AND_LEARNING.md#10-interview-key-points-how-to-talk-about-this)

**Goal:** upload / update documents → chunk + embed → **pgvector** → advanced grounded chat with optional live web search. Smarter pipeline: multi-query search, evidence grading, strict citations, one fix retry.

![Advanced chat intent router — “could you search on internet?” → LLM capability reply](../../docs/langgraph-lab-advanced-chatbot-intent.png)

## Phased roadmap

| Phase | Status | What |
|-------|--------|------|
| **A** | Ready now | Memory vector store, text upload/update/delete, rewrite→retrieve→generate→verify graph |
| **B** | Enabled | `VECTOR_BACKEND=pgvector` + Docker on port **5433** |
| **C** | Enabled | `OCR_PROVIDER=auto` — PDF text → **Tesseract** → Ollama vision → DeepSeek HTTP |
| **D** | Enabled | `docker compose -f deploy/docker-compose.yml up -d --build` → API on **:8001** |
| **E** | Next | Auth, streaming SSE, multi-turn checkpointer, Angular advanced UI polish |

```text
Upload (text | image/PDF)
        │
        ├─ text ────────────────────────► upsert chunks
        └─ OCR (auto cascade) ──────────► markdown ──► upsert chunks
             PDF text → Tesseract
             → Ollama vision → DeepSeek HTTP
                                              │
                                         memory | pgvector
                                              │
                         rewrite → retrieve → generate → verify
```

## Run (Phase A — today)

```bash
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

```bash
# status
curl http://localhost:8000/api/advanced-chat/status

# workspace + upload + ask
WID=$(curl -s -X POST http://localhost:8000/api/advanced-chat/workspaces | python -c "import sys,json; print(json.load(sys.stdin)['workspace_id'])")
curl -s -X POST "http://localhost:8000/api/advanced-chat/workspaces/$WID/upload" -F "file=@README.md"
curl -s -X POST http://localhost:8000/api/advanced-chat/ask \
  -H 'Content-Type: application/json' \
  -d "{\"workspace_id\":\"$WID\",\"question\":\"What is the phased roadmap?\"}"
```

Lab concept chip: **Advanced chat · OCR · pgvector** on `http://localhost:4200/chat`.

## Phase B — Postgres + pgvector

```bash
docker compose -f deploy/docker-compose.yml up -d db
pip install 'psycopg[binary]'
```

Add to `.env`:

```env
VECTOR_BACKEND=pgvector
DATABASE_URL=postgresql://langgraph:langgraph@localhost:5433/langgraph
EMBED_DIMS=768
```

> Host port is **5433** (avoids clashing with a local Postgres on 5432).
Restart the API. Same upload/ask APIs — store swaps under the hood.

## Phase C — OCR (enabled)

This Mac has **no NVIDIA GPU**, so Phase C runs locally with **Tesseract** (primary) via `OCR_PROVIDER=auto`.

```bash
brew install tesseract
pip install pytesseract pillow pypdf pypdfium2
# optional vision fallback:
# ollama pull moondream
```

`.env`:

```env
OCR_PROVIDER=auto
OLLAMA_VISION_MODEL=moondream
```

Upload `.png` / `.jpg` / `.pdf` in the lab — text is extracted, chunked, embedded into **pgvector**, then you can ask.

## Phase D — Deploy (Docker)

```bash
# full stack: API :8001 + Postgres :5433 + pgweb :8082 + Adminer :8083
docker compose -f deploy/docker-compose.yml up -d --build
curl http://localhost:8001/api/advanced-chat/status
```

| Service | URL |
|---------|-----|
| API | http://localhost:8001 |
| pgweb (visualize DB) | http://localhost:8082 |
| Adminer | http://localhost:8083 |

Step-by-step (local API vs full Docker): [`deploy/README.md`](../../deploy/README.md) and [FLOW §6](./FLOW_AND_LEARNING.md#6-how-to-start--deploy).

Container API is on **:8001** (local uvicorn can stay on :8000).

### Optional: DeepSeek-OCR on a GPU machine

DeepSeek-OCR / OCR-2 need a **self-hosted GPU** endpoint (vLLM/transformers), not the free DeepSeek chat API.

```env
OCR_PROVIDER=deepseek_http
DEEPSEEK_OCR_BASE_URL=http://YOUR_GPU_HOST:8001/v1
DEEPSEEK_OCR_MODEL=deepseek-ai/DeepSeek-OCR-2
```

## Layout

```
projects/advanced_chatbot/
  config.py      # VECTOR_BACKEND, OCR_*, DATABASE_URL
  store/         # memory.py | pgvector.py (same interface)
  ocr.py         # auto / tesseract / ollama_vision / deepseek_http
  service.py     # ingest_bytes (text | OCR)
  graph.py       # rewrite → retrieve → generate → verify
api/advanced_chat_routes.py
deploy/docker-compose.yml
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/advanced-chat/status` | Backend / OCR / phase flags |
| `POST` | `/api/advanced-chat/workspaces` | New workspace |
| `GET` | `/api/advanced-chat/workspaces/{id}` | List docs |
| `POST` | `/api/advanced-chat/workspaces/{id}/upload` | Upsert file (text or OCR) |
| `DELETE` | `/api/advanced-chat/workspaces/{id}/documents/{name}` | Delete + de-index |
| `POST` | `/api/advanced-chat/ask` | Grounded answer |
