# Deploy — Advanced chatbot API + Postgres/pgvector

Full start & deploy process for the advanced chatbot stack.

Also covered in: [`projects/advanced_chatbot/FLOW_AND_LEARNING.md`](../projects/advanced_chatbot/FLOW_AND_LEARNING.md) §6.

---

## What runs where

| Piece | Local (Option A) | Full Docker (Option B) |
|-------|------------------|------------------------|
| Ollama (chat + embeds) | Host `:11434` | Host `:11434` (via `host.docker.internal`) |
| Postgres + pgvector | Docker `:5433` | Docker `:5433` |
| FastAPI | Host uvicorn `:8000` | Container `:8001` |
| Angular lab | Host `:4200` | Host `:4200` (point `apiUrl` at API) |
| pgweb | Docker `:8082` | Docker `:8082` |
| Adminer | Docker `:8083` | Docker `:8083` |

---

## Option A — Local API (learning / day-to-day)

Best when you edit Python often (`--reload`).

### 1. One-time setup

```bash
# from repo root
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install 'psycopg[binary]' pytesseract pillow pypdf pypdfium2
brew install tesseract

ollama serve
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

### 2. Start DB + DB UIs

```bash
docker compose -f deploy/docker-compose.yml up -d db db-ui adminer
```

### 3. `.env` at repo root

```env
VECTOR_BACKEND=pgvector
DATABASE_URL=postgresql://langgraph:langgraph@localhost:5433/langgraph
EMBED_DIMS=768
OCR_PROVIDER=auto
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:8b
OLLAMA_EMBED_MODEL=nomic-embed-text
```

### 4. Start API + UI

```bash
# terminal 1
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000

# terminal 2
cd ui && npm start
```

- Lab: http://localhost:4200/chat → **Advanced chat · OCR · pgvector**
- API: http://localhost:8000
- pgweb: http://localhost:8082
- Adminer: http://localhost:8083 (Server `db`, user/password/db `langgraph`)

### 5. Smoke check

```bash
curl http://localhost:8000/api/advanced-chat/status
curl http://localhost:8000/api/health
```

---

## Option B — Full Docker deploy

API image includes Tesseract. Ollama stays on the host.

### 1. Ollama on host

```bash
ollama serve
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

### 2. Build & start everything

```bash
# from repo root
docker compose -f deploy/docker-compose.yml up -d --build
```

| Service | URL |
|---------|-----|
| API (Docker) | http://localhost:8001 |
| Postgres + pgvector | localhost:5433 |
| DB UI (pgweb) | http://localhost:8082 |
| Adminer | http://localhost:8083 |

### 3. Verify

```bash
curl http://localhost:8001/api/advanced-chat/status
curl http://localhost:8001/api/health
```

### 4. Point Angular at the container API (optional)

```ts
// ui/src/environments/environment.ts
apiUrl: 'http://localhost:8001'
```

Leave `apiUrl` on `:8000` if you still run local uvicorn and only use Docker for the DB.

### 5. Logs & stop

```bash
docker compose -f deploy/docker-compose.yml logs -f api
docker compose -f deploy/docker-compose.yml down       # keep DB volume
docker compose -f deploy/docker-compose.yml down -v    # wipe volumes
```

---

## Visualize pgvector

| UI | URL | Login |
|----|-----|-------|
| **pgweb** | http://localhost:8082 | Auto-connected |
| **Adminer** | http://localhost:8083 | PostgreSQL · Server `db` · User/Password/DB `langgraph` |

Useful tables after uploads: `documents`, `chunks`.

```bash
docker exec langgraph-pgvector psql -U langgraph -d langgraph \
  -c "SELECT filename, chunk_count, source_type FROM documents ORDER BY updated_at DESC LIMIT 10;"
```

---

## Notes

- Host Postgres port is **5433** (avoids clash with another Postgres on 5432).
- Container API is **8001** so local uvicorn can stay on **8000**.
- Chat + embeddings always call **Ollama on your Mac** — keep it running.
- DeepSeek-OCR GPU endpoint (optional later): set `DEEPSEEK_OCR_BASE_URL` / `OCR_PROVIDER=deepseek_http`.
