# Doc Upload Q&A

A small **project** (not a lesson) next to `Learning/`: users upload documents, they are chunked and embedded on upload, then questions are answered only from those docs.

Also available as the **Doc upload · ask** chip on the main lab (`http://localhost:4200/chat`):

![Lab with Doc upload · ask next to Complex RAG](../../docs/langgraph-lab-complex-rag.png)

## What it demonstrates

1. **Dynamic ingest** — upload `.md` / `.txt` at runtime
2. **Incremental upsert** — new file → chunk → embed → add; same filename replaces old chunks only
3. **Retrieve → generate** LangGraph over a per-workspace vector store
4. **UI + API** — Angular page + FastAPI endpoints

## Run

```bash
# terminal 1
source .venv/bin/activate
pip install -r requirements.txt   # includes python-multipart
uvicorn api.main:app --reload --port 8000

# terminal 2
cd ui && npm start
```

Open: [http://localhost:4200/chat/doc-rag](http://localhost:4200/chat/doc-rag)

Ollama models: `qwen3:8b` (chat) and `nomic-embed-text` (embeddings).

## API

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/doc-rag/workspaces` | Create workspace |
| `GET` | `/api/doc-rag/workspaces/{id}` | List docs / counts |
| `POST` | `/api/doc-rag/workspaces/{id}/upload` | Multipart file upload |
| `POST` | `/api/doc-rag/workspaces/{id}/seed` | Sample onboarding + refund docs |
| `DELETE` | `/api/doc-rag/workspaces/{id}/documents/{name}` | Remove one file + its chunks |
| `POST` | `/api/doc-rag/ask` | `{ workspace_id, question }` → answer + citations |

## Layout

```
projects/doc_upload_rag/
  store.py   # workspace files + InMemoryVectorStore upsert/delete
  graph.py   # retrieve → generate
api/doc_rag_routes.py
ui/.../doc-rag/   # upload + chat UI
```

Uploads live under `.data/doc-uploads/{workspace_id}/` (gitignored). Vectors stay in memory for the API process; if the API restarts, files on disk are re-indexed when the workspace is opened again.
