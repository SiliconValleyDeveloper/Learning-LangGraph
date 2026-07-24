# Incident Response Playbook

Use this playbook when a local AI service fails during a learning session.

## Severity levels

- SEV-3: lesson slow but usable
- SEV-2: one service down (API, UI, or Ollama)
- SEV-1: no answers and learners blocked

## First checks

1. Confirm Ollama is running: `ollama list`
2. Confirm chat model: `qwen3:8b`
3. Confirm embedding model: `nomic-embed-text`
4. Confirm API health: `GET http://127.0.0.1:8000/api/health`
5. Confirm UI: `http://localhost:4200/chat/rag`

## Recovery steps

Restart the FastAPI process with reload, then restart `ng serve` if the UI is
stale. If retrieval returns empty context, rebuild the vector store by
restarting the API so `build_vector_store` reloads documents.

## Communication

Post a short update in #ops-oncall with severity, impact, and ETA. After
recovery, note whether public docs or private docs were involved.
