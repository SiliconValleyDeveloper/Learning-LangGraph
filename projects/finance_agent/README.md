# Finance Agent — Markets analysis + autonomous LangGraph

India-first **analysis** platform (not a broker).

**Current phase: F7 (LangGraph L1–L2 analysis agent) implemented.** Next = F8 (watchlist).

## UI

```bash
uvicorn api.main:app --reload --port 8000
cd ui && npm start
```

Open **http://localhost:4200/finance/agent**.

## Analysis agent (F7)

```bash
# research:read is currently an admin scope
python -m projects.finance_agent.consumers create --name research-ui --tier admin

python -m projects.finance_agent.agent \
  "Give me a cited brief covering fundamentals, filings, and risks" \
  --symbol RELIANCE --level L2
```

API: `POST /api/finance/agent/analyse` with `question`, `symbol`, `exchange`,
and `level` (`L1` Observe or `L2` Analyze).

The LangGraph routes intent, executes read-only quote/fundamental/action/filing
tools, writes a grounded brief, verifies evidence IDs, and returns a tool trace.
Order requests are blocked; no brokerage tools exist.

## Filings RAG (F6)

```bash
python -m projects.finance_agent.filings ingest
python -m projects.finance_agent.filings search RELIANCE "What did management say about debt?"
```

API (requires `quotes:read`):

- `GET /api/finance/filings?symbol=RELIANCE`
- `GET /api/finance/filings/RELIANCE/search?q=debt`

Flow: embed query → pgvector top-N → **dynamic rerank** (`RERANK_BACKEND=auto`).

Embeddings: Ollama `nomic-embed-text` when available, else deterministic hash fallback (`FINANCE_EMBED_BACKEND=auto`).

## Completed phases

- F1–F5 (infra → fundamentals UI)
- F6 Filings → pgvector + dynamic rerank + Filings tab
- F7 LangGraph L1–L2 agent → cited briefs + tool trace + guardrails

## Next

> Implement finance **F8**: persisted analysis watchlists.
