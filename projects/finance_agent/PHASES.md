# Finance agent phases — track progress (see WORKFLOW.md)

**Product:** markets **analysis** platform — **not a broker**. No buy/sell.

- [x] F0 Workflow freeze (WORKFLOW.md + docs)
- [x] F1 Infra: Redis + finance_* Postgres schema
- [x] F2 Daily NSE/BSE ingest worker (lock, run audit, idempotent upserts)
- [x] F3 Zerodha quotes + Redis cache (market data only)
- [x] F3.5 Data service: API keys + scopes + Redis rate limit + usage audit
- [x] F4 Angular /finance shell (dashboard + symbol)
- [x] F5 Fundamentals (annual/quarterly) → Postgres + UI
- [x] F6 Filings RAG + dynamic rerank
- [x] F7 Agent L1–L2 (analysis briefs + citations)
- [ ] F8 Watchlist (analysis tracking)
- [ ] F9 Autonomous research goals → reports
- [ ] F10 Predict (optional, analysis/forecast only — no trading)
