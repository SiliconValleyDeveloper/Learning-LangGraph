# Finance Agent — Locked Workflow (as of now)

This document freezes the **enterprise workflow** before heavy coding.  
This is a **production enterprise application** — see §12 for cross-cutting security, auth, observability, reliability, and delivery requirements every phase must meet.  
Build phases below; do **not** jump to prediction until F0–F5 are solid.

**Status:** Architecture agreed · Implementation not started  
**Related:** [README.md](./README.md) · [FLOW_AND_LEARNING.md](./FLOW_AND_LEARNING.md)

---

## 1. Product goal

Build an **India-first markets analysis platform** (research / intelligence — **not a broker**):

- Dashboard (stocks, global indices, commodities, FX)
- Symbol detail (quote, fundamentals, filings, corp actions)
- Daily **NSE + BSE** knowledge ingest (companies, results, dividends, rules)
- **Autonomous LangGraph agent** (observe → research → analysis brief)
- Real-time India **market data** via **Zerodha Kite** (quotes only — no order placement)
- Global / commodities via Yahoo / MCP where useful
- **Zerodha → DB → your API** so other apps/users consume analysis data from you

**We do not buy or sell.** No order placement, no brokerage, no execution.  
**Not financial advice.** Agent must cite sources + dates. Output is for **analysis / education**.

---

## 2. Locked tech stack

| Layer | Choice | Role |
|-------|--------|------|
| UI | Angular (new `/finance/*` area) | Dashboard, detail, agent console |
| API | FastAPI | REST + job triggers + public data API |
| Orchestration | LangGraph + Ollama | Router, tools, autonomous **analysis** |
| Source of truth | **PostgreSQL** | Company master, EOD, corp actions, watchlists |
| Semantic filings | **pgvector** (same Postgres) | Annual-report / announcement chunks |
| Hot path | **Redis** | Quote cache, job locks, queues, rate limits |
| India market data | **Zerodha Kite** | Live quotes / ticks **only** (no buy/sell APIs) |
| Global / commodities | Yahoo / existing MCP servers | Indices, FX, gold, crude |
| OCR (optional) | Same cascade as advanced chat | Scanned result PDFs |
| RAG quality | **Dynamic rerank** every ask | Reuse `advanced_chatbot/rerank.py` pattern |

**Reuse existing deploy:** `deploy/docker-compose.yml` Postgres on `localhost:5433` — **add Redis** service.

```text
DATABASE_URL=postgresql://langgraph:langgraph@localhost:5433/langgraph
REDIS_URL=redis://localhost:6379/0
VECTOR_BACKEND=pgvector
RERANK_BACKEND=auto
RETRIEVE_CANDIDATES=12
RERANK_TOP_K=5
```

---

## 3. End-to-end architecture

```text
┌──────────────────────────────── Angular /finance/* ─────────────────────────────┐
│  Dashboard │ Symbol detail │ Watchlist │ Agent console │ Research               │
└─────────────────────────────────────┬───────────────────────────────────────────┘
                                      │ HTTP
┌─────────────────────────────────────▼───────────────────────────────────────────┐
│                     FastAPI (finance routes — analysis API only)                 │
└───┬─────────────────┬─────────────────┬─────────────────┬───────────────────────┘
    │                 │                 │                 │
    ▼                 ▼                 ▼                 ▼
 Redis            Postgres          Zerodha           LangGraph
 cache/queue      + pgvector        Kite quotes       analysis agent
    │                 │                 │                 │
    │                 │                 │                 ├── SQL tools
    │                 │                 │                 ├── quote tools
    │                 │                 │                 ├── filings RAG
    │                 │                 │                 │     retrieve → DYNAMIC RERANK
    │                 │                 │                 └── research brief (no orders)
    │                 │
    └──── daily worker: NSE/BSE download → parse → upsert ────┘
```

### Data split (important)

| Data | Store | Why |
|------|--------|-----|
| Balance sheet / P&L numbers | Postgres tables | Structured filter by FY / quarter |
| Watchlist, ingest runs | Postgres | Durable CRUD for analysis |
| Filing / announcement text | pgvector | “What did management say?” |
| Live quotes | Redis TTL + Zerodha | Seconds freshness for analysis |
| Job locks / rate limits | Redis | One daily ingest at a time |
| Graph thread memory | Postgres checkpointer (later) | Enterprise durability |

**Zerodha does not provide balance sheets.** Fundamentals come from Yahoo/filings APIs → Postgres; narrative PDFs → pgvector.  
**Zerodha is used only as a market-data feed** for this product — never to place orders.

---

## 3b. Zerodha → DB → expose to others (data-service pattern)

**Goal:** your backend is the **single Kite market-data client**. It pulls quotes once, stores them, then serves **many end users / apps** through **your own analysis API** — consumers never see your Kite token. You are **not** a broker; you expose **data + analysis**, not trading.

```text
        ONE market-data session               MANY analysis consumers
┌────────────────────────┐        ┌──────────────────────────────────┐
│  Zerodha Kite API      │        │  other users / apps / agent UI    │
│  (quotes / ticks only) │        └───────────────┬──────────────────┘
│  KITE_ACCESS_TOKEN     │                        │  your public REST/WS
└───────────┬────────────┘                        ▼
            │ poll / websocket        ┌──────────────────────────────┐
            ▼                         │  YOUR FastAPI (finance API)   │
┌────────────────────────┐           │  auth · rate limit · quotas   │
│  ingest worker         │           │  quotes · history · research  │
│  (ticker subscribe,    │           └───────────────┬──────────────┘
│   REST poll)           │        read models        │
└───────────┬────────────┘        ┌──────────────────▼──────────────┐
            │ write                │  Redis  (hot latest tick/quote) │
            ▼                      │  Postgres (history, snapshots)  │
      Redis + Postgres  ──────────▶│  → served for ANALYSIS only     │
                                   └─────────────────────────────────┘
```

### How it works

1. **Single producer:** one worker holds the Kite session, subscribes / polls **market data**.
2. **Write path:** ticks → Redis (`quote:{symbol}` latest, short TTL) + Postgres (`finance_eod_prices` / tick history).
3. **Serve path:** your endpoints read **Redis/Postgres**, never call Kite per user request.
   - `GET /api/finance/quote/{symbol}` → Redis first, Postgres fallback
   - `GET /api/finance/history/{symbol}` → Postgres
   - `WS /api/finance/stream` → fan out from Redis pub/sub
4. **Consumers** authenticate to **your** API — get quotes / history / research for analysis.

### Why this shape

| Benefit | Reason |
|---------|--------|
| Kite rate limits respected | 1 client, not N users hammering Kite |
| Token safety | `KITE_ACCESS_TOKEN` stays server-side only |
| Scale reads | Redis/Postgres serve many consumers |
| Analysis product | Same read models feed LangGraph agent + external apps |

### ⚠️ Compliance gate (before public “expose”)

Redistributing exchange market data to third parties is **not automatically allowed**:

- **Zerodha/Kite Connect terms** — check personal-use vs redistribution on your plan.
- **NSE/BSE data licensing** — real-time redistribution usually needs a data vendor license.
- **SEBI** — research/analysis product ≠ registered advisor; don’t present tips as advice.

**Practical stance for this lab:**
- v1 = you / internal demo; “expose” = architecture-ready.
- Delayed / EOD data is safer to share than real-time ticks.
- Confirm Kite plan + licensing before opening to real external users.

### Tables / keys touched

```text
Redis:    quote:{symbol}                 -- latest tick (TTL secs)
          quote:stream                   -- pub/sub channel
Postgres: finance_ticks (symbol, ts, ltp, vol, oi)   -- optional high-freq
          finance_eod_prices             -- daily snapshot
          finance_api_consumers          -- external API keys, tier, quota
          finance_api_usage              -- per-consumer request/rate audit
```

Build slot: **F3.5** — after Zerodha quote ingest (F3), before wide UI (F4).

---

## 4. Daily NSE / BSE knowledge workflow

Runs **after market** (target ~18:30–20:00 IST), worker — not inside a user HTTP request.

```text
Cron / Redis queue
      │
      ▼
 Acquire Redis lock: finance:ingest:daily
      │
      ▼
 Download NSE + BSE files
      │
      ├─ Securities / ISIN master     → finance_company_master
      ├─ Bhavcopy EOD                → finance_eod_prices
      ├─ Corporate actions           → finance_corp_actions
      │     (dividend, bonus, split, rights, …)
      ├─ Announcements / results     → finance_announcements
      │     + results calendar       → finance_results_calendar
      └─ Long text / PDFs            → chunk → embed → pgvector
      │
      ▼
 finance_ingest_runs (status, row counts, errors)
      │
      ▼
 Release lock · set Redis last_success · Dashboard “Data as of …”
```

### Agent tools fed by this job

- `list_companies` / search master  
- `corp_actions(symbol | date_range)`  
- `upcoming_results(watchlist)`  
- `search_announcements(query)` → retrieve → **dynamic rerank**  
- `eod_price(symbol, date)`  

---

## 5. LangGraph agent workflow (analysis only)

### Autonomy levels

| Level | Name | Allowed |
|-------|------|---------|
| L1 | Observe | Quotes, dashboard, calendars |
| L2 | Analyze | Research briefs + citations (**default / max target**) |

**No L3/L4 trading.** No propose-order, no Approve-to-buy/sell, no broker execution.

### Ask / research graph (per user question)

```text
START
  → route_intent
       │
       ├─ quote / markets     → quote_tools (Redis → Zerodha / Yahoo)
       ├─ fundamentals        → SQL Postgres (annual | quarterly)
       ├─ corp_actions        → SQL from daily NSE/BSE ingest
       ├─ filings / research  → pgvector retrieve → DYNAMIC RERANK → generate
       ├─ watchlist           → SQL + live marks (analysis lists)
       ├─ rules / disclaimer  → exchange / disclosure context (no invented law)
       └─ autonomous_goal     → planner → tool loop → analysis report
  → cite_sources
  → END
```

### Autonomous goal loop (agent console)

```text
User goal (e.g. "Flag watchlist names with results this week")
  → plan
  → execute tools (scan → fundamentals → announcements → rerank filings)
  → draft analysis brief + citations
  → publish to UI (Redis flag / poll)
```

### Dynamic rerank (mandatory on filing RAG)

```text
retrieve(RETRIEVE_CANDIDATES) → rerank(query, chunks) → keep RERANK_TOP_K
```

- Runs **per question** (not a static DB ranking).  
- Backend: `RERANK_BACKEND=auto` (CrossEncoder if installed, else lexical).  
- Skip rerank for pure SQL intents (quotes, numeric BS lines).

---

## 6. UI information architecture

```text
/finance
  /dashboard              India + global indices + commodities + FX tiles
  /markets/stocks
  /markets/indices
  /markets/commodities
  /markets/fx
  /symbol/:ticker         Overview | Fundamentals | Filings | Actions | Agent
  /watchlist              Symbols tracked for analysis (not a brokerage book)
  /agent                  Plans, tool trace, analysis briefs
  /research               Filing Q&A (RAG + rerank)
```

**Stock detail** pulls: Zerodha quote · Postgres fundamentals · corp actions · pgvector filings.

---

## 7. Study-before-predict (learning gate)

Do **not** ship prediction models until the team (and agent tools) understand:

1. Instruments (equity, F&O, indices, commodities, FX)  
2. How to read P&L, balance sheet, cash flow (annual vs quarterly)  
3. Exchange / disclosure basics (circuits, filings, calendars)  
4. Supply/demand ideas (volume, OI, FII/DII — later)  
5. Then features → backtest → **forecast for analysis only** (no live trading)

Document this in [FLOW_AND_LEARNING.md](./FLOW_AND_LEARNING.md).

---

## 8. Build phases (workflow order)

| Phase | Name | Deliverable | Done when |
|-------|------|-------------|-----------|
| **F0** | Workflow freeze | This doc + README + env stubs | ✅ |
| **F1** | Infra | Compose Redis + Postgres schemas `finance_*` | ✅ Redis service + `sql/001_finance_f1_schema.sql` + `python -m projects.finance_agent.migrate` |
| **F2** | Daily ingest | NSE/BSE worker + Redis lock | ✅ `ingest.py` (lock, `finance_ingest_runs`, idempotent upserts) → company master + corp actions (dividend/bonus/split/buyback) |
| **F3** | Quotes API | Zerodha **quotes** + Redis TTL | ✅ `quotes.py` + `kite_client.py` (sample/auto/kite) → Redis TTL + `finance_ticks` + `/api/finance/quote*` |
| **F3.5** | Data service | Zerodha → DB → **your analysis API** | ✅ API keys (`finance_api_consumers`), scopes, Redis RPM limit, `finance_api_usage` audit |
| **F4** | Angular shell | `/finance` routes + dashboard + symbol stub | ✅ Dashboard tiles, markets lists, symbol detail tabs, topbar link |
| **F5** | Fundamentals | Annual/quarterly → Postgres + detail tab | ✅ `fundamentals.py` + sample CSV → `finance_fundamentals` + `/api/finance/fundamentals/{symbol}` + symbol Fundamentals tab |
| **F6** | Filings RAG | Upload/ingest PDFs → pgvector → **dynamic rerank** | ✅ sample filings → `finance_documents`/`finance_chunks` + retrieve→rerank API + Filings tab |
| **F7** | Agent L1–L2 | LangGraph router + tools + console | ✅ intent router + read-only tools + evidence verification + `/finance/agent` cited briefs |
| **F8** | Watchlist | Postgres + UI (analysis tracking) | Persisted lists |
| **F9** | Autonomous goals | Multi-step research goals → reports | Goal → brief in agent console |
| **F10** | Predict (optional) | Supply/demand features + backtest | Forecasts for **analysis only** |

**Current position:** F7 complete. Next = **F8** (persisted analysis watchlists).

---

## 9. Suggested Postgres tables (F1)

```text
finance_company_master
finance_eod_prices
finance_corp_actions
finance_announcements
finance_results_calendar
finance_fundamentals          -- line_item, period_type, period, value, source
finance_watchlist             -- analysis tracking (not brokerage holdings)
finance_documents             -- filing metadata
finance_chunks                -- content + vector(768)
finance_ingest_runs
finance_ticks                 -- optional high-freq ticks (F3.5)
finance_api_consumers         -- external API keys / tier / quota (F3.5)
finance_api_usage             -- per-consumer rate + audit (F3.5)
```

---

## 10. Env vars (planned)

```bash
# Shared with advanced chatbot
DATABASE_URL=postgresql://langgraph:langgraph@localhost:5433/langgraph
VECTOR_BACKEND=pgvector
RERANK_BACKEND=auto
RETRIEVE_CANDIDATES=12
RERANK_TOP_K=5

# Finance (analysis platform — no trading keys beyond market-data session)
REDIS_URL=redis://localhost:6379/0
FINANCE_ENABLED=true
KITE_API_KEY=
KITE_API_SECRET=
KITE_ACCESS_TOKEN=          # daily session for QUOTES only — never commit
NSE_BSE_INGEST_ENABLED=true
FINANCE_INGEST_CRON=0 19 * * 1-5   # document only; implement in F2
```

---

## 11. Explicit non-goals

- **We are not a broker** — no buy, sell, place-order, or order-management APIs  
- Guaranteed returns / “tips” marketing  
- Unsupervised or HITL live trading  
- Replacing a SEBI-registered advisor  
- Storing API secrets in git  

---

## 12. Enterprise-grade requirements (cross-cutting)

This is a **production enterprise application**, so every phase must satisfy these — not bolt them on later.

### Security
- Secrets in a **vault / env**, never in git (`KITE_*`, DB creds). `.env` local only; managed secrets in prod.
- **Encryption**: TLS in transit; encrypt sensitive columns / disks at rest.
- **Input validation** on all API + agent tool params (Pydantic models).
- **Dependency + image scanning** (SCA), least-privilege DB roles.

### AuthN / AuthZ / multi-tenancy
- Consumer auth via **API keys / JWT / OAuth**; per-consumer tier + scopes (`finance_api_consumers`).
- **RBAC** (viewer / analyst / admin); tenant isolation on shared Postgres (row scoping or schema-per-tenant).
- **Quotas + rate limits** enforced in Redis per consumer/tenant.

### Observability
- **Structured logging** (JSON, request id, tenant id).
- **Metrics** (Prometheus): API latency, cache hit rate, ingest rows, agent tool latency, rerank time.
- **Tracing** (OpenTelemetry) across API → agent → DB/Redis/Kite.
- **Dashboards + alerts** (Grafana): ingest failures, stale data, Kite errors, queue backlog.

### Reliability / scale
- **Stateless API** behind load balancer; workers scale horizontally.
- **Redis** for cache + queue + locks; consider Redis Cluster / managed (ElastiCache) in prod.
- **Postgres**: connection pooling (PgBouncer), read replicas for heavy analytics, partition `finance_ticks`/`finance_eod_prices` by date.
- **Backpressure + retries** with idempotency on ingest; circuit breaker around Kite/Yahoo.
- **Graceful degradation**: serve last-good cached/EOD data when live feed down.

### Data governance & audit
- **Provenance** on every row (`source`, `fetched_at`); immutable **audit log** (`finance_api_usage`, ingest runs).
- **Data retention** + PII policy (consumer accounts).
- **Compliance gate** (see §3b) before any external data exposure.

### Delivery / ops
- **CI/CD**: lint, type-check, tests, migration check, image build/scan, deploy.
- **DB migrations** versioned (e.g. Alembic); no manual schema edits in prod.
- **Environments**: dev / staging / prod parity via the same Compose → K8s manifests.
- **Health/readiness probes**, blue-green or rolling deploys, DB backup + restore drills.
- **DR**: documented RPO/RTO; nightly backups; restore tested.

> Each build phase (F1–F10) is “done” only when its **security, observability, and migration** pieces are also in place.

---

## 13. Running the daily ingest (F2)

```bash
docker compose -f deploy/docker-compose.yml up -d db redis
source .venv/bin/activate
python -m projects.finance_agent.migrate          # schema (F1 + F2)
python -m projects.finance_agent.ingest           # run daily ingest
python -m projects.finance_agent.ingest --status  # counts + last run
```

- Default `FINANCE_INGEST_SOURCE=sample` (offline, bundled CSVs).
- Set `FINANCE_INGEST_SOURCE=live` + `NSE_EQUITY_URL` / `BSE_EQUITY_URL` /
  `CORP_ACTIONS_URL` for real files (see §3b compliance gate); live falls back to
  sample on download failure.
- Idempotent: re-running does not duplicate rows (company `ON CONFLICT (exchange,symbol)`;
  corp actions via `dedupe_key`). Each run is audited in `finance_ingest_runs`.
- Redis lock `finance:ingest:daily` prevents concurrent runs (set
  `FINANCE_INGEST_REQUIRE_LOCK=true` to hard-fail if Redis is down).

## 14. Quotes pipeline (F3)

```bash
python -m projects.finance_agent.quotes refresh
python -m projects.finance_agent.quotes get RELIANCE
python -m projects.finance_agent.quotes status
```

HTTP (with API running):

```bash
curl http://localhost:8000/api/finance/status
curl http://localhost:8000/api/finance/quote/RELIANCE
curl 'http://localhost:8000/api/finance/quotes?refresh=true'
curl -X POST http://localhost:8000/api/finance/quotes/refresh \
  -H 'Content-Type: application/json' \
  -d '{"symbols":["RELIANCE","TCS"]}'
```

- `FINANCE_QUOTE_SOURCE=auto` → Kite when `KITE_API_KEY` + `KITE_ACCESS_TOKEN` set; else sample.
- Redis key: `quote:{EXCHANGE}:{SYMBOL}` with TTL (`FINANCE_QUOTE_TTL`, default 15s).
- Optional tick history in `finance_ticks` (`FINANCE_QUOTE_PERSIST_TICKS=true`).
- Pub/sub channel `finance:quote:stream` for live UI later.
- **No order APIs** — quotes only.

## 15. API consumers / auth (F3.5)

Create a consumer (plaintext key shown **once**):

```bash
python -m projects.finance_agent.consumers create --name demo --tier free
python -m projects.finance_agent.consumers list
# python -m projects.finance_agent.consumers revoke --id 1
```

Call the analysis API:

```bash
export FINANCE_KEY='fk_...'   # from create output
curl -H "X-API-Key: $FINANCE_KEY" http://localhost:8000/api/finance/quote/RELIANCE
curl -H "Authorization: Bearer $FINANCE_KEY" \
  'http://localhost:8000/api/finance/quotes?refresh=true'
# refresh needs quotes:refresh scope → use --tier pro
curl -X POST -H "X-API-Key: $FINANCE_KEY" \
  -H 'Content-Type: application/json' \
  http://localhost:8000/api/finance/quotes/refresh \
  -d '{"symbols":["RELIANCE"]}'
```

| Tier | Default RPM | Scopes |
|------|-------------|--------|
| `free` | 60 | `quotes:read` |
| `pro` | 300 | `quotes:read`, `quotes:refresh` |
| `admin` | 1000 | + `research:read`, `admin` |

- Keys stored as SHA-256(`pepper` + key); pepper = `FINANCE_API_KEY_PEPPER`.
- Rate limit: Redis counter `finance:rl:{id}:{minute}`.
- Every authenticated call audited in `finance_api_usage`.
- `/api/finance/status` stays public; quote routes respect `FINANCE_API_AUTH_REQUIRED` (default true).
- Compliance gate in §3b still applies before public redistribution.

## 16. Angular UI (F4)

```bash
# terminal 1 — API
uvicorn api.main:app --reload --port 8000

# terminal 2 — UI
cd ui && npm start
```

Open **http://localhost:4200/finance**

1. Create a free data key: `python -m projects.finance_agent.consumers create --name ui --tier free`
   For the F7 agent, create an admin research key: `python -m projects.finance_agent.consumers create --name research-ui --tier admin`
2. Paste key into the dashboard **API key** field → Save
3. Click **Refresh quotes** — tiles populate (indices / stocks / commodities / FX)
4. Open a symbol for Overview + corporate actions (from F2 ingest)

Routes: `/finance`, `/finance/markets/:market`, `/finance/symbol/:symbol`, `/finance/agent`; watchlist remains a stub.

## 17. Next action

Start **F8**: persisted watchlists for analysis tracking (not brokerage holdings).

When ready: ask the agent to **“Implement finance F8”**.
