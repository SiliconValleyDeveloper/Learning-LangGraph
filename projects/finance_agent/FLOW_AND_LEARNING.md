# Finance Agent — Learning & Concepts (layman)

Companion to the locked build plan: [WORKFLOW.md](./WORKFLOW.md).

**We are not a broker.** This product does **analysis only** — no buy, no sell, no order placement.

---

## Why this project exists

A chat LLM that “guesses” stock prices is dangerous and useless for learning.  
This project teaches a **real analysis workflow**:

1. Know what instruments and reports mean  
2. Download exchange knowledge every day (NSE/BSE)  
3. Store facts in Postgres, documents in pgvector  
4. Cache live quotes in Redis (via Zerodha as a **data feed**)  
5. Let a LangGraph agent **research with tools** and **rerank evidence**  
6. Later: prediction / forecasts for **analysis** — still no trading  

---

## Big pieces in plain language

| Piece | Layman meaning |
|-------|----------------|
| **Zerodha Kite** | Live India **prices** for our DB (not used to buy/sell in this app) |
| **NSE/BSE daily files** | Official list of companies + dividends/results/actions |
| **Postgres** | Spreadsheet that never forgets (numbers, calendars) |
| **pgvector** | Search engine over report text (“find paragraphs about debt”) |
| **Redis** | Sticky notes / short-term memory (today’s quote, “job running”) |
| **Dynamic rerank** | After searching 12 chunks, carefully pick the best 5 for *this* question |
| **Autonomous agent** | You give a research goal; it plans tools and writes an **analysis brief** |
| **Your public API** | Others consume **quotes/history/research** from you — not brokerage |

---

## Annual vs quarterly (fundamentals)

| | Annual | Quarterly |
|--|--------|-----------|
| When | Once per FY (often Mar 31 year-end) | Every ~3 months |
| Balance sheet | Full audited picture | Often limited / presentation only |
| Best for | Debt, equity, cash structure | Earnings trend, surprises |

Agent must always show **period + source**.

---

## Filing RAG path (with dynamic rerank)

```text
Question: "What did Reliance say about debt in the annual report?"
  → embed question
  → pgvector: get ~12 similar chunks   (fast, approximate)
  → rerank: score each chunk vs THIS question
  → keep top 5
  → LLM answers only from those 5 + citations
```

Same idea as `projects/advanced_chatbot` — reuse that rerank pattern.

---

## DeepSeek OCR — when it appears

Only if you **upload scanned PDFs/images** of results or annual reports.  
Same cascade as advanced chat: PDF text → Tesseract → Ollama vision → DeepSeek HTTP (if configured).  
Not used for live prices or SQL fundamentals.

---

## Rules the agent must respect

- Don’t present unverified tips as advice  
- Cite exchange filings and dates  
- Circuit / surveillance flags from exchange data when available  
- Tax/STT: flag complexity; don’t act as CA  
- **Never place or propose brokerage orders** — analysis product only  

---

## Autonomy levels (remember)

1. **Observe** — see markets  
2. **Analyze** — written research + sources (**max target**)  

No trade / no “Approve to buy.”

---

## Phase checklist (study + build)

- [x] F0 Workflow document  
- [x] F1 Postgres + Redis infra  
- [x] F2 Daily NSE/BSE ingest  
- [x] F3 Zerodha quotes + cache  
- [x] F3.5 Zerodha → DB → analysis API (auth)  
- [x] F4 Angular dashboard / symbol  
- [x] F5 Fundamentals tables + UI  
- [x] F6 Filings + dynamic rerank  
- [x] F7 Agent L1–L2 analysis  
- [ ] F8 Watchlist  
- [ ] F9 Autonomous research goals  
- [ ] F10 Predict (optional, analysis only)  

---

## Interview one-liners

> “We build a **markets analysis** platform — not a broker. Structured fundamentals live in Postgres; filings go to pgvector with **dynamic CrossEncoder rerank**; live quotes flow Zerodha → Redis/Postgres → our API for other consumers; a LangGraph agent produces cited research briefs. We never place buy/sell orders.”
