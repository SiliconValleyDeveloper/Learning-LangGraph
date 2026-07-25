# Finance sample data (F2)

Small, **synthetic** seed files so the ingest pipeline runs offline — no NSE/BSE
network calls, no data-licensing concerns.

| File | Feeds table |
|------|-------------|
| `sample_company_master.csv` | `finance_company_master` |
| `sample_corp_actions.csv` | `finance_corp_actions` |
| `sample_quotes.csv` | Redis `quote:*` + `finance_ticks` (F3 offline) |

For real data set `FINANCE_INGEST_SOURCE=live` and the `*_URL` env vars
(see WORKFLOW.md §3b compliance gate). Live mode falls back to these samples
if a download fails, so the worker degrades gracefully.
