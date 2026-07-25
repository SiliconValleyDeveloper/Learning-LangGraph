-- Finance agent F1 schema (versioned migration)
-- Analysis platform only — no brokerage / order tables.
-- Apply via: python -m projects.finance_agent.migrate

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS finance_schema_migrations (
    version TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Least-privilege app roles (passwords set only in controlled environments)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'finance_app') THEN
        CREATE ROLE finance_app NOINHERIT LOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'finance_readonly') THEN
        CREATE ROLE finance_readonly NOINHERIT LOGIN;
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS finance_company_master (
    id              BIGSERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,
    exchange        TEXT NOT NULL CHECK (exchange IN ('NSE', 'BSE', 'OTHER')),
    isin            TEXT,
    name            TEXT NOT NULL,
    series          TEXT,
    status          TEXT NOT NULL DEFAULT 'active',
    sector          TEXT,
    industry        TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    source          TEXT NOT NULL DEFAULT 'unknown',
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (exchange, symbol)
);

CREATE INDEX IF NOT EXISTS idx_finance_company_isin
    ON finance_company_master (isin) WHERE isin IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_finance_company_name
    ON finance_company_master USING gin (to_tsvector('english', name));

CREATE TABLE IF NOT EXISTS finance_eod_prices (
    id              BIGSERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,
    exchange        TEXT NOT NULL CHECK (exchange IN ('NSE', 'BSE', 'OTHER')),
    trade_date      DATE NOT NULL,
    open            NUMERIC(18, 4),
    high            NUMERIC(18, 4),
    low             NUMERIC(18, 4),
    close           NUMERIC(18, 4),
    volume          BIGINT,
    delivery_pct    NUMERIC(8, 4),
    source          TEXT NOT NULL DEFAULT 'unknown',
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (exchange, symbol, trade_date, source)
);

CREATE INDEX IF NOT EXISTS idx_finance_eod_symbol_date
    ON finance_eod_prices (symbol, trade_date DESC);

CREATE TABLE IF NOT EXISTS finance_corp_actions (
    id              BIGSERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,
    exchange        TEXT NOT NULL CHECK (exchange IN ('NSE', 'BSE', 'OTHER')),
    action_type     TEXT NOT NULL,
    ex_date         DATE,
    record_date     DATE,
    ratio           TEXT,
    amount          NUMERIC(18, 6),
    currency        TEXT DEFAULT 'INR',
    details         JSONB NOT NULL DEFAULT '{}'::jsonb,
    source          TEXT NOT NULL DEFAULT 'unknown',
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_finance_corp_actions_symbol_ex
    ON finance_corp_actions (symbol, ex_date DESC);
CREATE INDEX IF NOT EXISTS idx_finance_corp_actions_type_ex
    ON finance_corp_actions (action_type, ex_date DESC);

CREATE TABLE IF NOT EXISTS finance_announcements (
    id              BIGSERIAL PRIMARY KEY,
    symbol          TEXT,
    exchange        TEXT CHECK (exchange IS NULL OR exchange IN ('NSE', 'BSE', 'OTHER')),
    category        TEXT NOT NULL,
    title           TEXT NOT NULL,
    body            TEXT,
    url             TEXT,
    published_at    TIMESTAMPTZ,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    source          TEXT NOT NULL DEFAULT 'unknown',
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_finance_announcements_symbol_pub
    ON finance_announcements (symbol, published_at DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_finance_announcements_category
    ON finance_announcements (category, published_at DESC NULLS LAST);

CREATE TABLE IF NOT EXISTS finance_results_calendar (
    id              BIGSERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,
    exchange        TEXT NOT NULL CHECK (exchange IN ('NSE', 'BSE', 'OTHER')),
    period          TEXT NOT NULL,
    result_date     DATE,
    status          TEXT NOT NULL DEFAULT 'scheduled',
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    source          TEXT NOT NULL DEFAULT 'unknown',
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (exchange, symbol, period, source)
);

CREATE INDEX IF NOT EXISTS idx_finance_results_date
    ON finance_results_calendar (result_date);

CREATE TABLE IF NOT EXISTS finance_fundamentals (
    id              BIGSERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,
    exchange        TEXT NOT NULL DEFAULT 'NSE'
                        CHECK (exchange IN ('NSE', 'BSE', 'OTHER')),
    period_type     TEXT NOT NULL CHECK (period_type IN ('annual', 'quarterly')),
    period          TEXT NOT NULL,
    statement       TEXT NOT NULL DEFAULT 'balance_sheet',
    line_item       TEXT NOT NULL,
    value           NUMERIC(24, 6),
    currency        TEXT NOT NULL DEFAULT 'INR',
    unit            TEXT,
    source          TEXT NOT NULL DEFAULT 'unknown',
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (exchange, symbol, period_type, period, statement, line_item, source)
);

CREATE INDEX IF NOT EXISTS idx_finance_fundamentals_lookup
    ON finance_fundamentals (symbol, period_type, period, statement);

CREATE TABLE IF NOT EXISTS finance_watchlist (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    user_id         TEXT NOT NULL DEFAULT 'default',
    list_name       TEXT NOT NULL DEFAULT 'default',
    symbol          TEXT NOT NULL,
    exchange        TEXT NOT NULL DEFAULT 'NSE'
                        CHECK (exchange IN ('NSE', 'BSE', 'OTHER')),
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, user_id, list_name, exchange, symbol)
);

CREATE INDEX IF NOT EXISTS idx_finance_watchlist_tenant
    ON finance_watchlist (tenant_id, user_id);

CREATE TABLE IF NOT EXISTS finance_documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    symbol          TEXT,
    exchange        TEXT CHECK (exchange IS NULL OR exchange IN ('NSE', 'BSE', 'OTHER')),
    doc_type        TEXT NOT NULL DEFAULT 'filing',
    title           TEXT NOT NULL,
    filename        TEXT,
    mime_type       TEXT NOT NULL DEFAULT 'application/pdf',
    source_url      TEXT,
    content_text    TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    source          TEXT NOT NULL DEFAULT 'upload',
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_finance_documents_symbol
    ON finance_documents (symbol, doc_type);

CREATE TABLE IF NOT EXISTS finance_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES finance_documents (id) ON DELETE CASCADE,
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    chunk_index     INT NOT NULL,
    content         TEXT NOT NULL,
    embedding       vector(768),
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    source          TEXT NOT NULL DEFAULT 'upload',
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_finance_chunks_document
    ON finance_chunks (document_id);
-- IVFFlat needs data; HNSW is fine empty on pgvector recent versions.
CREATE INDEX IF NOT EXISTS idx_finance_chunks_embedding
    ON finance_chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS finance_ingest_runs (
    id              BIGSERIAL PRIMARY KEY,
    source          TEXT NOT NULL,
    job_name        TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running', 'success', 'failed', 'partial')),
    rows_ok         INT NOT NULL DEFAULT 0,
    rows_failed     INT NOT NULL DEFAULT 0,
    file_names      TEXT[] NOT NULL DEFAULT '{}',
    error_summary   TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_finance_ingest_runs_started
    ON finance_ingest_runs (started_at DESC);

CREATE TABLE IF NOT EXISTS finance_ticks (
    id              BIGSERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,
    exchange        TEXT NOT NULL DEFAULT 'NSE'
                        CHECK (exchange IN ('NSE', 'BSE', 'OTHER')),
    ts              TIMESTAMPTZ NOT NULL,
    ltp             NUMERIC(18, 4),
    volume          BIGINT,
    oi              BIGINT,
    bid             NUMERIC(18, 4),
    ask             NUMERIC(18, 4),
    source          TEXT NOT NULL DEFAULT 'kite',
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_finance_ticks_symbol_ts
    ON finance_ticks (symbol, ts DESC);

CREATE TABLE IF NOT EXISTS finance_api_consumers (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    name            TEXT NOT NULL,
    api_key_hash    TEXT NOT NULL UNIQUE,
    tier            TEXT NOT NULL DEFAULT 'free',
    scopes          TEXT[] NOT NULL DEFAULT ARRAY['quotes:read', 'research:read'],
    rate_limit_rpm  INT NOT NULL DEFAULT 60,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_finance_api_consumers_tenant
    ON finance_api_consumers (tenant_id) WHERE is_active;

CREATE TABLE IF NOT EXISTS finance_api_usage (
    id              BIGSERIAL PRIMARY KEY,
    consumer_id     BIGINT REFERENCES finance_api_consumers (id) ON DELETE SET NULL,
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    endpoint        TEXT NOT NULL,
    method          TEXT NOT NULL DEFAULT 'GET',
    status_code     INT,
    latency_ms      INT,
    request_id      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_finance_api_usage_created
    ON finance_api_usage (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_finance_api_usage_consumer
    ON finance_api_usage (consumer_id, created_at DESC);

-- Grants: only finance_* tables (least privilege vs whole public schema)
GRANT SELECT, INSERT, UPDATE, DELETE ON
    finance_schema_migrations,
    finance_company_master,
    finance_eod_prices,
    finance_corp_actions,
    finance_announcements,
    finance_results_calendar,
    finance_fundamentals,
    finance_watchlist,
    finance_documents,
    finance_chunks,
    finance_ingest_runs,
    finance_ticks,
    finance_api_consumers,
    finance_api_usage
TO finance_app;

GRANT USAGE, SELECT ON SEQUENCE
    finance_company_master_id_seq,
    finance_eod_prices_id_seq,
    finance_corp_actions_id_seq,
    finance_announcements_id_seq,
    finance_results_calendar_id_seq,
    finance_fundamentals_id_seq,
    finance_watchlist_id_seq,
    finance_ingest_runs_id_seq,
    finance_ticks_id_seq,
    finance_api_consumers_id_seq,
    finance_api_usage_id_seq
TO finance_app;

GRANT SELECT ON
    finance_schema_migrations,
    finance_company_master,
    finance_eod_prices,
    finance_corp_actions,
    finance_announcements,
    finance_results_calendar,
    finance_fundamentals,
    finance_watchlist,
    finance_documents,
    finance_chunks,
    finance_ingest_runs,
    finance_ticks,
    finance_api_consumers,
    finance_api_usage
TO finance_readonly;

INSERT INTO finance_schema_migrations (version, description)
VALUES (
    '001_finance_f1_schema',
    'Finance F1: core analysis tables, pgvector chunks, API consumer audit, roles'
)
ON CONFLICT (version) DO NOTHING;

COMMIT;
