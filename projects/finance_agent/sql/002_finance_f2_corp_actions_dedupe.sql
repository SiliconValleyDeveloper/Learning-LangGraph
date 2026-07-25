-- Finance F2: make corp actions idempotent for repeated daily ingests.
-- Adds a natural-key dedupe_key so ON CONFLICT upserts are safe.

BEGIN;

ALTER TABLE finance_corp_actions
    ADD COLUMN IF NOT EXISTS dedupe_key TEXT;

-- Backfill any existing rows with a deterministic key.
UPDATE finance_corp_actions
SET dedupe_key = md5(
        coalesce(exchange, '') || '|' ||
        coalesce(symbol, '') || '|' ||
        coalesce(action_type, '') || '|' ||
        coalesce(ex_date::text, '') || '|' ||
        coalesce(source, '')
    )
WHERE dedupe_key IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_finance_corp_actions_dedupe
    ON finance_corp_actions (dedupe_key);

INSERT INTO finance_schema_migrations (version, description)
VALUES (
    '002_finance_f2_corp_actions_dedupe',
    'Finance F2: corp_actions dedupe_key + unique index for idempotent ingest'
)
ON CONFLICT (version) DO NOTHING;

COMMIT;
