CREATE SCHEMA IF NOT EXISTS shipping;

CREATE TABLE IF NOT EXISTS shipping.schema_migrations (
    version         TEXT PRIMARY KEY,
    description     TEXT NOT NULL,
    applied_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS shipping.customers (
    id              BIGSERIAL PRIMARY KEY,
    customer_code   TEXT NOT NULL UNIQUE,
    name            TEXT NOT NULL,
    country_code    CHAR(2) NOT NULL,
    email           TEXT,
    credit_status   TEXT NOT NULL DEFAULT 'approved'
                        CHECK (credit_status IN ('approved', 'hold', 'blocked')),
    credit_limit_usd NUMERIC(14, 2) NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS shipping.ports (
    id              BIGSERIAL PRIMARY KEY,
    unlocode        CHAR(5) NOT NULL UNIQUE,
    name            TEXT NOT NULL,
    country_code    CHAR(2) NOT NULL,
    timezone        TEXT NOT NULL DEFAULT 'UTC',
    active          BOOLEAN NOT NULL DEFAULT true
);

CREATE TABLE IF NOT EXISTS shipping.vessels (
    id              BIGSERIAL PRIMARY KEY,
    imo_number      CHAR(7) NOT NULL UNIQUE,
    name            TEXT NOT NULL,
    vessel_type     TEXT NOT NULL DEFAULT 'container',
    capacity_teu    INT NOT NULL CHECK (capacity_teu > 0),
    flag_country    CHAR(2),
    active          BOOLEAN NOT NULL DEFAULT true
);

CREATE TABLE IF NOT EXISTS shipping.sailings (
    id              BIGSERIAL PRIMARY KEY,
    vessel_id       BIGINT NOT NULL REFERENCES shipping.vessels(id),
    voyage_number   TEXT NOT NULL UNIQUE,
    origin_port_id  BIGINT NOT NULL REFERENCES shipping.ports(id),
    destination_port_id BIGINT NOT NULL REFERENCES shipping.ports(id),
    departure_at    TIMESTAMPTZ NOT NULL,
    arrival_at      TIMESTAMPTZ NOT NULL,
    status          TEXT NOT NULL DEFAULT 'scheduled'
                        CHECK (status IN ('scheduled', 'departed', 'arrived', 'cancelled')),
    available_teu   INT NOT NULL CHECK (available_teu >= 0),
    base_rate_20_usd NUMERIC(12, 2) NOT NULL CHECK (base_rate_20_usd >= 0),
    base_rate_40_usd NUMERIC(12, 2) NOT NULL CHECK (base_rate_40_usd >= 0),
    reefer_surcharge_usd NUMERIC(12, 2) NOT NULL DEFAULT 0,
    dangerous_goods_allowed BOOLEAN NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (arrival_at > departure_at),
    CHECK (origin_port_id <> destination_port_id)
);

CREATE INDEX IF NOT EXISTS sailings_route_departure_idx
    ON shipping.sailings (origin_port_id, destination_port_id, departure_at);

CREATE TABLE IF NOT EXISTS shipping.quotations (
    id              UUID PRIMARY KEY,
    quote_ref       TEXT NOT NULL UNIQUE,
    customer_id     BIGINT NOT NULL REFERENCES shipping.customers(id),
    sailing_id      BIGINT NOT NULL REFERENCES shipping.sailings(id),
    container_type  TEXT NOT NULL CHECK (container_type IN ('20GP', '40GP', '40HC', '40RF')),
    container_qty   INT NOT NULL CHECK (container_qty > 0),
    cargo_weight_kg NUMERIC(12, 2) NOT NULL CHECK (cargo_weight_kg > 0),
    cargo_description TEXT NOT NULL,
    dangerous_goods BOOLEAN NOT NULL DEFAULT false,
    ocean_freight_usd NUMERIC(14, 2) NOT NULL,
    surcharges_usd  NUMERIC(14, 2) NOT NULL DEFAULT 0,
    total_usd       NUMERIC(14, 2) NOT NULL,
    status          TEXT NOT NULL DEFAULT 'approved'
                        CHECK (status IN ('approved', 'accepted', 'expired', 'rejected')),
    valid_until     DATE NOT NULL,
    approval_id     UUID,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS shipping.bookings (
    id              UUID PRIMARY KEY,
    booking_ref     TEXT NOT NULL UNIQUE,
    quotation_id    UUID NOT NULL REFERENCES shipping.quotations(id),
    customer_id     BIGINT NOT NULL REFERENCES shipping.customers(id),
    sailing_id      BIGINT NOT NULL REFERENCES shipping.sailings(id),
    status          TEXT NOT NULL DEFAULT 'confirmed'
                        CHECK (status IN ('confirmed', 'gate_in', 'loaded', 'departed', 'arrived', 'delivered', 'cancelled')),
    approval_id     UUID,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS shipping.containers (
    id              BIGSERIAL PRIMARY KEY,
    container_number CHAR(11) NOT NULL UNIQUE,
    container_type  TEXT NOT NULL,
    booking_id      UUID REFERENCES shipping.bookings(id),
    seal_number     TEXT,
    gross_weight_kg NUMERIC(12, 2),
    status          TEXT NOT NULL DEFAULT 'available'
);

CREATE TABLE IF NOT EXISTS shipping.shipment_events (
    id              BIGSERIAL PRIMARY KEY,
    booking_id      UUID NOT NULL REFERENCES shipping.bookings(id) ON DELETE CASCADE,
    event_code      TEXT NOT NULL,
    event_time      TIMESTAMPTZ NOT NULL,
    port_id         BIGINT REFERENCES shipping.ports(id),
    description     TEXT NOT NULL,
    source          TEXT NOT NULL DEFAULT 'sample',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS shipment_events_booking_time_idx
    ON shipping.shipment_events (booking_id, event_time DESC);

CREATE TABLE IF NOT EXISTS shipping.approval_requests (
    id              UUID PRIMARY KEY,
    thread_id       TEXT NOT NULL UNIQUE,
    action          TEXT NOT NULL CHECK (action IN ('create_quotation', 'create_booking')),
    proposal        JSONB NOT NULL,
    risk_review     JSONB NOT NULL DEFAULT '{}'::jsonb,
    status          TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'approved', 'rejected', 'executed')),
    reviewer        TEXT,
    reviewer_note   TEXT,
    requested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_at      TIMESTAMPTZ,
    executed_at     TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS shipping.audit_log (
    id              BIGSERIAL PRIMARY KEY,
    thread_id       TEXT,
    actor           TEXT NOT NULL,
    action          TEXT NOT NULL,
    entity_type     TEXT,
    entity_id       TEXT,
    payload         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO shipping.customers
    (customer_code, name, country_code, email, credit_status, credit_limit_usd)
VALUES
    ('ACME-IN', 'ACME Industrial Exports', 'IN', 'logistics@acme.example', 'approved', 250000),
    ('GLOBEX-SG', 'Globex Trading Pte Ltd', 'SG', 'ops@globex.example', 'approved', 500000),
    ('INITECH-US', 'Initech Distribution', 'US', 'freight@initech.example', 'hold', 100000)
ON CONFLICT (customer_code) DO NOTHING;

INSERT INTO shipping.ports (unlocode, name, country_code, timezone)
VALUES
    ('INNSA', 'Nhava Sheva (JNPT)', 'IN', 'Asia/Kolkata'),
    ('SGSIN', 'Singapore', 'SG', 'Asia/Singapore'),
    ('AEDXB', 'Jebel Ali', 'AE', 'Asia/Dubai'),
    ('NLRTM', 'Rotterdam', 'NL', 'Europe/Amsterdam'),
    ('USNYC', 'New York', 'US', 'America/New_York')
ON CONFLICT (unlocode) DO NOTHING;

INSERT INTO shipping.vessels (imo_number, name, capacity_teu, flag_country)
VALUES
    ('9319466', 'MV Ocean Pioneer', 8500, 'SG'),
    ('9783459', 'MV Meridian Star', 14000, 'LR'),
    ('9876543', 'MV Eastern Bridge', 6200, 'IN')
ON CONFLICT (imo_number) DO NOTHING;

INSERT INTO shipping.sailings (
    vessel_id, voyage_number, origin_port_id, destination_port_id,
    departure_at, arrival_at, available_teu, base_rate_20_usd,
    base_rate_40_usd, reefer_surcharge_usd, dangerous_goods_allowed
)
SELECT v.id, x.voyage_number, op.id, dp.id,
       x.departure_at, x.arrival_at, x.available_teu, x.rate20,
       x.rate40, x.reefer, x.dg
FROM (
    VALUES
      ('9319466'::text, 'OP101E'::text, 'INNSA'::text, 'SGSIN'::text,
       now() + interval '7 days', now() + interval '14 days', 420, 980::numeric, 1550::numeric, 700::numeric, true),
      ('9876543', 'EB220E', 'INNSA', 'AEDXB',
       now() + interval '10 days', now() + interval '15 days', 180, 850, 1380, 620, false),
      ('9783459', 'MS330W', 'SGSIN', 'NLRTM',
       now() + interval '15 days', now() + interval '42 days', 760, 1450, 2320, 850, true),
      ('9783459', 'MS331W', 'INNSA', 'NLRTM',
       now() + interval '21 days', now() + interval '47 days', 510, 1750, 2780, 900, true),
      ('9319466', 'OP102E', 'SGSIN', 'USNYC',
       now() + interval '30 days', now() + interval '58 days', 350, 2100, 3350, 1100, false)
) AS x(imo, voyage_number, origin_code, destination_code, departure_at,
       arrival_at, available_teu, rate20, rate40, reefer, dg)
JOIN shipping.vessels v ON v.imo_number = x.imo
JOIN shipping.ports op ON op.unlocode = x.origin_code
JOIN shipping.ports dp ON dp.unlocode = x.destination_code
ON CONFLICT (voyage_number) DO NOTHING;

INSERT INTO shipping.schema_migrations (version, description)
VALUES ('001_shipping_schema', 'Shipping/logistics sample schema, approvals, and seed data')
ON CONFLICT (version) DO NOTHING;

