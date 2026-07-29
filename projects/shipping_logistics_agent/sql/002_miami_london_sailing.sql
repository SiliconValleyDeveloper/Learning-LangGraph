INSERT INTO shipping.ports (unlocode, name, country_code, timezone)
VALUES
    ('USMIA', 'Miami', 'US', 'America/New_York'),
    ('GBLGP', 'London Gateway', 'GB', 'Europe/London')
ON CONFLICT (unlocode) DO NOTHING;

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
      ('9783459'::text, 'MS410W'::text, 'USMIA'::text, 'GBLGP'::text,
       now() + interval '12 days', now() + interval '26 days',
       640, 1900::numeric, 3050::numeric, 980::numeric, false)
) AS x(imo, voyage_number, origin_code, destination_code, departure_at,
       arrival_at, available_teu, rate20, rate40, reefer, dg)
JOIN shipping.vessels v ON v.imo_number = x.imo
JOIN shipping.ports op ON op.unlocode = x.origin_code
JOIN shipping.ports dp ON dp.unlocode = x.destination_code
ON CONFLICT (voyage_number) DO NOTHING;

INSERT INTO shipping.schema_migrations (version, description)
VALUES (
    '002_miami_london_sailing',
    'Add Miami and London Gateway ports with one sample sailing'
)
ON CONFLICT (version) DO NOTHING;
