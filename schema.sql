-- Run once in the Supabase SQL Editor (Project -> SQL Editor -> New query)
-- to create the tables migrate_catalog_to_db.py populates.

CREATE TABLE IF NOT EXISTS catalog_items (
    id            TEXT PRIMARY KEY,
    category      TEXT,
    label         TEXT,
    dominant_hex  TEXT,
    dominant_hue  INTEGER,
    color_name    TEXT,
    gender        TEXT,
    source        TEXT,
    source_ref    TEXT,
    title         TEXT,
    link          TEXT
);

CREATE TABLE IF NOT EXISTS styling_catalog (
    item_id      TEXT PRIMARY KEY,
    garment_type TEXT,    -- 'shirt' | 'jeans'
    gender       TEXT,
    name         TEXT,
    type         TEXT,
    color        TEXT,
    pattern      TEXT,
    fit          TEXT,
    length       TEXT,
    fabric       TEXT,
    collar       TEXT,    -- shirts only
    sleeve       TEXT,    -- shirts only
    rise         TEXT,    -- jeans only
    style        JSONB,
    occasion     JSONB,
    season       JSONB,   -- shirts only
    image        TEXT
);

-- These tables are only ever written to by the backend (via the secret
-- key, which bypasses RLS), so Row Level Security stays off -- there's
-- no direct client/browser access path to lock down.
