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

-- One table per garment category (not one shared table with a
-- garment_type discriminator) -- each table only has the columns that
-- category actually uses, no unused NULL columns from other categories.
-- gender is a column within each table (not a separate table per
-- gender), since these files are self-labeled "female_*" today but the
-- same table is meant to also hold "male_*" rows once that data exists.
--
-- Superseded styling_catalog (single shared table) -- drop it if it's
-- still around from an earlier setup and nothing else depends on it:
DROP TABLE IF EXISTS styling_catalog;

CREATE TABLE IF NOT EXISTS shirts (
    item_id  TEXT PRIMARY KEY,
    gender   TEXT,
    name     TEXT,
    type     TEXT,
    color    TEXT,
    pattern  TEXT,
    fit      TEXT,
    length   TEXT,
    fabric   TEXT,
    collar   TEXT,
    sleeve_design  TEXT,  -- aesthetic style, e.g. "puff", "bishop", "bell" -- often unset
    sleeve_length  TEXT,  -- shoulder-to-cuff length, e.g. "half", "full"
    style    JSONB,
    occasion JSONB,
    season   JSONB,
    image    TEXT
);

CREATE TABLE IF NOT EXISTS jeans (
    item_id  TEXT PRIMARY KEY,
    gender   TEXT,
    name     TEXT,
    type     TEXT,
    color    TEXT,
    pattern  TEXT,
    fit      TEXT,
    length   TEXT,
    fabric   TEXT,
    rise     TEXT,
    style    JSONB,
    occasion JSONB,
    image    TEXT
);

CREATE TABLE IF NOT EXISTS tops (
    item_id    TEXT PRIMARY KEY,
    gender     TEXT,
    name       TEXT,
    type       TEXT,
    color      TEXT,
    pattern    TEXT,
    fit        TEXT,
    length     TEXT,
    fabric     TEXT,
    sleeve_design  TEXT,  -- aesthetic style, e.g. "puff", "bishop", "bell" -- often unset
    sleeve_length  TEXT,  -- shoulder-to-cuff length, e.g. "sleeveless", "short_sleeve", "long_sleeve"
    neckline   TEXT,
    properties JSONB,
    style      JSONB,
    occasion   JSONB,
    season     JSONB,
    image      TEXT
);

CREATE TABLE IF NOT EXISTS dresses (
    item_id    TEXT PRIMARY KEY,
    gender     TEXT,
    name       TEXT,
    type       TEXT,
    color      TEXT,
    pattern    TEXT,
    fit        TEXT,
    length     TEXT,
    neckline   TEXT,
    sleeve_design  TEXT,  -- aesthetic style, e.g. "puff", "bishop", "bell" -- often unset
    sleeve_length  TEXT,  -- shoulder-to-cuff length, e.g. "sleeveless", "short_sleeve", "long_sleeve"
    fabric     TEXT,
    properties JSONB,
    pockets    INTEGER,
    style      JSONB,
    occasion   JSONB,
    season     JSONB,
    image      TEXT
);

CREATE TABLE IF NOT EXISTS skirts (
    item_id    TEXT PRIMARY KEY,
    gender     TEXT,
    name       TEXT,
    type       TEXT,
    color      TEXT,
    pattern    TEXT,
    fit        TEXT,
    length     TEXT,
    rise       TEXT,
    fabric     TEXT,
    properties JSONB,
    pockets    INTEGER,
    style      JSONB,
    occasion   JSONB,
    season     JSONB,
    image      TEXT
);

CREATE TABLE IF NOT EXISTS trousers (
    item_id    TEXT PRIMARY KEY,
    gender     TEXT,
    name       TEXT,
    type       TEXT,
    color      TEXT,
    pattern    TEXT,
    fit        TEXT,
    length     TEXT,
    rise       TEXT,
    fabric     TEXT,
    properties JSONB,
    pockets    INTEGER,
    style      JSONB,
    occasion   JSONB,
    season     JSONB,
    image      TEXT
);

-- These tables are only ever written to by the backend (via the secret
-- key, which bypasses RLS), so Row Level Security stays off -- there's
-- no direct client/browser access path to lock down.

-- One-off migration for an ALREADY-CREATED database: CREATE TABLE IF NOT
-- EXISTS above won't add columns to tables that already exist. Run this
-- block once in the Supabase SQL Editor to split the old single `sleeve`
-- column (shirts/tops/dresses) into `sleeve_design` (aesthetic style,
-- e.g. "puff" -- often unset) and `sleeve_length` (shoulder-to-cuff
-- length, e.g. "half"/"full"/"sleeveless") -- they're different
-- attributes and were being conflated under one column. Safe to re-run.
ALTER TABLE shirts  ADD COLUMN IF NOT EXISTS sleeve_design TEXT;
ALTER TABLE shirts  ADD COLUMN IF NOT EXISTS sleeve_length TEXT;
ALTER TABLE tops    ADD COLUMN IF NOT EXISTS sleeve_design TEXT;
ALTER TABLE tops    ADD COLUMN IF NOT EXISTS sleeve_length TEXT;
ALTER TABLE dresses ADD COLUMN IF NOT EXISTS sleeve_design TEXT;
ALTER TABLE dresses ADD COLUMN IF NOT EXISTS sleeve_length TEXT;

ALTER TABLE shirts  DROP COLUMN IF EXISTS sleeve;
ALTER TABLE tops    DROP COLUMN IF EXISTS sleeve;
ALTER TABLE dresses DROP COLUMN IF EXISTS sleeve;

-- ─────────────────────────────────────────────────────────────────
-- App data tables: users, closet_items, ai_profiles, saved_looks.
-- These replace the local JSON files (users.json, closet_data.json,
-- ai_profiles.json, saved_looks.json) that used to live on Cloud Run's
-- disk -- which is ephemeral per container instance, so that data was
-- silently getting wiped on every cold start / new revision.
--
-- Same access pattern as the catalog tables above: only the backend
-- talks to these, via the secret key (bypasses RLS), so RLS stays off.
-- ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    email      TEXT PRIMARY KEY,
    user_id    TEXT UNIQUE NOT NULL,
    name       TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- One row per detected garment, not one JSON blob per user -- avoids the
-- old "read the whole list, mutate, write the whole list back" pattern
-- (closet_data.json's actual corruption/concurrency risk).
CREATE TABLE IF NOT EXISTS closet_items (
    id               BIGSERIAL PRIMARY KEY,
    user_id          TEXT NOT NULL,
    category         TEXT,
    label            TEXT,
    gender           TEXT,
    image_hash       TEXT NOT NULL,
    image_url        TEXT NOT NULL,
    dominant_hex     TEXT,
    dominant_hue     INTEGER,
    upload_timestamp TIMESTAMPTZ DEFAULT now(),
    UNIQUE (user_id, image_hash)
);
CREATE INDEX IF NOT EXISTS idx_closet_items_user_id ON closet_items(user_id);

-- The /analyze response saved here is a large, variably-shaped object
-- (body type, measurements, outfit suggestions, etc., whatever the
-- frontend happened to receive) -- genuinely a JSON document, not a
-- fixed set of columns, hence JSONB rather than a normalized table.
CREATE TABLE IF NOT EXISTS ai_profiles (
    user_id    TEXT PRIMARY KEY,
    profile    JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- catalog.py's shirts/jeans/tops/dresses/skirts/trousers rows store a
-- LOCAL FILE PATH in `image` (e.g. "female/fshirt/..."), not the image
-- itself -- every /analyze call was reading and base64-encoding all ~69
-- of those files off disk fresh, even for the ~54 that never end up in a
-- final look. image_url (once migrate_catalog_to_gcs.py has run) lets
-- catalog.py skip that local-disk read/encode entirely and serve a GCS
-- URL directly, same as closet_items/saved_looks.
ALTER TABLE shirts   ADD COLUMN IF NOT EXISTS image_url TEXT;
ALTER TABLE jeans    ADD COLUMN IF NOT EXISTS image_url TEXT;
ALTER TABLE tops     ADD COLUMN IF NOT EXISTS image_url TEXT;
ALTER TABLE dresses  ADD COLUMN IF NOT EXISTS image_url TEXT;
ALTER TABLE skirts   ADD COLUMN IF NOT EXISTS image_url TEXT;
ALTER TABLE trousers ADD COLUMN IF NOT EXISTS image_url TEXT;

CREATE TABLE IF NOT EXISTS saved_looks (
    look_id   TEXT PRIMARY KEY,
    user_id   TEXT NOT NULL,
    label     TEXT,
    gcs_path  TEXT NOT NULL,
    image_url TEXT NOT NULL,
    saved_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_saved_looks_user_id ON saved_looks(user_id);
