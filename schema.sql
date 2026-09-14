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
