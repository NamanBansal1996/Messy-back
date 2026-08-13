"""
db.py

Connection helper for the Supabase catalog database, using the
supabase-py client (REST, via PostgREST) instead of a raw Postgres
connection -- keyed off SUPABASE_URL + SUPABASE_SECRET_KEY.

Requires SUPABASE_URL and SUPABASE_SECRET_KEY in the environment (see
.env.example). The secret key is required (not the publishable/anon
key) because writes here need to bypass Row Level Security.

The two tables (catalog_items, styling_catalog) are NOT created by this
file -- supabase-py talks to PostgREST, which can't run arbitrary DDL.
Create them once via the Supabase SQL Editor using schema.sql, then run
migrate_catalog_to_db.py to populate them.
"""

import os

from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_SECRET_KEY")

_client = None


def get_client():
    global _client
    if _client is not None:
        return _client
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        raise RuntimeError(
            "SUPABASE_URL / SUPABASE_SECRET_KEY are not set. Add them to "
            ".env (see .env.example)."
        )
    _client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    return _client


def row_to_catalog_item(row):
    return {
        "id": row["id"],
        "category": row["category"],
        "label": row["label"],
        "dominant_hex": row["dominant_hex"],
        "dominant_hue": row["dominant_hue"],
        "color_name": row["color_name"],
        "gender": row["gender"],
        "source": row["source"],
        "source_ref": row["source_ref"],
        "title": row["title"],
        "link": row["link"],
    }


def row_to_styling_item(row):
    item = {
        "item_id": row["item_id"],
        "garment_type": row["garment_type"],
        "gender": row["gender"],
        "name": row["name"],
        "type": row["type"],
        "color": row["color"],
        "pattern": row["pattern"],
        "fit": row["fit"],
        "length": row["length"],
        "fabric": row["fabric"],
        "style": row["style"] or [],
        "occasion": row["occasion"] or [],
        "image": row["image"],
    }
    if row["garment_type"] == "shirt":
        item["collar"] = row["collar"]
        item["sleeve"] = row["sleeve"]
        item["season"] = row["season"] or []
    elif row["garment_type"] == "jeans":
        item["rise"] = row["rise"]
    return item
