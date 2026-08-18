"""
migrate_catalog_to_db.py

One-off script: loads the existing catalog JSON files and upserts them
into the Supabase database (see db.py). Safe to re-run -- upserts are
keyed on each table's primary key, so re-running after editing a JSON
file just refreshes the matching rows.

Requires the tables to already exist -- run schema.sql once via the
Supabase SQL Editor first.

Usage:
    python migrate_catalog_to_db.py
"""

import json
import os

from dotenv import load_dotenv

load_dotenv()

import db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CATALOG_ITEMS_FILE = os.path.join(BASE_DIR, "catalog_data.json")
FEMALE_SHIRTS_FILE = os.path.join(BASE_DIR, "catalog", "female", "fshirt", "female_shirts.json")
FEMALE_JEANS_FILE = os.path.join(BASE_DIR, "catalog", "female", "fjeans", "female_jeans.json")
FEMALE_TOPS_FILE = os.path.join(BASE_DIR, "catalog", "female", "ftop", "female_tops.json")


def _load_json(path):
    if not os.path.exists(path):
        print(f"  skip (not found): {path}")
        return []
    with open(path, "r") as f:
        return json.load(f)


def migrate_catalog_items(client):
    items = _load_json(CATALOG_ITEMS_FILE)
    if items:
        client.table("catalog_items").upsert(items).execute()
    print(f"  catalog_items: upserted {len(items)} row(s)")


def _base_row(item, gender):
    return {
        "item_id": item.get("item_id"),
        "gender": gender,
        "name": item.get("name"),
        "type": item.get("type"),
        "color": item.get("color"),
        "pattern": item.get("pattern"),
        "fit": item.get("fit"),
        "length": item.get("length"),
        "fabric": item.get("fabric"),
        "style": item.get("style") or [],
        "occasion": item.get("occasion") or [],
        "image": item.get("image"),
    }


def migrate_shirts(client, path, gender):
    items = _load_json(path)
    rows = []
    for item in items:
        row = _base_row(item, gender)
        row["collar"] = item.get("collar")
        row["sleeve"] = item.get("sleeve")
        row["season"] = item.get("season") or []
        rows.append(row)
    if rows:
        client.table("shirts").upsert(rows).execute()
    print(f"  shirts: upserted {len(rows)} row(s)")


def migrate_jeans(client, path, gender):
    items = _load_json(path)
    rows = []
    for item in items:
        row = _base_row(item, gender)
        row["rise"] = item.get("rise")
        rows.append(row)
    if rows:
        client.table("jeans").upsert(rows).execute()
    print(f"  jeans: upserted {len(rows)} row(s)")


def migrate_tops(client, path, gender):
    items = _load_json(path)
    rows = []
    for item in items:
        row = _base_row(item, gender)
        row["sleeve"] = item.get("sleeve")
        row["neckline"] = item.get("neckline")
        row["properties"] = item.get("properties") or []
        row["season"] = item.get("season") or []
        rows.append(row)
    if rows:
        client.table("tops").upsert(rows).execute()
    print(f"  tops: upserted {len(rows)} row(s)")


def main():
    client = db.get_client()

    print("Migrating catalog_data.json -> catalog_items...")
    migrate_catalog_items(client)

    # Female only for now -- male catalog folders don't exist yet.
    print("Migrating female_shirts.json -> shirts...")
    migrate_shirts(client, FEMALE_SHIRTS_FILE, gender="Female")

    print("Migrating female_jeans.json -> jeans...")
    migrate_jeans(client, FEMALE_JEANS_FILE, gender="Female")

    print("Migrating female_tops.json -> tops...")
    migrate_tops(client, FEMALE_TOPS_FILE, gender="Female")

    print("Done.")


if __name__ == "__main__":
    main()
