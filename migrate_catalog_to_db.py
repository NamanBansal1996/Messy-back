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


def migrate_styling_catalog(client, path, garment_type, gender):
    items = _load_json(path)
    rows = []
    for item in items:
        rows.append(
            {
                "item_id": item.get("item_id"),
                "garment_type": garment_type,
                "gender": gender,
                "name": item.get("name"),
                "type": item.get("type"),
                "color": item.get("color"),
                "pattern": item.get("pattern"),
                "fit": item.get("fit"),
                "length": item.get("length"),
                "fabric": item.get("fabric"),
                "collar": item.get("collar"),
                "sleeve": item.get("sleeve"),
                "rise": item.get("rise"),
                "style": item.get("style") or [],
                "occasion": item.get("occasion") or [],
                "season": item.get("season") or [],
                "image": item.get("image"),
            }
        )
    if rows:
        client.table("styling_catalog").upsert(rows).execute()
    print(f"  styling_catalog ({garment_type}): upserted {len(rows)} row(s)")


def main():
    client = db.get_client()

    print("Migrating catalog_data.json -> catalog_items...")
    migrate_catalog_items(client)

    print("Migrating female_shirts.json -> styling_catalog...")
    migrate_styling_catalog(client, FEMALE_SHIRTS_FILE, garment_type="shirt", gender="Female")

    print("Migrating female_jeans.json -> styling_catalog...")
    migrate_styling_catalog(client, FEMALE_JEANS_FILE, garment_type="jeans", gender="Female")

    print("Done.")


if __name__ == "__main__":
    main()
