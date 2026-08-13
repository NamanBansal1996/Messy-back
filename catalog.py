"""
catalog.py

Modular Product Catalog interface for the MeSS Recommendation Engine.

This reads from the catalog_items table in Supabase (Postgres) -- see
db.py for the connection/schema and migrate_catalog_to_db.py for the
one-off script that seeded it from the 7 products that used to live in
catalog_data.json (originally hardcoded in the frontend's Ads.jsx
"Shop the Look" sidebar).

get_catalog_items() is the ONLY function recommendation_engine.py depends on.
Swapping the internals for a different database or external API later
requires no changes anywhere else in the codebase -- that's the point of
keeping this behind one small interface.
"""

import db


def _load_catalog():
    client = db.get_client()
    response = client.table("catalog_items").select("*").execute()
    return [db.row_to_catalog_item(row) for row in response.data]


def get_catalog_items(category=None, gender=None):
    """
    Return catalog garments, optionally filtered by:
      category: "top" | "bottom" | "footwear" | "accessories"
      gender:   "Male" | "Female" (items tagged "Unisex" always match)

    Each item has the same shape wardrobe items already have (from
    closet_manager.get_user_closet()), so recommendation_engine.py can
    score both sources identically without special-casing either one:

        {
            "id": str,
            "category": str,
            "label": str,             # matches yolo_outfit_detect.py's label vocabulary
            "dominant_hex": str,
            "dominant_hue": int,
            "color_name": str,
            "gender": str,
            "source": "catalog",
            "source_ref": str,        # static frontend asset path, e.g. "/menlooset-shirtlevis.webp"
            "title": str,
            "link": str,               # affiliate link, unchanged from Ads.jsx
        }
    """
    items = _load_catalog()

    if category:
        items = [i for i in items if i.get("category") == category]
    if gender:
        items = [i for i in items if i.get("gender") in (gender, "Unisex")]

    return items
