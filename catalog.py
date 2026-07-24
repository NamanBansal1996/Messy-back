"""
catalog.py

Modular Product Catalog interface for the MeSS Recommendation Engine.

Today this reads from a static JSON file (catalog_data.json) seeded with the
7 products already hardcoded in the frontend's Ads.jsx ("Shop the Look"
sidebar) -- same images, same affiliate links, just made queryable from the
backend instead of living only inside a React component.

get_catalog_items() is the ONLY function recommendation_engine.py depends on.
Swapping the internals for a real product database or external API later
requires no changes anywhere else in the codebase -- that's the point of
keeping this behind one small interface.
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_FILE = os.path.join(BASE_DIR, "catalog_data.json")


def _load_catalog():
    if not os.path.exists(CATALOG_FILE):
        return []
    with open(CATALOG_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


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
