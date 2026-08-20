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

import base64
import colorsys
import os

import db
from color_utils import get_hex_for_color_name

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_DIR = os.path.join(BASE_DIR, "catalog")

# Each table's images live under a different catalog/ subfolder. jeans rows
# store "image" as a path that already includes its own "female/fjeans/"
# prefix (see female_jeans.json); shirts/tops store a bare filename instead
# (see female_shirts.json / female_tops.json) -- this map is how each
# table's image path actually resolves on disk, not a convention we get to
# assume is consistent.
_IMAGE_SUBDIR = {
    "shirts": os.path.join("female", "fshirt"),
    "jeans": None,
    "tops": os.path.join("female", "ftop"),
    "dresses": os.path.join("female", "fdresses"),
    "skirts": os.path.join("female", "fskirts"),
    "trousers": os.path.join("female", "ftrouser"),
}


def _image_to_b64(table_name, image_path):
    if not image_path:
        return None
    subdir = _IMAGE_SUBDIR.get(table_name)
    full_path = os.path.join(CATALOG_DIR, subdir, image_path) if subdir else os.path.join(CATALOG_DIR, image_path)
    if not os.path.exists(full_path):
        return None
    with open(full_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _hue_from_hex(hex_color):
    if not hex_color:
        return None
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    h, _, _ = colorsys.rgb_to_hsv(r, g, b)
    return round(h * 360)


def _row_to_garment(table_name, row, category, label):
    """
    Shapes a shirts/jeans/tops row into the same garment-dict shape
    recommendation_engine.py already expects from wardrobe/catalog_items
    (category, label, dominant_hex, source, image_b64, ...), plus the
    type/fit/rise/neckline/properties fields its body-shape scoring reads.
    image_b64 is base64-encoded server-side here (not a source_ref/static
    path) so the frontend's existing garmentToFile() -- which already
    handles image_b64 for wardrobe items -- needs no changes to use these.
    """
    color_name = (row.get("color") or "").replace("_", " ").title() or None
    dominant_hex = get_hex_for_color_name(row.get("color"))
    return {
        "id": row["item_id"],
        "category": category,
        "label": label,
        "dominant_hex": dominant_hex,
        "dominant_hue": _hue_from_hex(dominant_hex),
        "color_name": color_name,
        "gender": row.get("gender"),
        "source": "catalog",
        "source_ref": None,
        "title": row.get("name"),
        "link": None,
        "image_b64": _image_to_b64(table_name, row.get("image")),
        "type": row.get("type"),
        "fit": row.get("fit"),
        "rise": row.get("rise"),
        "neckline": row.get("neckline"),
        "sleeve": row.get("sleeve"),
        "properties": row.get("properties") or [],
        "pockets": row.get("pockets"),
    }


def _load_table(table_name, category, label, gender=None):
    client = db.get_client()
    response = client.table(table_name).select("*").execute()
    items = [_row_to_garment(table_name, row, category, label) for row in response.data]
    if gender:
        items = [i for i in items if i.get("gender") in (gender, "Unisex")]
    return items


def get_shirt_items(gender=None):
    return _load_table("shirts", category="top", label="shirt", gender=gender)


def get_jeans_items(gender=None):
    return _load_table("jeans", category="bottom", label="jeans", gender=gender)


def get_top_items(gender=None):
    return _load_table("tops", category="top", label="tshirt", gender=gender)


def get_dress_items(gender=None):
    return _load_table("dresses", category="dress", label="dress", gender=gender)


def get_skirt_items(gender=None):
    return _load_table("skirts", category="bottom", label="skirt", gender=gender)


def get_trouser_items(gender=None):
    return _load_table("trousers", category="bottom", label="trousers", gender=gender)


def get_styling_catalog_items(gender=None):
    """
    Combined shirts + jeans + tops + dresses + skirts + trousers -- the
    candidate pool recommendation_engine.py's body-shape/undertone scoring
    actually wants. Separate from get_catalog_items() below, which reads a
    different, unrelated table (the original 7-item "Shop the Look" seed
    data with real affiliate links) -- app.py passes both into
    generate_three_looks().
    """
    return (
        get_shirt_items(gender) + get_jeans_items(gender) + get_top_items(gender)
        + get_dress_items(gender) + get_skirt_items(gender) + get_trouser_items(gender)
    )


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
