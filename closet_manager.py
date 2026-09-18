import hashlib

from db import get_client
from gcs_storage import upload_base64_image

def generate_image_hash(image_b64):
    """Run MD5 hash on the base64 string to identify duplicates."""
    return hashlib.md5(image_b64.encode('utf-8')).hexdigest()

def add_items_to_closet(user_id, outfits_dict, gender="Unisex"):
    """
    Takes the dictionary of detected outfits (tops, bottoms, etc.)
    and adds each individual cropped item to the user's closet.
    """
    client = get_client()

    existing = client.table("closet_items").select("image_hash").eq("user_id", user_id).execute()
    existing_hashes = {row["image_hash"] for row in existing.data}

    added_count = 0
    duplicate_count = 0
    rows_to_insert = []

    # Look through all categories (top, bottom, footwear, accessories)
    for category, items in outfits_dict.items():
        if isinstance(items, list):
            for item in items:
                img_b64 = item.get("image")
                if not img_b64:
                    continue

                img_hash = generate_image_hash(img_b64)

                if img_hash in existing_hashes:
                    duplicate_count += 1
                    continue

                # Store the actual image in GCS rather than embedding it
                # as base64 -- closet_data.json used to grow huge (and
                # slow to read/write on every request) with every
                # detected garment's full image inlined.
                image_url = upload_base64_image(f"closet/{user_id}/{img_hash}.jpg", img_b64)

                rows_to_insert.append({
                    "user_id": user_id,
                    "category": category,
                    "label": item.get("label", "unknown"),
                    "gender": gender,
                    "image_hash": img_hash,
                    "image_url": image_url,
                    "dominant_hex": item.get("dominant_hex"),
                    "dominant_hue": item.get("dominant_hue"),
                })
                existing_hashes.add(img_hash)  # guard against dupes within this same batch
                added_count += 1

    if rows_to_insert:
        client.table("closet_items").insert(rows_to_insert).execute()

    return added_count, duplicate_count

CLOSET_ITEM_COLUMNS = "category,label,gender,image_hash,image_url,dominant_hex,dominant_hue,upload_timestamp"

def get_user_closet(user_id, gender=None):
    """
    Deliberately selects specific columns, not "*" -- recommendation_engine's
    _garment_id() treats a garment dict's "id" key as an authoritative id
    when present, preferring it over "image_hash". Postgres's own internal
    BIGSERIAL row id would leak in under that same "id" key with select("*"),
    silently colliding with that convention and mixing int/str ids with
    catalog items when sorted together (the exact TypeError that broke
    every /analyze call after this table went live).
    """
    client = get_client()
    items = client.table("closet_items").select(CLOSET_ITEM_COLUMNS).eq("user_id", user_id).execute().data
    if gender:
        return [item for item in items if item.get("gender") in (gender, "Unisex") or not item.get("gender")]
    return items

def migrate_closet_items(guest_id, user_id):
    """
    Migrates closet items from temporary guest_id to authenticated user_id.
    """
    client = get_client()
    guest_items = client.table("closet_items").select("*").eq("user_id", guest_id).execute().data
    if not guest_items:
        return 0

    existing = client.table("closet_items").select("image_hash").eq("user_id", user_id).execute()
    existing_hashes = {row["image_hash"] for row in existing.data}

    migrated_count = 0
    for item in guest_items:
        if item["image_hash"] in existing_hashes:
            client.table("closet_items").delete().eq("id", item["id"]).execute()
            continue
        client.table("closet_items").update({"user_id": user_id}).eq("id", item["id"]).execute()
        migrated_count += 1

    return migrated_count
