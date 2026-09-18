import hashlib
from concurrent.futures import ThreadPoolExecutor

from db import get_client
from gcs_storage import upload_base64_image

def generate_image_hash(image_b64):
    """Run MD5 hash on the base64 string to identify duplicates."""
    return hashlib.md5(image_b64.encode('utf-8')).hexdigest()

def add_items_to_closet(user_id, outfits_dict, gender="Unisex"):
    """
    Takes the dictionary of detected outfits (tops, bottoms, etc.) and adds
    each individual cropped item to the user's closet.

    Every item in outfits_dict -- new or a duplicate of something already in
    the closet -- is mutated in place to carry image_url instead of the
    original base64 "image" field. That lets the caller (app.py's /analyze)
    return the same URL in its JSON response instead of uploading the same
    image a second time just for display.

    New images upload to GCS in parallel via a thread pool -- each upload is
    I/O-bound (network), so several genuinely overlap instead of paying the
    latency of each one back-to-back.
    """
    client = get_client()

    existing = client.table("closet_items").select("image_hash,image_url").eq("user_id", user_id).execute()
    existing_by_hash = {row["image_hash"]: row["image_url"] for row in existing.data}

    # Group by hash first: handles both "already in the closet" duplicates
    # and "the same crop was detected twice in this one photo" duplicates
    # with one code path, and guarantees we never try to insert the same
    # (user_id, image_hash) pair twice in one batch.
    groups = {}
    for category, items in outfits_dict.items():
        if not isinstance(items, list):
            continue
        for item in items:
            img_b64 = item.get("image")
            if not img_b64:
                continue
            img_hash = generate_image_hash(img_b64)
            group = groups.setdefault(img_hash, {"category": category, "b64": img_b64, "items": []})
            group["items"].append(item)

    added_count = 0
    duplicate_count = 0
    to_upload = []

    for img_hash, group in groups.items():
        items = group["items"]
        if img_hash in existing_by_hash:
            url = existing_by_hash[img_hash]
            for item in items:
                item["image_url"] = url
                item.pop("image", None)
            duplicate_count += len(items)
        else:
            to_upload.append((img_hash, group))
            duplicate_count += len(items) - 1  # repeats of this same new hash within this one photo

    def _upload(entry):
        img_hash, group = entry
        url = upload_base64_image(f"closet/{user_id}/{img_hash}.jpg", group["b64"])
        return img_hash, group, url

    rows_to_insert = []
    if to_upload:
        with ThreadPoolExecutor(max_workers=min(8, len(to_upload))) as pool:
            for img_hash, group, url in pool.map(_upload, to_upload):
                for item in group["items"]:
                    item["image_url"] = url
                    item.pop("image", None)
                rows_to_insert.append({
                    "user_id": user_id,
                    "category": group["category"],
                    "label": group["items"][0].get("label", "unknown"),
                    "gender": gender,
                    "image_hash": img_hash,
                    "image_url": url,
                    "dominant_hex": group["items"][0].get("dominant_hex"),
                    "dominant_hue": group["items"][0].get("dominant_hue"),
                })
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
