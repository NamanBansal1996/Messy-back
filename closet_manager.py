import colorsys
import hashlib
from concurrent.futures import ThreadPoolExecutor

from db import get_client
from gcs_storage import upload_base64_image, delete_object, BUCKET_NAME

def generate_image_hash(image_b64):
    """Run MD5 hash on the base64 string to identify duplicates."""
    return hashlib.md5(image_b64.encode('utf-8')).hexdigest()

NEUTRAL_SATURATION_THRESHOLD = 0.15  # below this, hue is unstable/meaningless
NEUTRAL_VALUE_THRESHOLD = 0.15       # max value/lightness gap to count as the same neutral shade
CHROMATIC_HUE_THRESHOLD_DEGREES = 20

def _hex_to_hsv(hex_color):
    if not hex_color:
        return None
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return None
    try:
        r, g, b = (int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return None
    return colorsys.rgb_to_hsv(r, g, b)

def _is_similar_color(hex_a, hex_b):
    """
    Near-duplicate detection heuristic. Plain hue distance is unreliable
    for neutrals (black/white/gray) -- their saturation is near zero, so
    hue is essentially noise and a white shirt and a black shirt can land
    on coincidentally close hue values. So: only compare hue when BOTH
    colors are chromatic; when either is neutral, compare value/lightness
    instead (and only match if both are neutral -- a neutral is never
    "similar" to a chromatic color).
    """
    hsv_a = _hex_to_hsv(hex_a)
    hsv_b = _hex_to_hsv(hex_b)
    if not hsv_a or not hsv_b:
        return False
    h_a, s_a, v_a = hsv_a
    h_b, s_b, v_b = hsv_b
    neutral_a = s_a < NEUTRAL_SATURATION_THRESHOLD
    neutral_b = s_b < NEUTRAL_SATURATION_THRESHOLD
    if neutral_a or neutral_b:
        return neutral_a and neutral_b and abs(v_a - v_b) < NEUTRAL_VALUE_THRESHOLD
    hue_diff = abs(h_a - h_b) * 360
    circular_diff = min(hue_diff, 360 - hue_diff)
    return circular_diff <= CHROMATIC_HUE_THRESHOLD_DEGREES

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

    existing = client.table("closet_items").select(
        "image_hash,image_url,category,label,dominant_hex,possible_duplicate_of"
    ).eq("user_id", user_id).execute()
    existing_by_hash = {row["image_hash"]: row for row in existing.data}

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
            url = existing_by_hash[img_hash]["image_url"]
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
    rows_to_flag_existing = {}  # existing image_hash -> {"possible_duplicate_of": new_hash}
    if to_upload:
        with ThreadPoolExecutor(max_workers=min(8, len(to_upload))) as pool:
            for img_hash, group, url in pool.map(_upload, to_upload):
                for item in group["items"]:
                    item["image_url"] = url
                    item.pop("image", None)

                representative = group["items"][0]
                category = group["category"]
                label = representative.get("label", "unknown")
                dominant_hex = representative.get("dominant_hex")

                # Near-duplicate check (not the exact-hash match already
                # handled above) -- same category+label, similar color,
                # not already part of another flagged pair. Symmetric:
                # both rows end up pointing at each other.
                match_hash = None
                for candidate_hash, candidate in existing_by_hash.items():
                    if candidate.get("possible_duplicate_of"):
                        continue
                    if candidate.get("category") != category or candidate.get("label") != label:
                        continue
                    if _is_similar_color(dominant_hex, candidate.get("dominant_hex")):
                        match_hash = candidate_hash
                        break

                rows_to_insert.append({
                    "user_id": user_id,
                    "category": category,
                    "label": label,
                    "gender": gender,
                    "image_hash": img_hash,
                    "image_url": url,
                    "dominant_hex": dominant_hex,
                    "dominant_hue": representative.get("dominant_hue"),
                    "possible_duplicate_of": match_hash,
                })

                if match_hash:
                    rows_to_flag_existing[match_hash] = img_hash
                    # Don't match a second new item against the same
                    # existing one within this same batch.
                    existing_by_hash[match_hash]["possible_duplicate_of"] = img_hash

                added_count += 1

    if rows_to_insert:
        client.table("closet_items").insert(rows_to_insert).execute()

    for existing_hash, new_hash in rows_to_flag_existing.items():
        client.table("closet_items").update({"possible_duplicate_of": new_hash}).eq(
            "user_id", user_id
        ).eq("image_hash", existing_hash).execute()

    return added_count, duplicate_count

CLOSET_ITEM_COLUMNS = (
    "category,label,gender,image_hash,image_url,dominant_hex,dominant_hue,"
    "upload_timestamp,brand,possible_duplicate_of,duplicate_resolved"
)

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

def resolve_duplicate(user_id, image_hash, matched_image_hash, action, brand=None):
    """
    Resolves a flagged near-duplicate pair.
      keep_both  -- both items stay; mark the pair reviewed on both sides.
      keep_this  -- delete the row at matched_image_hash, keep image_hash.
      keep_other -- delete the row at image_hash, keep matched_image_hash.
    """
    client = get_client()

    if action == "keep_both":
        client.table("closet_items").update({"duplicate_resolved": True}).eq(
            "user_id", user_id
        ).eq("image_hash", image_hash).execute()
        client.table("closet_items").update({"duplicate_resolved": True}).eq(
            "user_id", user_id
        ).eq("image_hash", matched_image_hash).execute()
        if brand:
            client.table("closet_items").update({"brand": brand}).eq(
                "user_id", user_id
            ).eq("image_hash", image_hash).execute()
        return {"kept": [image_hash, matched_image_hash], "deleted": None}

    if action == "keep_this":
        keep_hash, discard_hash = image_hash, matched_image_hash
    elif action == "keep_other":
        keep_hash, discard_hash = matched_image_hash, image_hash
    else:
        raise ValueError(f"Unknown action: {action}")

    discarded = client.table("closet_items").select("image_url").eq(
        "user_id", user_id
    ).eq("image_hash", discard_hash).limit(1).execute().data

    client.table("closet_items").delete().eq("user_id", user_id).eq("image_hash", discard_hash).execute()

    update_payload = {"possible_duplicate_of": None, "duplicate_resolved": False}
    if brand:
        update_payload["brand"] = brand
    client.table("closet_items").update(update_payload).eq("user_id", user_id).eq("image_hash", keep_hash).execute()

    # Best-effort GCS cleanup -- the DB row is already gone either way, so a
    # stray object here isn't harmful, just untidy.
    if discarded and discarded[0].get("image_url"):
        try:
            prefix = f"https://storage.googleapis.com/{BUCKET_NAME}/"
            url = discarded[0]["image_url"]
            if url.startswith(prefix):
                delete_object(url[len(prefix):])
        except Exception:
            pass

    return {"kept": keep_hash, "deleted": discard_hash}
