import base64
import io
import os
import uuid
from datetime import datetime

from PIL import Image

from storage_utils import read_json, write_json_atomic

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_LOOKS_FILE = os.path.join(BASE_DIR, "saved_looks.json")
SAVED_LOOKS_IMAGE_DIR = os.path.join(BASE_DIR, "saved_looks_images")
os.makedirs(SAVED_LOOKS_IMAGE_DIR, exist_ok=True)


def get_saved_looks_data():
    return read_json(SAVED_LOOKS_FILE, {})


def _write_saved_looks_data(data):
    write_json_atomic(SAVED_LOOKS_FILE, data)


def get_saved_looks(user_id):
    return get_saved_looks_data().get(user_id, [])


def save_look(user_id, image_b64, label):
    """
    Decodes a base64 try-on render and writes it to disk as a JPEG under
    saved_looks_images/, storing only the lightweight filename (not the
    base64) in saved_looks.json -- a full-res render is ~1MB of base64, and
    embedding those inline the way closet_data.json does would bloat this
    file to the multi-MB range after just a handful of saves.
    """
    if not image_b64:
        raise ValueError("image_b64 is required")

    clean_b64 = image_b64.split(",", 1)[1] if image_b64.startswith("data:") else image_b64
    image_bytes = base64.b64decode(clean_b64)

    look_id = uuid.uuid4().hex
    filename = f"{look_id}.jpg"
    filepath = os.path.join(SAVED_LOOKS_IMAGE_DIR, filename)

    # Re-encode to JPEG at a fixed quality so disk usage stays predictable
    # regardless of the source render's original format/size.
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.save(filepath, "JPEG", quality=85)

    entry = {
        "look_id": look_id,
        "label": label or "Saved Look",
        "filename": filename,
        "saved_at": datetime.now().isoformat(),
    }

    data = get_saved_looks_data()
    data.setdefault(user_id, []).append(entry)
    _write_saved_looks_data(data)
    return entry


def delete_saved_look(user_id, look_id):
    data = get_saved_looks_data()
    looks = data.get(user_id, [])
    remaining = [item for item in looks if item.get("look_id") != look_id]
    if len(remaining) == len(looks):
        return False

    deleted = next((item for item in looks if item.get("look_id") == look_id), None)
    if deleted:
        filepath = os.path.join(SAVED_LOOKS_IMAGE_DIR, deleted["filename"])
        if os.path.exists(filepath):
            os.remove(filepath)

    data[user_id] = remaining
    _write_saved_looks_data(data)
    return True


def migrate_saved_looks(guest_id, user_id):
    """Same guest -> user migration pattern as migrate_closet_items /
    migrate_profile. Image files stay put on disk (named by look_id, not
    user_id) -- only the JSON ownership entries move."""
    data = get_saved_looks_data()
    guest_looks = data.get(guest_id)
    if not guest_looks:
        return 0

    data.setdefault(user_id, [])
    data[user_id].extend(guest_looks)
    del data[guest_id]
    _write_saved_looks_data(data)
    return len(guest_looks)
