import base64
import io
import os
import uuid
from datetime import datetime

from PIL import Image

from storage_utils import read_json, write_json_atomic
from gcs_storage import upload_bytes, delete_object

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_LOOKS_FILE = os.path.join(BASE_DIR, "saved_looks.json")


def get_saved_looks_data():
    return read_json(SAVED_LOOKS_FILE, {})


def _write_saved_looks_data(data):
    write_json_atomic(SAVED_LOOKS_FILE, data)


def get_saved_looks(user_id):
    return get_saved_looks_data().get(user_id, [])


def save_look(user_id, image_b64, label):
    """
    Decodes a base64 try-on render and uploads it to GCS as a JPEG, storing
    only the lightweight URL (not the base64) in saved_looks.json. This used
    to write to local disk, but Cloud Run's filesystem is ephemeral per
    container instance -- a look "saved" that way could vanish the moment
    the instance recycles, or simply not exist on whichever instance serves
    a later request.
    """
    if not image_b64:
        raise ValueError("image_b64 is required")

    clean_b64 = image_b64.split(",", 1)[1] if image_b64.startswith("data:") else image_b64
    image_bytes = base64.b64decode(clean_b64)

    look_id = uuid.uuid4().hex

    # Re-encode to JPEG at a fixed quality so storage size stays predictable
    # regardless of the source render's original format/size.
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=85)

    gcs_path = f"saved_looks/{user_id}/{look_id}.jpg"
    image_url = upload_bytes(gcs_path, buf.getvalue(), content_type="image/jpeg")

    entry = {
        "look_id": look_id,
        "label": label or "Saved Look",
        "gcs_path": gcs_path,
        "image_url": image_url,
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
    if deleted and deleted.get("gcs_path"):
        delete_object(deleted["gcs_path"])

    data[user_id] = remaining
    _write_saved_looks_data(data)
    return True


def migrate_saved_looks(guest_id, user_id):
    """Same guest -> user migration pattern as migrate_closet_items /
    migrate_profile. Images live in GCS keyed by look_id, not user_id, so
    only the JSON ownership entries need to move."""
    data = get_saved_looks_data()
    guest_looks = data.get(guest_id)
    if not guest_looks:
        return 0

    data.setdefault(user_id, [])
    data[user_id].extend(guest_looks)
    del data[guest_id]
    _write_saved_looks_data(data)
    return len(guest_looks)
