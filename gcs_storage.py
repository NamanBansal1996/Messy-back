import base64
import os
import uuid

from google.cloud import storage

# On Cloud Run this authenticates automatically via the instance's attached
# service account (Application Default Credentials) -- no key file needed.
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "mess_bucket1")

_client = None


def _bucket():
    global _client
    if _client is None:
        _client = storage.Client()
    return _client.bucket(BUCKET_NAME)


def upload_bytes(destination_path, data, content_type="image/jpeg"):
    """Uploads raw bytes to GCS and returns the public URL. Requires the
    bucket to grant allUsers Storage Object Viewer -- these are user photos
    meant to be displayed in the app, not sensitive documents."""
    blob = _bucket().blob(destination_path)
    blob.upload_from_string(data, content_type=content_type)
    return f"https://storage.googleapis.com/{BUCKET_NAME}/{destination_path}"


def upload_base64_image(destination_path, b64_data, content_type="image/jpeg"):
    clean_b64 = b64_data.split(",", 1)[1] if b64_data.startswith("data:") else b64_data
    raw_bytes = base64.b64decode(clean_b64)
    return upload_bytes(destination_path, raw_bytes, content_type)


def delete_object(destination_path):
    blob = _bucket().blob(destination_path)
    if blob.exists():
        blob.delete()


def upload_temp_scan(user_id, b64_data, content_type="image/png"):
    """
    For images that only matter for the current response and are never
    referenced by any persisted row -- e.g. the /analyze person cutout.
    Uploads under temp_scans/, which has a GCS Lifecycle Rule (see
    infra/README or the deploy notes) auto-deleting anything under that
    prefix after 7 days, so these don't accumulate forever the way a
    permanent closet/saved-looks upload would.
    """
    ext = "png" if content_type == "image/png" else "jpg"
    path = f"temp_scans/{user_id}/{uuid.uuid4().hex}.{ext}"
    return upload_base64_image(path, b64_data, content_type=content_type)
