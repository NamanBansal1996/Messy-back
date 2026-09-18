"""
migrate_catalog_to_gcs.py

One-off migration: uploads each shirts/jeans/tops/dresses/skirts/trousers
row's local catalog image file to GCS and writes the resulting URL back to
that row's new image_url column (see schema.sql). Safe to re-run -- rows
that already have image_url are skipped.

Run via POST /admin/migrate-catalog-images (only reachable where the local
catalog/ image files actually exist, i.e. the deployed container -- the
whole point is these files live on Cloud Run's disk today and this moves
them to GCS).
"""

import os

from db import get_client
from gcs_storage import upload_bytes
from catalog import CATALOG_DIR, _IMAGE_SUBDIR

TABLES = ["shirts", "jeans", "tops", "dresses", "skirts", "trousers"]


def _resolve_local_path(table_name, image_path):
    if not image_path:
        return None
    subdir = _IMAGE_SUBDIR.get(table_name)
    normalized = image_path.replace(os.sep, "/")
    if subdir and normalized.startswith(subdir.replace(os.sep, "/") + "/"):
        return os.path.join(CATALOG_DIR, image_path)
    return os.path.join(CATALOG_DIR, subdir, image_path) if subdir else os.path.join(CATALOG_DIR, image_path)


def _content_type_for(ext):
    ext = ext.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def migrate_table_images(table_name):
    client = get_client()
    rows = client.table(table_name).select("item_id,image,image_url").execute().data

    migrated = 0
    skipped = 0
    failed = []

    for row in rows:
        if row.get("image_url"):
            skipped += 1
            continue

        local_path = _resolve_local_path(table_name, row.get("image"))
        if not local_path or not os.path.exists(local_path):
            failed.append({"item_id": row.get("item_id"), "reason": f"file not found: {local_path}"})
            continue

        with open(local_path, "rb") as f:
            data = f.read()

        ext = os.path.splitext(local_path)[1] or ".jpg"
        gcs_path = f"catalog/{table_name}/{row['item_id']}{ext}"
        url = upload_bytes(gcs_path, data, content_type=_content_type_for(ext))

        client.table(table_name).update({"image_url": url}).eq("item_id", row["item_id"]).execute()
        migrated += 1

    return {"table": table_name, "migrated": migrated, "skipped": skipped, "failed": failed}


def migrate_all_catalog_images():
    return [migrate_table_images(t) for t in TABLES]
