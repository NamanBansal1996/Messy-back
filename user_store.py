import uuid

from db import get_client


def resolve_user_id(email, name=None):
    """
    Returns the canonical user_id for an email, creating one the first time
    this email is seen. Signup and login both call this so the same person
    always resolves to the same user_id, instead of each flow generating its
    own (previously: a random UUID at signup vs. the email prefix at login).
    """
    email_key = email.strip().lower()
    client = get_client()

    existing = client.table("users").select("*").eq("email", email_key).limit(1).execute()
    if existing.data:
        record = existing.data[0]
        resolved_name = record.get("name") or name
        if name and not record.get("name"):
            client.table("users").update({"name": name}).eq("email", email_key).execute()
        return record["user_id"], resolved_name, False

    user_id = "user_" + uuid.uuid4().hex[:8]
    client.table("users").insert({"email": email_key, "user_id": user_id, "name": name}).execute()
    return user_id, name, True
