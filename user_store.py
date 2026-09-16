import os
import uuid

from storage_utils import read_json, write_json_atomic

USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")


def get_users_data():
    return read_json(USERS_FILE, {})


def save_users_data(data):
    write_json_atomic(USERS_FILE, data)


def resolve_user_id(email, name=None):
    """
    Returns the canonical user_id for an email, creating one the first time
    this email is seen. Signup and login both call this so the same person
    always resolves to the same user_id, instead of each flow generating its
    own (previously: a random UUID at signup vs. the email prefix at login).
    """
    email_key = email.strip().lower()
    data = get_users_data()

    record = data.get(email_key)
    if record:
        if name and not record.get("name"):
            record["name"] = name
            save_users_data(data)
        return record["user_id"], record.get("name") or name, False

    user_id = "user_" + uuid.uuid4().hex[:8]
    data[email_key] = {"user_id": user_id, "email": email, "name": name}
    save_users_data(data)
    return user_id, name, True
