import uuid

from werkzeug.security import generate_password_hash, check_password_hash

from db import get_client

MIN_PASSWORD_LENGTH = 8


class InvalidCredentialsError(Exception):
    """Existing account, but the submitted password doesn't match (or the
    account predates password_hash existing at all)."""


class PasswordRequiredError(Exception):
    """New account being created with no password."""


class PasswordTooShortError(Exception):
    """New account's password doesn't meet the minimum length."""


def resolve_user_id(email, password, name=None):
    """
    Returns the canonical user_id for an email -- verifying the password
    against the stored hash for an existing account, or creating a new
    account (hashing the given password) if this email hasn't been seen
    before.

    Both signup and login call this same function: whether the email
    already has a password_hash is what decides create-vs-verify, so the
    frontend doesn't need to declare which flow it's in, and signup and
    login can never again silently resolve to two different user_ids for
    the same person (the original bug this whole identity system replaced).
    """
    email_key = email.strip().lower()
    client = get_client()

    existing = client.table("users").select("*").eq("email", email_key).limit(1).execute()
    if existing.data:
        record = existing.data[0]
        stored_hash = record.get("password_hash")
        if not stored_hash or not check_password_hash(stored_hash, password or ""):
            raise InvalidCredentialsError("Incorrect email or password.")

        resolved_name = record.get("name") or name
        if name and not record.get("name"):
            client.table("users").update({"name": name}).eq("email", email_key).execute()
        return record["user_id"], resolved_name, False

    if not password:
        raise PasswordRequiredError("A password is required to create an account.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise PasswordTooShortError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")

    user_id = "user_" + uuid.uuid4().hex[:8]
    client.table("users").insert({
        "email": email_key,
        "user_id": user_id,
        "name": name,
        "password_hash": generate_password_hash(password),
    }).execute()
    return user_id, name, True
