from db import get_client


def get_profile(user_id):
    client = get_client()
    result = client.table("ai_profiles").select("profile").eq("user_id", user_id).limit(1).execute()
    if result.data:
        return result.data[0]["profile"]
    return None


def save_profile(user_id, profile_data):
    client = get_client()
    client.table("ai_profiles").upsert({"user_id": user_id, "profile": profile_data}).execute()
    return profile_data


def migrate_profile(guest_id, user_id):
    """
    Carries an AI profile (body shape/skin tone/face shape from /analyze)
    computed while browsing as a guest over to the authenticated account,
    same pattern as migrate_closet_items in closet_manager.py.
    """
    client = get_client()
    result = client.table("ai_profiles").select("profile").eq("user_id", guest_id).limit(1).execute()
    if not result.data:
        return False

    profile_data = result.data[0]["profile"]
    client.table("ai_profiles").upsert({"user_id": user_id, "profile": profile_data}).execute()
    client.table("ai_profiles").delete().eq("user_id", guest_id).execute()
    return True
