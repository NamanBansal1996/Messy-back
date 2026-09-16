import os

from storage_utils import read_json, write_json_atomic

PROFILE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_profiles.json")


def get_profiles_data():
    return read_json(PROFILE_FILE, {})


def get_profile(user_id):
    return get_profiles_data().get(user_id)


def save_profile(user_id, profile_data):
    data = get_profiles_data()
    data[user_id] = profile_data
    write_json_atomic(PROFILE_FILE, data)
    return data[user_id]


def migrate_profile(guest_id, user_id):
    """
    Carries an AI profile (body shape/skin tone/face shape from /analyze)
    computed while browsing as a guest over to the authenticated account,
    same pattern as migrate_closet_items in closet_manager.py.
    """
    data = get_profiles_data()
    if guest_id not in data:
        return False

    data[user_id] = data[guest_id]
    del data[guest_id]
    write_json_atomic(PROFILE_FILE, data)
    return True
