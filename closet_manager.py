import json
import os
import hashlib
from datetime import datetime

CLOSET_FILE = "closet_data.json"

def get_closet_data():
    if not os.path.exists(CLOSET_FILE):
        return {}
    with open(CLOSET_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_closet_data(data):
    with open(CLOSET_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def generate_image_hash(image_b64):
    """Run MD5 hash on the base64 string to identify duplicates."""
    return hashlib.md5(image_b64.encode('utf-8')).hexdigest()

def add_items_to_closet(user_id, outfits_dict):
    """
    Takes the dictionary of detected outfits (tops, bottoms, etc.)
    and adds each individual cropped item to the user's closet.
    """
    data = get_closet_data()
    
    if user_id not in data:
        data[user_id] = []
        
    user_closet = data[user_id]
    added_count = 0
    duplicate_count = 0

    # Look through all categories (top, bottom, footwear, accessories)
    for category, items in outfits_dict.items():
        if isinstance(items, list):
            for item in items:
                img_b64 = item.get("image")
                if not img_b64:
                    continue
                    
                img_hash = generate_image_hash(img_b64)
                
                # Check for duplicates
                is_duplicate = False
                for existing_item in user_closet:
                    if existing_item.get("image_hash") == img_hash:
                        is_duplicate = True
                        duplicate_count += 1
                        break
                
                if not is_duplicate:
                    # Add new item
                    new_item = {
                        "category": category,
                        "label": item.get("label", "unknown"),
                        "image_hash": img_hash,
                        "image_b64": img_b64,
                        "upload_timestamp": datetime.now().isoformat(),
                        "dominant_hex": item.get("dominant_hex"),
                        "dominant_hue": item.get("dominant_hue")
                    }
                    user_closet.append(new_item)
                    added_count += 1

    if added_count > 0:
        save_closet_data(data)
        
    return added_count, duplicate_count

def get_user_closet(user_id):
    data = get_closet_data()
    return data.get(user_id, [])
