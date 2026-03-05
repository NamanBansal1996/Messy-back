# # yolo_outfit_detect.py
# import time
# import os
# import cv2
# import base64
# from ultralytics import YOLO
#
# # =====================================================
# # 🔹 LOAD TRAINED YOLO MODEL (best.pt)
# # =====================================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
#
# model = YOLO(MODEL_PATH)
#
# # =====================================================
# # 🔹 CATEGORY DEFINITIONS (EXPANDED & ROBUST)
# # These MUST match labels printed by YOLO exactly
# # =====================================================
# TOP = {
#     "tshirt",
#     "shirt",
#     "short sleeve shirt",
#     "long sleeve shirt",
#     "sweater",
#     "jacket",
#     "dress"
# }
#
# BOTTOM = {
#     "pants",
#     "jeans",
#     "trousers",
#     "skirt"
# }
#
# FOOTWEAR = {
#     "shoes"
# }
#
# ACCESSORIES = {
#     "bag"
# }
#
# def categorize(label: str):
#     if label in TOP:
#         return "top"
#     elif label in BOTTOM:
#         return "bottom"
#     elif label in FOOTWEAR:
#         return "footwear"
#     elif label in ACCESSORIES:
#         return "accessories"
#     return None
#
#
# # =====================================================
# # 🔹 MAIN FUNCTION CALLED BY app.py
# # =====================================================
# def detect_outfits(image):
#     """
#     image: OpenCV image (numpy array)
#     return: dict with detected outfit categories + cropped images
#     """
#
#     results = model(image)[0]
#
#     outfits = {
#         "top": [],
#         "bottom": [],
#         "footwear": [],
#         "accessories": []
#     }
#
#     for box in results.boxes:
#         cls_id = int(box.cls[0])
#         label = results.names[cls_id].lower()
#
#         category = categorize(label)
#         if not category:
#             continue
#
#         # Confidence threshold (keeps results clean)
#         if box.conf[0] < 0.4:
#             continue
#
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         crop = image[y1:y2, x1:x2]
#
#         if crop.size == 0:
#             continue
#
#         _, buffer = cv2.imencode(".jpg", crop)
#         crop_b64 = base64.b64encode(buffer).decode("utf-8")
#
#         outfits[category].append({
#             "label": label,
#             "confidence": round(float(box.conf[0]), 2),
#             "image": crop_b64
#         })
#
#     return outfits

#-------------
# yolo_outfit_detect.py

import time
import os
import cv2
import base64
from ultralytics import YOLO
from color_utils import get_dominant_color

# =====================================================
# 🔹 LOAD TRAINED YOLO MODEL (best.pt)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

model = YOLO(MODEL_PATH)

# =====================================================
# 🔹 DIGITAL WARDROBE BASE DIR
# =====================================================
WARDROBE_DIR = os.path.join(BASE_DIR, "digital_wardrobe")

# =====================================================
# 🔹 CATEGORY DEFINITIONS (MATCH YOLO LABELS)
# =====================================================
TOP = {
    "tshirt",
    "shirt",
    "short sleeve shirt",
    "long sleeve shirt",
    "sweater",
    "jacket",
    "dress"
}

BOTTOM = {
    "pants",
    "jeans",
    "trousers",
    "skirt"
}

FOOTWEAR = {
    "shoes"
}

ACCESSORIES = {
    "bag"
}

def categorize(label: str):
    if label in TOP:
        return "top"
    elif label in BOTTOM:
        return "bottom"
    elif label in FOOTWEAR:
        return "footwear"
    elif label in ACCESSORIES:
        return "accessories"
    return None


# =====================================================
# 🔹 MAIN FUNCTION CALLED BY app.py
# =====================================================
def detect_outfits(image):
    """
    image: OpenCV image (numpy array)
    return: dict with detected outfit categories + cropped images
    """

    results = model(image)[0]

    outfits = {
        "top": [],
        "bottom": [],
        "footwear": [],
        "accessories": []
    }

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names[cls_id].lower()

        category = categorize(label)
        if not category:
            continue

        # Confidence threshold
        if box.conf[0] < 0.4:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # =================================================
        # 🔹 SAVE CROPPED IMAGE TO DIGITAL WARDROBE
        # =================================================
        timestamp = int(time.time() * 1000)
        filename = f"{label}_{timestamp}.jpg"

        category_dir = os.path.join(WARDROBE_DIR, category)
        os.makedirs(category_dir, exist_ok=True)

        save_path = os.path.join(category_dir, filename)
        cv2.imwrite(save_path, crop)

        # =================================================
        # 🔹 BASE64 FOR FRONTEND PREVIEW
        # =================================================
        _, buffer = cv2.imencode(".jpg", crop)
        crop_b64 = base64.b64encode(buffer).decode("utf-8")

        # =================================================
        # 🔹 EXTRACT DOMINANT COLOR
        # =================================================
        dominant_hex, dominant_hue = get_dominant_color(crop)

        outfits[category].append({
            "label": label,
            "confidence": round(float(box.conf[0]), 2),
            "image": crop_b64,
            "saved_path": save_path,
            "dominant_hex": dominant_hex,
            "dominant_hue": dominant_hue
        })

    return outfits
