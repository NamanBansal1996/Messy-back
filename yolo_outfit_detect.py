# yolo_outfit_detect.py


import time
import os
import cv2
import base64
import numpy as np
from color_utils import get_dominant_color
from segformer_parser import mask_out_skin_and_bg, parse_human

# =====================================================
# 🔹 DIGITAL WARDROBE BASE DIR
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WARDROBE_DIR = os.path.join(BASE_DIR, "digital_wardrobe")

# =====================================================
# 🔹 CATEGORY DEFINITIONS
#
# Garment categorization comes from SegFormer's real per-pixel human-parsing
# classes (see segformer_parser.py), not from an object-detection model --
# yolov8n.pt (previously used here) is the stock COCO checkpoint, which has
# no clothing classes at all, so it could never actually detect a shirt,
# pants, or a dress; every request silently fell through to a "split the
# person box into a top 45% / bottom 55%" guess regardless of what was
# actually being worn. SegFormer already runs once per request (for body
# shape/skin tone) and already distinguishes Upper-clothes/Skirt/Pants/
# Dress/Belts/Shoes/Headwear/Bag/Scarf at the pixel level, so real
# categorization -- including a real "dress" category -- comes for free.
# =====================================================

# class_id -> (category, label)
#
# Class 11 (Headwear) is deliberately excluded: testing against 6 real
# photos, none wearing a hat, showed it firing every single time (0.82-0.96
# confidence) -- it appears to confuse hair/hairline regions with headwear
# rather than reliably detecting actual hats. Re-add it if/when that's
# verified against real hat-wearing photos.
SEGFORMER_GARMENT_CLASSES = {
    4:  ("top", "shirt"),
    5:  ("bottom", "skirt"),
    6:  ("bottom", "pants"),
    7:  ("dress", "dress"),
    8:  ("accessories", "belt"),
    12: ("accessories", "bag"),
    13: ("accessories", "scarf"),
}
# Left-shoe (9) and Right-shoe (10) are merged into one "shoes" item rather
# than reported as two separate footwear entries.
FOOTWEAR_CLASSES = {9, 10}

# Filters out small segmentation-noise islands (a known artifact of
# patch-based transformer segmentation at class boundaries) -- a detected
# region must cover at least this fraction of the image, with an absolute
# floor so a tiny image can't produce a near-zero threshold.
MIN_ITEM_AREA_FRACTION = 0.003
MIN_ITEM_AREA_FLOOR_PX = 200


def _bbox_from_mask(mask, pad_frac=0.15):
    """
    Returns a 15%-padded bounding box (x1, y1, x2, y2) around every nonzero
    pixel in a binary mask, or None if the mask is empty. Padding is applied
    by the caller so it can be clipped against the image bounds.
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    pad_x, pad_y = int((x2 - x1) * pad_frac), int((y2 - y1) * pad_frac)
    return x1, y1, x2, y2, pad_x, pad_y


# =====================================================
# 🔹 MAIN FUNCTION CALLED BY app.py
# =====================================================
def detect_outfits(image, label_map=None, confidence_map=None):
    """
    image: OpenCV BGR image (numpy array)
    label_map / confidence_map: optional precomputed SegFormer parse_human()
        output, passed in by the caller when it already ran segmentation for
        this request (avoids running the transformer model twice on the same
        image). Computed internally if not provided.
    return: dict with detected outfit categories (top/bottom/dress/footwear/
        accessories) + cropped images
    """
    if label_map is None:
        label_map, confidence_map = parse_human(image)
    elif confidence_map is None:
        # A caller passed a label_map without a confidence_map (e.g. an
        # older/simplified call site) -- fall back to a neutral confidence
        # rather than crashing.
        confidence_map = np.full(label_map.shape, 0.85, dtype=np.float32)

    # Full transparent RGBA image of only the clothes, for saved/preview crops
    image_rgba = mask_out_skin_and_bg(image, label_map=label_map)

    img_h, img_w = label_map.shape
    min_area = max(MIN_ITEM_AREA_FLOOR_PX, int(img_h * img_w * MIN_ITEM_AREA_FRACTION))

    outfits = {
        "top": [],
        "bottom": [],
        "dress": [],
        "footwear": [],
        "accessories": [],
    }

    # Each entry: (category, label, binary mask of exactly that garment)
    detected_groups = []
    for class_id, (category, label) in SEGFORMER_GARMENT_CLASSES.items():
        class_mask = (label_map == class_id)
        if class_mask.sum() >= min_area:
            detected_groups.append((category, label, class_mask))

    footwear_mask = np.isin(label_map, list(FOOTWEAR_CLASSES))
    if footwear_mask.sum() >= min_area:
        detected_groups.append(("footwear", "shoes", footwear_mask))

    for category, label, item_mask in detected_groups:
        bbox = _bbox_from_mask(item_mask)
        if bbox is None:
            continue
        x1, y1, x2, y2, pad_x, pad_y = bbox
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img_w, x2 + pad_x)
        y2 = min(img_h, y2 + pad_y)

        # We crop using the RGBA image so it natively drops skin and background
        crop_rgba = image_rgba[y1:y2, x1:x2]
        crop_bgr = image[y1:y2, x1:x2]

        if crop_rgba.size == 0 or crop_bgr.size == 0:
            continue

        # Confidence = mean per-pixel softmax probability over this exact
        # garment's mask -- a real measurement, not a guessed constant.
        confidence = float(confidence_map[item_mask].mean())

        # =================================================
        # 🔹 SAVE CROPPED IMAGE TO DIGITAL WARDROBE
        # =================================================
        timestamp = int(time.time() * 1000)
        filename = f"{label}_{timestamp}.png"

        category_dir = os.path.join(WARDROBE_DIR, category)
        os.makedirs(category_dir, exist_ok=True)

        save_path = os.path.join(category_dir, filename)
        cv2.imwrite(save_path, crop_rgba)

        # =================================================
        # 🔹 BASE64 FOR FRONTEND PREVIEW
        # =================================================
        _, buffer = cv2.imencode(".png", crop_rgba)
        crop_b64 = base64.b64encode(buffer).decode("utf-8")

        # =================================================
        # 🔹 EXTRACT DOMINANT COLOR FROM THE EXACT GARMENT PIXELS
        # Note: we pass crop_bgr (not the RGBA cutout) so color extraction
        # isn't affected by the RGBA alpha layer, and we pass the item's own
        # mask (cropped to the same box) so skin/hair/other-garment/
        # background pixels that happen to fall inside the bounding box are
        # excluded from the sample entirely.
        # =================================================
        crop_item_mask = item_mask[y1:y2, x1:x2].astype(np.uint8)
        dominant_hex, dominant_hue, color_name = get_dominant_color(crop_bgr, mask=crop_item_mask)

        outfits[category].append({
            "label": label,
            "confidence": round(confidence, 2),
            "image": crop_b64,
            "saved_path": save_path,
            "dominant_hex": dominant_hex,
            "dominant_hue": dominant_hue,
            "color_name": color_name,
            "bounding_box": {
                "x_pct": (x1 / img_w) * 100,
                "y_pct": (y1 / img_h) * 100,
                "w_pct": ((x2 - x1) / img_w) * 100,
                "h_pct": ((y2 - y1) / img_h) * 100
            }
        })

    return outfits
