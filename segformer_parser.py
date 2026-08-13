import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import cv2
import os

print("loading SegFormer fashion parser model...")
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
model.eval()
print("SegFormer loaded successfully.")

# Real labels in mattmdjaga/segformer_b2_clothes (verified against the
# loaded model's own config.id2label -- the model actually has 18 classes,
# not 14, and every ID from 1 onward previously assumed here was wrong,
# e.g. class 11 is Face, not Headwear, which is why garment-only cutouts
# were keeping faces fully opaque instead of stripping them):
# 0: Background, 1: Hat, 2: Hair, 3: Sunglasses
# 4: Upper-clothes, 5: Skirt, 6: Pants, 7: Dress, 8: Belt
# 9: Left-shoe, 10: Right-shoe, 11: Face
# 12: Left-leg, 13: Right-leg, 14: Left-arm, 15: Right-arm
# 16: Bag, 17: Scarf
BACKGROUND = 0
HAT = 1
HAIR = 2
SUNGLASSES = 3
UPPER_CLOTHES = 4
SKIRT = 5
PANTS = 6
DRESS = 7
BELT = 8
LEFT_SHOE = 9
RIGHT_SHOE = 10
FACE = 11
LEFT_LEG = 12
RIGHT_LEG = 13
LEFT_ARM = 14
RIGHT_ARM = 15
BAG = 16
SCARF = 17

# Hat (1) is deliberately left out for now -- unlike every other class here,
# it's never been validated against real hat-wearing photos (the previous
# "headwear" exclusion in yolo_outfit_detect.py was actually reacting to
# Face, class 11, misread as headwear, so it never tested the real Hat
# class at all). Leave disabled until verified separately.
CLOTHES_CLASSES = {
    UPPER_CLOTHES, SKIRT, PANTS, DRESS, BELT, LEFT_SHOE, RIGHT_SHOE, BAG, SCARF,
}


def parse_human(image_bgr):
    """
    Runs SegFormer human parsing ONCE on a BGR image and returns:
      - label_map: per-pixel class-id map (same H, W as the input image)
      - confidence_map: per-pixel softmax probability of the winning class,
        a genuine confidence signal (used e.g. by yolo_outfit_detect.py to
        score detected garments, instead of a hardcoded/guessed value)

    Callers should compute this once per request and reuse it (via the
    label_map params below) instead of calling this repeatedly -- each
    call is a full transformer forward pass.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.cpu()

    # Upsample back to original image size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=pil_image.size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False,
    )

    probs = torch.nn.functional.softmax(upsampled_logits, dim=1)
    label_map = probs.argmax(dim=1)[0].numpy()
    confidence_map = probs.max(dim=1).values[0].numpy()
    return label_map, confidence_map


SKIN_LIKE_CLASSES = {HAIR, FACE, LEFT_LEG, RIGHT_LEG, LEFT_ARM, RIGHT_ARM}


def _seal_enclosed_skin_holes(label_map, clothes_mask):
    """
    Fills enclosed gaps inside the clothing silhouette (e.g. skin visible
    through an open V-neck/collar) so the garment cutout keeps a solid,
    continuous shape instead of a floating transparent hole -- WITHOUT
    touching real background gaps that happen to also be geometrically
    "enclosed" (e.g. the space between someone's legs, or between an arm
    held slightly away from the torso).

    Purely geometric hole-filling (cv2.findContours + fill, or
    cv2.morphologyEx close) can't tell these two cases apart -- both are
    just "a hole surrounded by clothing pixels" topologically. The fix:
    only fill an enclosed hole if the pixels inside it are actually
    classified as skin/hair/glasses, not background. Verified against
    real photos: a naive geometric fill painted a true background gap
    between the legs solid black; this class-aware version leaves it
    alone and only fills genuine skin islands.
    """
    non_clothes = (~clothes_mask).astype(np.uint8)
    num_labels, components = cv2.connectedComponents(non_clothes, connectivity=8)

    border_labels = (
        set(components[0, :]) | set(components[-1, :])
        | set(components[:, 0]) | set(components[:, -1])
    )
    border_labels.discard(0)

    sealed = clothes_mask.copy()
    for component_id in range(1, num_labels):
        if component_id in border_labels:
            continue  # touches the image edge -> real exterior background/face/hands
        island = components == component_id
        skin_fraction = np.isin(label_map[island], list(SKIN_LIKE_CLASSES)).mean()
        if skin_fraction > 0.5:
            sealed |= island

    return sealed


def mask_out_skin_and_bg(image_bgr, label_map=None):
    """
    Takes an OpenCV BGR image. Returns an OpenCV RGBA image with skin, hair,
    and bg made fully transparent -- except for skin/hair pixels fully
    enclosed inside the clothing silhouette (e.g. a V-neck/collar gap),
    which are kept natural rather than punched into a floating hole.

    An already-computed label_map (from parse_human) can be passed in to
    avoid re-running inference.
    """
    if label_map is None:
        label_map = parse_human(image_bgr)

    # Create a binary mask targeting strictly clothing pixels
    clothes_mask = np.isin(label_map, list(CLOTHES_CLASSES))
    clothes_mask = _seal_enclosed_skin_holes(label_map, clothes_mask)
    mask = clothes_mask.astype(np.uint8) * 255

    # Smooth the mask to remove jagged transformer patch boundaries
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Apply mask via alpha channel
    b, g, r = cv2.split(image_bgr)
    rgba = cv2.merge([b, g, r, mask])

    return rgba


def remove_background_only(image_bgr, label_map=None):
    """
    Takes an OpenCV BGR image. Returns an OpenCV RGBA image with only the
    external background made transparent -- face, hair, skin, and clothing
    all stay fully intact. For full-person previews and Virtual Try-On base
    photos, where (unlike mask_out_skin_and_bg) skin should never be
    stripped out.

    An already-computed label_map (from parse_human) can be passed in to
    avoid re-running inference.
    """
    if label_map is None:
        label_map = parse_human(image_bgr)

    mask = get_person_mask(label_map).astype(np.uint8) * 255
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    b, g, r = cv2.split(image_bgr)
    rgba = cv2.merge([b, g, r, mask])

    return rgba


def get_person_mask(label_map):
    """Binary (0/1) mask of every pixel classified as part of the person (hair + skin + clothing)."""
    return (label_map != BACKGROUND).astype(np.uint8)


def get_skin_mask(label_map):
    """Binary (0/1) mask of pixels classified as face."""
    return (label_map == FACE).astype(np.uint8)


def get_hair_mask(label_map):
    """Binary (0/1) mask of pixels classified as hair."""
    return (label_map == HAIR).astype(np.uint8)
