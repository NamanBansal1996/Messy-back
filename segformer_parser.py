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

# Labels in mattmdjaga/segformer_b2_clothes:
# 0: Background, 1: Hair, 2: Face/Skin, 3: Glasses
# 4: Upper-clothes, 5: Skirt, 6: Pants, 7: Dress, 8: Belts
# 9: Left-shoe, 10: Right-shoe, 11: Headwear
# 12: Bag, 13: Scarf
BACKGROUND = 0
HAIR = 1
FACE_SKIN = 2
GLASSES = 3
CLOTHES_CLASSES = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13}


def parse_human(image_bgr):
    """
    Runs SegFormer human parsing ONCE on a BGR image and returns the raw
    per-pixel class-id map (same H, W as the input image).

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

    return upsampled_logits.argmax(dim=1)[0].numpy()


def mask_out_skin_and_bg(image_bgr, label_map=None):
    """
    Takes an OpenCV BGR image. Returns an OpenCV RGBA image with skin, hair,
    and bg made fully transparent.

    An already-computed label_map (from parse_human) can be passed in to
    avoid re-running inference.
    """
    if label_map is None:
        label_map = parse_human(image_bgr)

    # Create a binary mask targeting strictly clothing pixels
    mask = np.isin(label_map, list(CLOTHES_CLASSES)).astype(np.uint8) * 255

    # Smooth the mask to remove jagged transformer patch boundaries
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Apply mask via alpha channel
    b, g, r = cv2.split(image_bgr)
    rgba = cv2.merge([b, g, r, mask])

    return rgba


def get_person_mask(label_map):
    """Binary (0/1) mask of every pixel classified as part of the person (hair + skin + clothing)."""
    return (label_map != BACKGROUND).astype(np.uint8)


def get_skin_mask(label_map):
    """Binary (0/1) mask of pixels classified as face/skin."""
    return (label_map == FACE_SKIN).astype(np.uint8)


def get_hair_mask(label_map):
    """Binary (0/1) mask of pixels classified as hair."""
    return (label_map == HAIR).astype(np.uint8)
