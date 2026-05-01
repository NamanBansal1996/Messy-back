import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import cv2
import os

print("loading SegFormer fashion parser model...")
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
print("SegFormer loaded successfully.")

# Target Classes we want to KEEP for digital closet
# Labels in mattmdjaga/segformer_b2_clothes:
# 0: Background, 1: Hair, 2: Face/Skin, 3: Glasses
# 4: Upper-clothes, 5: Skirt, 6: Pants, 7: Dress, 8: Belts
# 9: Left-shoe, 10: Right-shoe, 11: Headwear
# 12: Bag, 13: Scarf
CLOTHES_CLASSES = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13}

def mask_out_skin_and_bg(image_bgr):
    """
    Takes an OpenCV BGR image.
    Passes it through SegFormer.
    Returns an OpenCV RGBA image with skin, hair, and bg made fully transparent.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # 1. Inference
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits.cpu()

    # 2. Upsample back to original image size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=pil_image.size[::-1], # (height, width)
        mode="bilinear",
        align_corners=False,
    )
    
    # 3. Label prediction map
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    
    # 4. Create binary mask targeting strictly clothing pixels
    mask = np.isin(pred_seg, list(CLOTHES_CLASSES)).astype(np.uint8) * 255
    
    # 5. Smooth the mask to remove jagged transformer patch boundaries
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 6. Apply mask via alpha channel
    b, g, r = cv2.split(image_bgr)
    rgba = cv2.merge([b, g, r, mask])
    
    return rgba
