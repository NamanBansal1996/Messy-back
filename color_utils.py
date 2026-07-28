import cv2
import numpy as np
from sklearn.cluster import KMeans

def get_dominant_color(image_crop, mask=None):
    """
    Extracts the dominant color from a clothing crop.
    Uses HSV color space to pre-filter noise (shadows, highlights, background walls),
    then runs KMeans clustering on the vibrant pixels.

    mask: optional array, same H x W as image_crop, where nonzero marks a
    real garment pixel. When provided, only those pixels are sampled --
    skin/hair/other-garment/background pixels that happen to fall inside
    the bounding-box crop are excluded before the HSV filtering below even
    runs. When None, the whole crop is sampled (previous behavior).
    """
    if image_crop is None or image_crop.size == 0:
        return None, None, None

    # 1. Convert BGR to HSV
    hsv_image = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)

    # Reshape for filtering, restricting to the exact garment mask if given
    if mask is not None:
        pixels_bgr = image_crop[mask > 0]
        pixels_hsv = hsv_image[mask > 0]
    else:
        pixels_bgr = image_crop.reshape(-1, 3)
        pixels_hsv = hsv_image.reshape(-1, 3)

    if len(pixels_bgr) == 0:
        # Mask excluded every pixel (shouldn't normally happen) -- fall back
        # to the full crop rather than returning nothing.
        pixels_bgr = image_crop.reshape(-1, 3)
        pixels_hsv = hsv_image.reshape(-1, 3)

    # 2. Pre-filter noisy pixels mathematically
    # We want to remove:
    # - Saturation < 15 (Grays, whites, washed out background walls)
    # - Value < 20 (Pure black, heavy shadows, dark unlit areas)
    # - Value > 95 (for 0-100 scale, but OpenCV V is 0-255. So V > 240)

    # OpenCV HSV scales: H: 0-179, S: 0-255, V: 0-255
    min_sat = int(255 * (5 / 100)) # Lowered to 5% to keep gray/white clothing
    min_val = int(255 * (10 / 100)) # Lowered to 10% to keep dark gray/black clothing
    max_val = int(255 * (98 / 100)) # Raised to 98% to keep bright white clothing

    # Create a boolean mask of "good" pixels
    hsv_filter_mask = (pixels_hsv[:, 1] > min_sat) & (pixels_hsv[:, 2] > min_val) & (pixels_hsv[:, 2] < max_val)

    filtered_pixels = pixels_bgr[hsv_filter_mask]

    # 3. Safety Fallback
    # If the garment itself was black, white, or gray, the filter might remove everything.
    if len(filtered_pixels) < 500:
        # Use a vastly simplified filter (just remove extreme dark/light) or fallback to all pixels
        basic_mask = (pixels_hsv[:, 2] > 10) & (pixels_hsv[:, 2] < 245)
        filtered_pixels = pixels_bgr[basic_mask]

        # If STILL too few, just use the raw crop
        if len(filtered_pixels) < 100:
            filtered_pixels = pixels_bgr

    # 4. K-Means Clustering on the robust pixel set
    # Using 3 clusters to find the most dominant block of color -- but a
    # tightly-masked small garment (e.g. a belt) can leave fewer pixels than
    # clusters, which KMeans can't handle, so shrink cluster count to fit.
    n_clusters = min(3, len(filtered_pixels))
    if n_clusters < 1:
        return None, None, None
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    kmeans.fit(filtered_pixels)

    # 5. Select the cluster with the highest pixel count
    labels = kmeans.labels_
    counts = np.bincount(labels)
    dominant_cluster_idx = np.argmax(counts)

    dominant_bgr = kmeans.cluster_centers_[dominant_cluster_idx]

    # Ensure it's 8-bit integer
    dominant_bgr = np.uint8(dominant_bgr)

    # Convert BGR to Hex
    hex_color = "#{:02x}{:02x}{:02x}".format(dominant_bgr[2], dominant_bgr[1], dominant_bgr[0]) # RGB hex

    # Convert BGR to HSV to get the Hue for our band filtering
    dominant_bgr_array = np.uint8([[dominant_bgr]])
    dominant_hsv = cv2.cvtColor(dominant_bgr_array, cv2.COLOR_BGR2HSV)[0][0]

    # OpenCV Hue is 0-179. Standard Hue is 0-360. Kept as-is: recommendation_engine.py's
    # _approximate_color_name() relies on this exact hue value for wardrobe items that
    # don't have a persisted color_name.
    true_hue = int(dominant_hsv[0] * 2)

    color_name = get_color_name(dominant_bgr)

    return hex_color, true_hue, color_name


# =====================================================
# 🔹 REFERENCE PALETTE FOR NEAREST-COLOR-NAME MATCHING
#
# Precise fashion color naming from a single dominant RGB doesn't work well
# as a cascade of hand-tuned hue/saturation/value thresholds (the previous
# approach here): the "saturation < 35% -> some shade of Gray" rule fired
# for ANY desaturated color regardless of hue, so pale/washed-out fabric
# colors -- light-wash denim, sage green, dusty pink -- all silently lost
# their hue and got called "Gray". Coarse hue bins (e.g. one single 85-degree
# range from 65-150 all called "Green") also couldn't distinguish visually
# distinct colors like yellow-green from teal-green.
#
# This replaces that cascade with a standard nearest-neighbor lookup: each
# named color below is converted to CIE Lab once at import time (a
# perceptually uniform color space, unlike HSV, where Euclidean distance
# actually tracks how different two colors LOOK to a person), and a sampled
# color is matched to whichever palette entry is closest in that space.
# This is the same general approach used by most "nearest CSS color name"
# tools, and naturally fixes the desaturated-color problem above: a pale
# blue is simply closer to "Light Blue"/"Denim Blue" in Lab space than to
# "Gray", no saturation special-casing required.
# =====================================================
_COLOR_PALETTE_RGB = {
    # Neutrals
    "Black": (20, 20, 20),
    "Charcoal": (54, 54, 58),
    "Dark Gray": (89, 89, 89),
    "Gray": (128, 128, 128),
    "Light Gray": (195, 195, 195),
    "Silver": (211, 211, 211),
    "White": (250, 250, 248),
    "Ivory": (255, 255, 240),
    "Cream": (255, 253, 208),

    # Beige / tan / brown family
    "Beige": (222, 201, 168),
    "Tan": (210, 180, 140),
    "Khaki": (189, 183, 107),
    "Camel": (193, 154, 107),
    "Taupe": (150, 133, 120),
    "Brown": (101, 67, 33),
    "Chocolate": (89, 55, 34),
    "Chestnut": (149, 69, 53),
    "Rust": (183, 65, 14),

    # Red family
    "Red": (200, 30, 30),
    "Crimson": (180, 20, 50),
    "Maroon": (114, 47, 55),
    "Burgundy": (106, 13, 34),
    "Coral": (255, 127, 80),
    "Salmon": (250, 128, 114),

    # Pink family
    "Pink": (255, 182, 193),
    "Hot Pink": (255, 105, 180),
    "Light Pink": (255, 214, 220),
    "Rose": (224, 120, 140),
    "Magenta": (200, 40, 150),

    # Orange / yellow family
    "Orange": (240, 120, 30),
    "Peach": (255, 204, 153),
    "Mustard": (210, 170, 40),
    "Yellow": (235, 220, 60),
    "Gold": (212, 175, 55),

    # Green family
    "Olive Green": (110, 110, 40),
    "Green": (40, 130, 60),
    "Dark Green": (20, 80, 40),
    "Forest Green": (34, 80, 34),
    "Sage Green": (150, 168, 140),
    "Mint Green": (160, 220, 180),
    "Lime Green": (140, 200, 60),

    # Blue family (Teal/Turquoise included here since they sit on the
    # blue-green boundary and read as "blue-ish" in clothing contexts)
    "Teal": (20, 120, 120),
    "Turquoise": (64, 190, 180),
    "Sky Blue": (120, 180, 220),
    "Light Blue": (170, 210, 235),
    "Denim Blue": (70, 110, 150),
    "Blue": (40, 90, 190),
    "Navy Blue": (20, 40, 90),
    "Royal Blue": (40, 60, 180),

    # Purple family
    "Purple": (110, 60, 150),
    "Lavender": (190, 175, 220),
    "Plum": (120, 60, 110),
    "Violet": (150, 80, 190),
}

_PALETTE_NAMES = list(_COLOR_PALETTE_RGB.keys())
_palette_bgr_array = np.uint8([[list(rgb)[::-1] for rgb in _COLOR_PALETTE_RGB.values()]])  # RGB -> BGR
_PALETTE_LAB = cv2.cvtColor(_palette_bgr_array, cv2.COLOR_BGR2LAB)[0].astype(np.float32)



# Lightness (L) is weighted down relative to hue/chroma (a, b) when computing
# color distance below. Plain unweighted Euclidean distance in Lab (CIE76)
# was tested against real photo crops and misfired specifically on
# near-white fabric: a slightly-warm off-white (common under indoor/studio
# lighting, which rarely renders "white" as perfectly bright and neutral as
# an idealized reference) matched closer to "Light Pink" than to "White",
# purely because it happened to sit a few lightness points nearer to the
# Light Pink reference than the White reference, despite its hue being
# barely tinted at all. This mirrors why real perceptual color-difference
# formulas (CIE94/CIEDE2000) don't weight L, a, b equally either -- L varies
# with lighting/exposure far more than it reflects a garment's actual color.
# 0.5 was chosen by testing against real photo crops: it fixes the
# near-white misclassification above while still correctly keeping "Light
# Blue" (not "Sky Blue") as the nearest match for pale denim -- the case
# that motivated this rewrite in the first place.
_LIGHTNESS_WEIGHT = 0.5


def get_color_name(bgr_pixel):
    """
    Given a single BGR color (uint8 3-tuple/array), returns the name of the
    closest color in the reference palette above, using (lightness-weighted)
    Euclidean distance in CIE Lab space -- a perceptually uniform space, so
    "closest" tracks what a person would actually call the color, unlike
    raw HSV threshold cascades.
    """
    sample_bgr = np.uint8([[bgr_pixel]])
    sample_lab = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2LAB)[0][0].astype(np.float32)

    diff = _PALETTE_LAB - sample_lab
    diff[:, 0] *= _LIGHTNESS_WEIGHT
    distances = np.sum(diff ** 2, axis=1)
    nearest_idx = int(np.argmin(distances))
    return _PALETTE_NAMES[nearest_idx]
