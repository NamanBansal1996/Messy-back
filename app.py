from dotenv import load_dotenv
load_dotenv()  # loads .env (git-ignored) before anything below reads env vars, e.g. ANTHROPIC_API_KEY

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import os
import base64
import mediapipe as mp
import math
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
from PIL import Image, ImageOps
import numpy as np
import traceback
import logging

logging.basicConfig(level=logging.INFO)

# 🔹 Import YOLO outfit detection
from yolo_outfit_detect import detect_outfits
from garment_classifier import enrich_outfits_with_attributes
from closet_manager import add_items_to_closet, get_user_closet, migrate_closet_items, resolve_duplicate
from user_store import resolve_user_id, InvalidCredentialsError, PasswordRequiredError, PasswordTooShortError
from profile_store import get_profile, save_profile, migrate_profile
from saved_looks_store import (
    get_saved_looks,
    save_look,
    delete_saved_look,
    migrate_saved_looks,
)
from gcs_storage import upload_temp_scan
from styling_rules import get_styling_recommendations
from virtual_tryon import generate_tryon
from recommendation_engine import generate_three_looks, generate_new_outfit_suggestions
from catalog import get_catalog_items, get_styling_catalog_items
from weather_service import get_current_weather
from segformer_parser import parse_human, get_person_mask, get_skin_mask, get_hair_mask

# ---------------- APP SETUP ----------------
app = Flask(__name__)

# Locked down to the actual frontend origins -- this used to be "*", which
# is fine for an unreachable localhost backend but not for a publicly
# invokable Cloud Run service. FLASK_DEBUG (already the local-dev toggle)
# also opts into allowing the Vite dev server origin.
ALLOWED_ORIGINS = [
    "https://themessy.co.in",
    "https://www.themessy.co.in",
]
if os.environ.get("FLASK_DEBUG", "false").lower() == "true":
    ALLOWED_ORIGINS.append("http://localhost:5173")

CORS(
    app,
    resources={r"/*": {"origins": ALLOWED_ORIGINS}}
)
UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection


@app.errorhandler(Exception)
def handle_unexpected_error(e):
    """
    Without this, an unhandled exception anywhere just falls through to
    Flask's default 500 handling -- which, on Cloud Run under gunicorn, was
    producing a bare 500 with no traceback logged anywhere (confirmed via
    Cloud Logging: the request log showed status 500, but no application
    log line, no traceback, nothing -- indistinguishable from an OOM kill).
    This guarantees every failure at least gets a real traceback in the logs.
    """
    if isinstance(e, HTTPException):
        return e
    app.logger.error(
        "Unhandled exception on %s %s:\n%s",
        request.method, request.path, traceback.format_exc()
    )
    return jsonify({"error": "Internal server error"}), 500

# ---------------- UTILS ----------------
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def load_image_exif_safe(filepath):
    """
    Loads an image and applies EXIF orientation before handing it to OpenCV.
    cv2.imread ignores EXIF rotation tags, so a portrait phone photo stored
    with a rotation tag (very common) would otherwise be processed sideways,
    silently breaking every landmark-based measurement downstream.
    Falls back to a plain cv2.imread if PIL can't open the file.
    """
    try:
        pil_img = Image.open(filepath)
        pil_img = ImageOps.exif_transpose(pil_img)
        rgb = np.array(pil_img.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return cv2.imread(filepath)


def assess_image_exposure(image):
    """
    Cheap whole-image brightness check. Very dark/blown-out photos degrade
    both skin-tone sampling (color is unreliable) and silhouette/edge based
    measurements. Returns a confidence multiplier in (0, 1] and an optional
    warning string.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())
    if mean_brightness < 40:
        return 0.6, "Photo looks quite dark - better, even lighting will improve accuracy."
    if mean_brightness > 215:
        return 0.6, "Photo looks overexposed/washed out - better, even lighting will improve accuracy."
    return 1.0, None


def build_face_region_mask(landmarks, h, w, pad_ratio=0.35):
    """Rough padded bounding-box mask around all face landmarks (always available, even without segmentation)."""
    xs = [p[0] for p in landmarks.values()]
    ys = [p[1] for p in landmarks.values()]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    pad_x = (x2 - x1) * pad_ratio
    pad_y = (y2 - y1) * pad_ratio
    x1, x2 = max(0, int(x1 - pad_x)), min(w, int(x2 + pad_x))
    y1, y2 = max(0, int(y1 - pad_y)), min(h, int(y2 + pad_y))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def compute_gray_world_scale(image_bgr, exclude_mask=None, strength=0.6):
    """
    Estimates a per-channel white-balance correction from the gray-world
    assumption (average scene reflectance ~= gray), computed over pixels
    OUTSIDE exclude_mask.

    Excluding the face/skin region is essential: skin is not actually
    neutral gray (it has a genuine, real warm/cool bias from melanin and
    blood flow). Computing the "should be gray" reference from the whole
    image -- including the face itself -- systematically washes real
    undertone signal toward Neutral whenever skin fills a large fraction of
    the frame, which is the common case for a close-up selfie. Using only
    background/hair/clothing pixels for the illumination estimate avoids
    that collapse.

    `strength` partially damps the correction (1.0 = full correction, 0.0 =
    none) because a frame dominated by one strongly saturated color
    (a bright shirt, a colored wall) can itself violate the gray-world
    assumption; a partial correction is safer than a full one when the
    scene content is unknown.
    """
    if exclude_mask is not None:
        keep = exclude_mask == 0
        if keep.sum() > 500:
            region_pixels = image_bgr[keep]
        else:
            region_pixels = image_bgr.reshape(-1, 3)
    else:
        region_pixels = image_bgr.reshape(-1, 3)

    region_pixels = region_pixels.astype(np.float32)
    means = region_pixels.mean(axis=0)  # B, G, R
    mean_gray = float(means.mean())
    eps = 1e-6
    raw_scale = mean_gray / (means + eps)
    scale = 1.0 + strength * (raw_scale - 1.0)
    return scale.astype(np.float32)


def apply_white_balance(image_bgr, scale):
    img = image_bgr.astype(np.float32)
    img[..., 0] *= scale[0]
    img[..., 1] *= scale[1]
    img[..., 2] *= scale[2]
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------- BODY SHAPE CLASSIFIER ----------------

def get_body_width_at_y(image, y_row, x_center, search_half_width=200):
    """
    Last-resort fallback: scans a horizontal row of Canny edges to find the
    leftmost/rightmost non-background pixel near the body center. Kept only
    for the case where segmentation is unavailable, since it's noisy (a single
    stray background edge pixel throws the whole measurement off).
    """
    h, w = image.shape[:2]
    y = int(np.clip(y_row, 0, h - 1))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 80)

    x_left = int(max(0, x_center - search_half_width))
    x_right = int(min(w, x_center + search_half_width))

    row_edges = edges[y, x_left:x_right]
    edge_positions = np.where(row_edges > 0)[0] + x_left

    if len(edge_positions) < 2:
        return None

    return int(edge_positions[-1] - edge_positions[0])


def get_body_mask_grabcut(image, pose_landmarks, h, w):
    """
    GrabCut seeded by the pose bounding box. Kept only as a secondary fallback
    for when SegFormer human-parsing is unavailable/fails -- it's slower and
    far more sensitive to background clutter than a real segmentation model.
    """
    lm = pose_landmarks.landmark

    xs = [int(l.x * w) for l in lm if l.visibility > 0.5]
    ys = [int(l.y * h) for l in lm if l.visibility > 0.5]

    if not xs or not ys:
        return None

    x1, x2 = max(0, min(xs) - 20), min(w, max(xs) + 20)
    y1, y2 = max(0, min(ys) - 20), min(h, max(ys) + 20)

    rect = (x1, y1, x2 - x1, y2 - y1)
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        return final_mask
    except Exception:
        return None


def get_silhouette_width_at_y(body_mask, y_row, x_bounds=None):
    """
    Given a binary (0/1 or 0/255) body mask, returns silhouette width at row
    y. When x_bounds=(min_x, max_x) is given, the row is clipped to that
    range before measuring -- used to exclude an arm hanging beside the
    torso from the waist/hip width when check_arms_occlude_torso() has
    flagged that risk (see classify_body_type_v2).
    """
    h, w = body_mask.shape
    y = int(np.clip(y_row, 0, h - 1))
    row = body_mask[y, :]
    if x_bounds is not None:
        x_min = int(np.clip(x_bounds[0], 0, w))
        x_max = int(np.clip(x_bounds[1], 0, w))
        if x_max <= x_min:
            return None
        row = row[x_min:x_max]
        cols = np.where(row > 0)[0]
        if len(cols) < 2:
            return None
        return int(cols[-1] - cols[0])
    cols = np.where(row > 0)[0]
    if len(cols) < 2:
        return None
    return int(cols[-1] - cols[0])


def get_band_median_width(mask, y_row, band=3, x_bounds=None):
    """
    Median silhouette width across a small vertical band of rows around
    y_row, instead of trusting a single row. A single row is vulnerable to
    one noisy/misclassified pixel; a median over a few neighboring rows is
    far more outlier-resistant.
    """
    if mask is None:
        return None
    h = mask.shape[0]
    widths = []
    for dy in range(-band, band + 1):
        y = y_row + dy
        if 0 <= y < h:
            width = get_silhouette_width_at_y(mask, y, x_bounds=x_bounds)
            if width is not None and width > 5:
                widths.append(width)
    if not widths:
        return None
    return float(np.median(widths))


def assess_pose_quality(pose_landmarks):
    """
    Checks whether enough of the body is actually visible/frontal to trust
    the measurements at all, instead of silently classifying whatever
    landmarks happen to be there.

    Returns (is_usable, warnings, visibility_avg, frontal_score).
    """
    lm = pose_landmarks.landmark
    required = [
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE,
    ]
    visibilities = [lm[p.value].visibility for p in required]
    visibility_avg = float(np.mean(visibilities))

    warnings = []
    is_usable = True
    if visibility_avg < 0.35:
        is_usable = False
        warnings.append(
            "Body isn't clearly visible enough to measure (cropped, occluded, or too far away). "
            "Use a full-length, front-facing photo with good lighting."
        )
    elif visibility_avg < 0.6:
        warnings.append("Some body landmarks are only partially visible; body-shape confidence is reduced.")

    ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    lh = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
    rh = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
    z_asymmetry = abs(ls.z - rs.z) + abs(lh.z - rh.z)
    frontal_score = float(np.clip(1.0 - z_asymmetry * 4.0, 0.0, 1.0))
    if frontal_score < 0.5:
        warnings.append("Body appears angled rather than facing the camera directly; measurements may be less accurate.")

    return is_usable, warnings, visibility_avg, frontal_score


def check_arms_occlude_torso(pose_landmarks, w, h, y_waist, y_hip):
    """
    Heuristic: if a wrist sits at roughly waist/hip height, that arm is
    probably hanging alongside the torso in the photo, which means the
    measured waist/hip width likely includes the arm rather than being pure
    torso silhouette. This can't be corrected without a limb-aware model, but
    it can and should be flagged instead of silently trusted.
    """
    lm = pose_landmarks.landmark
    band_top = min(y_waist, y_hip) - 20
    band_bottom = max(y_waist, y_hip) + 40
    for wrist_idx in (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST):
        wrist = lm[wrist_idx.value]
        if wrist.visibility < 0.4:
            continue
        wrist_y = wrist.y * h
        if band_top <= wrist_y <= band_bottom:
            return True
    return False


def estimate_mask_quality(person_mask, x1, y1, x2, y2):
    """
    Sanity-checks the segmentation mask against the pose bounding box: a real
    person silhouette should cover a moderate-to-high fraction of their own
    bounding box. Near-zero coverage means segmentation likely missed the
    person; this is used to down-weight confidence, not to hard-fail.
    """
    if person_mask is None:
        return 0.35
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    region = person_mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if region.size == 0:
        return 0.35
    coverage = float(region.mean())
    if coverage < 0.15:
        return 0.35
    return float(np.clip(0.5 + coverage * 0.5, 0.5, 1.0))


def classify_body_type_v2(image, pose_landmarks, gender="Female", person_mask=None):
    """
    Body type classifier using:
    1. Skeleton landmarks for Y positions
    2. SegFormer human-parsing silhouette for actual widths (GrabCut/edge-scan
       as fallback only)
    3. 5 body type classification, with gender-adjusted thresholds and a
       confidence score computed from measurement margin + pose/segmentation
       quality (not a fixed constant per category)

    NOTE ON ACCURACY: this remains a 2D, single-photo heuristic. Loose
    clothing will always visually widen whatever it covers -- no silhouette
    or edge based method can see through fabric to the body underneath.
    Getting true body measurements from clothed photos would require a
    body-mesh regression model (e.g. SMPL-based shape estimation), which is
    out of scope for this rule-based pipeline.
    """
    h, w, _ = image.shape
    lm = pose_landmarks.landmark

    def coord(idx):
        return int(lm[idx].x * w), int(lm[idx].y * h)

    warnings = []

    is_usable, pose_warnings, visibility_avg, frontal_score = assess_pose_quality(pose_landmarks)
    warnings.extend(pose_warnings)

    # ── Key Y-level positions from skeleton ──
    ls = coord(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    rs = coord(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    lh = coord(mp_pose.PoseLandmark.LEFT_HIP.value)
    rh = coord(mp_pose.PoseLandmark.RIGHT_HIP.value)
    lk = coord(mp_pose.PoseLandmark.LEFT_KNEE.value)
    rk = coord(mp_pose.PoseLandmark.RIGHT_KNEE.value)

    x_center = int((ls[0] + rs[0] + lh[0] + rh[0]) / 4)

    y_shoulder = int((ls[1] + rs[1]) / 2)
    y_hip      = int((lh[1] + rh[1]) / 2)
    y_knee     = int((lk[1] + rk[1]) / 2)
    body_height = y_hip - y_shoulder

    y_waist    = int(y_shoulder + 0.40 * body_height)
    y_high_hip = int(y_shoulder + 0.65 * body_height)
    y_bust     = int(y_shoulder + 0.20 * body_height)
    y_thigh    = int(y_hip + 0.15 * (y_knee - y_hip))

    arms_occlude = check_arms_occlude_torso(pose_landmarks, w, h, y_waist, y_hip)
    if arms_occlude:
        warnings.append(
            "Arms appear to be resting near the waist/hip in this photo, which can inflate those "
            "measurements; for best results stand with arms slightly away from the body."
        )

    # When arms are occluding the waist/hip band, an unbounded silhouette
    # scan measures arm-to-arm width rather than torso width (person_mask is
    # "not background" and doesn't distinguish arm pixels from torso pixels;
    # in a normal arms-at-sides pose the arm is mask-connected to the torso,
    # so there's no gap to exploit via connected components either). Bound
    # the scan to a torso-plausible x-range instead, derived from the
    # skeleton landmarks -- padded generously enough to still capture real
    # body curve (a genuine Pear's flared hips, an Apple's waist bulge)
    # while excluding most of a hanging forearm/hand's lateral reach.
    torso_x_bounds = None
    if arms_occlude:
        skeleton_shoulder_w = euclidean_distance(ls, rs)
        pad = 0.30 * skeleton_shoulder_w
        torso_x_bounds = (
            min(ls[0], rs[0], lh[0], rh[0]) - pad,
            max(ls[0], rs[0], lh[0], rh[0]) + pad,
        )

    # ── Get widths: SegFormer silhouette first, GrabCut then edge-scan as fallback ──
    grabcut_mask = None  # computed lazily, only if segmentation is unavailable

    def get_width(y_row, x_bounds=None):
        width = get_band_median_width(person_mask, y_row, x_bounds=x_bounds)
        if width is not None:
            return width
        nonlocal grabcut_mask
        if grabcut_mask is None:
            grabcut_mask = get_body_mask_grabcut(image, pose_landmarks, h, w)
        width = get_band_median_width(grabcut_mask, y_row, x_bounds=x_bounds)
        if width is not None:
            return width
        return get_body_width_at_y(image, y_row, x_center)

    W_shoulder = get_width(y_shoulder) or euclidean_distance(ls, rs)
    W_bust     = get_width(y_bust)     or W_shoulder
    W_waist    = get_width(y_waist, x_bounds=torso_x_bounds)    or (euclidean_distance(ls, rs) * 0.75)
    W_high_hip = get_width(y_high_hip, x_bounds=torso_x_bounds) or euclidean_distance(lh, rh)
    W_hip      = get_width(y_hip, x_bounds=torso_x_bounds)      or euclidean_distance(lh, rh)
    W_thigh    = get_width(y_thigh)    or (W_hip * 0.55)

    def norm(val):
        return val / W_shoulder if W_shoulder > 0 else 0

    n_bust     = norm(W_bust)
    n_waist    = norm(W_waist)
    n_high_hip = norm(W_high_hip)
    n_hip      = norm(W_hip)
    n_thigh    = norm(W_thigh)

    hip_to_shoulder  = W_hip / W_shoulder   if W_shoulder > 0 else 1
    waist_to_hip     = W_waist / W_hip      if W_hip > 0 else 1
    waist_to_shoulder= W_waist / W_shoulder if W_shoulder > 0 else 1
    bust_to_hip      = W_bust / W_hip       if W_hip > 0 else 1

    # ── Gender-adjusted thresholds ──
    # Heuristic, not clinically validated: average male waist-to-hip/shoulder
    # differential runs shallower than female, so the "defined waist" cutoffs
    # are relaxed slightly for a male-presenting body. Same 7-category system
    # either way -- only the numeric cutoffs shift.
    is_male = bool(gender) and gender.strip().lower().startswith("m")
    hourglass_waist_cut = 0.80 if is_male else 0.75
    pear_waist_cut       = 0.88 if is_male else 0.85
    apple_cut            = 0.90 if is_male else 0.85

    # ── 5-Type Classification with priority order (unchanged decision structure) ──
    body_type = "Rectangle"
    logic     = ""
    margin    = 0.4

    if (0.90 <= hip_to_shoulder <= 1.10
            and waist_to_hip < hourglass_waist_cut
            and waist_to_shoulder < hourglass_waist_cut):
        body_type = "Hourglass"
        logic = "Shoulder ≈ Hip, strongly defined waist"
        margin = min(
            1 - abs(hip_to_shoulder - 1.00) / 0.10,
            (hourglass_waist_cut - waist_to_hip) / hourglass_waist_cut,
            (hourglass_waist_cut - waist_to_shoulder) / hourglass_waist_cut,
        )

    elif hip_to_shoulder > 1.10 and waist_to_hip < pear_waist_cut:
        body_type = "Pear"
        logic = "Hips wider than shoulders"
        margin = min((hip_to_shoulder - 1.10) / 0.30, (pear_waist_cut - waist_to_hip) / pear_waist_cut)

    elif hip_to_shoulder < 0.90:
        body_type = "Inverted Triangle"
        logic = "Shoulders wider than hips"
        margin = (0.90 - hip_to_shoulder) / 0.90

    elif waist_to_hip >= apple_cut and waist_to_shoulder >= apple_cut:
        body_type = "Apple"
        logic = "Wide waist relative to hips and shoulders"
        margin = min((waist_to_hip - apple_cut) / (1.0 - apple_cut), (waist_to_shoulder - apple_cut) / (1.0 - apple_cut))

    else:
        body_type = "Rectangle"
        logic = "Measurements fairly uniform, minimal waist definition"
        margin = 0.4

    margin = float(np.clip(margin, 0.0, 1.0))

    x1, y1 = min(ls[0], rs[0], lh[0], rh[0]), y_shoulder
    x2, y2 = max(ls[0], rs[0], lh[0], rh[0]), y_hip
    mask_quality = estimate_mask_quality(person_mask, x1, y1, x2, y2)

    confidence = 0.45 + 0.40 * margin
    confidence *= (0.5 + 0.5 * visibility_avg)
    confidence *= (0.5 + 0.5 * frontal_score)
    confidence *= (0.6 + 0.4 * mask_quality)
    if arms_occlude:
        confidence *= 0.85
    confidence = float(np.clip(confidence, 0.05, 0.97))

    measurements = {
        "shoulder_px"     : round(W_shoulder, 1),
        "bust_px"         : round(W_bust, 1),
        "waist_px"        : round(W_waist, 1),
        "high_hip_px"     : round(W_high_hip, 1),
        "hip_px"          : round(W_hip, 1),
        "thigh_px"        : round(W_thigh, 1),
        "hip_to_shoulder" : round(hip_to_shoulder, 3),
        "waist_to_hip"    : round(waist_to_hip, 3),
        "waist_to_shoulder": round(waist_to_shoulder, 3),
        "bust_to_hip"     : round(bust_to_hip, 3),
    }

    return body_type, logic, confidence, measurements, warnings, is_usable


# ---------------- FACE SHAPE CLASSIFIER ----------------

# Generic 3D face model (arbitrary units) + matching MediaPipe FaceMesh
# landmark ids, used only for a coarse (uncalibrated) head-pose estimate to
# gate obviously non-frontal photos. This is the standard model used in
# common OpenCV head-pose tutorials -- it is NOT a per-user calibration and
# should not be read as a precise pose measurement.
_HEAD_POSE_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye, left corner
    (225.0, 170.0, -135.0),   # Right eye, right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)
_HEAD_POSE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


def estimate_head_pose(landmarks, w, h):
    """
    Coarse yaw/pitch/roll (degrees) via solvePnP with an approximated camera
    matrix (no real camera calibration available from a single uploaded
    photo). Good enough to flag/penalize clearly non-frontal faces; not
    precise enough to correct measurements.
    """
    if not all(idx in landmarks for idx in _HEAD_POSE_LANDMARK_IDS):
        return None
    image_points = np.array([landmarks[idx] for idx in _HEAD_POSE_LANDMARK_IDS], dtype=np.float64)
    focal_length = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, translation_vec = cv2.solvePnP(
        _HEAD_POSE_MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    proj_matrix = np.hstack((rotation_mat, translation_vec))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch, yaw, roll = [float(a[0]) for a in euler_angles]
    return {"yaw": yaw, "pitch": pitch, "roll": roll}


def detect_hairline_y(hair_mask, x_left, x_right, forehead_y, search_up=150):
    """
    Uses the SegFormer hair-class mask to find the real hairline within the
    forehead column, instead of assuming a fixed offset above landmark 10
    (which breaks for bangs, receding hairlines, buns, or short/no hair).
    Returns None (caller falls back to the fixed-offset heuristic) if no
    clear hair region is found in that column.
    """
    if hair_mask is None:
        return None
    h, mask_w = hair_mask.shape
    x_left = max(0, int(x_left))
    x_right = min(mask_w, int(x_right))
    if x_right <= x_left:
        return None
    y_top = max(0, int(forehead_y) - search_up)
    y_bottom = int(forehead_y)
    if y_bottom <= y_top:
        return None
    strip = hair_mask[y_top:y_bottom, x_left:x_right]
    if strip.size == 0:
        return None
    row_hair_frac = strip.mean(axis=1)
    hair_rows = np.where(row_hair_frac > 0.35)[0]
    if len(hair_rows) == 0:
        return None
    hairline_row = hair_rows.max()
    return float(y_top + hairline_row)


def assess_face_quality(landmarks, image, h):
    """Flags faces too small or too blurry in-frame to trust for shape geometry."""
    xs = [p[0] for p in landmarks.values()]
    ys = [p[1] for p in landmarks.values()]
    x1, x2 = max(0, min(xs)), max(xs)
    y1, y2 = max(0, min(ys)), max(ys)

    warnings = []
    quality = 1.0

    size_ratio = (y2 - y1) / h if h > 0 else 0
    if size_ratio < 0.15:
        quality *= 0.6
        warnings.append("Face is small relative to the photo; move closer for a more reliable face-shape result.")

    crop = image[int(y1):int(y2), int(x1):int(x2)]
    if crop.size > 0:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 30:
            quality *= 0.7
            warnings.append("Photo appears blurry; a sharper photo will improve accuracy.")

    return quality, warnings


def _crop_upscale_and_mesh(image, x1, y1, x2, y2, target_min_dim=400):
    """
    Crops the given box out of image, upscales it so its shorter side
    reaches target_min_dim, runs FaceMesh on just that crop, and remaps any
    resulting landmarks back into ORIGINAL image pixel coordinates. Returns
    None if the box is degenerate or no face is found in the crop.
    """
    h, w = image.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]
    if crop_h == 0 or crop_w == 0:
        return None

    scale = max(1.0, target_min_dim / min(crop_h, crop_w))
    crop_up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1.0 else crop
    crop_up_rgb = cv2.cvtColor(crop_up, cv2.COLOR_BGR2RGB)
    crop_up_h, crop_up_w = crop_up.shape[:2]

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        result = face_mesh.process(crop_up_rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0]
    lm_pixels = {}
    for idx, lm in enumerate(landmarks.landmark):
        orig_crop_px_x = (lm.x * crop_up_w) / scale
        orig_crop_px_y = (lm.y * crop_up_h) / scale
        lm_pixels[idx] = (int(orig_crop_px_x + x1), int(orig_crop_px_y + y1))
    return lm_pixels


def locate_face_landmarks(image, rgb, pose_results=None):
    """
    3-tier cascade for locating face landmarks reliably even when the face
    is small relative to the frame -- FaceMesh's internal face detector is
    hardcoded to a short-range model tuned for large-in-frame faces (no
    way to configure it otherwise), so it silently misses faces in exactly
    the full-body photos this app requires for body-type detection, even
    though a face is clearly, visibly present. Verified against real
    photos: FaceMesh alone failed on full-body shots where a face was
    plainly visible; each tier below recovered a different failure case.

    Returns lm_pixels (landmark index -> (x, y) in ORIGINAL image pixel
    coordinates) or None if no face could be located by any tier.
    """
    h, w = image.shape[:2]

    # Tier 1: direct FaceMesh on the full image -- cheapest, most precise,
    # covers any photo where the face is already large in frame (e.g. a
    # genuine close-up selfie).
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        result = face_mesh.process(rgb)
    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0]
        return {idx: (int(lm.x * w), int(lm.y * h)) for idx, lm in enumerate(landmarks.landmark)}

    # Tier 2: the standalone short-range FaceDetection model sometimes
    # finds a face on the full image that FaceMesh's own internal detector
    # missed on the identical image. If it does, crop generously around
    # that (precise) box, upscale, and retry FaceMesh on just the crop.
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:
        fd_result = face_detection.process(rgb)
    if fd_result.detections:
        bbox = fd_result.detections[0].location_data.relative_bounding_box
        bx1, by1 = bbox.xmin * w, bbox.ymin * h
        bw, bh = bbox.width * w, bbox.height * h
        pad_x, pad_y = bw * 0.6, bh * 0.6
        lm_pixels = _crop_upscale_and_mesh(image, bx1 - pad_x, by1 - pad_y, bx1 + bw + pad_x, by1 + bh + pad_y)
        if lm_pixels is not None:
            return lm_pixels

    # Tier 3: fall back to mp_pose's nose landmark as a coarse head-location
    # guess, for faces too small for even the dedicated detector to find
    # (e.g. a small figure in a large, padded full-body photo). Reuses
    # pose_results if the caller already computed it (as /analyze does);
    # otherwise runs pose fresh, only on this fallback path.
    if pose_results is None:
        with mp_pose.Pose(static_image_mode=True) as pose:
            pose_results = pose.process(rgb)
    if pose_results and pose_results.pose_landmarks:
        nose = pose_results.pose_landmarks.landmark[0]
        nx, ny = nose.x * w, nose.y * h
        half = h * 0.12
        lm_pixels = _crop_upscale_and_mesh(image, nx - half, ny - half, nx + half, ny + half)
        if lm_pixels is not None:
            return lm_pixels

    return None


def classify_face_shape(landmarks, image, hair_mask=None):
    """
    Face shape classifier using MediaPipe landmark indices, a segmentation-
    derived hairline (falls back to a fixed heuristic), head-pose gating, and
    a confidence computed from measurement margin + pose/image quality
    (not a fixed constant per category).
    """
    h, w, _ = image.shape

    LANDMARKS = {
        'chin'          : 152,
        'forehead_mid'  : 10,
        'left_cheek'    : 123,
        'right_cheek'   : 352,
        'left_jaw'      : 172,
        'right_jaw'     : 397,
        'left_temple'   : 70,
        'right_temple'  : 300,
        'left_jawangle' : 234,
        'right_jawangle': 454,
        'nose_tip'      : 4,
    }

    required = list(LANDMARKS.values())
    for idx in required:
        if idx not in landmarks:
            return "Unknown", 0.0, ["Face landmarks incomplete."]

    def pt(name):
        return np.array(landmarks[LANDMARKS[name]], dtype=float)

    chin          = pt('chin')
    forehead_mid  = pt('forehead_mid')
    left_cheek    = pt('left_cheek')
    right_cheek   = pt('right_cheek')
    left_jaw      = pt('left_jaw')
    right_jaw     = pt('right_jaw')
    left_temple   = pt('left_temple')
    right_temple  = pt('right_temple')

    warnings = []

    # ── Head pose gating ──
    pose_factor = 0.8  # default (moderate) when pose can't be estimated at all
    head_pose = estimate_head_pose(landmarks, w, h)
    if head_pose is not None:
        yaw, pitch = head_pose["yaw"], head_pose["pitch"]
        angle_off = max(abs(yaw), abs(pitch))
        if angle_off > 25:
            pose_factor = 0.5
            warnings.append("Face is significantly angled relative to the camera; face shape result is unreliable.")
        elif angle_off > 12:
            pose_factor = 0.75
            warnings.append("Face is slightly angled relative to the camera; face shape confidence is reduced.")
        else:
            pose_factor = 1.0

    quality_factor, quality_warnings = assess_face_quality(landmarks, image, h)
    warnings.extend(quality_warnings)

    # ── Face length via segmentation-detected hairline (fallback: fixed 12% heuristic) ──
    raw_face_height = abs(chin[1] - forehead_mid[1])
    x_left_temple, x_right_temple = left_temple[0], right_temple[0]
    hairline_y = detect_hairline_y(hair_mask, min(x_left_temple, x_right_temple),
                                    max(x_left_temple, x_right_temple), forehead_mid[1])
    if hairline_y is None:
        hairline_y = forehead_mid[1] - (raw_face_height * 0.12)
    hairline_pt = np.array([forehead_mid[0], hairline_y])
    face_length = np.linalg.norm(chin - hairline_pt)

    cheek_width     = np.linalg.norm(left_cheek    - right_cheek)
    jaw_width       = np.linalg.norm(left_jaw      - right_jaw)
    forehead_width  = np.linalg.norm(left_temple   - right_temple)

    if cheek_width < 1 or jaw_width < 1 or forehead_width < 1:
        return "Unknown", 0.0, warnings + ["Face measurements degenerate (near-zero width); check image quality."]

    len_to_cheek     = face_length   / cheek_width
    jaw_to_forehead  = jaw_width     / forehead_width
    cheek_to_jaw     = cheek_width   / jaw_width
    jaw_to_cheek     = jaw_width     / cheek_width
    cheek_to_forehead= cheek_width   / forehead_width

    margin = 0.4
    if len_to_cheek > 1.50 and abs(jaw_to_forehead - 1.0) < 0.15:
        face_shape = "Oblong"
        margin = min((len_to_cheek - 1.50) / 0.30, (0.15 - abs(jaw_to_forehead - 1.0)) / 0.15)

    elif len_to_cheek > 1.35 and jaw_to_forehead < 0.80:
        face_shape = "Heart"
        margin = min((len_to_cheek - 1.35) / 0.30, (0.80 - jaw_to_forehead) / 0.80)

    elif len_to_cheek < 1.10 and cheek_to_jaw < 1.10:
        face_shape = "Round"
        margin = min((1.10 - len_to_cheek) / 1.10, (1.10 - cheek_to_jaw) / 1.10)

    elif 1.10 <= len_to_cheek <= 1.30 and jaw_to_cheek > 0.85:
        face_shape = "Square"
        margin = min(1 - abs(len_to_cheek - 1.20) / 0.10, (jaw_to_cheek - 0.85) / 0.85)

    elif cheek_to_jaw > 1.20 and cheek_to_forehead > 1.20:
        face_shape = "Diamond"
        margin = min((cheek_to_jaw - 1.20) / 0.50, (cheek_to_forehead - 1.20) / 0.50)

    else:
        face_shape = "Oval"
        margin = 0.4

    margin = float(np.clip(margin, 0.0, 1.0))

    confidence = 0.45 + 0.40 * margin
    confidence *= pose_factor
    confidence *= quality_factor
    confidence = float(np.clip(confidence, 0.05, 0.95))

    return face_shape, confidence, warnings


# ---------------- SKIN TONE DETECTOR ----------------
def classify_skin_tone(image, landmarks, skin_mask=None):
    """
    Skin tone/undertone classifier sampling multiple facial regions.
    Uses (when available) the SegFormer skin-class mask intersected with each
    landmark ROI, so hair/eyebrows/shadow-edges/glasses inside a patch don't
    get averaged into the skin color. Applies gray-world white balance first
    to reduce (not eliminate) lighting color-cast bias. Takes the MEDIAN
    across regions so one shadowed/blemished patch can't dominate, and skips
    any region without enough valid skin pixels rather than guessing.

    NOTE ON ACCURACY: without a physical color-reference card in frame, LAB
    values from an uncalibrated phone photo are only relatively meaningful --
    ambient lighting still legitimately shifts the same person's reading
    across different photos. This mitigates but does not remove that limit.
    """
    h, w, _ = image.shape
    # 103/332 (upper-forehead, left/right) were added and verified against a
    # bearded/angled test photo this session: they kept surviving when the
    # cheek/chin points got correctly excluded by beard detection, giving
    # more redundancy against both facial-hair occlusion and shadow bias.
    target_landmarks = [234, 454, 10, 152, 6, 103, 332]
    patch_half = 15

    face_region_mask = build_face_region_mask(landmarks, h, w)
    exclude_mask = face_region_mask
    if skin_mask is not None:
        exclude_mask = np.clip(skin_mask.astype(np.uint8) + face_region_mask, 0, 1)
    wb_scale = compute_gray_world_scale(image, exclude_mask=exclude_mask, strength=0.6)
    image_wb = apply_white_balance(image, wb_scale)
    lab_image = cv2.cvtColor(image_wb, cv2.COLOR_BGR2LAB).astype(np.float32)

    patch_lab_means = []

    for idx in target_landmarks:
        if idx not in landmarks:
            continue
        px, py = landmarks[idx]
        x1, y1 = max(0, px - patch_half), max(0, py - patch_half)
        x2, y2 = min(w, px + patch_half), min(h, py + patch_half)
        if x2 <= x1 or y2 <= y1:
            continue

        lab_patch = lab_image[y1:y2, x1:x2].reshape(-1, 3)
        if lab_patch.shape[0] == 0:
            continue

        if skin_mask is not None:
            mask_patch = skin_mask[y1:y2, x1:x2].reshape(-1)
            valid = lab_patch[mask_patch > 0]
        else:
            valid = lab_patch

        min_required = max(10, int(0.25 * lab_patch.shape[0]))
        if valid.shape[0] < min_required:
            continue  # likely occluded (hair, shadow, glasses) or segmentation missed this spot

        patch_lab_means.append(valid.mean(axis=0))

    if len(patch_lab_means) < 2:
        return "Unknown", "Unknown", 0.0, [
            "Could not find enough clear, unobstructed patches of facial skin (facial hair, "
            "an extreme angle, or occlusion) -- try a well-lit, front-facing photo."
        ]

    patch_lab_means = np.array(patch_lab_means)

    # Lightness uses the brightest valid sample (capped, to guard against an
    # overexposed/blown-out pixel being mistaken for skin) rather than the
    # median: verified against a real photo where beard detection correctly
    # excluded most sample points, leaving very few, and the face was angled
    # toward a light source -- one remaining point landed on the shadowed
    # side and read L*=32 while another read L*=62 from the same forehead.
    # A shadowed patch can only read darker than true skin, never lighter,
    # so the brightest genuine skin sample is the more reliable estimate
    # under directional/uneven lighting. Hue/undertone (a*, b*) stay on the
    # median across samples -- shadowing affects lightness far more than hue.
    L_CAP_OPENCV = 92 * 255.0 / 100.0  # ~234.6, guards against blown-out pixels
    l_opencv = float(min(np.max(patch_lab_means[:, 0]), L_CAP_OPENCV))
    a_opencv, b_opencv = np.median(patch_lab_means[:, 1:], axis=0)

    l_true = (l_opencv * 100.0) / 255.0
    a_true = a_opencv - 128.0
    b_true = b_opencv - 128.0

    if l_true > 70:
        skin_tone = "Fair"
    elif 50 < l_true <= 70:
        skin_tone = "Medium"
    else:
        skin_tone = "Dark"

    threshold = 2.0
    if (b_true - a_true) > threshold:
        undertone = "Warm"
    elif (a_true - b_true) > threshold:
        undertone = "Cool"
    else:
        undertone = "Neutral"

    coverage_ratio = len(patch_lab_means) / len(target_landmarks)
    spread = float(np.mean(np.std(patch_lab_means, axis=0))) if len(patch_lab_means) > 1 else 0.0
    consistency = float(np.clip(1.0 - spread / 20.0, 0.2, 1.0))
    confidence = float(np.clip(0.35 + 0.35 * coverage_ratio + 0.30 * consistency, 0.05, 0.95))

    warnings = []
    if coverage_ratio < 0.6:
        warnings.append("Several skin-sample regions were occluded or unclear; result may be less reliable.")

    return skin_tone, undertone, confidence, warnings

# ---------------- API ----------------
@app.route("/")
def home():
    return "Backend is running successfully"


@app.route("/analyze", methods=["POST","GET"])
def analyze_image():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    gender = request.form.get("gender", "Female")
    lat = request.form.get("lat", type=float)
    lon = request.form.get("lon", type=float)

    image_file = request.files["image"]
    filename = secure_filename(image_file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image_file.save(filepath)

    image = load_image_exif_safe(filepath)
    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    quality_warnings = []
    exposure_factor, exposure_warning = assess_image_exposure(image)
    if exposure_warning:
        quality_warnings.append(exposure_warning)

    # ── Run SegFormer human parsing ONCE and reuse it for body silhouette,
    #    skin sampling, hairline detection, and outfit masking. ──
    try:
        label_map, confidence_map = parse_human(image)
        person_mask = get_person_mask(label_map)
        skin_mask = get_skin_mask(label_map)
        hair_mask = get_hair_mask(label_map)
    except Exception:
        label_map = None
        confidence_map = None
        person_mask = None
        skin_mask = None
        hair_mask = None
        quality_warnings.append("Segmentation model unavailable for this request; falling back to less accurate edge-based measurements.")

    # =====================================================
    # 🔵 PART 1: BODY TYPE DETECTION
    # =====================================================
    with mp_pose.Pose(static_image_mode=True) as pose:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if not results.pose_landmarks:
            return jsonify({"error": "No person detected"}), 400

        body_type, logic_used, confidence, measurements, body_warnings, body_usable = classify_body_type_v2(
            image,
            results.pose_landmarks,
            gender,
            person_mask=person_mask,
        )
        quality_warnings.extend(body_warnings)

        if not body_usable:
            return jsonify({
                "error": "Body not clearly visible enough to analyze",
                "details": body_warnings,
            }), 400

    # =====================================================
    # 🟣 PART 2: FACE & SKIN ANALYSIS
    # =====================================================
    face_shape = "Unknown"
    face_confidence = 0.0
    skin_tone = "Unknown"
    undertone = "Unknown"
    skin_confidence = 0.0

    lm_pixels = locate_face_landmarks(image, rgb, pose_results=results)

    if lm_pixels is not None:
        face_shape, face_confidence, face_warnings = classify_face_shape(lm_pixels, image, hair_mask=hair_mask)
        skin_tone, undertone, skin_confidence, skin_warnings = classify_skin_tone(image, lm_pixels, skin_mask=skin_mask)

        face_confidence = float(np.clip(face_confidence * exposure_factor, 0.05, 0.95))
        skin_confidence = float(np.clip(skin_confidence * exposure_factor, 0.05, 0.95))

        quality_warnings.extend(face_warnings)
        quality_warnings.extend(skin_warnings)
    else:
        quality_warnings.append("No face detected; face shape and skin tone are unavailable for this photo.")

    # =====================================================
    # 🟢 PART 3: OUTFIT DETECTION (SegFormer garment classes)
    # =====================================================
    outfits = detect_outfits(image, label_map=label_map, confidence_map=confidence_map)
    person_rgba = outfits.pop("person_rgba", None)
    outfits = enrich_outfits_with_attributes(outfits)

    # =====================================================
    # 🟣 PART 3: SAVE DETECTED CLOTHES TO CLOSET
    # =====================================================
    user_id = request.form.get("user_id", "guest_user")
    added_count, duplicate_count = add_items_to_closet(user_id, outfits, gender=gender)

    # person_rgba has no persisted row anywhere -- it's only ever used for
    # this response, so it uploads to the auto-expiring temp_scans/ prefix
    # rather than embedding ~1-2MB of base64 directly in the JSON (the
    # actual cause of "Response size was too large" / dropped mobile
    # connections on /analyze).
    person_rgba_url = None
    if person_rgba:
        person_rgba_url = upload_temp_scan(user_id, person_rgba, content_type="image/png")

    message = ""
    if added_count > 0:
        message = f"Added {added_count} new item(s) to closet."
        if duplicate_count > 0:
            message += f" ({duplicate_count} duplicates skipped)."
    elif duplicate_count > 0:
        message = f"No new items added. ({duplicate_count} already in closet)."
    else:
        message = "No valid clothing items detected to save."

    # =====================================================
    # 🟣 PART 4: GENERATE STYLING RECOMMENDATIONS
    # =====================================================
    detected_colors = []
    for category_items in outfits.values():
        for item in category_items:
            if "color_name" in item:
                detected_colors.append(item["color_name"])

    styling_recommendations = get_styling_recommendations(
        body_type=body_type,
        face_shape=face_shape,
        skin_tone=skin_tone,
        undertone=undertone,
        outfits=outfits,
        gender=gender
    )

    weather = get_current_weather(lat, lon)

    profile = {
        "body_type": body_type,
        "face_shape": face_shape,
        "skin_tone": skin_tone,
        "undertone": undertone,
        "gender": gender,
        "weather": weather,
    }

    current_outfit_items = [
        item for category_items in outfits.values() if isinstance(category_items, list) for item in category_items
    ]
    wardrobe_items = get_user_closet(user_id)
    # Two separate, unrelated catalog sources: the original 7-item "Shop the
    # Look" seed data (real affiliate links, source_ref-based) and the
    # richer shirts/jeans/tops tables (body-shape/undertone scoring data,
    # image_b64-based) -- both are valid look candidates, so both go in.
    catalog_items = get_catalog_items(gender=gender) + get_styling_catalog_items(gender=gender)
    recommendation = generate_three_looks(profile, current_outfit_items, wardrobe_items, catalog_items, request_id=filename)

    # New Outfit Suggestions: a separate, display-only set of 3 outfit cards
    # (not read by Virtual Try-On) -- see recommendation_engine.py for why
    # this is intentionally independent of the `recommendation` looks above.
    new_outfit_suggestions = generate_new_outfit_suggestions(
        profile, current_outfit_items, wardrobe_items, catalog_items, request_id=filename
    )

    # ---------------- FINAL RESPONSE ----------------
    return jsonify({
        "body_type": body_type,
        "face_shape": face_shape,
        "face_shape_confidence": face_confidence,
        "skin_tone": skin_tone,
        "undertone": undertone,
        "skin_confidence": skin_confidence,
        "gender": gender,
        "logic_used": logic_used,
        "confidence_score": confidence,
        "measurements": measurements,
        "quality_warnings": quality_warnings,
        "outfits": outfits,
        "person_rgba_url": person_rgba_url,
        "styling_recommendations": styling_recommendations,
        "recommended_looks": recommendation["looks"],
        "styling": recommendation["styling"],
        "outfit_suggestions": new_outfit_suggestions["looks"],
        "closet_info": {
            "saved": added_count > 0,
            "added_count": added_count,
            "duplicate_count": duplicate_count,
            "message": message
        }
    })


@app.route("/analyze-face", methods=["POST"])
def analyze_face_only():
    """
    Lean retry endpoint for when /analyze couldn't get a reliable face shape
    or skin tone from the main photo (no face detected, bad angle, poor
    lighting, occlusion). Takes a second, dedicated selfie and runs ONLY
    face-mesh + face-shape + skin-tone -- no body pose, outfit detection,
    closet saving, or recommendations, since the retry photo may not be a
    full-body shot at all. Reuses the exact same helpers/functions as
    /analyze so this can't drift out of sync with the main flow's logic.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    filename = secure_filename(image_file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image_file.save(filepath)

    image = load_image_exif_safe(filepath)
    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    quality_warnings = []
    exposure_factor, exposure_warning = assess_image_exposure(image)
    if exposure_warning:
        quality_warnings.append(exposure_warning)

    try:
        label_map, confidence_map = parse_human(image)
        skin_mask = get_skin_mask(label_map)
        hair_mask = get_hair_mask(label_map)
    except Exception:
        skin_mask = None
        hair_mask = None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lm_pixels = locate_face_landmarks(image, rgb)

    if lm_pixels is None:
        quality_warnings.append(
            "No face detected in this photo either -- try a well-lit, front-facing selfie "
            "without sunglasses, a hat, or heavy shadow."
        )
        return jsonify({
            "face_shape": "Unknown",
            "face_shape_confidence": 0.0,
            "skin_tone": "Unknown",
            "undertone": "Unknown",
            "skin_confidence": 0.0,
            "quality_warnings": quality_warnings,
        })

    face_shape, face_confidence, face_warnings = classify_face_shape(lm_pixels, image, hair_mask=hair_mask)
    skin_tone, undertone, skin_confidence, skin_warnings = classify_skin_tone(image, lm_pixels, skin_mask=skin_mask)

    face_confidence = float(np.clip(face_confidence * exposure_factor, 0.05, 0.95))
    skin_confidence = float(np.clip(skin_confidence * exposure_factor, 0.05, 0.95))

    quality_warnings.extend(face_warnings)
    quality_warnings.extend(skin_warnings)

    return jsonify({
        "face_shape": face_shape,
        "face_shape_confidence": face_confidence,
        "skin_tone": skin_tone,
        "undertone": undertone,
        "skin_confidence": skin_confidence,
        "quality_warnings": quality_warnings,
    })


@app.route("/closet/<user_id>", methods=["GET"])

def get_closet(user_id):
    items = get_user_closet(user_id)
    return jsonify({"user_id": user_id, "closet": items})


@app.route("/closet/<user_id>/resolve-duplicate", methods=["POST"])
def resolve_closet_duplicate(user_id):
    data = request.json or {}
    image_hash = data.get("image_hash")
    matched_image_hash = data.get("matched_image_hash")
    action = data.get("action")
    brand = data.get("brand")

    if not image_hash or not matched_image_hash or action not in ("keep_both", "keep_this", "keep_other"):
        return jsonify({"error": "image_hash, matched_image_hash, and a valid action are required"}), 400

    result = resolve_duplicate(user_id, image_hash, matched_image_hash, action, brand=brand)
    return jsonify({"success": True, "result": result})


@app.route("/profile/<user_id>", methods=["GET"])
def get_ai_profile(user_id):
    """Persisted body shape/skin tone/face shape from a prior /analyze call,
    so the Profile UI survives a page refresh instead of showing fake
    defaults."""
    profile = get_profile(user_id)
    return jsonify({"user_id": user_id, "profile": profile})


@app.route("/profile/<user_id>", methods=["POST"])
def save_ai_profile(user_id):
    profile_data = request.json or {}
    saved = save_profile(user_id, profile_data)
    return jsonify({"success": True, "user_id": user_id, "profile": saved})


@app.route("/saved-looks/<user_id>", methods=["GET"])
def list_saved_looks(user_id):
    return jsonify({"user_id": user_id, "looks": get_saved_looks(user_id)})


@app.route("/saved-looks/<user_id>", methods=["POST"])
def create_saved_look(user_id):
    data = request.json or {}
    image_b64 = data.get("image_b64")
    label = data.get("label")

    if not image_b64:
        return jsonify({"error": "image_b64 is required"}), 400

    try:
        entry = save_look(user_id, image_b64, label)
    except Exception as e:
        return jsonify({"error": f"Failed to save look: {e}"}), 400

    return jsonify({"success": True, "look": entry})


@app.route("/saved-looks/<user_id>/<look_id>", methods=["DELETE"])
def remove_saved_look(user_id, look_id):
    deleted = delete_saved_look(user_id, look_id)
    if not deleted:
        return jsonify({"error": "Look not found"}), 404
    return jsonify({"success": True})


@app.route("/admin/migrate-catalog-images", methods=["POST"])
def admin_migrate_catalog_images():
    """
    One-off: uploads catalog/*.jpg local files to GCS and writes image_url
    back onto their Supabase rows (see migrate_catalog_to_gcs.py). Gated on
    a header matching SUPABASE_SECRET_KEY -- an existing backend-only
    secret already never shipped to the frontend, reused here rather than
    introducing a whole auth system for a single internal one-time action.
    Safe to call more than once; already-migrated rows are skipped.
    """
    admin_key = request.headers.get("X-Admin-Key")
    if not admin_key or admin_key != os.environ.get("SUPABASE_SECRET_KEY"):
        return jsonify({"error": "Unauthorized"}), 403

    from migrate_catalog_to_gcs import migrate_all_catalog_images
    results = migrate_all_catalog_images()
    return jsonify({"results": results})


@app.route("/weather", methods=["GET"])
def weather():
    """
    Standalone weather lookup for frontend widgets (e.g. the homepage hero
    banner) that need a current temperature without running a full /analyze
    image upload. Thin wrapper around weather_service.get_current_weather,
    which already handles missing coordinates/API key/network failure by
    falling back to a default reading -- never raises.
    """
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    return jsonify(get_current_weather(lat, lon))



# ──────────────────────────────────────────────────────────────────
# ROUTE 1: Single garment try-on
# POST /tryon
# ──────────────────────────────────────────────────────────────────

@app.route("/tryon", methods=["POST", "OPTIONS"])
def tryon_single():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if "person_image" not in request.files:
        return jsonify({"error": "person_image is required"}), 400
    if "garment_image" not in request.files:
        return jsonify({"error": "garment_image is required"}), 400

    garment_type = request.form.get("garment_type", "upper")

    person_file = request.files["person_image"]
    person_bytes = person_file.read()
    person_b64 = base64.b64encode(person_bytes).decode("utf-8")

    garment_file = request.files["garment_image"]
    garment_bytes = garment_file.read()
    garment_b64 = base64.b64encode(garment_bytes).decode("utf-8")

    result = generate_tryon(person_b64, garment_b64, garment_type)

    if result["success"]:
        return jsonify({
            "success": True,
            "result_image": result["image_b64"],
            "model_used": result["model_used"],
            "garment_type": garment_type
        })
    else:
        return jsonify({
            "success": False,
            "fallback_image_b64": result.get("fallback_image_b64"),
            "error": result.get("error", "Virtual try-on model busy."),
            "fallback": True,
            "model_used": result.get("model_used", "Fallback")
        }), 200


# ──────────────────────────────────────────────────────────────────
# ROUTE 2: Generate all 3 looks at once (Look A, B, C)
# POST /tryon/all-looks
# ──────────────────────────────────────────────────────────────────

@app.route("/tryon/all-looks", methods=["POST", "OPTIONS"])
def tryon_all_looks():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if "person_image" not in request.files:
        return jsonify({"error": "person_image is required"}), 400

    person_file = request.files["person_image"]
    person_bytes = person_file.read()
    person_b64 = base64.b64encode(person_bytes).decode("utf-8")

    top_b64 = None
    bottom_b64 = None

    if "top_garment" in request.files:
        top_bytes = request.files["top_garment"].read()
        top_b64 = base64.b64encode(top_bytes).decode("utf-8")

    if "bottom_garment" in request.files:
        bottom_bytes = request.files["bottom_garment"].read()
        bottom_b64 = base64.b64encode(bottom_bytes).decode("utf-8")

    results = {}

    if bottom_b64:
        look_a = generate_tryon(person_b64, bottom_b64, "lower")
        results["look_a"] = {
            "success": look_a["success"],
            "image": look_a.get("image_b64"),
            "fallback_image_b64": look_a.get("fallback_image_b64"),
            "label": "Your Top + New Bottom",
            "model_used": look_a.get("model_used")
        }
    else:
        results["look_a"] = {"success": False, "image": None, "label": "Your Top + New Bottom"}

    if top_b64:
        look_b = generate_tryon(person_b64, top_b64, "upper")
        results["look_b"] = {
            "success": look_b["success"],
            "image": look_b.get("image_b64"),
            "fallback_image_b64": look_b.get("fallback_image_b64"),
            "label": "New Top + Your Bottom",
            "model_used": look_b.get("model_used")
        }
    else:
        results["look_b"] = {"success": False, "image": None, "label": "New Top + Your Bottom"}

    if top_b64 and bottom_b64:
        step1 = generate_tryon(person_b64, top_b64, "upper")
        step1_img = step1.get("image_b64") or step1.get("fallback_image_b64")
        if step1_img:
            step2 = generate_tryon(step1_img, bottom_b64, "lower")
            results["look_c"] = {
                "success": step2["success"],
                "image": step2.get("image_b64"),
                "fallback_image_b64": step2.get("fallback_image_b64") or step1.get("fallback_image_b64"),
                "label": "Full New Outfit",
                "model_used": f"{step1.get('model_used')} + {step2.get('model_used')}"
            }
        else:
            results["look_c"] = {"success": False, "image": None, "label": "Full New Outfit"}
    else:
        results["look_c"] = {"success": False, "image": None, "label": "Full New Outfit"}

    return jsonify({
        "success": True,
        "looks": results
    })

# ──────────────────────────────────────────────────────────────────
# ROUTE 3: Resolve (or create) the canonical user_id for an email, then
# merge the guest session's closet into that account.
# POST /auth/session
#
# Called by both signup and login so the two flows always resolve the same
# email to the same user_id -- previously signup minted a random UUID while
# login derived an id from the email prefix, silently orphaning data on the
# very common signup -> logout -> login path.
# ──────────────────────────────────────────────────────────────────

@app.route("/auth/session", methods=["POST", "OPTIONS"])
def auth_session():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.json or {}
    guest_id = data.get("guest_id")
    email = data.get("email")
    name = data.get("name")
    password = data.get("password")

    if not guest_id or not email:
        return jsonify({"error": "guest_id and email are required"}), 400

    try:
        user_id, resolved_name, is_new_user = resolve_user_id(email, password, name)
    except InvalidCredentialsError as e:
        return jsonify({"error": str(e)}), 401
    except (PasswordRequiredError, PasswordTooShortError) as e:
        return jsonify({"error": str(e)}), 400

    migrated_count = migrate_closet_items(guest_id, user_id)
    migrate_profile(guest_id, user_id)
    migrate_saved_looks(guest_id, user_id)

    return jsonify({
        "success": True,
        "user_id": user_id,
        "name": resolved_name,
        "email": email,
        "is_new_user": is_new_user,
        "migrated_items": migrated_count,
        "message": f"Successfully migrated {migrated_count} items from guest session to user account."
    })

if __name__ == "__main__":
    # Cloud Run (and most PaaS hosts) inject PORT and expect the app to bind
    # to it; FLASK_DEBUG opts into the interactive debugger for local dev
    # only -- leaving debug=True on by default is a known RCE risk once a
    # server is reachable from the internet.
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", debug=debug_mode, port=port, threaded=True)
