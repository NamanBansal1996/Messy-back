from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import os
import base64
import mediapipe as mp
import math
from werkzeug.utils import secure_filename
import numpy as np

# 🔹 Import YOLO outfit detection
from yolo_outfit_detect import detect_outfits
from closet_manager import add_items_to_closet, get_user_closet
from styling_rules import get_styling_recommendations
from virtual_tryon import generate_tryon
from recommendation_engine import generate_three_looks
from catalog import get_catalog_items
from weather_service import get_current_weather

# ---------------- APP SETUP ----------------
app = Flask(__name__)
# CORS(app, resources={r"/analyze": {"origins": "*"}})
CORS(
    app,
    resources={r"/*": {"origins": "*"}}
)
UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# ---------------- UTILS ----------------
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# ---------------- BODY SHAPE CLASSIFIER ----------------

def get_body_width_at_y(image, y_row, x_center, search_half_width=200):
    """
    Scans a horizontal row to find the leftmost and rightmost
    non-background pixel near the body center.
    Returns width in pixels.
    """
    h, w = image.shape[:2]
    y = int(np.clip(y_row, 0, h - 1))

    # Convert to grayscale and get the row
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use Canny edges on full image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 80)

    # Search within bounds around center
    x_left = int(max(0, x_center - search_half_width))
    x_right = int(min(w, x_center + search_half_width))

    row_edges = edges[y, x_left:x_right]
    edge_positions = np.where(row_edges > 0)[0] + x_left

    if len(edge_positions) < 2:
        # Fallback: use fixed shoulder/hip skeleton width ± 10%
        return None

    return int(edge_positions[-1] - edge_positions[0])

def get_body_mask(image, pose_landmarks, h, w):
    """
    Uses GrabCut seeded by pose bounding box to isolate body silhouette.
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

def get_silhouette_width_at_y(body_mask, y_row):
    """
    Given a binary body mask, returns actual silhouette width at row y.
    """
    h, w = body_mask.shape
    y = int(np.clip(y_row, 0, h - 1))
    row = body_mask[y, :]
    cols = np.where(row > 0)[0]
    if len(cols) < 2:
        return None
    return int(cols[-1] - cols[0])

def classify_body_type_v2(image, pose_landmarks, gender="Female"):
    """
    Accurate body type classifier using:
    1. Skeleton landmarks for Y positions
    2. Silhouette scanning for actual widths
    3. 7 body type classification
    """
    h, w, _ = image.shape
    lm = pose_landmarks.landmark

    def coord(idx):
        return int(lm[idx].x * w), int(lm[idx].y * h)

    # ── Key Y-level positions from skeleton ──
    ls = coord(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    rs = coord(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    lh = coord(mp_pose.PoseLandmark.LEFT_HIP.value)
    rh = coord(mp_pose.PoseLandmark.RIGHT_HIP.value)
    lk = coord(mp_pose.PoseLandmark.LEFT_KNEE.value)
    rk = coord(mp_pose.PoseLandmark.RIGHT_KNEE.value)

    # Body center X
    x_center = int((ls[0] + rs[0] + lh[0] + rh[0]) / 4)

    # Y levels
    y_shoulder = int((ls[1] + rs[1]) / 2)
    y_hip      = int((lh[1] + rh[1]) / 2)
    y_knee     = int((lk[1] + rk[1]) / 2)
    body_height = y_hip - y_shoulder

    # Waist is ~38-42% down from shoulder to hip
    y_waist    = int(y_shoulder + 0.40 * body_height)
    # High hip is ~65% down from shoulder to hip
    y_high_hip = int(y_shoulder + 0.65 * body_height)
    # Bust is ~20% down from shoulder to hip
    y_bust     = int(y_shoulder + 0.20 * body_height)
    # Thigh just below hip
    y_thigh    = int(y_hip + 0.15 * (y_knee - y_hip))

    # ── Get body mask via GrabCut ──
    body_mask = get_body_mask(image, pose_landmarks, h, w)

    def get_width(y_row):
        """Try silhouette mask first, fallback to edge scan."""
        width = None
        if body_mask is not None:
            width = get_silhouette_width_at_y(body_mask, y_row)
        if width is None or width < 10:
            width = get_body_width_at_y(image, y_row, x_center)
        return width

    # ── Measure widths at key levels ──
    W_shoulder = get_width(y_shoulder) or euclidean_distance(ls, rs)
    W_bust     = get_width(y_bust)     or W_shoulder
    W_waist    = get_width(y_waist)    or (euclidean_distance(ls, rs) * 0.75)
    W_high_hip = get_width(y_high_hip) or euclidean_distance(lh, rh)
    W_hip      = get_width(y_hip)      or euclidean_distance(lh, rh)
    W_thigh    = get_width(y_thigh)    or (W_hip * 0.55)

    # ── Normalize by shoulder width ──
    def norm(val):
        return val / W_shoulder if W_shoulder > 0 else 0

    n_bust     = norm(W_bust)
    n_waist    = norm(W_waist)
    n_high_hip = norm(W_high_hip)
    n_hip      = norm(W_hip)
    n_thigh    = norm(W_thigh)

    # ── Ratios ──
    hip_to_shoulder  = W_hip / W_shoulder   if W_shoulder > 0 else 1
    waist_to_hip     = W_waist / W_hip      if W_hip > 0 else 1
    waist_to_shoulder= W_waist / W_shoulder if W_shoulder > 0 else 1
    bust_to_hip      = W_bust / W_hip       if W_hip > 0 else 1

    # ── 7-Type Classification with priority order ──
    body_type = "Rectangle"
    logic     = ""
    confidence= 0.70

    # 1. Hourglass: shoulder ≈ hip, waist narrow
    if (0.90 <= hip_to_shoulder <= 1.10
            and waist_to_hip < 0.75
            and waist_to_shoulder < 0.75):
        body_type  = "Hourglass"
        logic      = "Shoulder ≈ Hip, strongly defined waist"
        confidence = 0.92

    # 2. Pear / Triangle: hips noticeably wider than shoulders
    elif hip_to_shoulder > 1.10 and waist_to_hip < 0.85:
        body_type  = "Pear"
        logic      = "Hips wider than shoulders"
        confidence = 0.88

    # 3. Inverted Triangle: shoulders wider, hips narrow
    elif hip_to_shoulder < 0.90 and waist_to_shoulder > 0.80:
        body_type  = "Inverted Triangle"
        logic      = "Shoulders wider, narrow hips"
        confidence = 0.87

    # 4. Apple: waist ≥ hips, weight at centre
    elif waist_to_hip >= 0.85 and waist_to_shoulder >= 0.85:
        body_type  = "Apple"
        logic      = "Wide waist relative to hips and shoulders"
        confidence = 0.80

    # 5. Spoon: high-hip wider, defined waist-to-high-hip curve
    elif (W_high_hip > W_shoulder * 1.05
              and waist_to_hip < 0.80):
        body_type  = "Spoon"
        logic      = "High hip wider than shoulders with waist definition"
        confidence = 0.78

    # 6. Diamond: narrow shoulders AND narrow hips, wide middle
    elif (hip_to_shoulder < 0.95
              and W_shoulder < W_waist * 1.05
              and W_hip < W_waist * 1.05):
        body_type  = "Diamond"
        logic      = "Narrow shoulders and hips, widest at waist"
        confidence = 0.78

    # 7. Rectangle: everything roughly equal, little waist
    else:
        body_type  = "Rectangle"
        logic      = "Measurements fairly uniform, minimal waist definition"
        confidence = 0.72

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

    return body_type, logic, confidence, measurements


# ---------------- FACE SHAPE CLASSIFIER ----------------
def classify_face_shape(landmarks, image_shape=None):
    """
    Corrected face shape classifier using accurate MediaPipe landmark indices.
    """
    LANDMARKS = {
        'chin'          : 152,
        'forehead_mid'  : 10,
        'left_cheek'    : 123,   # zygomatic prominence
        'right_cheek'   : 352,
        'left_jaw'      : 172,
        'right_jaw'     : 397,
        'left_temple'   : 70,    # temporal region
        'right_temple'  : 300,
        'left_jawangle' : 234,   # repurposed: jaw angle (was wrongly called cheek)
        'right_jawangle': 454,
        'nose_tip'      : 4,     # used for lower-third split
    }

    # ── Safety check ────────────────────────────────────────────────────
    required = list(LANDMARKS.values())
    for idx in required:
        if idx not in landmarks:
            return "Unknown", 0.0

    def pt(name):
        return np.array(landmarks[LANDMARKS[name]], dtype=float)

    # ── Key points ──────────────────────────────────────────────────────
    chin          = pt('chin')
    forehead_mid  = pt('forehead_mid')
    left_cheek    = pt('left_cheek')
    right_cheek   = pt('right_cheek')
    left_jaw      = pt('left_jaw')
    right_jaw     = pt('right_jaw')
    left_temple   = pt('left_temple')
    right_temple  = pt('right_temple')
    nose_tip      = pt('nose_tip')

    # ── Face length correction ───────────────────────────────────────────
    # lm[10] is on the forehead surface, not the hairline.
    # Estimate hairline by extending upward by ~12% of face height.
    raw_face_height = abs(chin[1] - forehead_mid[1])
    hairline_y      = forehead_mid[1] - (raw_face_height * 0.12)
    hairline_pt     = np.array([forehead_mid[0], hairline_y])
    face_length     = np.linalg.norm(chin - hairline_pt)

    # ── Five measurements ────────────────────────────────────────────────
    cheek_width     = np.linalg.norm(left_cheek    - right_cheek)
    jaw_width       = np.linalg.norm(left_jaw      - right_jaw)
    forehead_width  = np.linalg.norm(left_temple   - right_temple)
    
    # Lower-third: width at jaw-angle level (below cheekbones)
    jaw_angle_width = np.linalg.norm(
        np.array(landmarks[234], dtype=float) - 
        np.array(landmarks[454], dtype=float)
    )

    # Guard against zero division
    if cheek_width < 1 or jaw_width < 1 or forehead_width < 1:
        return "Unknown", 0.0

    # ── Ratios ──────────────────────────────────────────────────────────
    len_to_cheek     = face_length   / cheek_width
    jaw_to_forehead  = jaw_width     / forehead_width
    cheek_to_jaw     = cheek_width   / jaw_width
    jaw_to_cheek     = jaw_width     / cheek_width
    cheek_to_forehead= cheek_width   / forehead_width
    len_to_forehead  = face_length   / forehead_width

    # ── Classification (priority order matters) ──────────────────────────
    if len_to_cheek > 1.50 and abs(jaw_to_forehead - 1.0) < 0.15:
        face_shape = "Oblong"
        confidence = 0.82

    elif len_to_cheek > 1.35 and jaw_to_forehead < 0.80:
        face_shape = "Heart"
        confidence = 0.85

    elif len_to_cheek < 1.10 and cheek_to_jaw < 1.10:
        face_shape = "Round"
        confidence = 0.83

    elif 1.10 <= len_to_cheek <= 1.30 and jaw_to_cheek > 0.85:
        face_shape = "Square"
        confidence = 0.82

    elif cheek_to_jaw > 1.20 and cheek_to_forehead > 1.20:
        face_shape = "Diamond"
        confidence = 0.80

    else:
        face_shape = "Oval"
        confidence = 0.75

    return face_shape, confidence

# ---------------- SKIN TONE DETECTOR ----------------
def classify_skin_tone(image, landmarks):
    h, w, _ = image.shape
    
    # We use Left Cheek (234), Right Cheek (454), and Lower Forehead (10)
    target_landmarks = [234, 454, 10]
    patches = []
    patch_size = 15 # slightly smaller to ensure we only get skin
    
    for idx in target_landmarks:
        if idx in landmarks:
            px, py = landmarks[idx]
            x1, y1 = max(0, px - patch_size), max(0, py - patch_size)
            x2, y2 = min(w, px + patch_size), min(h, py + patch_size)
            patch = image[y1:y2, x1:x2]
            if patch.size > 0:
                patches.append(patch)
                
    if not patches:
        return "Unknown", "Unknown"
        
    # Average all patches to get one unified RGB color
    avg_color_bgr = np.mean([np.mean(p.reshape(-1, 3), axis=0) for p in patches], axis=0)
    avg_color_bgr_uint8 = np.uint8([[avg_color_bgr]])
    
    # Convert exactly from BGR to LAB
    lab_color = cv2.cvtColor(avg_color_bgr_uint8, cv2.COLOR_BGR2LAB)[0][0]
    l_opencv, a_opencv, b_opencv = lab_color
    
    # Mathematical conversion to standard CIE L*a*b* (L: 0-100, a/b: ~ -128 to 127)
    l_true = (l_opencv * 100.0) / 255.0
    a_true = a_opencv - 128.0
    b_true = b_opencv - 128.0
    
    # Skin Tone classification based on standard Luminance (L)
    if l_true > 70:
        skin_tone = "Fair"
    elif 50 < l_true <= 70:
        skin_tone = "Medium"
    else:
        skin_tone = "Dark"
        
    # Undertone classification based on A (Red/Green) and B (Yellow/Blue) channels
    # A positive threshold means the color strongly leans towards Yellow or Red
    threshold = 2.0
    
    if (b_true - a_true) > threshold:
        undertone = "Warm"
    elif (a_true - b_true) > threshold:
        undertone = "Cool"
    else:
        undertone = "Neutral"
        
    return skin_tone, undertone

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

    image = cv2.imread(filepath)
    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    # =====================================================
    # 🔵 PART 1: BODY TYPE DETECTION
    # =====================================================
    with mp_pose.Pose(static_image_mode=True) as pose:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if not results.pose_landmarks:
            return jsonify({"error": "No person detected"}), 400

        body_type, logic_used, confidence, measurements = classify_body_type_v2(
            image,
            results.pose_landmarks,
            gender
        )

    # =====================================================
    # 🟣 PART 2: FACE & SKIN ANALYSIS
    # =====================================================
    face_shape = "Unknown"
    face_confidence = 0.0
    skin_tone = "Unknown"
    undertone = "Unknown"

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.2) as face_mesh:
        results_face = face_mesh.process(rgb)
        
        if results_face.multi_face_landmarks:
            face_landmarks = results_face.multi_face_landmarks[0]
            
            # Convert landmarks to pixel coordinates needed for our functions
            h, w, _ = image.shape
            lm_pixels = {}
            for idx, lm in enumerate(face_landmarks.landmark):
                lm_pixels[idx] = (int(lm.x * w), int(lm.y * h))
            
            face_shape, face_confidence = classify_face_shape(lm_pixels)
            skin_tone, undertone = classify_skin_tone(image, lm_pixels)

    # =====================================================
    # 🟢 PART 3: OUTFIT DETECTION (YOLO MODULE)
    # =====================================================
    outfits = detect_outfits(image)

    # =====================================================
    # 🟣 PART 3: SAVE DETECTED CLOTHES TO CLOSET
    # =====================================================
    user_id = "guest_user"
    added_count, duplicate_count = add_items_to_closet(user_id, outfits)
    
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
    catalog_items = get_catalog_items(gender=gender)
    recommendation = generate_three_looks(profile, current_outfit_items, wardrobe_items, catalog_items, request_id=filename)

    # ---------------- FINAL RESPONSE ----------------
    return jsonify({
        "body_type": body_type,
        "face_shape": face_shape,
        "face_shape_confidence": face_confidence,
        "skin_tone": skin_tone,
        "undertone": undertone,
        "gender": gender,
        "logic_used": logic_used,
        "confidence_score": confidence,
        "measurements": measurements,
        "outfits": outfits,
        "styling_recommendations": styling_recommendations,
        "recommended_looks": recommendation["looks"],
        "styling": recommendation["styling"],
        "closet_info": {
            "saved": added_count > 0,
            "added_count": added_count,
            "duplicate_count": duplicate_count,
            "message": message
        }
    })

@app.route("/closet/<user_id>", methods=["GET"])

def get_closet(user_id):
    items = get_user_closet(user_id)
    return jsonify({"user_id": user_id, "closet": items})



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
            "error": result.get("error", "Virtual try-on failed."),
            "fallback": True
        }), 503


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
            "label": "New Top + Your Bottom",
            "model_used": look_b.get("model_used")
        }
    else:
        results["look_b"] = {"success": False, "image": None, "label": "New Top + Your Bottom"}

    if top_b64 and bottom_b64:
        step1 = generate_tryon(person_b64, top_b64, "upper")
        if step1["success"]:
            step2 = generate_tryon(step1["image_b64"], bottom_b64, "lower")
            results["look_c"] = {
                "success": step2["success"],
                "image": step2.get("image_b64"),
                "label": "Full New Outfit",
                "model_used": f"{step1.get('model_used')} + {step2.get('model_used')}"
            }
        else:
            step1b = generate_tryon(person_b64, bottom_b64, "lower")
            results["look_c"] = {
                "success": step1b["success"],
                "image": step1b.get("image_b64"),
                "label": "Full New Outfit (partial)",
                "model_used": step1b.get("model_used")
            }
    else:
        results["look_c"] = {"success": False, "image": None, "label": "Full New Outfit"}

    return jsonify({
        "success": True,
        "looks": results
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
