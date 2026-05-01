
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

# ---------------- APP SETUP ----------------
app = Flask(__name__)
CORS(app, resources={r"/analyze": {"origins": "*"}})

UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# ---------------- UTILS ----------------
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# ---------------- BODY SHAPE CLASSIFIER ----------------
def classify_body_type(shoulder_ratio, waist_ratio, highhip_waist_ratio, gender="Female"):

    # Standard male build in a suit can easily hit 1.40 due to shoulder pads. Require strict 1.50.
    inv_tri_thresh = 1.50 if gender.lower() == "male" else 1.20

    if abs(shoulder_ratio - 1.0) <= 0.08 and waist_ratio <= 0.75:
        return "hourglass", "Balanced shoulders & hips with narrow waist", 0.90

    elif shoulder_ratio < 0.90 and waist_ratio <= 0.78:
        return "triangle", "Hips wider than shoulders", 0.85

    elif shoulder_ratio > inv_tri_thresh and waist_ratio <= 0.78:
        return "inverted_triangle", "Shoulders wider than hips", 0.85

    elif waist_ratio > 0.85:
        return "apple", "Waist not well defined", 0.75

    else:
        return "rectangle", "Measurements fairly balanced", 0.70


# ---------------- FACE SHAPE CLASSIFIER ----------------
def classify_face_shape(landmarks):
    # Landmarks indices
    LANDMARKS_ID = {
        'chin': 152, 'forehead': 10,
        'left_cheekbone': 234, 'right_cheekbone': 454,
        'left_jaw': 172, 'right_jaw': 397,
        'left_temple': 162, 'right_temple': 389
    }
    
    # Helper to get coord
    def get_pt(name):
        return np.array(landmarks[LANDMARKS_ID[name]])

    # Calculate distances
    face_length = np.linalg.norm(get_pt('forehead') - get_pt('chin'))
    cheekbone_width = np.linalg.norm(get_pt('left_cheekbone') - get_pt('right_cheekbone'))
    jaw_width = np.linalg.norm(get_pt('left_jaw') - get_pt('right_jaw'))
    forehead_width = np.linalg.norm(get_pt('left_temple') - get_pt('right_temple'))

    # Ratios
    if cheekbone_width == 0 or jaw_width == 0 or forehead_width == 0:
        return "Unknown"
        
    length_to_cheekbone = face_length / cheekbone_width
    jaw_to_forehead = jaw_width / forehead_width
    cheekbone_to_jaw = cheekbone_width / jaw_width

    # Logic
    if length_to_cheekbone > 1.35:
        if jaw_to_forehead < 0.85: return "Heart"
        elif jaw_to_forehead > 1.15: return "Triangle"
        else: return "Oblong"
    elif length_to_cheekbone < 1.1:
        if abs(cheekbone_width - jaw_width) < 0.1 * face_length: return "Round"
        else: return "Square"
    elif cheekbone_to_jaw > 1.15:
        return "Diamond"
    elif abs(cheekbone_width - jaw_width) < 0.08 * face_length:
        return "Square"
    else:
        return "Oval"

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
@app.route("/analyze", methods=["POST"])
def analyze_image():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    gender = request.form.get("gender", "Female")

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

        h, w, _ = image.shape
        lm = results.pose_landmarks.landmark

        def coord(idx):
            return int(lm[idx].x * w), int(lm[idx].y * h)

        ls = coord(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        rs = coord(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        lh = coord(mp_pose.PoseLandmark.LEFT_HIP.value)
        rh = coord(mp_pose.PoseLandmark.RIGHT_HIP.value)

        wl = (
            int((ls[0] + lh[0]) / 2),
            int((ls[1] + lh[1]) / 2)
        )
        wr = (
            int((rs[0] + rh[0]) / 2),
            int((rs[1] + rh[1]) / 2)
        )

        shoulder_width = euclidean_distance(ls, rs)
        hip_width = euclidean_distance(lh, rh)
        waist_width = euclidean_distance(wl, wr)

        shoulder_ratio = shoulder_width / hip_width
        waist_ratio = waist_width / shoulder_width
        highhip_waist_ratio = hip_width / waist_width

        body_type, logic_used, confidence = classify_body_type(
            shoulder_ratio,
            waist_ratio,
            highhip_waist_ratio,
            gender
        )

    # =====================================================
    # 🟣 PART 2: FACE & SKIN ANALYSIS
    # =====================================================
    face_shape = "Unknown"
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
            
            face_shape = classify_face_shape(lm_pixels)
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

    # ---------------- FINAL RESPONSE ----------------
    return jsonify({
        "body_type": body_type,
        "face_shape": face_shape,
        "skin_tone": skin_tone,
        "undertone": undertone,
        "gender": gender,
        "logic_used": logic_used,
        "confidence_score": confidence,
        "measurements": {
            "shoulder_ratio": round(shoulder_ratio, 2),
            "waist_ratio": round(waist_ratio, 2),
            "highhip_waist_ratio": round(highhip_waist_ratio, 2)
        },
        "outfits": outfits,
        "styling_recommendations": styling_recommendations,
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
