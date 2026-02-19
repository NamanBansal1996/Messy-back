#
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import cv2
# import os
# import base64
# import mediapipe as mp
# import math
# from werkzeug.utils import secure_filename
# from ultralytics import YOLO
#
# # ---------------- APP SETUP ----------------
# app = Flask(__name__)
# CORS(app, resources={r"/analyze": {"origins": "*"}})
#
# UPLOAD_FOLDER = "temp_uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
#
# mp_pose = mp.solutions.pose
# yolo_model = YOLO("yolov8n.pt")
#
# # ---------------- UTILS ----------------
# def euclidean_distance(p1, p2):
#     return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
#
# # ---------------- BODY SHAPE CLASSIFIER ----------------
# def classify_body_type(shoulder_ratio, waist_ratio, highhip_waist_ratio):
#
#     if abs(shoulder_ratio - 1.0) <= 0.08 and waist_ratio <= 0.75:
#         return "hourglass", "Balanced shoulders & hips with narrow waist", 0.90
#
#     elif shoulder_ratio < 0.90 and waist_ratio <= 0.78:
#         return "triangle", "Hips wider than shoulders", 0.85
#
#     elif shoulder_ratio > 1.20 and waist_ratio <= 0.78:
#         return "inverted_triangle", "Shoulders wider than hips", 0.85
#
#     elif waist_ratio > 0.85:
#         return "apple", "Waist not well defined", 0.75
#
#     else:
#         return "rectangle", "Measurements fairly balanced", 0.70
#
#
# # ---------------- OUTFIT CATEGORIES ----------------
# TOP = set()
# BOTTOM = set()
# ACCESSORIES = {"handbag", "tie", "backpack"}
#
# def categorize(label):
#     if label in TOP:
#         return "top"
#     elif label in BOTTOM:
#         return "bottom"
#     elif label in ACCESSORIES:
#         return "accessories"
#     return "unknown"
#
#
# # ---------------- API ----------------
# @app.route("/analyze", methods=["POST"])
# def analyze_image():
#
#     if "image" not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400
#
#     image_file = request.files["image"]
#     filename = secure_filename(image_file.filename)
#     filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#     image_file.save(filepath)
#
#     image = cv2.imread(filepath)
#     if image is None:
#         return jsonify({"error": "Invalid image"}), 400
#
#     # =====================================================
#     # 🔵 PART 1: BODY TYPE (IMPROVED)
#     # =====================================================
#     with mp_pose.Pose(static_image_mode=True) as pose:
#         rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = pose.process(rgb)
#
#         if not results.pose_landmarks:
#             return jsonify({"error": "No person detected"}), 400
#
#         h, w, _ = image.shape
#         lm = results.pose_landmarks.landmark
#
#         def coord(idx):
#             return int(lm[idx].x * w), int(lm[idx].y * h)
#
#         ls = coord(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
#         rs = coord(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
#         lh = coord(mp_pose.PoseLandmark.LEFT_HIP.value)
#         rh = coord(mp_pose.PoseLandmark.RIGHT_HIP.value)
#
#         # Waist estimation (midpoint between shoulders & hips)
#         wl = (
#             int((ls[0] + lh[0]) / 2),
#             int((ls[1] + lh[1]) / 2)
#         )
#         wr = (
#             int((rs[0] + rh[0]) / 2),
#             int((rs[1] + rh[1]) / 2)
#         )
#
#         shoulder_width = euclidean_distance(ls, rs)
#         hip_width = euclidean_distance(lh, rh)
#         waist_width = euclidean_distance(wl, wr)
#
#         shoulder_ratio = shoulder_width / hip_width
#         waist_ratio = waist_width / shoulder_width
#         highhip_waist_ratio = hip_width / waist_width
#
#         body_type, logic_used, confidence = classify_body_type(
#             shoulder_ratio,
#             waist_ratio,
#             highhip_waist_ratio
#         )
#
#     # =====================================================
#     # 🟢 PART 2: OUTFIT DETECTION (UNCHANGED)
#     # =====================================================
#     yolo_results = yolo_model(image)
#
#     outfits = {"top": [], "bottom": [], "accessories": []}
#
#     for box in yolo_results[0].boxes:
#         cls_id = int(box.cls[0])
#         label = yolo_results[0].names[cls_id].lower()
#
#         category = categorize(label)
#         if category == "unknown":
#             continue
#
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         cropped = image[y1:y2, x1:x2]
#
#         _, buffer = cv2.imencode(".jpg", cropped)
#         crop_b64 = base64.b64encode(buffer).decode("utf-8")
#
#         outfits[category].append({
#             "label": label,
#             "image": crop_b64
#         })
#
#     # ---------------- FINAL RESPONSE ----------------
#     return jsonify({
#         "body_type": body_type,
#         "logic_used": logic_used,
#         "confidence_score": confidence,
#         "measurements": {
#             "shoulder_ratio": round(shoulder_ratio, 2),
#             "waist_ratio": round(waist_ratio, 2),
#             "highhip_waist_ratio": round(highhip_waist_ratio, 2)
#         },
#         "outfits": outfits
#     })
#
#
# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import os
import base64
import mediapipe as mp
import math
from werkzeug.utils import secure_filename

# 🔹 Import YOLO outfit detection
from yolo_outfit_detect import detect_outfits

# ---------------- APP SETUP ----------------
app = Flask(__name__)
CORS(app, resources={r"/analyze": {"origins": "*"}})

UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

mp_pose = mp.solutions.pose

# ---------------- UTILS ----------------
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# ---------------- BODY SHAPE CLASSIFIER ----------------
def classify_body_type(shoulder_ratio, waist_ratio, highhip_waist_ratio):

    if abs(shoulder_ratio - 1.0) <= 0.08 and waist_ratio <= 0.75:
        return "hourglass", "Balanced shoulders & hips with narrow waist", 0.90

    elif shoulder_ratio < 0.90 and waist_ratio <= 0.78:
        return "triangle", "Hips wider than shoulders", 0.85

    elif shoulder_ratio > 1.20 and waist_ratio <= 0.78:
        return "inverted_triangle", "Shoulders wider than hips", 0.85

    elif waist_ratio > 0.85:
        return "apple", "Waist not well defined", 0.75

    else:
        return "rectangle", "Measurements fairly balanced", 0.70


# ---------------- API ----------------
@app.route("/analyze", methods=["POST"])
def analyze_image():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

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
            highhip_waist_ratio
        )

    # =====================================================
    # 🟢 PART 2: OUTFIT DETECTION (YOLO MODULE)
    # =====================================================
    outfits = detect_outfits(image)

    # ---------------- FINAL RESPONSE ----------------
    return jsonify({
        "body_type": body_type,
        "logic_used": logic_used,
        "confidence_score": confidence,
        "measurements": {
            "shoulder_ratio": round(shoulder_ratio, 2),
            "waist_ratio": round(waist_ratio, 2),
            "highhip_waist_ratio": round(highhip_waist_ratio, 2)
        },
        "outfits": outfits
    })


if __name__ == "__main__":
    app.run(debug=True)
