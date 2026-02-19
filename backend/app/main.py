# # main.py
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# import cv2
# import numpy as np
# import mediapipe as mp
# from typing import Tuple
# from io import BytesIO
#
# app = FastAPI()
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
# )
#
# mp_pose = mp.solutions.pose
# mp_face_detection = mp.solutions.face_detection
#
#
# def get_body_type(image: np.ndarray) -> str:
#     with mp_pose.Pose(static_image_mode=True) as pose:
#         results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         if not results.pose_landmarks:
#             print("No pose landmarks detected.")
#             return "Unknown"
#
#         lm = results.pose_landmarks.landmark
#
#         # Widths using X-coordinates
#         shoulder_width = abs(lm[11].x - lm[12].x)
#         hip_width = abs(lm[23].x - lm[24].x)
#
#         # Optional: use knees to scale distances
#         knee_height = abs(lm[25].y - lm[26].y)
#         if knee_height == 0:
#             return "Unknown"
#
#         ratio = shoulder_width / hip_width
#         print(f"[DEBUG] Shoulder Width: {shoulder_width:.2f}, Hip Width: {hip_width:.2f}, Ratio: {ratio:.2f}")
#         if ratio >= 1.2:
#             return "Inverted Triangle"
#         elif ratio <= 0.8:
#             return "Pear"
#         elif abs(shoulder_width - hip_width) < 0.05:
#             return "Rectangle"
#         else:
#             return "Hourglass"
#
#
# def get_skin_tone_with_face(image: np.ndarray) -> str:
#     with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
#         results = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#
#         if not results.detections:
#             print("No face detected.")
#             return "Unknown"
#
#         # Get face bounding box from first detection
#         detection = results.detections[0]
#         bbox = detection.location_data.relative_bounding_box
#         h, w, _ = image.shape
#
#         x1 = int(bbox.xmin * w)
#         y1 = int(bbox.ymin * h)
#         x2 = int((bbox.xmin + bbox.width) * w)
#         y2 = int((bbox.ymin + bbox.height) * h)
#
#         # Expand crop to include more face area
#         padding = 10
#         x1 = max(0, x1 - padding)
#         y1 = max(0, y1 - padding)
#         x2 = min(w, x2 + padding)
#         y2 = min(h, y2 + padding)
#
#         face_crop = image[y1:y2, x1:x2]
#
#         # Convert to LAB and average L channel (brightness)
#         lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
#         l_avg = cv2.mean(lab)[0]
#
#         if l_avg < 60:
#             return "Deep"
#         elif l_avg < 130:
#             return "Medium"
#         else:
#             return "Fair"
#
#
# def suggest_outfit(body_type: str, skin_tone: str) -> str:
#     suggestion = {
#         "Inverted Triangle": "Wear A-line dresses, avoid shoulder pads.",
#         "Pear": "Try bright tops and structured shoulders.",
#         "Rectangle": "Use belts and layers to add curves.",
#         "Hourglass": "Bodycon and wrap dresses work well.",
#         "Unknown": "Try basic well-fitted clothes."
#     }
#     tone_note = {
#         "Fair": "Pastel and earthy tones suit you.",
#         "Medium": "Jewel tones and warm colors work best.",
#         "Deep": "Bright and bold colors will pop on you."
#     }
#     return f"{suggestion.get(body_type, '')} {tone_note.get(skin_tone, '')}"
#
#
# @app.post("/analyze")
# async def analyze_image(image: UploadFile = File(...)):
#     contents = await image.read()
#     np_img = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
#
#     body_type = get_body_type(img)
#     skin_tone = get_skin_tone_with_face(img)
#     outfit = suggest_outfit(body_type, skin_tone)
#
#     return {
#         "body_type": body_type,
#         "skin_tone": skin_tone,
#         "outfit_suggestion": outfit
#
#     }


from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from typing import Dict
from .schemas import AnalysisResult
from .services.detectors import detect_all

app = FastAPI(title="Style Analyzer API", version="0.1.0")

app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

@app.get("/health")
def health() -> Dict[str, str]:
  return {"status": "ok"}

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(...)) -> AnalysisResult:
  if not file.content_type or not file.content_type.startswith("image/"):
    raise HTTPException(status_code=400, detail="Only image uploads are supported")
  try:
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
  except Exception as exc:
    raise HTTPException(status_code=400, detail="Invalid image data") from exc
  return detect_all(image)