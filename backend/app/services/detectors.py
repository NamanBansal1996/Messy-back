from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
# Optionally: import cv2, mediapipe as mp
from ..schemas import AnalysisResult

def detect_skin_tone(image: Image.Image) -> Tuple[str, Dict[str, float]]:
  sample = image.resize((256, 256)).convert("L")
  luma = np.asarray(sample, dtype=np.float32)
  mean_luma = float(luma.mean())  # 0..255

  if mean_luma >= 175:
    category = "fair"
  elif mean_luma >= 125:
    category = "medium"
  else:
    category = "dark"
  return category, {"mean_luma": mean_luma}

def detect_face_type(_image: Image.Image) -> str:
  # TODO: Use MediaPipe face mesh ratios -> {oval, round, square, heart, oblong}
  return "unknown"

def detect_body_type(_image: Image.Image) -> str:
  # TODO: Use pose landmarks (shoulder/waist/hip widths) -> {rectangle, triangle, inverted_triangle, hourglass, oval}
  return "unknown"

def detect_outfits(_image: Image.Image) -> List[str]:
  # TODO: Use clothing segmentation/classifier -> ["t-shirt", "jeans", ...]
  return []

def detect_all(image: Image.Image) -> AnalysisResult:
  skin_tone, debug = detect_skin_tone(image)
  face_type = detect_face_type(image)
  body_type = detect_body_type(image)
  outfits = detect_outfits(image)
  return AnalysisResult(
    skin_tone=skin_tone,
    face_type=face_type,
    body_type=body_type,
    outfits=outfits,
    debug=debug,
  )