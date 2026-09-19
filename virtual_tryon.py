import base64
import time
import os
import tempfile
import traceback
import requests
import cv2
import mediapipe as mp
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError

try:
    from gradio_client import Client, handle_file
except ImportError as e:
    print(f"WARNING: gradio_client not installed or failed to import. Error: {e}")

mp_pose = mp.solutions.pose

# mediapipe Pose landmark indices (BlazePose topology)
_LEFT_SHOULDER, _RIGHT_SHOULDER = 11, 12
_LEFT_HIP, _RIGHT_HIP = 23, 24
_LEFT_ANKLE, _RIGHT_ANKLE = 27, 28

def base64_to_tempfile(b64_string, suffix=".jpg"):
    img_bytes = base64.b64decode(b64_string)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(img_bytes)
    tmp.close()
    return tmp.name

def file_to_base64(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def is_space_alive(space_id="yisol/IDM-VTON"):
    try:
        url = f"https://huggingface.co/api/spaces/{space_id}"
        r = requests.get(url, timeout=3)
        stage = r.json().get("runtime", {}).get("stage", "")
        print(f"[HF Space] Status: {stage}")
        return stage == "RUNNING"
    except Exception as e:
        print(f"[HF Space] Status check failed: {e}")
        return False

def _detect_pose_landmarks(person_img):
    """Returns pixel-coordinate shoulder/hip/ankle landmarks, or None if no
    person is detected in this image."""
    try:
        h, w = person_img.shape[:2]
        rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        with mp_pose.Pose(static_image_mode=True) as pose:
            results = pose.process(rgb)
        if not results.pose_landmarks:
            return None
        lm = results.pose_landmarks.landmark

        def px(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        return {
            "left_shoulder": px(_LEFT_SHOULDER),
            "right_shoulder": px(_RIGHT_SHOULDER),
            "left_hip": px(_LEFT_HIP),
            "right_hip": px(_RIGHT_HIP),
            "left_ankle": px(_LEFT_ANKLE),
            "right_ankle": px(_RIGHT_ANKLE),
        }
    except Exception as e:
        print(f"[FallbackOverlay] Pose detection failed: {e}")
        return None


def _landmark_placement(landmarks, garment_type, frame_w, frame_h):
    """
    Real-landmark-driven placement instead of a fixed percentage of the
    frame. Returns (center_x, center_y, box_w, box_h, angle_deg) -- angle
    comes from the same landmark pair used for width, so a tilted person
    gets a tilted garment instead of one pasted perfectly upright.
    """
    if garment_type == "lower":
        left, right = landmarks["left_hip"], landmarks["right_hip"]
        ankle_y = max(landmarks["left_ankle"][1], landmarks["right_ankle"][1])
        top_y = min(left[1], right[1])
        hip_width = np.linalg.norm(right - left)
        box_w = hip_width * 2.3
        box_h = ankle_y - top_y
        center = np.array([(left[0] + right[0]) / 2, (top_y + ankle_y) / 2])
    else:
        left, right = landmarks["left_shoulder"], landmarks["right_shoulder"]
        hip_top_y = min(landmarks["left_hip"][1], landmarks["right_hip"][1])
        shoulder_top_y = min(left[1], right[1])
        shoulder_width = np.linalg.norm(right - left)
        box_w = shoulder_width * 1.9
        box_h = (hip_top_y - shoulder_top_y) * 1.15
        center = np.array([(left[0] + right[0]) / 2, (shoulder_top_y + hip_top_y) / 2])

    dx, dy = right[0] - left[0], right[1] - left[1]
    angle_deg = -np.degrees(np.arctan2(dy, dx))
    # A person roughly facing the camera has a near-horizontal shoulder/hip
    # line, so this should only ever correct a slight camera/pose roll. A
    # large value here means the landmark pair isn't reliably left-right
    # (e.g. the subject is bent over or turned side-on) -- rotating by that
    # raw angle can flip the garment upside down, so treat anything beyond
    # a plausible roll as unreliable and skip rotation instead.
    if abs(angle_deg) > 30:
        angle_deg = 0.0

    box_w = float(np.clip(box_w, 10, frame_w * 1.5))
    box_h = float(np.clip(box_h, 10, frame_h * 1.5))
    return center[0], center[1], box_w, box_h, angle_deg


def _rotate_with_expanded_canvas(img, angle_deg):
    """Rotates img by angle_deg onto a larger canvas so corners don't clip."""
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    m[0, 2] += (new_w / 2) - center[0]
    m[1, 2] += (new_h / 2) - center[1]
    border_value = (0, 0, 0, 0) if img.ndim == 3 and img.shape[2] == 4 else (0, 0, 0)
    return cv2.warpAffine(img, m, (new_w, new_h), borderValue=border_value)


def create_fallback_overlay(person_image_b64, garment_image_b64, garment_type="upper"):
    """
    Not a real garment-fitting model -- pastes a scaled/rotated garment
    cutout onto the person photo. Uses real body landmarks (mediapipe Pose:
    shoulders/hips/ankles) to size, position, and tilt the garment when a
    person is detected in this photo; falls back to the original fixed
    percentage-of-frame box if pose detection can't find a person at all,
    so this never hard-fails regardless of photo quality.
    """
    try:
        person_bytes = base64.b64decode(person_image_b64)
        person_np = np.frombuffer(person_bytes, dtype=np.uint8)
        person_img = cv2.imdecode(person_np, cv2.IMREAD_COLOR)

        garment_bytes = base64.b64decode(garment_image_b64)
        garment_np = np.frombuffer(garment_bytes, dtype=np.uint8)
        garment_img = cv2.imdecode(garment_np, cv2.IMREAD_UNCHANGED)

        if person_img is None or garment_img is None:
            return person_image_b64

        h, w, _ = person_img.shape
        landmarks = _detect_pose_landmarks(person_img)
        angle_deg = 0.0

        if landmarks:
            cx, cy, box_w, box_h, angle_deg = _landmark_placement(landmarks, garment_type, w, h)
        else:
            box_w = w * (0.6 if garment_type == "lower" else 0.65)
            box_h = h * 0.45
            cx = w / 2
            cy = (h * 0.48 + box_h / 2) if garment_type == "lower" else (h * 0.15 + box_h / 2)

        garment_resized = cv2.resize(
            garment_img, (max(1, int(box_w)), max(1, int(box_h))), interpolation=cv2.INTER_AREA
        )
        garment_final = (
            _rotate_with_expanded_canvas(garment_resized, angle_deg)
            if abs(angle_deg) > 1.0
            else garment_resized
        )

        rh, rw = garment_final.shape[:2]
        x_offset = int(cx - rw / 2)
        y_offset = int(cy - rh / 2)

        y1, y2 = max(0, y_offset), min(h, y_offset + rh)
        x1, x2 = max(0, x_offset), min(w, x_offset + rw)
        gy1, gx1 = y1 - y_offset, x1 - x_offset
        gh, gw = y2 - y1, x2 - x1

        if gh <= 0 or gw <= 0:
            return person_image_b64

        garment_crop = garment_final[gy1:gy1 + gh, gx1:gx1 + gw]
        result_img = person_img.copy()

        if garment_crop.ndim == 3 and garment_crop.shape[2] == 4:
            alpha = (garment_crop[:, :, 3] / 255.0)[:, :, np.newaxis]
            bgr = garment_crop[:, :, 0:3]
            result_img[y1:y2, x1:x2] = (alpha * bgr + (1.0 - alpha) * result_img[y1:y2, x1:x2]).astype(np.uint8)
        else:
            bgr = garment_crop[:, :, 0:3] if garment_crop.ndim == 3 else garment_crop
            result_img[y1:y2, x1:x2] = cv2.addWeighted(bgr, 0.85, result_img[y1:y2, x1:x2], 0.15, 0)

        _, buffer = cv2.imencode(".jpg", result_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        print(f"[FallbackOverlay] Error generating fallback overlay: {e}")
        return person_image_b64

def try_on_with_idmvton(person_image_b64, garment_image_b64, garment_type="upper"):
    person_path = None
    garment_path = None
    try:
        print("[IDM-VTON] Saving temp files...")
        person_path = base64_to_tempfile(person_image_b64)
        garment_path = base64_to_tempfile(garment_image_b64)

        print("[IDM-VTON] Connecting to HF Space...")
        client = Client("yisol/IDM-VTON")

        print("[IDM-VTON] Sending request...")
        person_dict = {
            "background": handle_file(person_path),
            "layers": [],
            "composite": None
        }

        garment_desc = "a pair of pants" if garment_type == "lower" else "a shirt"

        result = client.predict(
            person_dict,
            handle_file(garment_path),
            garment_desc,
            True,
            False,
            30,
            42,
            api_name="/tryon"
        )

        print(f"[IDM-VTON] Result received: {result}")

        if isinstance(result, tuple) and len(result) > 0:
            output_path = result[0]
            if output_path and os.path.exists(output_path):
                print("[IDM-VTON] Success!")
                return file_to_base64(output_path), None
            else:
                print(f"[IDM-VTON] Bad output path: {output_path}")
        elif isinstance(result, str) and os.path.exists(result):
            return file_to_base64(result), None
        else:
            print(f"[IDM-VTON] Unexpected result format: {type(result)} = {result}")

    except Exception as e:
        error_msg = str(e)
        print(f"[IDM-VTON] EXCEPTION: {error_msg}")
        traceback.print_exc()
        return None, error_msg
    finally:
        if person_path and os.path.exists(person_path): os.unlink(person_path)
        if garment_path and os.path.exists(garment_path): os.unlink(garment_path)

    return None, "Try-on model failed to return an image"

def generate_tryon(person_image_b64, garment_image_b64, garment_type="upper"):
    print(f"[TryOn] garment_type={garment_type}")

    if is_space_alive():
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(try_on_with_idmvton, person_image_b64, garment_image_b64, garment_type)
            result, err = future.result(timeout=60.0)
            executor.shutdown(wait=False)
            if result:
                return {"success": True, "image_b64": result, "model_used": "IDM-VTON", "error": None}
            print(f"[TryOn] IDM-VTON attempt failed: {err}. Generating local overlay fallback...")
        except TimeoutError:
            print("[TryOn] IDM-VTON model request timed out after 60 seconds. Generating local overlay fallback...")
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception as ex:
            print(f"[TryOn] IDM-VTON error: {ex}. Generating local overlay fallback...")
            executor.shutdown(wait=False)
    else:
        print("[TryOn] IDM-VTON Space is sleeping/unavailable. Generating local overlay fallback...")

    fallback_b64 = create_fallback_overlay(person_image_b64, garment_image_b64, garment_type)
    return {
        "success": False,
        "image_b64": None,
        "fallback_image_b64": fallback_b64,
        "model_used": "Fallback Overlay",
        "error": "IDM-VTON model busy or unavailable. Showing preview."
    }
