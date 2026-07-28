import base64
import time
import os
import tempfile
import traceback
import requests
import cv2
import numpy as np

try:
    from gradio_client import Client, handle_file
except ImportError as e:
    print(f"WARNING: gradio_client not installed or failed to import. Error: {e}")

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
        r = requests.get(url, timeout=10)
        stage = r.json().get("runtime", {}).get("stage", "")
        print(f"[HF Space] Status: {stage}")
        return stage == "RUNNING"
    except Exception as e:
        print(f"[HF Space] Status check failed: {e}")
        return False

def create_fallback_overlay(person_image_b64, garment_image_b64, garment_type="upper"):
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

        if garment_type == "lower":
            overlay_w = int(w * 0.6)
            overlay_h = int(h * 0.45)
            x_offset = int((w - overlay_w) / 2)
            y_offset = int(h * 0.48)
        else:
            overlay_w = int(w * 0.65)
            overlay_h = int(h * 0.45)
            x_offset = int((w - overlay_w) / 2)
            y_offset = int(h * 0.15)

        garment_resized = cv2.resize(garment_img, (overlay_w, overlay_h), interpolation=cv2.INTER_AREA)

        y1, y2 = max(0, y_offset), min(h, y_offset + overlay_h)
        x1, x2 = max(0, x_offset), min(w, x_offset + overlay_w)
        gh, gw = y2 - y1, x2 - x1

        if gh <= 0 or gw <= 0:
            return person_image_b64

        garment_crop = garment_resized[0:gh, 0:gw]
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

        result = client.predict(
            person_dict,
            handle_file(garment_path),
            "clothing item",
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
        result, err = try_on_with_idmvton(person_image_b64, garment_image_b64, garment_type)
        if result:
            return {"success": True, "image_b64": result, "model_used": "IDM-VTON", "error": None}
        print(f"[TryOn] IDM-VTON attempt failed: {err}. Generating local overlay fallback...")
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
