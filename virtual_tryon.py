import base64
import time
import os
import tempfile
import traceback
import requests

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

def try_on_with_idmvton(person_image_b64, garment_image_b64, garment_type="upper"):
    person_path = None
    garment_path = None
    try:
        print("[IDM-VTON] Saving temp files...")
        person_path = base64_to_tempfile(person_image_b64)
        garment_path = base64_to_tempfile(garment_image_b64)

        print("[IDM-VTON] Connecting to HF Space...")
        # Old versions of gradio_client do not support the timeout arg
        client = Client("yisol/IDM-VTON")

        print("[IDM-VTON] Sending request...")
        person_dict = {
            "background": handle_file(person_path),
            "layers": [],
            "composite": None
        }

        # Pass POSITIONALLY
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

    # Check space before wasting time
    if not is_space_alive():
        return {
            "success": False,
            "image_b64": None,
            "model_used": None,
            "error": "IDM-VTON Space is sleeping or unavailable. Please try again in 1-2 minutes."
        }

    result, err = try_on_with_idmvton(person_image_b64, garment_image_b64, garment_type)
    if result:
        return {"success": True, "image_b64": result, "model_used": "IDM-VTON", "error": None}

    return {
        "success": False,
        "image_b64": None,
        "model_used": None,
        "error": err or "Try-on model failed. The space may be overloaded."
    }

