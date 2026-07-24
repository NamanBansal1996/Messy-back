import requests

try:
    print("Testing CatVTON...")
    resp = requests.get("https://zhengchong-catvton.hf.space", timeout=5)
    print(f"CatVTON status: {resp.status_code}")
except Exception as e:
    print(f"CatVTON error: {e}")

try:
    print("Testing IDM-VTON...")
    resp = requests.get("https://yisol-idm-vton.hf.space", timeout=5)
    print(f"IDM-VTON status: {resp.status_code}")
except Exception as e:
    print(f"IDM-VTON error: {e}")

