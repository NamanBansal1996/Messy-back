import cv2
import sys
from yolo_outfit_detect import detect_outfits

def test_yolo(filepath):
    print(f"Testing YOLO on {filepath}...")
    image = cv2.imread(filepath)
    if image is None:
        print("Failed to load image.")
        return
    
    outfits = detect_outfits(image)
    for category, items in outfits.items():
        print(f"Category: {category}")
        for item in items:
            print(f"  - Label: {item['label']}, Conf: {item['confidence']}")
            if 'bounding_box' in item:
                print(f"    - BBox: {item['bounding_box']}")

if __name__ == "__main__":
    test_yolo("temp_uploads/penny_photo1.jpg")
