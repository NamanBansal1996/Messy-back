# MESS Backend: AI Styling & Analysis API

This is the backend for **MESS (Minimum Effort in Styling Sexy)**. It serves as an AI-powered digital personal stylist engine, processing user images to detect body types, face shapes, skin tones, undertones, and clothing items.

The frontend for this application is located in the `Messy` project.

---

## 🧠 Core Features

- **Body Type Detection:** Uses MediaPipe Pose to analyze shoulder, waist, and hip ratios to classify body shapes (Hourglass, Triangle, Inverted Triangle, Apple, Rectangle).
- **Face Shape Classification:** Uses MediaPipe Face Mesh to measure facial landmarks and determine face shape (Heart, Round, Oval, Square, etc.).
- **Skin Tone & Undertone Analysis:** Extracts skin patches from the face to calculate CIE L*a*b* values, classifying skin tone (Fair, Medium, Dark) and undertone (Warm, Cool, Neutral).
- **Outfit Detection & Parsing:** Utilizes **YOLO** for outfit detection and **SegFormer** for high-precision clothing segmentation (removing background/skin/hair).
- **Digital Closet Manager:** Automatically saves segmented clothing items as transparent PNGs into a digital closet and prevents duplicates.
- **Styling Recommendations:** A rule-based engine providing customized advice based on the user's analyzed features and wardrobe.

---

## 🛠️ Tech Stack (Backend)

- **Framework:** Flask (Python)
- **Machine Learning & Vision:**
  - OpenCV
  - MediaPipe (Pose & Face Mesh)
  - YOLO (Ultralytics)
  - SegFormer (Hugging Face Transformers)
- **Data Handling:** NumPy, SciPy, Pillow
- **Deployment:** Ready for deployment (e.g., via Railway or Docker)

---

## 🚀 How to Run Locally

### Prerequisites
Make sure you have Python 3.9+ installed.

### Installation
1. Navigate to the `Messy-back` directory.
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Server
```bash
python3 app.py
```
The server will start on `http://localhost:5001`. 

---

## 📡 API Endpoints

### `POST /analyze`
Analyzes a full-body image to determine user styling characteristics and detect outfits.
- **Form Data:**
  - `image`: The uploaded image file (JPEG/PNG).
  - `gender`: (Optional) "Female" or "Male". Defaults to "Female".
- **Returns:** JSON object containing body type, face shape, skin tone, undertone, confidence scores, detected outfits, styling recommendations, and closet update status.

### `GET /closet/<user_id>`
Retrieves the digital closet items for a specific user.
- **Returns:** JSON object with a list of saved clothing items.

---

## 📁 Directory Structure
- `app.py`: Main Flask application and API routes.
- `yolo_outfit_detect.py`: Logic for YOLO-based clothing item detection.
- `segformer_parser.py`: SegFormer integration for precise clothing segmentation.
- `closet_manager.py`: Manages saving items, deduplication, and retrieving user closet data.
- `styling_rules.py`: Generates personalized recommendations.
- `color_utils.py` & `face_shape_test.py`: Utilities for advanced color and shape processing.
- `uploads/` & `temp_uploads/`: Directories for processing images and storing wardrobe PNGs.

---

MESS – Minimum Effort, Maximum Style.
