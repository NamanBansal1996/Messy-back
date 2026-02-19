import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define more accurate landmark indices
LANDMARKS = {
    'chin': 152,  # Bottom of chin
    'forehead': 10,  # Center of forehead
    'left_cheekbone': 234,  # Left cheek extreme
    'right_cheekbone': 454,  # Right cheek extreme
    'left_jaw': 172,  # Left jaw corner
    'right_jaw': 397,  # Right jaw corner
    'left_temple': 162,  # Left temple/forehead side
    'right_temple': 389  # Right temple/forehead side
}

# For smoothing results
SHAPE_HISTORY = deque(maxlen=10)


def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def classify_face_shape(landmarks):
    # Get all landmark points
    chin = landmarks[LANDMARKS['chin']]
    forehead = landmarks[LANDMARKS['forehead']]
    left_cheekbone = landmarks[LANDMARKS['left_cheekbone']]
    right_cheekbone = landmarks[LANDMARKS['right_cheekbone']]
    left_jaw = landmarks[LANDMARKS['left_jaw']]
    right_jaw = landmarks[LANDMARKS['right_jaw']]
    left_temple = landmarks[LANDMARKS['left_temple']]
    right_temple = landmarks[LANDMARKS['right_temple']]

    # Calculate key measurements
    face_length = calculate_distance(forehead, chin)
    cheekbone_width = calculate_distance(left_cheekbone, right_cheekbone)
    jaw_width = calculate_distance(left_jaw, right_jaw)
    forehead_width = calculate_distance(left_temple, right_temple)

    # Calculate normalized ratios (0-1 range)
    length_to_cheekbone = face_length / cheekbone_width
    jaw_to_forehead = jaw_width / forehead_width
    cheekbone_to_jaw = cheekbone_width / jaw_width

    # Debug output
    print(f"\n[DEBUG] Ratios:")
    print(f"Length/Cheekbone: {length_to_cheekbone:.2f}")
    print(f"Jaw/Forehead: {jaw_to_forehead:.2f}")
    print(f"Cheekbone/Jaw: {cheekbone_to_jaw:.2f}")

    # Classification logic with improved thresholds
    if length_to_cheekbone > 1.35:
        if jaw_to_forehead < 0.85:
            shape = "Heart"
        elif jaw_to_forehead > 1.15:
            shape = "Triangle"
        else:
            shape = "Oblong"
    elif length_to_cheekbone < 1.1:
        if abs(cheekbone_width - jaw_width) < 0.1 * face_length:
            shape = "Round"
        else:
            shape = "Square"
    elif cheekbone_to_jaw > 1.15 and cheekbone_width > forehead_width * 1.1:
        shape = "Diamond"
    elif abs(cheekbone_width - jaw_width) < 0.08 * face_length and abs(
            cheekbone_width - forehead_width) < 0.08 * face_length:
        shape = "Square"
    else:
        shape = "Oval"

    # Add to history for smoothing
    SHAPE_HISTORY.append(shape)

    # Return the most frequent shape in history
    return max(set(SHAPE_HISTORY), key=SHAPE_HISTORY.count)


def main():
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb_frame)

            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    h, w, _ = frame.shape
                    landmarks = {}

                    # Extract landmarks
                    for name, idx in LANDMARKS.items():
                        lm = face_landmarks.landmark[idx]
                        landmarks[idx] = (int(lm.x * w), int(lm.y * h))

                    # Draw landmarks and measurements
                    for pt in landmarks.values():
                        cv2.circle(frame, pt, 3, (0, 0, 255), -1)

                    # Draw key facial lines
                    cv2.line(frame, landmarks[LANDMARKS['forehead']], landmarks[LANDMARKS['chin']], (255, 0, 0), 2)
                    cv2.line(frame, landmarks[LANDMARKS['left_cheekbone']], landmarks[LANDMARKS['right_cheekbone']],
                             (0, 255, 0), 2)
                    cv2.line(frame, landmarks[LANDMARKS['left_jaw']], landmarks[LANDMARKS['right_jaw']], (0, 255, 255),
                             2)
                    cv2.line(frame, landmarks[LANDMARKS['left_temple']], landmarks[LANDMARKS['right_temple']],
                             (255, 0, 255), 2)

                    # Classify and display
                    if len(landmarks) == len(LANDMARKS):
                        shape = classify_face_shape(landmarks)
                        cv2.putText(frame, f"Face Shape: {shape}", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Draw face mesh
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style()
                    )

            cv2.imshow("Face Shape Detection", frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()