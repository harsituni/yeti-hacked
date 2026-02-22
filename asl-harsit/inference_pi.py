import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
from pathlib import Path
import time
import os
import platform
import argparse
from collections import deque

# --- CUSTOM DRAWING LOGIC ---
# Standard hand landmark connections for visualization
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),    # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)          # Palm
]

def draw_landmarks(frame, hand_landmarks):
    """Draws a custom hand skeleton using OpenCV."""
    h, w, _ = frame.shape
    # Convert normalized landmarks to pixel coordinates
    points = []
    for lm in hand_landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        points.append((px, py))
        cv2.circle(frame, (px, py), 4, (0, 255, 0), -1) # Green dots

    # Draw connection lines
    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], (255, 255, 255), 2) # White lines

# --- CONFIGURATION ---
CONFIDENCE_THRESHOLD = 0.40 # Increased sensitivity (Lower threshold)
TTS_COOLDOWN_SEC = 2.5
STABILITY_WINDOW = 3 # Increased sensitivity (Fewer frames required)
SEQUENCE_LENGTH = 30 # Must match train_model.py
INPUT_DIM = 63 # 21 landmarks * (x,y,z)

# --- UTILS ---
def normalize_landmarks(landmarks_list):
    """Makes landmarks relative to wrist and scales to hand size."""
    points = np.array(landmarks_list).reshape(21, 3)
    wrist = points[0]
    relative_points = points - wrist
    scale = np.linalg.norm(relative_points[9])
    if scale > 0:
        relative_points = relative_points / scale
    return relative_points.flatten()

MODEL_DIR = Path(__file__).parent / "models"
MODEL_TASK_PATH = MODEL_DIR / "hand_landmarker.task"

# Performance/Visuals
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (0, 0, 255)

def say_text(text):
    """Speaks the text using the best available method."""
    print(f"SPEAKING: {text}")
    try:
        if platform.system() == "Darwin": # macOS
            os.system(f"say '{text}' &")
        else: # Linux (Pi) or Windows
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    # Load LSTM Model, Scaler, and Label Encoder
    try:
        model = keras.models.load_model(MODEL_DIR / "asl_model.keras")
        scaler = joblib.load(MODEL_DIR / "scaler.joblib")
        label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
        print(f"Loaded LSTM model with {len(label_encoder.classes_)} classes.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize Hand Landmarker
    base_options = python.BaseOptions(model_asset_path=str(MODEL_TASK_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera with index {args.camera}.")
        print("TIP: Try changing the camera index with --camera 1 or --camera 2.")
        return
    else:
        print(f"SUCCESS: Camera {args.camera} opened.")

    last_tts_time = 0
    
    # buffers
    sequence_data = deque(maxlen=SEQUENCE_LENGTH) # For LSTM motion window
    stability_buffer = deque(maxlen=STABILITY_WINDOW) # For result smoothing
    
    print("\nASL Temporal Inference Running...")
    print("Press 'q' to quit.")

    try:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                print("ERROR: Failed to capture frame from camera. Camera is active but not sending data.")
                print("ACTION: Make sure no other app (Zoom, Teams, etc.) is using your camera.")
                print("ACTION: Try running with a different index: python inference_pi.py --camera 1")
                break
            
            frame_count += 1
            if frame_count == 1:
                print("SUCCESS: Received first frame. Starting prediction loop...")

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Use timestamp for VIDEO mode
            timestamp_ms = int(time.time() * 1000)
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)

            current_label = "Scanning..."
            confidence = 0

            hand_detected = False
            if detection_result.hand_landmarks:
                hand_detected = True
                # 1. Extract Landmarks
                lms_raw = []
                for lm in detection_result.hand_landmarks[0]:
                    lms_raw.extend([lm.x, lm.y, lm.z])
                
                # 2. Normalize (Wrist-Relative + Scale)
                lms = normalize_landmarks(lms_raw)
                
                # 3. Add to sequence window
                sequence_data.append(lms)
                
                # 4. If window is full, predict
                if len(sequence_data) == SEQUENCE_LENGTH:
                    input_data = np.array(sequence_data)
                    # Scale (reshape to flat for scaler, then back to 3D)
                    input_flat = input_data.reshape(-1, 63)
                    input_scaled = scaler.transform(input_flat)
                    input_lstm = input_scaled.reshape(1, SEQUENCE_LENGTH, 63)
                    
                    predictions = model.predict(input_lstm, verbose=0)
                    pred_idx = np.argmax(predictions[0])
                    confidence = predictions[0][pred_idx]
                    
                    if confidence > args.threshold:
                        label = label_encoder.inverse_transform([pred_idx])[0]
                        stability_buffer.append(label)
                    else:
                        stability_buffer.append(None)

                # Check stability buffer for TTS (Now INSIDE hand detection block)
                most_common = None
                if len(stability_buffer) == STABILITY_WINDOW:
                    counts = {}
                    for l in stability_buffer:
                        if l: counts[l] = counts.get(l, 0) + 1
                    if counts:
                        best_l = max(counts, key=counts.get)
                        if counts[best_l] >= STABILITY_WINDOW - 1: # High consistency
                            most_common = best_l
                            current_label = best_l

                # TTS Trigger (Now INSIDE hand detection block)
                if most_common and not args.no_tts:
                    if time.time() - last_tts_time > TTS_COOLDOWN_SEC:
                        say_text(most_common)
                        last_tts_time = time.time()

                # Visual Feedback: Draw the full hand skeleton
                for hand_landmarks in detection_result.hand_landmarks:
                    draw_landmarks(frame, hand_landmarks)

            else:
                hand_detected = False
                # IMPORTANT: Reset buffers when hand is lost to prevent "jumping" or phantom words
                sequence_data.clear()
                stability_buffer.clear()

            # --- DEBUG UI OVERLAY ---
            # 1. Main Result
            cv2.putText(frame, f"Sign: {current_label}", (10, 40), FONT, 1, COLOR_GREEN, 2)
            
            # 2. Hand Status
            status_color = COLOR_GREEN if hand_detected else COLOR_RED
            status_text = "Hand: OK" if hand_detected else "Hand: MISSING"
            cv2.putText(frame, status_text, (10, 75), FONT, 0.7, status_color, 2)
            
            # 3. Buffer Progress
            buf_len = len(sequence_data)
            cv2.putText(frame, f"Buffer: {buf_len}/{SEQUENCE_LENGTH}", (10, 105), FONT, 0.6, COLOR_WHITE, 1)
            
            # 4. Raw Confidence (Always visible for debugging)
            cv2.putText(frame, f"Raw Conf: {confidence:.2f}", (10, 135), FONT, 0.6, COLOR_WHITE, 1)
            
            # 5. Stability Lock-in
            if len(stability_buffer) > 0 and stability_buffer[-1] is not None:
                cv2.putText(frame, "LOCKING IN...", (10, 165), FONT, 0.6, COLOR_RED, 1)

            cv2.imshow("ASL Temporal Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Runtime Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
