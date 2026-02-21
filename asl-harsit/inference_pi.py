"""
ASL Live Inference Script (Raspberry Pi)

Runs the trained model on live webcam feed. Displays predicted letter/phrase
with confidence. Uses text-to-speech to announce when confidence exceeds threshold.
Press 'q' to quit.
"""

import argparse
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow import keras

# Match data_collection.py structure
NUM_LANDMARKS = 21
DIMENSIONS = 3
FEATURES_PER_HAND = NUM_LANDMARKS * DIMENSIONS  # 63

# Confidence threshold to trigger TTS (0-1)
CONFIDENCE_THRESHOLD = 0.85
# Minimum seconds between speaking the same label again
TTS_COOLDOWN_SEC = 2.0

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def get_hand_model_path():
    """Download hand landmarker model if needed; return local path."""
    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "hand_landmarker.task"
    if not model_path.exists():
        print("Downloading hand landmarker model (first run only)...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("Download complete.")
    return str(model_path)


def extract_hand_features(hand_landmarks):
    """Flatten hand landmarks to a list of 63 values (21 * x,y,z)."""
    features = []
    for lm in hand_landmarks:
        features.extend([lm.x, lm.y, lm.z])
    return features


def parse_args():
    parser = argparse.ArgumentParser(description="ASL live inference with TTS")
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for TTS (0-1). Default: {CONFIDENCE_THRESHOLD}",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable text-to-speech (e.g. for headless Pi)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index. Default: 0",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(__file__).parent / "models"
    model_path = model_dir / "asl_model.keras"
    scaler_path = model_dir / "scaler.joblib"
    labels_path = model_dir / "label_encoder.joblib"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Run train_model.py first to train the model.")
        return

    print("Loading model and artifacts...")
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(labels_path)
    print(f"Classes: {list(label_encoder.classes_)}")

    # Initialize MediaPipe HandLandmarker
    hand_model_path = get_hand_model_path()
    base_options = python.BaseOptions(model_asset_path=hand_model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        running_mode=vision.RunningMode.VIDEO,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hand_connections = mp.solutions.hands.HAND_CONNECTIONS

    # TTS
    tts_engine = None
    if not args.no_tts:
        try:
            import pyttsx3
            tts_engine = pyttsx3.init()
            tts_engine.setProperty("rate", 120)
            print("Text-to-speech enabled.")
        except Exception as e:
            print(f"TTS unavailable ({e}). Run with --no-tts to suppress.")

    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # TTS debouncing
    last_spoken_label = None
    last_spoken_time = 0.0
    tts_lock = threading.Lock()
    tts_busy = [False]  # list so we can mutate from nested function

    def speak(text):
        with tts_lock:
            if tts_busy[0]:
                return
            tts_busy[0] = True
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        finally:
            with tts_lock:
                tts_busy[0] = False

    print("=" * 50)
    print("ASL Live Inference")
    print("=" * 50)
    print(f"Confidence threshold for TTS: {args.threshold:.0%}")
    print("Press 'q' to quit")
    print("=" * 50)

    frame_count = 0
    current_label = "—"
    current_confidence = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = rgb_frame.copy() if not rgb_frame.flags["C_CONTIGUOUS"] else rgb_frame

            # Hand detection
            frame_timestamp_ms = int(frame_count * 1000 / 30)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
            frame_count += 1

            if detection_result.hand_landmarks:
                # The tasks API returns a list of NormalizedLandmark lists, one per hand.
                # mp_drawing expects a NormalizedLandmarkList message.
                from mediapipe.framework.formats import landmark_pb2
                
                for hand_landmarks in detection_result.hand_landmarks:
                    # Convert the list of python objects back to the proto message format
                    # that the legacy drawing solution expects
                    proto_landmarks = landmark_pb2.NormalizedLandmarkList()
                    proto_landmarks.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
                        for lm in hand_landmarks
                    ])
                    
                    mp_drawing.draw_landmarks(
                        frame,
                        proto_landmarks,
                        hand_connections,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                # Run inference
                features = extract_hand_features(detection_result.hand_landmarks[0])
                X = scaler.transform([features])
                probs = model.predict(X, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                current_label = label_encoder.inverse_transform([pred_idx])[0]
                current_confidence = float(probs[pred_idx])

                # TTS when confidence exceeds threshold (runs in thread so video doesn't freeze)
                if (
                    tts_engine
                    and not tts_busy[0]
                    and current_confidence >= args.threshold
                    and (current_label != last_spoken_label or time.time() - last_spoken_time > TTS_COOLDOWN_SEC)
                ):
                    last_spoken_label = current_label
                    last_spoken_time = time.time()
                    threading.Thread(target=speak, args=(current_label,), daemon=True).start()
            else:
                current_label = "—"
                current_confidence = 0.0

            # Overlay prediction
            text = f"{current_label}  {current_confidence:.0%}"
            color = (0, 255, 0) if current_confidence >= args.threshold else (0, 255, 255)
            cv2.putText(
                frame,
                text,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                3,
            )
            cv2.putText(
                frame,
                f"Threshold: {args.threshold:.0%} | q to quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            cv2.imshow("ASL Inference", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
