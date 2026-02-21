"""
ASL Dual Live Inference Script 

Runs both the letter-level model and the word-level model on a live webcam feed. 
Displays predicted letter and word with confidence.
Press 'q' to quit.
"""

import argparse
import threading
import time
import urllib.request
from pathlib import Path

import collections
import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from tensorflow import keras

# Match data_collection.py structure
NUM_LANDMARKS = 21
DIMENSIONS = 3
FEATURES_PER_HAND = NUM_LANDMARKS * DIMENSIONS  # 63

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
    parser = argparse.ArgumentParser(description="ASL dual model live inference")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Default: 0")
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.85,
        help="Confidence threshold for TTS (0-1). Default: 0.85",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable text-to-speech",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(__file__).parent / "models"
    
    # Paths for Letter Model
    letter_model_path = model_dir / "letter_model.keras"
    letter_scaler_path = model_dir / "letter_scaler.joblib"
    letter_labels_path = model_dir / "letter_label_encoder.joblib"
    
    # Paths for Word Model
    word_model_path = model_dir / "word_model.keras"
    word_scaler_path = model_dir / "word_scaler.joblib"
    word_labels_path = model_dir / "word_label_encoder.joblib"

    if not letter_model_path.exists() or not word_model_path.exists():
        print(f"Error: Missing models in {model_dir}")
        print("Run `train_model.py --type letter` and `train_model.py --type word` first.")
        return

    print("Loading models and artifacts...")
    letter_model = keras.models.load_model(letter_model_path)
    letter_scaler = joblib.load(letter_scaler_path)
    letter_encoder = joblib.load(letter_labels_path)
    
    word_model = keras.models.load_model(word_model_path)
    word_scaler = joblib.load(word_scaler_path)
    word_encoder = joblib.load(word_labels_path)

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

    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
    from mediapipe.python.solutions import hands as mp_hands
    hand_connections = mp_hands.HAND_CONNECTIONS

    # TTS Setup
    tts_engine = None
    if not args.no_tts:
        try:
            import pyttsx3
            tts_engine = pyttsx3.init()
            tts_engine.setProperty("rate", 120)
            print("Text-to-speech enabled.")
        except Exception as e:
            print(f"TTS unavailable ({e}). Run with --no-tts to suppress.")
            
    # TTS debouncing logic
    last_spoken_label = None
    last_spoken_time = 0.0
    tts_cooldown_sec = 2.0
    tts_lock = threading.Lock()
    tts_busy = [False]

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

    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("=" * 50)
    print("ASL Dual Live Inference (Letters & Words)")
    print("=" * 50)
    print(f"Confidence threshold for TTS: {args.threshold:.0%}")
    print("Press 'q' to quit")
    print("=" * 50)

    frame_count = 0
    letter_label = "—"
    letter_conf = 0.0
    word_label = "—"
    word_conf = 0.0

    # Rolling window for the LSTM word model (20 frames of 63 features)
    sequence_length = 20
    frame_buffer = collections.deque(maxlen=sequence_length)

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
                for hand_landmarks_data in detection_result.hand_landmarks:
                    # Convert to legacy proto message format for drawing
                    proto_landmarks = landmark_pb2.NormalizedLandmarkList()
                    proto_landmarks.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
                        for lm in hand_landmarks_data
                    ])
                    
                    mp_drawing.draw_landmarks(
                        frame,
                        proto_landmarks,
                        hand_connections,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                # Extract features for prediction
                features = extract_hand_features(detection_result.hand_landmarks[0])
                frame_buffer.append(features)
                
                # Letter Prediction Pipeline (Static - 1 frame)
                l_features = letter_scaler.transform([features])
                l_preds = letter_model.predict(l_features, verbose=0)[0]
                l_idx = np.argmax(l_preds)
                letter_conf = l_preds[l_idx]
                letter_label = letter_encoder.inverse_transform([l_idx])[0]
                
                # Word Prediction Pipeline (Dynamic - 20 frames)
                if len(frame_buffer) == sequence_length:
                    # The scaler expects 63 features per row. Scaler transforms the (20, 63) array.
                    sequence_array = np.array(frame_buffer)
                    w_features_scaled = word_scaler.transform(sequence_array)
                    # Reshape to 3D for LSTM: (1 sample, 20 timesteps, 63 features)
                    w_features = w_features_scaled.reshape(1, sequence_length, FEATURES_PER_HAND)
                    
                    w_preds = word_model.predict(w_features, verbose=0)[0]
                    w_idx = np.argmax(w_preds)
                    word_conf = w_preds[w_idx]
                    word_label = word_encoder.inverse_transform([w_idx])[0]
                else:
                    word_label = "Buffering..."
                    word_conf = 0.0

                # --- TTS Trigger Logic ---
                if tts_engine and not tts_busy[0]:
                    best_label = None
                    # Prioritize words over letters if both are highly confident
                    if word_conf >= args.threshold:
                        best_label = word_label
                    elif letter_conf >= args.threshold:
                        best_label = letter_label
                    
                    if best_label:
                        current_time = time.time()
                        if best_label != last_spoken_label or (current_time - last_spoken_time) > tts_cooldown_sec:
                            last_spoken_label = best_label
                            last_spoken_time = current_time
                            threading.Thread(target=speak, args=(best_label,), daemon=True).start()

            # Display predictions on frame
            cv2.putText(
                frame,
                f"Letter: {letter_label} ({letter_conf:.0%})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if letter_conf > 0.7 else (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Word: {word_label} ({word_conf:.0%})",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0) if word_conf > 0.7 else (0, 0, 255),
                2,
            )

            cv2.imshow("ASL Live Dual Inference", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
