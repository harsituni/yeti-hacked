"""
ASL Data Collection Script

Captures hand landmarks from webcam using MediaPipe Tasks API.
- Letter mode: Press a letter key (a-z) to record the current hand pose for that letter.
- Phrase mode: Use --phrase "hello" (or enter at startup). Press Space to record with that label.
Press 'q' to quit.
"""

import argparse
import csv
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe Hands gives 21 landmarks per hand, each with x, y, z
NUM_LANDMARKS = 21
DIMENSIONS = 3
FEATURES_PER_HAND = NUM_LANDMARKS * DIMENSIONS  # 63

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def get_model_path():
    """Download hand landmarker model if needed; return local path."""
    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "hand_landmarker.task"
    if not model_path.exists():
        print("Downloading hand landmarker model (first run only)...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("Download complete.")
    return str(model_path)


def get_landmark_columns():
    """Generate column names for CSV: label, then x0,y0,z0, x1,y1,z1, ..."""
    columns = ["label"]
    for i in range(NUM_LANDMARKS):
        for d in ["x", "y", "z"]:
            columns.append(f"{d}{i}")
    return columns


def extract_hand_features(hand_landmarks):
    """Flatten hand landmarks to a list of 63 values (21 * x,y,z).
    hand_landmarks: list of NormalizedLandmark from HandLandmarker result.
    """
    features = []
    for lm in hand_landmarks:
        features.extend([lm.x, lm.y, lm.z])
    return features


def parse_args():
    parser = argparse.ArgumentParser(description="ASL hand pose data collection")
    parser.add_argument(
        "--phrase", "-p",
        type=str,
        default=None,
        help="Label for recordings (e.g. 'hello', 'thank you'). Press Space to record. Omit for letter mode (a-z).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    phrase_label = args.phrase

    # If no phrase from CLI, prompt for optional phrase/word
    if phrase_label is None:
        user_input = input("Enter label (letter or phrase, or press Enter for letter mode): ").strip()
        phrase_label = user_input if user_input else None

    # Create data directory
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "asl_data.csv"

    # Initialize MediaPipe HandLandmarker (Tasks API)
    model_path = get_model_path()
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Drawing utilities
    mp_drawing = mp.tasks.vision.drawing_utils
    mp_drawing_styles = mp.tasks.vision.drawing_styles
    hand_connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create or append to CSV
    file_exists = csv_path.exists()
    columns = get_landmark_columns()

    print("=" * 50)
    print("ASL Data Collection")
    print("=" * 50)
    if phrase_label:
        print(f"Phrase mode: label = '{phrase_label}'")
        print("Press SPACE to record | 'q' to quit")
    else:
        print("Letter mode: Press a letter key (a-z) to record hand pose")
        print("Press 'q' to quit")
    print("=" * 50)

    frame_count = 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(columns)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Ensure contiguous array for MediaPipe
            rgb_frame = rgb_frame.copy() if not rgb_frame.flags["C_CONTIGUOUS"] else rgb_frame

            # Timestamp in ms for video mode
            frame_timestamp_ms = int(frame_count * 1000 / 30)  # Assume ~30 fps
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
            frame_count += 1

            # Draw hand landmarks if detected
            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        hand_connections,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # Show instructions on frame
            if phrase_label:
                cv2.putText(
                    frame,
                    f"Label: '{phrase_label}' | SPACE to record | q to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Press letter (a-z) to record | q to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("ASL Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            # Determine label and whether to record
            record_label = None
            if phrase_label:
                if key == ord(" "):  # Space bar
                    record_label = phrase_label
            else:
                if 97 <= key <= 122:  # a-z
                    record_label = chr(key)

            if record_label is not None:
                if detection_result.hand_landmarks:
                    features = extract_hand_features(detection_result.hand_landmarks[0])
                    row = [record_label] + features
                    writer.writerow(row)
                    f.flush()
                    print(f"Recorded: {record_label}")
                else:
                    print("No hand detected - try again when hand is visible")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nData saved to {csv_path}")
    if csv_path.exists():
        with open(csv_path) as f:
            lines = sum(1 for _ in f) - 1  # minus header
        print(f"Total samples: {lines}")


if __name__ == "__main__":
    main()
