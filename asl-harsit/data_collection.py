import argparse
import csv
import time
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
SEQUENCE_LENGTH = 30
MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"
OUTPUT_CSV = Path(__file__).parent / "data" / "asl_data_auto.csv"

def normalize_landmarks(landmarks_list):
    """Makes landmarks relative to the wrist (index 0) and scales them."""
    points = np.array(landmarks_list).reshape(21, 3)
    wrist = points[0]
    relative_points = points - wrist
    scale = np.linalg.norm(relative_points[9])
    if scale > 0:
        relative_points = relative_points / scale
    return relative_points.flatten().tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phrase", "-p", type=str, default=None)
    args = parser.parse_args()
    
    label = args.phrase
    if label is None:
        label = input("Enter label (letter or word): ").strip().lower()

    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    
    # Initialize MediaPipe
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    
    recording_counter = 0
    current_sequence = []

    print(f"\nTarget Label: {label}")
    print("Commands:")
    print("  's' - Record 1-second dynamic movement (30 frames)")
    print("  'l' - Record static letter pose (simulates 30 frames)")
    print("  'n' - Change to a NEW label")
    print("  'q' - Quit")

    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            timestamp_ms = int(time.time() * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)
            
            # --- UI FEEDBACK ---
            status = "IDLE"
            color = (0, 255, 0)
            
            if recording_counter > 0:
                status = f"RECORDING... {recording_counter}"
                color = (0, 0, 255) # Red for recording
                
                if result.hand_landmarks:
                    lms = []
                    for lm in result.hand_landmarks[0]:
                        lms.extend([lm.x, lm.y, lm.z])
                    current_sequence.append(normalize_landmarks(lms))
                    recording_counter -= 1
                    
                    if recording_counter == 0:
                        if len(current_sequence) == SEQUENCE_LENGTH:
                            flat_seq = [item for sublist in current_sequence for item in sublist]
                            writer.writerow([label] + flat_seq)
                            f.flush()
                            print(f"Captured dynamic sequence for: {label}")
                        current_sequence = []
                else:
                    print("Warning: Hand lost during recording. Restarting...")
                    recording_counter = 0
                    current_sequence = []

            # Draw UI
            cv2.putText(frame, f"Label: {label} | Status: {status}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, "'s':Record | 'l':Static | 'n':New Label | 'q':Quit", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if result.hand_landmarks:
                # Custom simple drawing to avoid mp_drawing bloat
                for lm in result.hand_landmarks[0]:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            cv2.imshow("Personal ASL Collector", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'): # Change label
                print("\n[PAUSED] Switch to your terminal to enter new label.")
                new_label = input("Enter next label (letter or word): ").strip().lower()
                if new_label:
                    label = new_label
                    print(f"Label updated to: {label}")
            elif key == ord('s') and recording_counter == 0:
                recording_counter = SEQUENCE_LENGTH
                current_sequence = []
            elif key == ord('l'): # Static letter capture
                if result.hand_landmarks:
                    lms = []
                    for lm in result.hand_landmarks[0]:
                        lms.extend([lm.x, lm.y, lm.z])
                    norm_lms = normalize_landmarks(lms)
                    # Duplicate to fill 30-frame sequence
                    flat_seq = norm_lms * SEQUENCE_LENGTH
                    writer.writerow([label] + flat_seq)
                    f.flush()
                    print(f"Captured static pose for: {label}")
                else:
                    print("Error: No hand detected!")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
