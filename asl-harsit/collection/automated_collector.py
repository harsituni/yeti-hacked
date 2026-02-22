import json
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import kagglehub
import os
from pathlib import Path
import csv
import numpy as np

# --- PATHS ---
OUTPUT_CSV = Path(__file__).parent.parent / "data" / "asl_data_auto.csv"
MODEL_PATH = Path(__file__).parent.parent / "models" / "hand_landmarker.task"

# --- CONFIGURATION ---
SEQUENCE_LENGTH = 30 # Number of frames per gesture sequence
# ASL Alphabet (Images)
ALPHABET_SIGNS = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ["del", "nothing", "space"]
LIMIT_ALPHABET = 50 # 50 samples per letter is plenty for normalized static signs

# WLASL (Videos)
# Optimized for WLASL-300 (Top 300 most common words)
WLASL_LIMIT_GLOSSES = 5 # Instant verification set (5 words)
LIMIT_WLASL = 3 # 3 videos per word for max speed

def get_alphabet_path():
    print("Locating ASL Alphabet dataset...")
    path = Path(kagglehub.dataset_download("grassknoted/asl-alphabet"))
    return path / "asl_alphabet_train" / "asl_alphabet_train"

def get_wlasl_path():
    print("Locating WLASL dataset...")
    path = Path(kagglehub.dataset_download("risangbaskoro/wlasl-processed"))
    return path / "WLASL_v0.3.json", path / "videos"

# Initialize MediaPipe HandLandmarker
base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE,
)
detector = vision.HandLandmarker.create_from_options(options)

def normalize_landmarks(landmarks_list):
    """
    Makes landmarks relative to the wrist (index 0) and 
    scales them so the hand size is consistent.
    Input: List of 21 landmarks [x, y, z, ...]
    """
    # 1. Reshape to points (21, 3)
    points = np.array(landmarks_list).reshape(21, 3)
    
    # 2. Subtract Wrist (index 0)
    wrist = points[0]
    relative_points = points - wrist
    
    # 3. Calculate Scale (Distance between wrist and middle finger base index 9)
    # This makes the hand 'size independent'
    scale = np.linalg.norm(relative_points[9])
    if scale > 0:
        relative_points = relative_points / scale
        
    return relative_points.flatten().tolist()

def get_existing_labels():
    """Returns a set of labels already present in the CSV."""
    if not Path(OUTPUT_CSV).exists():
        return set()
    try:
        with open(OUTPUT_CSV, 'r') as f:
            reader = csv.reader(f)
            return {row[0] for row in reader if row}
    except Exception:
        return set()

def process_alphabet(writer, existing_labels):
    train_dir = get_alphabet_path()
    if not train_dir.exists():
        print(f"Error: Alphabet directory not found at {train_dir}")
        return

    print(f"Alphabet processing (Sequences): {len(ALPHABET_SIGNS)} signs.")

    for label in ALPHABET_SIGNS:
        low_label = label.lower()
        if low_label in existing_labels:
            print(f"  Skipping Alphabet: {label} (already collected)")
            continue

        folder_path = train_dir / label
        if not folder_path.exists():
            continue

        print(f"  Folder: {label}")
        image_files = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files = image_files[:LIMIT_ALPHABET]
        
        count = 0
        for img_name in image_files:
            img_path = folder_path / img_name
            try:
                frame = cv2.imread(str(img_path))
                if frame is None: continue
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = detector.detect(mp_image)

                if detection_result.hand_landmarks:
                    raw_lms = []
                    for lm in detection_result.hand_landmarks[0]:
                        raw_lms.extend([lm.x, lm.y, lm.z])
                    
                    # Apply Relative + Scale Normalization
                    normalized_lms = normalize_landmarks(raw_lms)

                    # For static images, duplicate landmarks to fill the sequence
                    sequence = normalized_lms * SEQUENCE_LENGTH
                    writer.writerow([label.lower()] + sequence)
                    count += 1
            except Exception as e:
                pass
        print(f"  Captured {count} sequences for {label}")

def process_wlasl(writer, existing_labels):
    json_path, videos_dir = get_wlasl_path()
    if not json_path.exists():
        print(f"Error: WLASL JSON not found at {json_path}")
        return

    with open(json_path, 'r') as f:
        entries = json.load(f)

    print(f"WLASL processing (Sequences): Top {WLASL_LIMIT_GLOSSES} glosses.")

    for i, entry in enumerate(entries):
        if i >= WLASL_LIMIT_GLOSSES:
            break
            
        gloss = entry['gloss'].lower()
        if gloss in existing_labels:
            print(f"  Skipping WLASL: {gloss} (already collected)")
            continue

        print(f"  Word: {gloss}")
        vids_processed = 0
        for instance in entry['instances']:
            if vids_processed >= LIMIT_WLASL:
                break
                
            video_path = videos_dir / f"{instance['video_id']}.mp4"
            if not video_path.exists():
                continue

            cap = cv2.VideoCapture(str(video_path))
            video_landmarks = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = detector.detect(mp_image)

                if detection_result.hand_landmarks:
                    lms = []
                    for lm in detection_result.hand_landmarks[0]:
                        lms.extend([lm.x, lm.y, lm.z])
                    
                    # Apply Relative + Scale Normalization
                    lms = normalize_landmarks(lms)
                    video_landmarks.append(lms)
                # If no hand detected, we skip this frame but maintain sequence integrity by not adding it.
                # WLASL usually has the hand clear. If missing, we might have gaps.
            
            cap.release()
            
            # Process the collected video frames into a fixed-length sequence
            if len(video_landmarks) >= 10: # Minimum frames to be useful
                # Resample to SEQUENCE_LENGTH
                indices = np.linspace(0, len(video_landmarks)-1, SEQUENCE_LENGTH, dtype=int)
                final_sequence = []
                for idx in indices:
                    final_sequence.extend(video_landmarks[idx])
                
                writer.writerow([gloss] + final_sequence)
                vids_processed += 1
        
        print(f"  Processed {vids_processed} videos for {gloss}")

def main():
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    
    existing_labels = get_existing_labels()
    print(f"Resuming collection. {len(existing_labels)} signs already in CSV.")

    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        
        print("Starting Alphabet collection (Static Sequences)...")
        process_alphabet(writer, existing_labels)
        
        print("\nStarting WLASL collection (Motion Sequences)...")
        process_wlasl(writer, existing_labels)

    print(f"\nUnified sequence dataset saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()