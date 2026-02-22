import json
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import kagglehub

# Download latest version
path = kagglehub.dataset_download("risangbaskoro/wlasl-processed")

print("Path to dataset files:", path)

# 1. Setup the Options
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# 2. To get landmarks from a frame in your loop:
#image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
#detection_result = detector.detect(image)
import yt_dlp
import csv
from pathlib import Path

import kagglehub

# --- CONFIGURATION ---
TARGET_SIGNS = ["hello", "thank you", "book", "drink"] # WLASL glosses
LIMIT_PER_SIGN = 100 # How many videos per sign to process
OUTPUT_CSV = "asl-harsit/data/asl_data_auto.csv"

# Download/Locate WLASL Dataset
print("Checking WLASL dataset...")
WLASL_PATH = Path(kagglehub.dataset_download("risangbaskoro/wlasl-processed"))
JSON_INPUT = WLASL_PATH / "WLASL_v0.3.json"
VIDEOS_DIR = WLASL_PATH / "videos"

# Initialize MediaPipe HandLandmarker (Tasks API)
model_path = Path(__file__).parent / "models" / "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=str(model_path))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO,
)
detector = vision.HandLandmarker.create_from_options(options)

# Global timestamp tracker to ensure strictly increasing timestamps across all calls
global_timestamp_ms = 0

def process_dataset():
    global global_timestamp_ms
    
    # Ensure data directory exists
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    
    if not JSON_INPUT.exists():
        print(f"Error: JSON input not found at {JSON_INPUT}")
        return

    with open(JSON_INPUT, 'r') as f:
        entries = json.load(f)

    # Count samples collected per sign
    counts = {sign: 0 for sign in TARGET_SIGNS}

    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.writer(f)

        for entry in entries:
            gloss = entry['gloss'].lower()
            
            # WLASL structure: entry has 'gloss' and 'instances'
            if gloss in TARGET_SIGNS and counts[gloss] < LIMIT_PER_SIGN:
                for instance in entry['instances']:
                    if counts[gloss] >= LIMIT_PER_SIGN:
                        break
                        
                    video_id = instance['video_id']
                    video_path = VIDEOS_DIR / f"{video_id}.mp4"
                    
                    if not video_path.exists():
                        print(f"Warning: Video file not found for {gloss} (ID: {video_id}) at {video_path}. Skipping.")
                        continue

                    print(f"Processing: {gloss} (ID: {video_id})")
                    
                    try:
                        cap = cv2.VideoCapture(str(video_path))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if fps <= 0: fps = 30 # Fallback
                        
                        frame_interval_ms = 1000 / fps
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret: break

                            # MediaPipe Processing (Tasks API)
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            rgb_frame = rgb_frame.copy() if not rgb_frame.flags["C_CONTIGUOUS"] else rgb_frame
                            
                            # Increment global timestamp to satisfy "strictly increasing" requirement
                            # Even if we switch videos, we keep increasing
                            global_timestamp_ms += int(frame_interval_ms)
                            
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                            detection_result = detector.detect_for_video(mp_image, global_timestamp_ms)

                            if detection_result.hand_landmarks:
                                landmarks = []
                                for lm in detection_result.hand_landmarks[0]:
                                    landmarks.extend([lm.x, lm.y, lm.z])
                                
                                writer.writerow([gloss] + landmarks)
                                f.flush()
                        
                        cap.release()
                        counts[gloss] += 1
                        print(f"Done with {video_id}. Progress for '{gloss}': {counts[gloss]}/{LIMIT_PER_SIGN}")
                    except Exception as e:
                        print(f"Skipping video {video_id} due to error: {e}")

if __name__ == "__main__":
    process_dataset()