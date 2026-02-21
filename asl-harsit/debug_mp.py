import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

# Pick one of the videos that failed (e.g., 'book' video 69241 or 65225)
VIDEO_ID = "69241"
VIDEO_PATH = f"/Users/dhirpatel/.cache/kagglehub/datasets/risangbaskoro/wlasl-processed/versions/5/videos/{VIDEO_ID}.mp4"

model_path = Path(__file__).parent / "models" / "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=str(model_path))

# Try lowering confidence thresholds
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.1,  # Lowered from 0.5
    min_hand_presence_confidence=0.1,   # Lowered from 0.5
    min_tracking_confidence=0.1,        # Lowered from 0.5
    running_mode=vision.RunningMode.VIDEO,
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Failed to open {VIDEO_PATH}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Testing video {VIDEO_ID} at {fps} FPS")

frame_count = 0
found_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(frame_count * 1000 / (fps if fps > 0 else 30))
    
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)
    
    if detection_result.hand_landmarks:
        found_count += 1
        
    frame_count += 1

print(f"Result: Found hands in {found_count} out of {frame_count} frames.")
cap.release()
