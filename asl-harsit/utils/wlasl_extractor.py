import csv
import json
import os
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def get_detector():
    """Initialize MediaPipe HandLandmarker (Image Mode)"""
    model_path = Path(__file__).parent.parent / "models" / "hand_landmarker.task"
    if not model_path.exists():
        raise FileNotFoundError(f"Hand landmarker model not found at {model_path}")
        
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.1,
        min_tracking_confidence=0.1,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.HandLandmarker.create_from_options(options)

def process_video(video_path, detector, bbox=None, target_frames=20):
    """
    Process a single video and return a fixed-length sequence of hand landmarks.
    We uniformly sample `target_frames` across the video. If no hands are found
    in some frames, we use the last valid frame, or zero-pad if no hands were 
    found at all yet. Returns a flat list of (target_frames * 63) values.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    # First pass: collect all frames
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30 # fallback
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Crop to bounding box if provided
        if bbox and len(bbox) == 4:
            ymin, xmin, ymax, xmax = bbox
            h, w = frame.shape[:2]
            
            ymin, ymax = max(0, ymin), min(h, ymax)
            xmin, xmax = max(0, xmin), min(w, xmax)
            
            margin_y = int((ymax - ymin) * 0.2)
            margin_x = int((xmax - xmin) * 0.2)
            
            ymin = max(0, ymin - margin_y)
            ymax = min(h, ymax + margin_y)
            xmin = max(0, xmin - margin_x)
            xmax = min(w, xmax + margin_x)
            
            if ymax > ymin and xmax > xmin:
                frame = frame[ymin:ymax, xmin:xmax]
        
        frames.append(frame)
    cap.release()

    if not frames:
        return None

    # Step 2: Uniformly sample `target_frames` from the collected frames
    total_frames = len(frames)
    indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    
    sequence_features = []
    last_valid_landmarks = [0.0] * 63 # 21 landmarks * 3 coords
    
    for idx in indices:
        frame = frames[idx]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        try:
            detection_result = detector.detect(mp_image)
            if detection_result.hand_landmarks:
                landmarks = []
                # Grab the first hand
                for lm in detection_result.hand_landmarks[0]:
                    landmarks.extend([lm.x, lm.y, lm.z])
                last_valid_landmarks = landmarks
        except Exception:
            pass # ignore, fallback to last valid

        sequence_features.extend(last_valid_landmarks)

    # Validate we got exactly target_frames * 63 values
    expected_length = target_frames * 63
    if len(sequence_features) != expected_length:
        return None

    return sequence_features

def extract_wlasl_features(dataset_dir, output_dir, max_words=25):
    """
    Extract features from the WLASL dataset and save to CSV.
    Extract sequence features from the WLASL dataset and save to CSV.
    Each row will contain the label followed by (20 frames * 63 landmarks) = 1260 columns.
    """
    dataset_path = Path(dataset_dir)
    json_path = dataset_path / "WLASL_v0.3.json"
    missing_txt_path = dataset_path / "missing.txt"
    videos_dir = dataset_path / "videos"
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "wlasl_data_naive.csv"
    
    print("Loading missing videos list...")
    missing_videos = set()
    if missing_txt_path.exists():
        with open(missing_txt_path, 'r') as f:
            missing_videos = set([line.strip() for line in f.readlines()])
    
    print("Loading JSON annotations...")
    with open(json_path, 'r') as f:
        wlasl_data = json.load(f)
        
    # Sort by number of instances (videos) to get the most common words
    wlasl_data.sort(key=lambda x: len(x['instances']), reverse=True)
    target_words = wlasl_data[:max_words]
    
    print(f"Targeting top {max_words} words...")
    detector = get_detector()
    
    total_processed = 0
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write sequence header (1260 features)
        target_frames = 20
        columns = ["label"]
        for i in range(target_frames):
            columns.extend([f"f{i}_x{j}" for j in range(21)])
            columns.extend([f"f{i}_y{j}" for j in range(21)])
            columns.extend([f"f{i}_z{j}" for j in range(21)])
        writer.writerow(columns)
        
        for word_data in target_words:
            word = word_data['gloss']
            instances = word_data['instances']
            print(f"Processing word: '{word}' ({len(instances)} potential videos)")
            
            valid_count = 0
            for instance in instances:
                vid_id = instance['video_id']
                if vid_id in missing_videos:
                    continue
                    
                vid_file = videos_dir / f"{vid_id}.mp4"
                if not vid_file.exists():
                    continue
                    
                bbox = instance.get('bbox')
                landmarks = process_video(vid_file, detector, bbox)
                if landmarks:
                    writer.writerow([word] + landmarks)
                    valid_count += 1
                    total_processed += 1
            
            print(f" -> Found landmarks for {valid_count} videos.")
            
    return csv_path
