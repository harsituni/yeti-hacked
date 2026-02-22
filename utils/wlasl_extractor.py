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


def normalize_features(landmarks_list):
    """Apply wrist-relative, scale-invariant normalization to a 63-value feature list."""
    if len(landmarks_list) != 63:
        return None
    wrist_x, wrist_y, wrist_z = landmarks_list[0], landmarks_list[1], landmarks_list[2]
    
    max_dist = 0.0
    for j in range(21):
        lx = landmarks_list[j*3]
        ly = landmarks_list[j*3 + 1]
        dist = ((lx - wrist_x)**2 + (ly - wrist_y)**2)**0.5
        if dist > max_dist:
            max_dist = dist
    if max_dist == 0:
        max_dist = 1.0
    
    normalized = []
    for j in range(21):
        normalized.append((landmarks_list[j*3] - wrist_x) / max_dist)
        normalized.append((landmarks_list[j*3 + 1] - wrist_y) / max_dist)
        normalized.append((landmarks_list[j*3 + 2] - wrist_z) / max_dist)
    return normalized


def extract_static_frames(video_path, detector, bbox=None, samples_per_video=5):
    """
    Extract multiple individual static frames from a video.
    Returns a list of 63-value feature lists (one per successfully detected frame).
    Each frame is independently normalized (wrist-relative, scale-invariant).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
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
        return []

    # Sample evenly spaced frames
    indices = np.linspace(0, len(frames) - 1, samples_per_video, dtype=int)
    
    results = []
    for idx in indices:
        rgb_frame = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        try:
            detection_result = detector.detect(mp_image)
            if detection_result.hand_landmarks:
                raw = []
                for lm in detection_result.hand_landmarks[0]:
                    raw.extend([lm.x, lm.y, lm.z])
                normalized = normalize_features(raw)
                if normalized:
                    results.append(normalized)
        except Exception:
            pass
    
    return results


def extract_wlasl_features(dataset_dir, output_dir, max_words=25):
    """Legacy sequence extractor (kept for backward compatibility)."""
    return extract_wlasl_static(dataset_dir, output_dir, max_words)


def extract_wlasl_static(dataset_dir, output_dir, max_words=100):
    """
    Extract STATIC single-frame features from the WLASL dataset.
    Each row = label + 63 normalized features (same format as letter data).
    Samples multiple frames per video to maximize training data.
    """
    dataset_path = Path(dataset_dir)
    json_path = dataset_path / "WLASL_v0.3.json"
    missing_txt_path = dataset_path / "missing.txt"
    videos_dir = dataset_path / "videos"
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "wlasl_data_static.csv"
    
    print("Loading missing videos list...")
    missing_videos = set()
    if missing_txt_path.exists():
        with open(missing_txt_path, 'r') as f:
            missing_videos = set([line.strip() for line in f.readlines()])
    
    print("Loading JSON annotations...")
    with open(json_path, 'r') as f:
        wlasl_data = json.load(f)
        
    wlasl_data.sort(key=lambda x: len(x['instances']), reverse=True)
    target_words = wlasl_data[:max_words]
    
    print(f"Targeting top {max_words} words (static single-frame mode)...")
    detector = get_detector()
    
    total_processed = 0
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # 63-feature header (same as letter data)
        columns = ["label"]
        for j in range(21):
            columns.extend([f"x{j}", f"y{j}", f"z{j}"])
        writer.writerow(columns)
        
        for word_data in target_words:
            word = word_data['gloss']
            instances = word_data['instances']
            print(f"Processing word: '{word}' ({len(instances)} potential videos)")
            
            frame_count = 0
            for instance in instances:
                vid_id = instance['video_id']
                if vid_id in missing_videos:
                    continue
                    
                vid_file = videos_dir / f"{vid_id}.mp4"
                if not vid_file.exists():
                    continue
                    
                bbox = instance.get('bbox')
                static_samples = extract_static_frames(vid_file, detector, bbox, samples_per_video=5)
                for sample in static_samples:
                    writer.writerow([word] + sample)
                    frame_count += 1
                    total_processed += 1
            
            print(f" -> Extracted {frame_count} static frames.")
            
    print(f"Total static samples extracted: {total_processed}")
    return csv_path
