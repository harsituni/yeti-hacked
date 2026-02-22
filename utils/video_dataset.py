"""
Video Dataset Generator for LRCN Training.

Loads WLASL videos on-the-fly, samples fixed-length frame sequences,
resizes to 224x224, and applies video augmentations.
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


class VideoDataGenerator(tf.keras.utils.Sequence):
    """
    Keras Sequence generator that loads video clips on-the-fly.
    
    Each sample is a fixed-length sequence of frames (e.g., 16 frames of 224x224x3).
    Videos are uniformly sampled to produce exactly `sequence_length` frames.
    """
    
    def __init__(
        self, 
        video_paths, 
        labels, 
        num_classes,
        sequence_length=16, 
        target_size=(224, 224),
        batch_size=8,
        augment=False,
        shuffle=True,
    ):
        self.video_paths = video_paths
        self.labels = labels
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(self.video_paths))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return max(1, len(self.video_paths) // self.batch_size)
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        batch_sequences = []
        batch_labels = []
        
        for i in batch_indices:
            sequence = self._load_video(self.video_paths[i])
            if sequence is not None:
                batch_sequences.append(sequence)
                batch_labels.append(self.labels[i])
        
        if not batch_sequences:
            # Fallback: return zeros if all videos failed
            batch_sequences = [np.zeros((self.sequence_length, *self.target_size, 3), dtype=np.float32)]
            batch_labels = [0]
        
        X = np.array(batch_sequences, dtype=np.float32)
        y = np.array(batch_labels, dtype=np.int32)
        
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _load_video(self, video_path):
        """Load a video and sample `sequence_length` frames uniformly."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if len(frames) < 4:
            return None
        
        # Uniform sampling of sequence_length frames
        total = len(frames)
        
        if self.augment:
            # Temporal jitter: slight random offset to sampling indices
            jitter = random.randint(-2, 2)
            indices = np.linspace(0, total - 1, self.sequence_length, dtype=float)
            indices = np.clip(indices + jitter, 0, total - 1).astype(int)
        else:
            indices = np.linspace(0, total - 1, self.sequence_length, dtype=int)
        
        # Augmentation params (consistent across all frames in sequence)
        do_flip = self.augment and random.random() > 0.5
        brightness_delta = random.uniform(-0.15, 0.15) if self.augment else 0.0
        
        # Random crop params
        if self.augment:
            crop_frac = random.uniform(0.85, 1.0)
        else:
            crop_frac = 1.0
        
        sequence = []
        for idx in indices:
            frame = frames[idx]
            
            # Random crop (same crop for all frames)
            if crop_frac < 1.0:
                h, w = frame.shape[:2]
                new_h, new_w = int(h * crop_frac), int(w * crop_frac)
                top = random.randint(0, h - new_h)
                left = random.randint(0, w - new_w)
                frame = frame[top:top+new_h, left:left+new_w]
            
            # Resize to target
            frame = cv2.resize(frame, self.target_size)
            
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Horizontal flip
            if do_flip:
                frame = np.fliplr(frame)
            
            # Normalize to [0, 1] and apply brightness
            frame = frame.astype(np.float32) / 255.0
            frame = np.clip(frame + brightness_delta, 0.0, 1.0)
            
            sequence.append(frame)
        
        return np.array(sequence)


def build_dataset_from_wlasl(dataset_dir, max_words=100, min_videos=5):
    """
    Parse WLASL JSON and return lists of (video_paths, labels, class_names).
    Only includes words with at least `min_videos` existing video files.
    """
    dataset_path = Path(dataset_dir)
    json_path = dataset_path / "WLASL_v0.3.json"
    missing_txt_path = dataset_path / "missing.txt"
    videos_dir = dataset_path / "videos"
    
    # Load missing videos list
    missing_videos = set()
    if missing_txt_path.exists():
        with open(missing_txt_path, 'r') as f:
            missing_videos = set(line.strip() for line in f.readlines())
    
    # Load annotations
    with open(json_path, 'r') as f:
        wlasl_data = json.load(f)
    
    # Sort by number of instances, take top words
    wlasl_data.sort(key=lambda x: len(x['instances']), reverse=True)
    
    video_paths = []
    labels = []
    class_names = []
    class_idx = 0
    
    for word_data in wlasl_data:
        if class_idx >= max_words:
            break
            
        word = word_data['gloss']
        found_videos = []
        
        for instance in word_data['instances']:
            vid_id = instance['video_id']
            if vid_id in missing_videos:
                continue
            vid_file = videos_dir / f"{vid_id}.mp4"
            if vid_file.exists():
                found_videos.append(str(vid_file))
        
        if len(found_videos) >= min_videos:
            video_paths.extend(found_videos)
            labels.extend([class_idx] * len(found_videos))
            class_names.append(word)
            class_idx += 1
    
    return video_paths, labels, class_names
