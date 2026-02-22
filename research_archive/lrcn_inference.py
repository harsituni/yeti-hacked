"""
LRCN Live Inference — Real-time ASL gesture recognition from webcam.

Uses a rolling 16-frame buffer fed through the LRCN model (MobileNetV3 + LSTM)
to predict words from hand gestures captured by the webcam.

Usage:
    python lrcn_inference.py
    python lrcn_inference.py --no-tts --threshold 0.5
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Suppress TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    parser = argparse.ArgumentParser(description="LRCN ASL Live Inference")
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='Confidence threshold for TTS')
    parser.add_argument('--no-tts', action='store_true', help='Disable TTS')
    parser.add_argument('--predict_every', type=int, default=8,
                        help='Run prediction every N frames (for speed)')
    args = parser.parse_args()
    
    model_dir = Path(__file__).parent / 'models'
    
    # Load model and config
    model_path = model_dir / 'lrcn_word_model.keras'
    encoder_path = model_dir / 'lrcn_label_encoder.joblib'
    config_path = model_dir / 'lrcn_config.json'
    
    for p in [model_path, encoder_path, config_path]:
        if not p.exists():
            print(f"Error: {p} not found. Run train_lrcn.py first.")
            return
    
    print("Loading LRCN model...")
    model = keras.models.load_model(model_path)
    label_encoder = joblib.load(encoder_path)
    
    with open(config_path) as f:
        config = json.load(f)
    
    seq_len = config['sequence_length']
    img_size = config['image_size']
    class_names = config['class_names']
    
    print(f"Model loaded: {len(class_names)} word classes")
    print(f"Sequence: {seq_len} frames at {img_size}x{img_size}")
    
    # TTS setup
    use_tts = not args.no_tts
    last_spoken = None
    last_spoken_time = 0.0
    tts_cooldown = 3.0
    
    # Webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("=" * 50)
    print("LRCN ASL Live Gesture Inference")
    print("=" * 50)
    print(f"Confidence threshold: {args.threshold:.0%}")
    print(f"Predicting every {args.predict_every} frames")
    print("Press 'q' to quit")
    print("=" * 50)
    
    # Rolling frame buffer
    frame_buffer = []
    frame_count = 0
    
    word_label = "—"
    word_conf = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Preprocess frame for model
            resized = cv2.resize(frame, (img_size, img_size))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            
            frame_buffer.append(normalized)
            if len(frame_buffer) > seq_len:
                frame_buffer.pop(0)
            
            frame_count += 1
            
            # Only predict when buffer is full and at interval
            if len(frame_buffer) == seq_len and frame_count % args.predict_every == 0:
                sequence = np.array(frame_buffer)
                sequence = np.expand_dims(sequence, axis=0)  # (1, seq_len, H, W, 3)
                
                preds = model.predict(sequence, verbose=0)[0]
                idx = np.argmax(preds)
                word_conf = preds[idx]
                word_label = label_encoder.inverse_transform([idx])[0]
                
                # TTS
                if use_tts and word_conf >= args.threshold:
                    now = time.time()
                    if word_label != last_spoken or (now - last_spoken_time) > tts_cooldown:
                        last_spoken = word_label
                        last_spoken_time = now
                        subprocess.Popen(["say", word_label])
            
            # Display
            if len(frame_buffer) < seq_len:
                status = f"Buffering... ({len(frame_buffer)}/{seq_len})"
                color = (128, 128, 128)
            else:
                status = f"Word: {word_label} ({word_conf:.0%})"
                color = (0, 255, 0) if word_conf > 0.5 else (0, 165, 255)
            
            cv2.putText(display_frame, status, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Show buffer indicator
            buf_pct = len(frame_buffer) / seq_len
            bar_width = int(200 * buf_pct)
            cv2.rectangle(display_frame, (10, 60), (10 + bar_width, 70), (0, 255, 0), -1)
            cv2.rectangle(display_frame, (10, 60), (210, 70), (255, 255, 255), 1)
            
            cv2.imshow("LRCN ASL Gesture Recognition", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Inference stopped.")


if __name__ == '__main__':
    main()
