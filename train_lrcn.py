"""
LRCN (Long-term Recurrent Convolutional Network) Training Script.

Architecture: TimeDistributed(MobileNetV3Small) → LSTM(256) → Multi-task Heads
- Word Head: classifies the full gesture from the temporal sequence
- Pose Head: classifies the static hand shape from the middle frame's features

Usage:
    python train_lrcn.py --max_words 100 --epochs 30
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from utils.video_dataset import VideoDataGenerator, build_dataset_from_wlasl


# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def build_lrcn_model(num_classes, sequence_length=16, image_size=224):
    """
    Build LRCN: TimeDistributed CNN (MobileNetV3Small) + LSTM + Dense heads.
    
    The CNN backbone is frozen (ImageNet weights) to act as a feature extractor.
    Only the LSTM and classification heads are trained.
    """
    # --- CNN Backbone (frozen) ---
    backbone = keras.applications.MobileNetV3Small(
        include_top=False,
        weights='imagenet',
        input_shape=(image_size, image_size, 3),
        pooling='avg',  # GlobalAveragePooling → 576-dim vector per frame
    )
    # Fine-tune the top layers of the backbone
    backbone.trainable = True
    # Freeze the bottom layers, unfreeze the top 30
    for layer in backbone.layers[:-30]:
        layer.trainable = False
    
    # --- LRCN Architecture ---
    video_input = layers.Input(shape=(sequence_length, image_size, image_size, 3), name='video_input')
    
    # Apply CNN to each frame independently
    x = layers.TimeDistributed(backbone, name='cnn_features')(video_input)
    # x shape: (batch, sequence_length, 576)
    
    # Temporal modeling with LSTM
    x = layers.LSTM(256, return_sequences=False, name='temporal_lstm')(x)
    x = layers.Dropout(0.4)(x)
    
    # --- Word Classification Head (primary) ---
    word_out = layers.Dense(128, activation='relu', name='word_dense')(x)
    word_out = layers.Dropout(0.3)(word_out)
    word_out = layers.Dense(num_classes, activation='softmax', name='word_output')(word_out)
    
    model = keras.Model(inputs=video_input, outputs=word_out, name='LRCN_ASL')
    
    return model


def main():
    parser = argparse.ArgumentParser(description="LRCN ASL Gesture Recognition Training")
    parser.add_argument('--dataset_dir', type=str, 
                        default=os.path.expanduser('~/.cache/kagglehub/datasets/risangbaskoro/wlasl-processed/versions/5'),
                        help='Path to WLASL dataset')
    parser.add_argument('--max_words', type=int, default=20,
                        help='Number of top words to train on')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (keep small for CPU)')
    parser.add_argument('--sequence_length', type=int, default=16,
                        help='Number of frames per video sequence')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Frame resize dimension')
    args = parser.parse_args()
    
    model_dir = Path(__file__).parent / 'models'
    model_dir.mkdir(exist_ok=True)
    
    # --- Build Dataset ---
    print("=" * 60)
    print("LRCN ASL Gesture Recognition Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset_dir}")
    print(f"Max words: {args.max_words}")
    print(f"Sequence length: {args.sequence_length} frames")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print()
    
    video_paths, labels, class_names = build_dataset_from_wlasl(
        args.dataset_dir,
        max_words=args.max_words,
        min_videos=5,
    )
    
    num_classes = len(class_names)
    print(f"Found {len(video_paths)} videos across {num_classes} classes")
    print(f"Classes: {class_names[:20]}{'...' if num_classes > 20 else ''}")
    
    if num_classes < 2:
        print("Error: Need at least 2 classes to train.")
        return
    
    # --- Label Encoding ---
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    
    # Convert integer labels to class name labels for the encoder
    label_names = [class_names[i] for i in labels]
    encoded_labels = label_encoder.transform(label_names)
    
    # --- Train/Val Split ---
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_paths, encoded_labels, 
        test_size=0.2, 
        random_state=42,
        stratify=encoded_labels,
    )
    
    print(f"Train: {len(train_paths)} videos | Val: {len(val_paths)} videos")
    print()
    
    # --- Data Generators ---
    train_gen = VideoDataGenerator(
        train_paths, train_labels, num_classes,
        sequence_length=args.sequence_length,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        augment=True,
        shuffle=True,
    )
    
    val_gen = VideoDataGenerator(
        val_paths, val_labels, num_classes,
        sequence_length=args.sequence_length,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        augment=False,
        shuffle=False,
    )
    
    # --- Build Model ---
    print("Building LRCN model...")
    model = build_lrcn_model(
        num_classes=num_classes,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    model.summary()
    print()
    
    # --- Training ---
    print("Training...")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    
    # --- Evaluation ---
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)
    
    val_preds = []
    val_trues = []
    for i in range(len(val_gen)):
        X_batch, y_batch = val_gen[i]
        preds = model.predict(X_batch, verbose=0)
        val_preds.extend(np.argmax(preds, axis=1))
        val_trues.extend(y_batch)
    
    val_preds = np.array(val_preds)
    val_trues = np.array(val_trues)
    
    print(classification_report(
        label_encoder.inverse_transform(val_trues),
        label_encoder.inverse_transform(val_preds),
    ))
    
    # --- Save Model & Artifacts ---
    model_path = model_dir / 'lrcn_word_model.keras'
    encoder_path = model_dir / 'lrcn_label_encoder.joblib'
    
    model.save(model_path)
    joblib.dump(label_encoder, encoder_path)
    
    # Save config for inference
    config = {
        'sequence_length': args.sequence_length,
        'image_size': args.image_size,
        'num_classes': num_classes,
        'class_names': class_names,
    }
    config_path = model_dir / 'lrcn_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nModel saved to {model_path}")
    print(f"Label encoder saved to {encoder_path}")
    print(f"Config saved to {config_path}")
    
    best_val_acc = max(history.history.get('val_accuracy', [0]))
    print(f"\nBest validation accuracy: {best_val_acc:.2%}")


if __name__ == '__main__':
    main()
