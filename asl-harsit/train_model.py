"""
ASL Temporal Model Training Script (LSTM)

Loads sequence-based hand landmark data from CSVs, 
reshapes them into (samples, time_steps, features),
and trains a Recurrent Neural Network (LSTM).
"""

import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configuration
SEQUENCE_LENGTH = 30 # Must match automated_collector.py
NUM_LANDMARKS = 21
DIMENSIONS = 3
FEATURES_PER_FRAME = NUM_LANDMARKS * DIMENSIONS # 63
INPUT_DIM = FEATURES_PER_FRAME # Features per timestep

def load_data(csv_paths):
    """Load and merge ASL sequence data from multiple CSVs. Returns X, y, label_encoder."""
    dfs = []
    for path in csv_paths:
        print(f"  Reading {path.name}...")
        df = pd.read_csv(path, header=None)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)

    if len(df) < 5:
        raise ValueError(f"Not enough data ({len(df)} samples). Collect more.")

    y = df.iloc[:, 0].values   # Label column
    X = df.iloc[:, 1:].values  # Landmark columns (usually SEQUENCE_LENGTH * FEATURES_PER_FRAME)

    # Reshape X for LSTM: (samples, time_steps, features)
    # Ensure X has exactly expected columns
    expected_cols = SEQUENCE_LENGTH * FEATURES_PER_FRAME
    if X.shape[1] != expected_cols:
        print(f"Warning: Expected {expected_cols} feature columns, found {X.shape[1]}. Truncating/Padding.")
        if X.shape[1] > expected_cols:
            X = X[:, :expected_cols]
        else:
            X = np.pad(X, ((0,0), (0, expected_cols - X.shape[1])), mode='constant')

    X = X.reshape(-1, SEQUENCE_LENGTH, FEATURES_PER_FRAME)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder

def build_lstm_model(num_classes):
    """Build a Recurrent Neural Network (LSTM) for gesture recognition."""
    model = keras.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, INPUT_DIM)),
        # Using tanh (default) for LSTM to prevent exploding gradients/NaNs
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(128, return_sequences=False),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    parser = argparse.ArgumentParser(description="ASL Model Training")
    parser.add_argument(
        "--type", 
        type=str, 
        choices=["letter", "word"], 
        default="word",
        help="Type of model to train (affects input CSV and output model names)"
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated list of labels to exclude (e.g., 'del,nothing')"
    )
    args = parser.parse_args()
    
    exclude_list = [x.strip().lower() for x in args.exclude.split(",") if x.strip()]
    
    data_dir = Path(__file__).parent / "data"
    model_dir = Path(__file__).parent / "models"
    
    # Only load the main ASL sequence data (other CSVs have incompatible formats)
    csv_paths = [data_dir / "asl_data.csv"]
    csv_paths = [p for p in csv_paths if p.exists()]
    if not csv_paths:
        print(f"Error: asl_data.csv not found in {data_dir}")
        print("Run automated_collector.py first to collect gesture data.")
        return

    print(f"Loading sequence data from {len(csv_paths)} files...")
    X, y_raw, label_encoder = load_data(csv_paths)

    # Apply exclusions if any
    if exclude_list:
        print(f"Excluding labels: {exclude_list}")
        # Need to re-filter X and y_raw before encoding
        # However, load_data already encoded y. Let's fix load_data or handle it here.
        active_mask = ~np.isin(label_encoder.inverse_transform(y_raw), exclude_list)
        X = X[active_mask]
        y_raw = y_raw[active_mask]
        
        # Re-encode to remove gaps in target indices
        actual_labels = label_encoder.inverse_transform(y_raw)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(actual_labels)
    else:
        y = y_raw

    num_classes = len(label_encoder.classes_)
    print(f"Classes: {num_classes}")
    print(f"Samples: {len(X)}")

    # Filter out classes with too few samples to prevent train_test_split errors
    min_samples = 3
    class_counts = pd.Series(y).value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    
    valid_mask = np.isin(y, valid_classes)
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(np.unique(y)) < 2:
        print("Not enough diverse data to train yet. Let the extraction script finish.")
        return

    # Split data without strict stratification in case of small datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, INPUT_DIM)
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_train = X_train_scaled.reshape(-1, SEQUENCE_LENGTH, INPUT_DIM)

    X_test_flat = X_test.reshape(-1, INPUT_DIM)
    X_test_scaled = scaler.transform(X_test_flat)
    X_test = X_test_scaled.reshape(-1, SEQUENCE_LENGTH, INPUT_DIM)

    # Build and compile model with Gradient Clipping for stability
    model = build_lstm_model(num_classes)
    optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\nTraining LSTM Model (Stable Tanh + Clip)...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ]
    )

    # Save
    model_dir.mkdir(exist_ok=True)
    model.save(model_dir / "asl_model.keras")
    joblib.dump(scaler, model_dir / "scaler.joblib")
    joblib.dump(label_encoder, model_dir / "label_encoder.joblib")

    print(f"\nModel and artifacts saved to {model_dir}")

if __name__ == "__main__":
    main()
