"""
ASL Model Training Script

Loads collected hand landmark data from CSV, preprocesses with scikit-learn,
and trains a neural network with TensorFlow.
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


# Match data_collection.py structure
NUM_LANDMARKS = 21
DIMENSIONS = 3
FEATURES_PER_HAND = NUM_LANDMARKS * DIMENSIONS  # 63


def load_data(csv_path, is_sequence=False):
    """Load ASL data from CSV. Returns X (features), y (labels), label_encoder."""
    df = pd.read_csv(csv_path, header=None, on_bad_lines='skip')

    if len(df) < 10:
        raise ValueError(
            f"Not enough data ({len(df)} samples). Collect at least 10 samples per letter."
        )

    # First row might be header
    if isinstance(df.iloc[0, 1], str) and "f0_" in df.iloc[0, 1]:
        df = df.iloc[1:].copy()
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()

    X = df.iloc[:, 1:].values  # All columns except 'label'
    y = df.iloc[:, 0].values   # Label column
    
    if is_sequence:
        # Reshape flat 1260 array into (samples, 20 time steps, 63 features)
        X = X.reshape(-1, 20, 63)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder


def build_model(num_classes, input_shape, is_sequence=False):
    """Build a neural network based on the input type."""
    if is_sequence:
        # LSTM Architecture for dynamic 20-frame spatial tracking
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(64, return_sequences=True),
            layers.LSTM(128),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ])
    else:
        # Normal dense architecture for static 1-frame poses
        model = keras.Sequential([
            layers.Input(shape=(input_shape[0],)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
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
    args = parser.parse_args()
    
    data_dir = Path(__file__).parent / "data"
    model_dir = Path(__file__).parent / "models"
    
    if args.type == "letter":
        csv_path = data_dir / "asl_data_auto.csv"
        prefix = "letter"
    else:
        csv_path = data_dir / "wlasl_data_static.csv"
        prefix = "word"

    if not csv_path.exists():
        print(f"Error: No data found at {csv_path}")
        print("Run extraction or collection scripts first.")
        return

    # Both letter and word now use static Dense architecture
    is_sequence = False

    print(f"Loading data from {csv_path}...")
    X, y, label_encoder = load_data(csv_path, is_sequence=is_sequence)

    num_classes = len(label_encoder.classes_)
    print(f"Classes: {num_classes} labels")
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

    # Scale features with scikit-learn
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    input_shape = (X.shape[1],)

    # Build and compile model
    model = build_model(num_classes, input_shape, is_sequence=is_sequence)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\nTraining...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            ),
        ],
    )

    # Evaluate
    print("\n" + "=" * 50)
    print("Evaluation")
    print("=" * 50)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(
        label_encoder.inverse_transform(y_test),
        label_encoder.inverse_transform(y_pred),
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model and artifacts (joblib for sklearn objects)
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{prefix}_model.keras"
    scaler_path = model_dir / f"{prefix}_scaler.joblib"
    labels_path = model_dir / f"{prefix}_label_encoder.joblib"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, labels_path)

    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Label encoder saved to {labels_path}")


if __name__ == "__main__":
    main()
