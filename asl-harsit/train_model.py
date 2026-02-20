"""
ASL Model Training Script

Loads collected hand landmark data from CSV, preprocesses with scikit-learn,
and trains a neural network with TensorFlow.
"""

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


def load_data(csv_path):
    """Load ASL data from CSV. Returns X (features), y (labels), label_encoder."""
    df = pd.read_csv(csv_path)

    if len(df) < 10:
        raise ValueError(
            f"Not enough data ({len(df)} samples). Collect at least 10 samples per letter."
        )

    X = df.iloc[:, 1:].values  # All columns except 'label'
    y = df.iloc[:, 0].values   # Label column

    # Encode labels (a-z -> 0-25)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder


def build_model(num_classes, input_dim):
    """Build a simple feedforward neural network."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def main():
    data_dir = Path(__file__).parent / "data"
    model_dir = Path(__file__).parent / "models"
    csv_path = data_dir / "asl_data.csv"

    if not csv_path.exists():
        print(f"Error: No data found at {csv_path}")
        print("Run data_collection.py first to collect hand pose data.")
        return

    print("Loading data...")
    X, y, label_encoder = load_data(csv_path)

    num_classes = len(label_encoder.classes_)
    print(f"Classes: {list(label_encoder.classes_)} ({num_classes} letters)")
    print(f"Samples: {len(X)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features with scikit-learn
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build and compile model
    model = build_model(num_classes, X.shape[1])
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
    model_path = model_dir / "asl_model.keras"
    scaler_path = model_dir / "scaler.joblib"
    labels_path = model_dir / "label_encoder.joblib"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, labels_path)

    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Label encoder saved to {labels_path}")


if __name__ == "__main__":
    main()
