import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import os

# ── 1. Load Data ─────────────────────────────────────────────────────────────
df = pd.read_csv('data/asl_dataset.csv')
print(f"Loaded {len(df)} samples across {df['label'].nunique()} classes")
print(f"Classes: {sorted(df['label'].unique())}\n")

X = df.drop('label', axis=1).values
y = df['label'].values

# Encode labels ('A' -> 0, 'B' -> 1, etc.)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ── 2. Build a Lightweight Neural Network ────────────────────────────────────
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(63,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ── 3. Train ─────────────────────────────────────────────────────────────────
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

print("Training model...")
model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# ── 4. Evaluate ──────────────────────────────────────────────────────────────
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\n── Classification Report ──────────────────────────")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# ── 5. Save model + label mapping ───────────────────────────────────────────
if not os.path.exists('models'):
    os.makedirs('models')

# Save label classes so the Pi script can load them automatically
np.save('models/label_classes.npy', encoder.classes_)
print(f"Label classes saved to models/label_classes.npy")

# Convert to TFLite for Raspberry Pi
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('models/asl_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Model saved to models/asl_model.tflite")