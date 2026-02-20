import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# 1. Load Data
df = pd.read_csv('data/asl_dataset.csv')
X = df.drop('label', axis=1).values
y = df['label'].values

# Encode labels (e.g., 'A', 'B' -> 0, 1)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 2. Build a Lightweight Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(63,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. Train
print("Training model...")
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# 4. Save mapping for the Pi
classes_str = ",".join(encoder.classes_)
print(f"CLASS MAPPING: {classes_str} (Save this for your Pi script!)")

# 5. Convert to TFLite for Raspberry Pi
if not os.path.exists('models'):
    os.makedirs('models')
    
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('models/asl_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Model saved to models/asl_model.tflite")