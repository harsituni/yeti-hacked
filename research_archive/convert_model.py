import tensorflow as tf
from pathlib import Path

# Paths based on your structure
model_path = Path("models/asl_model.keras")
tflite_path = Path("models/asl_model.tflite")

print(f"Loading {model_path}...")
model = tf.keras.models.load_model(model_path)

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"Success! TFLite model saved to {tflite_path}")
