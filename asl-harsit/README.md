# ASL Super-Vision: Real-Time Hand Language Recognition

An advanced American Sign Language (ASL) recognition system that translates hand gestures into text and speech. This project combines the **ASL Alphabet** with the massive **WLASL (World-Level ASL)** dataset to recognize over 2000+ unique signs in real-time.

![ASL Recognition Demo](https://images.unsplash.com/photo-1610484826967-09c57193796d?auto=format&fit=crop&w=800&q=80) 
*(Example image - replace with your own screenshot/gif)*

## 🚀 Features

- **Massive Vocabulary**: Recognizes the full ASL Alphabet (A-Z) plus over 2000 words from the WLASL dataset.
- **Real-Time Inference**: Optimized for Raspberry Pi and Desktop using MediaPipe's modern HandLandmarker Tasks API.
- **Smart Text-to-Speech**: Native macOS `say` integration and `pyttsx3` fallback with prediction stabilization (prevents "flickering" audio).
- **Automated Data Pipeline**: One-script dataset collection that pulls directly from Kaggle and extracts hand landmarks into a unified CSV.
- **Lock-in Mechanism**: Visual and logic stabilization that ensures a sign is held before triggering speech.

## 🛠️ Technology Stack

- **Computer Vision**: MediaPipe (HandLandmarker Tasks API), OpenCV
- **Deep Learning**: TensorFlow / Keras (Feedforward Neural Network)
- **Data Engineering**: Kagglehub, Scikit-learn, Pandas, Joblib
- **Speech Synthesis**: Native macOS TTS / pyttsx3

## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/asl-super-vision.git
   cd asl-super-vision
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **MediaPipe Model**: The project will automatically download the `hand_landmarker.task` file on its first run.

## 📂 Project Structure

- `automated_collector.py`: The "Super Collector". Hybrid script that processes thousands of images (Alphabet) and videos (WLASL) into a single landmark dataset.
- `train_model.py`: High-capacity training script optimized for 2000+ classes.
- `inference_pi.py`: The main live application. Features camera feed, landing overlay, and stabilized TTS.
- `data/`: Contains the generated `asl_data_auto.csv`.
- `models/`: Stores the trained `.keras` model, scaler, and label encoder.

## 🚀 How to Use

### 1. Unified Data Collection
Build your own massive dataset locally. This script handles the downloading and landmark extraction automatically.
```bash
python automated_collector.py
```
*Note: This processes ~2000 glosses and may take a significant amount of time.*

### 2. Training the Model
Train the deep neural network on the extracted landmark data:
```bash
python train_model.py
```
The script uses **Early Stopping** to get the best possible accuracy and saves the model to the `models/` directory.

### 3. Real-Time Recognition
Run the live translator:
```bash
python inference_pi.py
```
- **"LOCKING IN..."**: Appears when the system is holding a prediction for stability.
- **"TALKING..."**: Appears when the sign is being translated into speech.
- **Press 'q'**: To exit the application.

## 📈 Performance
The current model architecture achieves over **99% accuracy** on the Alphabet validation set and high consistency on word detection due to the stabilization buffer.

## 🤝 Credits
- Dataset: [ASL Alphabet (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- Dataset: [WLASL (World-Level ASL)](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed)
- Hand Tracking: [MediaPipe](https://google.github.io/mediapipe/)

---
Created with ❤️ by [Harsit / Yeti-Hacked]
