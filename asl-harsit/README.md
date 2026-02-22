# ASL Super-Vision: Real-Time Hand Language Recognition

An advanced American Sign Language (ASL) recognition system that translates hand gestures into text and speech. This project uses **Wrist-Relative Normalization**, making it robust to hand position and distance from the camera.

## 🚀 Features

- **Personalized Training**: Teach the AI *your* specific hand style using the built-in collection tool.
- **Position Independent**: Works anywhere in the camera frame thanks to landmark normalization.
- **Real-Time Inference**: Optimized for Raspberry Pi and Desktop using MediaPipe's modern HandLandmarker Tasks API.
- **Stabilized TTS**: Integrated Text-to-Speech with logic that prevents "flickering" audio.

## ⚙️ Quick Start (For Friends)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/harsituni/yeti-hacked.git
   cd yeti-hacked/asl-harsit
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the recognition:**
   ```bash
   python inference_pi.py
   ```
   *The system will automatically download the required MediaPipe model on its first run.*

## 🎨 How to Personalize (Teach it your signs)

If you want the AI to learn your specific hand gestures:

1. **Record your data:**
   ```bash
   python data_collection.py
   ```
   - Press **'l'** to save a static letter.
   - Press **'s'** to record a 1-second motion (for words).
   - Press **'n'** to switch to a new label mid-session.

2. **Retrain the brain:**
   ```bash
   python train_model.py
   ```
   *The AI will now recognize your custom hand style!*

## 📂 Project Structure

- `inference_pi.py`: The main live application.
- `data_collection.py`: The personalization tool for recording your own signs.
- `train_model.py`: The training engine that generates the AI's "brain."
- `automated_collector.py`: Processes the massive ASL Alphabet and WLASL datasets.
- `models/`: Stores the trained `.keras` model and normalization artifacts.

## 📈 Performance
The current model uses a **Dual-Stage LSTM** architecture, achieving over **97% accuracy** on personalized datasets while maintaining high frame rates on low-power devices like the Raspberry Pi.

---
Created with ❤️ by [Harsit / Yeti-Hacked]
