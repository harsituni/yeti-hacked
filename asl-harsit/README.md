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

2. **Mandatory System Libraries (Linux/Pi only):**
   ```bash
   sudo apt update
   sudo apt install -y libgl1-mesa-glx libglib2.0-0 libespeak-ng1
   ```

3. **Install Python dependencies:**
   ```bash
   # Create a virtual environment (Recommended)
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # .venv\Scripts\activate   # Windows

   pip install -r requirements.txt
   ```

4. **Run the recognition:**
   ```bash
   python inference_pi.py
   ```

## 🎨 How to Personalize (Teach it your signs)

If you want the AI to learn your specific hand gestures:

1. **Record your data:**
   ```bash
   python collection/data_collection.py
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

```text
asl-harsit/
├── collection/               # Data tools
│   ├── automated_collector.py
│   └── data_collection.py
├── models/                   # The AI "Brain"
│   ├── asl_model.keras       # Retrained LSTM model
│   ├── hand_landmarker.task  # MediaPipe model
│   ├── label_encoder.joblib  # Mapping of IDs to labels
│   └── scaler.joblib         # Landmark scaling metadata
├── data/                     # Training data
│   └── asl_data_auto.csv
├── inference_pi.py           # Main Translation Application
├── train_model.py            # Neural Network Trainer
├── requirements.txt          # Dependency List
├── README.md                 # Instructions & Pitch
└── .gitignore                # Repository hygiene
```

- `inference_pi.py`: The main live application.
- `train_model.py`: The training engine that generates the AI's "brain."
- `collection/`:
    - `data_collection.py`: The personalization tool for recording your own signs.
    - `automated_collector.py`: Processes the massive ASL Alphabet and WLASL datasets.
- `models/`: Stores the trained `.keras` model and normalization artifacts.

## 📈 Performance
The current model uses a **Dual-Stage LSTM** architecture, achieving over **97% accuracy** on personalized datasets while maintaining high frame rates on low-power devices like the Raspberry Pi.

## 🛠 Troubleshooting & Raspberry Pi Tips

### ❌ Error: "Could not find a version that satisfies the requirement mediapipe"
This is the most common issue on Raspberry Pi. **MediaPipe requires a 64-bit OS.**
1. **Check your OS**: Run `getconf LONG_BIT` in your terminal.
2. **If it says `32`**: You must reinstall your Raspberry Pi OS using the **64-bit version**.
3. **If it says `64`**: Ensure you are using Python 3.9, 3.10, or 3.11.

### 🔇 No Speech on Raspberry Pi?
If you can see the text but can't hear the voice:
1. **Install audio libraries**: 
   ```bash
   sudo apt-get install libespeak-ng1
   ```
2. **Force Audio Jack**: Run `sudo raspi-config` -> System Options -> Audio -> choose the Headphones/Jack if not using HDMI.

---
