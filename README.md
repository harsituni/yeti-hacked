# 🌉 SignBridge: Giving Voice to Your Signs

### *Breaking Communication Barriers with Real-Time ASL Translation*

**SignBridge** is a cutting-edge American Sign Language (ASL) recognition system designed for immediate, real-time translation. By combining advanced **Temporal LSTM Networks** with **Wrist-Relative Normalization**, SignBridge provides a robust and personalized bridge between the signing and non-signing communities.

---

## 🌟 Why SignBridge?

Most sign language models fail when the user moves their hand or stands at a different distance. **SignBridge is built differently.**

*   **📏 Position & Scale Invariant**: Our custom "Wrist-Relative" normalization ensures accuracy whether you're right in front of the camera or across the room.
*   **🧠 Personalized Intelligence**: Don't just adapt to the model—make the model adapt to *you*. Use the built-in teaching tool to record your unique signing style.
*   **🥧 Raspberry Pi Optimized**: Engineered to run efficiently on low-power devices, making portable, real-time translation a reality.
*   **🔊 Dual Output**: Instant text overlays and stabilized text-to-speech for seamless conversations.

---

## 🚀 Quick Start

### 1. Prepare Your Environment

**🐧 Linux & Raspberry Pi**
```bash
git clone https://github.com/harsituni/yeti-hacked.git
cd yeti-hacked
sudo apt update && sudo apt install -y libgl1-mesa-glx libglib2.0-0 libespeak-ng1
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

**🍎 macOS**
```bash
git clone https://github.com/harsituni/yeti-hacked.git
cd yeti-hacked
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

**🪟 Windows**
```powershell
git clone https://github.com/harsituni/yeti-hacked.git
cd yeti-hacked
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Bridging

**🐧 Linux & Raspberry Pi / 🍎 macOS**
```bash
python3 inference_pi.py
```

**🪟 Windows**
```powershell
python inference_pi.py
```

---

## 🎨 Teach it Your Style (Personalization)

Every hand is unique. SignBridge allows you to build a custom dictionary in minutes.

1.  **Record**: Run `python3 collection/data_collection.py` (or `python` on Windows). Press **'l'** for letters or **'s'** for motion-based words.
2.  **Train**: Run `python3 train_model.py` (or `python` on Windows). The AI will automatically rebuild its "brain" with your data.
3.  **Deploy**: Your new `SignBridge` is ready!

---

## 📂 Project Architecture

```text
SignBridge/ (Root)
├── collection/               # Data Acquisition Suite
│   ├── automated_collector.py # Dataset processor (WLASL/Alphabet)
│   └── data_collection.py     # Personalization tool
├── models/                   # The AI Brain
│   ├── asl_model.keras       # The trained Neural Network
│   ├── hand_landmarker.task  # MediaPipe Vision Core
│   └── *.joblib              # Meta-parameters & Encoding
├── data/                     # Your personal data vaults
├── inference_pi.py           # The Mission Control (Live App)
├── train_model.py            # The Training Engine
├── requirements.txt          # The Blueprint
├── research_archive/         # Archived development history
└── README.md                 # Giving Voice to Your Signs
```

---

## 📈 Performance & Tech Stack
*   **Architecture**: Dual-Stage LSTM (Long Short-Term Memory)
*   **Accuracy**: >97% on personalized gesture sets
*   **CV Engine**: MediaPipe Tasks API
*   **Logic**: Python 3 / TensorFlow / OpenCV

---

## 🛠 Troubleshooting

*   **MediaPipe Missing?** Ensure you are on a **64-bit OS** (`getconf LONG_BIT`).
*   **No Sound?** Run `sudo apt install libespeak-ng1` to enable the speech engine.
*   **Python Version?** SignBridge loves Python **3.10 or 3.11**.

---


*SignBridge: Built for developers, designed for people.* 🌉✨
