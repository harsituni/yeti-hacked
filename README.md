# ASL Hand Recognition Project

Convert camera hand movements to American Sign Language letters using MediaPipe, TensorFlow, and scikit-learn.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Collection

Run the data collection script to record hand poses:

```bash
python data_collection.py
```

**Letter mode** (default if you press Enter at the prompt):
- **Press a letter key (a-z)** to record the current hand pose for that letter

**Phrase/word mode** (enter a phrase at startup or use `--phrase`):
```bash
python data_collection.py --phrase "hello"
# or: python data_collection.py -p "thank you"
```
- **Press SPACE** to record the current hand pose with that label
- Useful for collecting signs for words like "hello", "thank you", "bye", etc.

- **Press 'q'** to quit
- Data is saved to `data/asl_data.csv`

### 2. Training

After collecting enough samples per letter, train the model:

```bash
python train_model.py
```

- Loads data from `data/asl_data.csv`
- Trains a neural network (TensorFlow) with scikit-learn for preprocessing
- Saves model to `models/asl_model.keras`

### 3. Live Inference (Raspberry Pi / Desktop)

Run real-time ASL recognition with text-to-speech:

```bash
python inference_pi.py
```

- **Live video** from webcam with hand landmarks overlay
- **Prediction + confidence** shown on screen (e.g., `a  92%`)
- **Text-to-speech** announces the letter/phrase when confidence exceeds 85%
- **Press 'q'** to quit

**Options:**
```bash
python inference_pi.py --threshold 0.9    # Require 90% confidence for TTS
python inference_pi.py --no-tts          # Disable speech (e.g. headless Pi)
python inference_pi.py --camera 1         # Use camera index 1
```

**Raspberry Pi setup:** Install `espeak` for TTS: `sudo apt install espeak`
