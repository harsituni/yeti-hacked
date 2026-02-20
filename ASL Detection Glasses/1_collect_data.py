import cv2
import mediapipe as mp
import csv
import os
from collections import defaultdict

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# ── CSV setup ────────────────────────────────────────────────────────────────
DATA_FILE = 'data/asl_dataset.csv'

# Build descriptive headers: label, x0, y0, z0, x1, y1, z1, …, x20, y20, z20
HEADERS = ['label']
for i in range(21):
    HEADERS.extend([f'x{i}', f'y{i}', f'z{i}'])

if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, mode='w', newline='') as f:
        csv.writer(f).writerow(HEADERS)

# ── Camera ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
sample_counts = defaultdict(int)  # track how many samples per letter

print("Hold up a hand sign, then press the matching letter key (a-z) to save.")
print("Press 'q' to quit.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # ── Single waitKey call (fixes the dropped-keypress bug) ─────────────
    key = cv2.waitKey(1) & 0xFF

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 63 coordinates (21 landmarks × 3 axes)
            row = []
            for landmark in hand_landmarks.landmark:
                row.extend([landmark.x, landmark.y, landmark.z])

            # Label + save when a letter key (a-z) is pressed
            if ord('a') <= key <= ord('z'):
                label = chr(key).upper()
                with open(DATA_FILE, mode='a', newline='') as f:
                    csv.writer(f).writerow([label] + row)
                sample_counts[label] += 1
                print(f"Captured: {label}  (total: {sample_counts[label]})")

        # ── On-screen feedback ───────────────────────────────────────────
        info = "  ".join(f"{k}:{v}" for k, v in sorted(sample_counts.items()))
        if info:
            cv2.putText(frame, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('ASL Data Collection', frame)

    if key == ord('q'):
        break

# ── Cleanup ──────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()

if sample_counts:
    print("\n── Session summary ─────────────────")
    for label, count in sorted(sample_counts.items()):
        print(f"  {label}: {count} samples")
    print(f"  Total: {sum(sample_counts.values())} samples")
else:
    print("\nNo samples captured this session.")