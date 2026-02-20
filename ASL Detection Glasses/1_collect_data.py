import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

DATA_FILE = 'data/asl_dataset.csv'

# Create file and headers if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        # 21 landmarks, 3 coordinates (x, y, z) each = 63 features + 1 label
        writer.writerow(['label'] + [f'pt{i}' for i in range(63)])

cap = cv2.VideoCapture(0)
print("Press 'A', 'B', 'C' etc. to capture data for that class. Press 'Q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract the 63 coordinates
            row = []
            for landmark in hand_landmarks.landmark:
                row.extend([landmark.x, landmark.y, landmark.z])

            # Wait for key press to label and save the data
            key = cv2.waitKey(1) & 0xFF
            if ord('a') <= key <= ord('z'):
                label = chr(key).upper()
                with open(DATA_FILE, mode='a', newline='') as f:
                    csv.writer(f).writerow([label] + row)
                print(f"Captured: {label}")

    cv2.imshow('ASL Data Collection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()