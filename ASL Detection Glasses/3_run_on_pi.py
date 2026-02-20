import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tflite_runtime.interpreter as tflite # Use this on Pi instead of standard tensorflow
import time

# Initialize Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 150) # Speed of speech

# Load TFLite model
interpreter = tflite.Interpreter(model_path="models/asl_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# IMPORTANT: Update this list based on what the training script printed!
CLASSES = ['A', 'B', 'C'] 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
last_spoken = ""
cooldown_time = 0

print("Glasses online. Looking for ASL...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract 63 coordinates
            row = []
            for landmark in hand_landmarks.landmark:
                row.extend([landmark.x, landmark.y, landmark.z])
            
            # Prepare data for TFLite model
            input_data = np.array([row], dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Get prediction
            confidence = np.max(output_data)
            class_idx = np.argmax(output_data)
            prediction = CLASSES[class_idx]

            # Output logic: Only speak if confident, and don't spam the speaker
            if confidence > 0.8:
                cv2.putText(frame, f'{prediction} ({confidence:.2f})', (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Speak if it's a new letter and cooldown has passed
                if prediction != last_spoken and time.time() > cooldown_time:
                    print(f"Speaking: {prediction}")
                    engine.say(prediction)
                    engine.runAndWait()
                    last_spoken = prediction
                    cooldown_time = time.time() + 2  # Wait 2 seconds before speaking a new letter

    cv2.imshow('Smart ASL Glasses', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()