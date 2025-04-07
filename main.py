import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("sign_language_model_CNN.h5")
# Define labels corresponding to letters A-Z
labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Initialize webcam
cap = cv2.VideoCapture(0)
# Initialize Mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Detect hands using Mediapipe
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # Get the bounding box of the hand
        for landmarks in results.multi_hand_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            x_min, x_max, y_min, y_max = 1000, 0, 1000, 0
            for landmark in landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(
                    landmark.y * frame.shape[0]
                )
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y
            # Extract hand region
            hand_frame = frame[y_min:y_max, x_min:x_max]

            if hand_frame.shape[0] > 0 and hand_frame.shape[1] > 0:
                # Preprocess the hand frame
                resized_hand_frame = cv2.resize(hand_frame, (28, 28))
                grayscale_hand_frame = cv2.cvtColor(
                    resized_hand_frame, cv2.COLOR_BGR2GRAY
                )
                normalized_hand_frame = grayscale_hand_frame / 255.0
                # Make a prediction
                input_frame = normalized_hand_frame.reshape(
                    1, 28, 28, 1
                )  # Reshape for a single image

                prediction = model.predict(input_frame)
                predicted_label = np.argmax(prediction)
                # Get the corresponding letter
                letter = labels[predicted_label]
                # Display the prediction on the hand frame
                cv2.putText(
                    hand_frame,
                    letter,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                # Display both frames in separate windows
                cv2.imshow("Original Frame", frame)
                # cv2.imshow('Hand Frame', hand_frame)
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
