import mediapipe
import cv2
import numpy as np

mp_drawing = mediapipe.solutions.drawing_utils
mp_hands = mediapipe.solutions.hands
mp_drawing_styles = mediapipe.solutions.drawing_styles


def main():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        cv2.imshow("Hand Tracking", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
    hands.close()
    cap.release()
    cv2.destroyAllWindows()


main()
