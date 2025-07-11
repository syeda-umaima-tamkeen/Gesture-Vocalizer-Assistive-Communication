import joblib
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# ✅ Load model
data = joblib.load('./model.pkl')
model = data['model']

# ✅ Define class names (according to label encoding during training)
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G',
    'H', 'I', 'J', 'K', 'L', 'M', 'N',
    'O', 'P', 'Q', 'R', 'S', 'T', 'U',
    'V', 'W', 'X', 'Y', 'Z',
    '1', '2', '3', '4', '5',
    '6', '7', '8', '9', '0'
]

# ✅ Text-to-speech
engine = pyttsx3.init()

# ✅ Webcam setup
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("❌ Camera index 2 failed. Trying index 0.")
    cap = cv2.VideoCapture(0)

# ✅ Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ✅ State tracking
hand_present = False
gesture_spoken = False
last_gesture = None

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W, _ = frame.shape
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        if not hand_present:
            hand_present = True
            gesture_spoken = False
            last_gesture = None

        data_aux = []
        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

        try:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_index = int(prediction[0])
            current_gesture = class_names[predicted_index]

            if current_gesture != last_gesture:
                last_gesture = current_gesture
                gesture_spoken = False

            if not gesture_spoken:
                print(f"🗣️ Speaking: {current_gesture}")
                engine.say(current_gesture)
                engine.runAndWait()
                gesture_spoken = True

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, current_gesture, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)

        except Exception as e:
            print("Prediction error:", e)

    else:
        if hand_present:
            print("🙌 Hand removed. Resetting state.")
            hand_present = False
            gesture_spoken = False
            last_gesture = None

    cv2.putText(frame, f"Last: {last_gesture if last_gesture else 'None'}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 2)

    cv2.imshow("Sign Language Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
