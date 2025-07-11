import streamlit as st
import cv2
import numpy as np
import joblib
import mediapipe as mp
import pyttsx3

# === Load model ===
model_dict = joblib.load(open('./model.pkl', 'rb'))  # Corrected: joblib instead of pickle
model = model_dict['model']

# === Labels (based on your training) ===
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
    7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N',
    14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
    21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: '1', 27: '2', 28: '3', 29: '4', 30: '5',
    31: '6', 32: '7', 33: '8', 34: '9', 35: '0'
}

# === Mediapipe setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# === Text-to-speech ===
engine = pyttsx3.init()

# === Sentence buffer ===
sentence = []

# === Streamlit UI ===
st.title("🖐 Sign Language Detection with Voice")
st.write("Detect hand signs and speak them in real-time!")

run = st.checkbox('✅ Start Webcam')
clear_sentence = st.button('🧹 Clear Sentence')

if clear_sentence:
    sentence.clear()

FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("🚫 Could not open webcam. Please check your camera connection.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to grab frame!")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            predicted_char = ""

            if hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        x_.append(lm.x)
                        y_.append(lm.y)

                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min(x_))
                        data_aux.append(lm.y - min(y_))

                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_char = labels_dict[int(prediction[0])]

                    cv2.putText(frame, predicted_char, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

                    if not sentence or predicted_char != sentence[-1]:
                        sentence.append(predicted_char)
                        engine.say(predicted_char)
                        engine.runAndWait()

                except Exception as e:
                    st.error(f"Prediction error: {e}")

            FRAME_WINDOW.image(frame, channels="BGR")
            st.write("📝 Current Sentence: ", "".join(sentence))

            # Streamlit auto-break condition
            if not run:
                break

        cap.release()
        cv2.destroyAllWindows()
else:
    st.write("👈 Click the checkbox to start webcam")
