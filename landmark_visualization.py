import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

DATA_DIR = './data'
OUTPUT_DIR = './landmark_output'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

with mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) as hands:
    for class_folder in os.listdir(DATA_DIR):
        class_path = os.path.join(DATA_DIR, class_folder)
        if not os.path.isdir(class_path):
            continue

        # Create output subfolder for each class
        output_class_folder = os.path.join(OUTPUT_DIR, class_folder)
        if not os.path.exists(output_class_folder):
            os.makedirs(output_class_folder)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Skipping unreadable image: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Save output image with landmarks drawn
                output_path = os.path.join(output_class_folder, img_name)
                cv2.imwrite(output_path, img)
                print(f"✅ Saved: {output_path}")
