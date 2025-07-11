import os
import pickle
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
import sys

DATA_DIR = './landmark_output'
OUTPUT_PICKLE = 'data.pickle'

data = []
labels = []
images_for_preview = {}

mp_hands = mp.solutions.hands

with mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) as hands:
    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(dir_path):
            continue
        if not os.listdir(dir_path):
            print(f"⚠️ Folder {dir_path} is empty.")
            continue

        for img_name in os.listdir(dir_path):
            data_aux = []
            x_ = []
            y_ = []

            img_path = os.path.join(dir_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Skipping unreadable image: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]  # Use only one hand
                if len(hand_landmarks.landmark) != 21:
                    print(f"⚠️ Incomplete hand in {img_path}. Skipping.")
                    continue

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                min_x, min_y = min(x_), min(y_)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)

                data.append(data_aux)
                labels.append(dir_)

                if dir_ not in images_for_preview:
                    images_for_preview[dir_] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Save
with open(OUTPUT_PICKLE, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"✅ Saved {len(data)} samples to {OUTPUT_PICKLE}")

# === Visualization ===
if len(labels) == 0 or len(data) == 0:
    print("⚠️ No data to visualize.")
    sys.exit()

# 1. Class Distribution
label_counts = Counter(labels)
label_keys = sorted(label_counts.keys())
label_vals = [label_counts[k] for k in label_keys]

plt.figure(figsize=(10, 5))
sns.barplot(x=label_keys, y=label_vals)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Sample Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("fig_1_class_distribution.png", dpi=300)
plt.show()

# 2. Sample Images
num_classes = len(images_for_preview)
cols = 5
rows = math.ceil(num_classes / cols)

plt.figure(figsize=(15, rows * 3))
for i, (label, img) in enumerate(images_for_preview.items()):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.title(label)
    plt.axis('off')

plt.suptitle("Sample Image from Each Class")
plt.tight_layout()
plt.savefig("fig_2_sample_images.png", dpi=300)
plt.show()
