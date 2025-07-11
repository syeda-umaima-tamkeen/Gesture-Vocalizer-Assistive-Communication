import os
import cv2
import time

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 1000 # images per class
class_index = 0


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

print("Press 'n' to start capturing images for a NEW class.")
print("Press 'ESC' anytime to exit.")

while True:
    print(f"\nWaiting to start class {class_index} capture...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            continue

        cv2.putText(frame, f'Press "n" for next class, ESC to quit', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'):
            break
        elif key == 27:  # ESC
            print("Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    class_dir = os.path.join(DATA_DIR, str(class_index))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Starting capture for class {class_index} in 2 seconds...')
    time.sleep(2)

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            continue

        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to interrupt early
            print("Interrupted by user.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    print(f"Captured {counter} images for class {class_index}")
    class_index += 1
