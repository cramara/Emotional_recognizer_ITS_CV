import cv2
import os
import time
import numpy as np

# --- MODEL SETUP (TO BE UNCOMMENTED LATER) ---

# --- INITIALIZATION ---
# set to false if needed
SAVE_CAPTURES = True
OUTPUT_FOLDER = 'captured_faces'
if SAVE_CAPTURES and not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("Error loading cascade file.")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- VARIABLES ---
PREDICTION_INTERVAL = 1.0  # seconds
last_prediction_time = time.time()
current_emotion = "Processing..."

print("Webcam started. Press 'q' to quit.")

# --- MAIN APPLICATION LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # <--- THIS LINE IS NOW CORRECTED
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if time.time() - last_prediction_time >= PREDICTION_INTERVAL:
            last_prediction_time = time.time()
            roi_gray = gray_frame[y:y+h, x:x+w]

            # --- RESIZE THE CROPPED FACE TO 48x48 ---
            roi_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            # --- MODEL INFERENCE PLACEHOLDER ---

            # --- END OF PLACEHOLDER ---

            if SAVE_CAPTURES:
                timestamp = int(time.time())
                filename = f"face_{timestamp}.jpg"
                filepath = os.path.join(OUTPUT_FOLDER, filename)
                # --- SAVE THE RESIZED 48x48 IMAGE ---
                cv2.imwrite(filepath, roi_resized)
                print(f"📸 Saved {filename} (48x48)")
                # change this according to necessary pixels

        cv2.putText(frame, current_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Facial Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
print("Quitting program.")
cap.release()
cv2.destroyAllWindows()