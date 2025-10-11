import cv2
import os
import time
import numpy as np
import tensorflow as tf

# --- MODEL SETUP ---
# Load the trained emotion recognition model
model_path = os.path.join('..', 'model', 'final_model.keras')
emotion_model = tf.keras.models.load_model(model_path)

# Define emotion classes (same order as training)
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def preprocess_face_image(face_image):
    """
    Preprocess face image for emotion prediction
    Args:
        face_image: numpy array of shape (48, 48) - grayscale face image
    Returns:
        preprocessed image ready for model prediction
    """
    # Normalize pixel values to [0, 1]
    face_normalized = face_image.astype('float32') / 255.0
    
    # Add batch dimension and channel dimension for model input
    face_reshaped = np.expand_dims(face_normalized, axis=0)  # Add batch dimension
    face_reshaped = np.expand_dims(face_reshaped, axis=-1)   # Add channel dimension
    
    return face_reshaped

def predict_emotion(face_image):
    """
    Predict emotion from face image
    Args:
        face_image: numpy array of shape (48, 48) - grayscale face image
    Returns:
        tuple: (predicted_emotion, confidence)
    """
    # Preprocess the image
    processed_image = preprocess_face_image(face_image)
    
    # Make prediction
    predictions = emotion_model.predict(processed_image, verbose=0)
    
    # Get the predicted class and confidence
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_emotion = emotion_classes[predicted_class_idx]
    
    return predicted_emotion, confidence

# --- INITIALIZATION ---
# set to false if needed
SAVE_CAPTURES = False
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
print(f"Emotion model loaded successfully. Classes: {emotion_classes}")
print("Real-time emotion recognition is now active!")

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

            # --- MODEL INFERENCE ---
            try:
                predicted_emotion, confidence = predict_emotion(roi_resized)
                current_emotion = f"{predicted_emotion} ({confidence:.2f})"
            except Exception as e:
                current_emotion = "Error in prediction"
                print(f"Prediction error: {e}")
            # --- END OF INFERENCE ---

            if SAVE_CAPTURES:
                timestamp = int(time.time())
                filename = f"face_{timestamp}.jpg"
                filepath = os.path.join(OUTPUT_FOLDER, filename)
                # --- SAVE THE RESIZED 48x48 IMAGE ---
                cv2.imwrite(filepath, roi_resized)
                print(f"📸 Saved {filename} (48x48)")
                # change this according to necessary pixels

        # Display emotion prediction with better formatting
        emotion_text = current_emotion
        font_scale = 0.8
        font_thickness = 2
        text_color = (0, 255, 0)  # Green color
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        # Draw background rectangle for better text visibility
        cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), (0, 0, 0), -1)
        
        # Draw the emotion text
        cv2.putText(frame, emotion_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    cv2.imshow('Facial Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
print("Quitting program.")
cap.release()
cv2.destroyAllWindows()