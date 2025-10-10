import cv2
import os
import time

# --- Configuration ---
DATASET_DIR = "facial_dataset" 
EMOTIONS = {'neutral': ("Neutral", 'n'), 'smile': ("Smile", 's'), 'sad': ("Sad", 'a'), 'angry': ("Angry", 'g'), 'surprise': ("Surprise", 'u')}
ANGLES = {'front': ("Front", 'f'), 'left_45': ("Left 45", 'l'), 'right_45': ("Right 45", 'r')}
IMAGES_PER_POSE = 30
COUNTDOWN_SEC = 3
# --- End of Configuration ---

def draw_text(frame, text, pos, font_scale=1.0, color=(255, 255, 255), thickness=2):
    """Draws text with a black background for better readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, pos, (pos[0] + text_w + 10, pos[1] - text_h - 10), (0, 0, 0), -1)
    cv2.putText(frame, text, (pos[0] + 5, pos[1] - 5), font, font_scale, color, thickness)

def main():
    # --- 1. Initialize Application State & Webcam ---
    app_state = 'GET_NAME' # GET_NAME, MENU_EMOTION, MENU_ANGLE, COUNTDOWN, CAPTURE, POST_CAPTURE
    subject_name = ""
    subject_dir = ""
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    selected_emotion = None
    selected_angle = None
    countdown_timer = 0
    capture_count = 0
    last_frame_time = time.time()
    error_message = ""
    error_timer = 0

    # --- 2. Start Main Application Loop ---
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) # Flip horizontally for a mirror-like view
        key = cv2.waitKey(1) & 0xFF

        # --- State Machine Logic ---
        if app_state == 'GET_NAME':
            draw_text(frame, "Enter Subject Name:", (400, 300))
            cursor = "|" if int(time.time() * 2) % 2 == 0 else " "
            draw_text(frame, f"{subject_name}{cursor}", (400, 350))
            draw_text(frame, "Press ENTER to confirm", (400, 400), font_scale=0.7)
            
            if key == 13: # ENTER key
                if subject_name:
                    subject_dir = os.path.join(DATASET_DIR, subject_name)
                    os.makedirs(subject_dir, exist_ok=True)
                    app_state = 'MENU_EMOTION'
                else:
                    error_message = "Name cannot be empty!"
                    error_timer = time.time() + 2
            elif key == 8: subject_name = subject_name[:-1]
            elif 32 <= key <= 126: subject_name += chr(key)
            
            if error_message and time.time() < error_timer:
                draw_text(frame, error_message, (400, 450), color=(0, 0, 255))
            else: error_message = ""

        elif app_state == 'MENU_EMOTION':
            y_pos = 50
            draw_text(frame, "--- SELECT EMOTION ---", (30, y_pos))
            y_pos += 50
            for e_key, (e_text, e_hotkey) in EMOTIONS.items():
                draw_text(frame, f"({e_hotkey.upper()}) {e_text}", (30, y_pos), font_scale=0.8)
                y_pos += 40
            draw_text(frame, "(Q) Quit", (30, y_pos), font_scale=0.8)

            if key != 255: # A key was pressed
                for e_key, (_, e_hotkey) in EMOTIONS.items():
                    if key == ord(e_hotkey):
                        selected_emotion = e_key
                        app_state = 'MENU_ANGLE'
                        break
                if key == ord('q'): break

        elif app_state == 'MENU_ANGLE':
            y_pos = 50
            draw_text(frame, f"EMOTION: {EMOTIONS[selected_emotion][0]}", (30, y_pos))
            y_pos += 50
            draw_text(frame, "--- SELECT ANGLE ---", (30, y_pos))
            y_pos += 50
            for a_key, (a_text, a_hotkey) in ANGLES.items():
                draw_text(frame, f"({a_hotkey.upper()}) {a_text}", (30, y_pos), font_scale=0.8)
                y_pos += 40
            draw_text(frame, "(B)ack to Emotions", (30, y_pos), font_scale=0.8)
            
            if key != 255:
                for a_key, (_, a_hotkey) in ANGLES.items():
                    if key == ord(a_hotkey):
                        selected_angle = a_key
                        app_state = 'COUNTDOWN'
                        countdown_timer = time.time() + COUNTDOWN_SEC
                        break
                if key == ord('b'): app_state = 'MENU_EMOTION'

        elif app_state == 'COUNTDOWN':
            remaining_time = int(countdown_timer - time.time())
            if remaining_time > 0:
                e_text, _ = EMOTIONS[selected_emotion]
                a_text, _ = ANGLES[selected_angle]
                draw_text(frame, f"Get Ready: {e_text} ({a_text})", (350, 300))
                draw_text(frame, str(remaining_time), (600, 400), font_scale=2.0)
            else:
                app_state = 'CAPTURE'
                capture_count = 0

        elif app_state == 'CAPTURE':
            feedback_text = f"Capturing: {capture_count}/{IMAGES_PER_POSE}"
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

            if len(faces) > 0:
                (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                if time.time() - last_frame_time > 0.1:
                    # --- CHANGED: Smart Square Cropping Logic ---
                    # 1. Find the center of the original bounding box
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # 2. Determine the size of the new square crop (make it larger)
                    size = max(w, h)
                    crop_size = int(size * 1.5) # Add a 50% buffer for better framing

                    # 3. Calculate the new top-left corner, ensuring it's within bounds
                    new_x = max(0, center_x - crop_size // 2)
                    new_y = max(0, center_y - crop_size // 2)
                    
                    # 4. Define the bottom-right corner
                    new_x2 = new_x + crop_size
                    new_y2 = new_y + crop_size
                    
                    # 5. Crop the face from the clean grayscale frame
                    face_crop_gray = gray[new_y:new_y2, new_x:new_x2]
                    # --- End of Change ---

                    img_name = f"{subject_name}_{selected_emotion}_{selected_angle}_{capture_count}.png"
                    img_path = os.path.join(subject_dir, img_name)
                    cv2.imwrite(img_path, face_crop_gray)
                    print(f"Saved: {img_path}")
                    capture_count += 1
                    last_frame_time = time.time()
            else: feedback_text = "No face detected!"

            draw_text(frame, feedback_text, (500, 50))
            if capture_count >= IMAGES_PER_POSE: app_state = 'POST_CAPTURE'

        elif app_state == 'POST_CAPTURE':
            draw_text(frame, "Capture Complete!", (500, 300))
            draw_text(frame, "(R)edo this pose", (500, 350))
            draw_text(frame, "(M)ain Menu", (500, 400))

            if key == ord('r'):
                app_state = 'COUNTDOWN'
                countdown_timer = time.time() + COUNTDOWN_SEC
            elif key == ord('m'):
                app_state = 'MENU_EMOTION'

        cv2.imshow('Interactive Dataset Creator', frame)

    # --- Cleanup ---
    print("\n🎉 Dataset creation session finished!")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()