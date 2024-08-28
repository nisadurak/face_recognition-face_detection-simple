import cv2
import os
import time

def detect_and_process_faces(frame, face_counter, first_seen_time, capture_duration=5):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    current_time = time.time()
    face_locations = []
    
    for (x, y, w, h) in faces:
        face_key = f"{x}_{y}_{w}_{h}"
        
        if face_key not in first_seen_time:
            first_seen_time[face_key] = current_time

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if current_time - first_seen_time[face_key] <= capture_duration:
            face_folder = os.path.join("data/detected_faces", f"Yuz{face_counter}")
            os.makedirs(face_folder, exist_ok=True)
            face_counter += 1

            face_img = frame[y:y + h, x:x + w]
            photo_path = os.path.join(face_folder, f"face_{face_counter}.jpg")
            cv2.imwrite(photo_path, face_img)
            print(f"Yüz algılandı ve kaydedildi: {photo_path}")

        face_locations.append((y, x + w, y + h, x))

    return face_counter, first_seen_time, face_locations
