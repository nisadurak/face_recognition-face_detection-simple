import face_recognition
import cv2
import os
import time
import numpy as np  # Use np for numpy

def recognize_and_process_faces(frame, known_encodings, known_names, first_seen_time, capture_duration, unknown_counter):
    """
    Görüntüden yüz tanıma, tanınan kişilere göre fotoğraf kaydetme ve kare içine alma işlemini yapar.
    
    Args:
        frame (ndarray): Kameradan alınan anlık görüntü.
        known_encodings (List): Bilinen yüzlerin kodlamaları.
        known_names (List): Bilinen yüzlerin isimleri.
        first_seen_time (dict): Her yüzün ilk algılandığı zamanı tutan sözlük.
        capture_duration (int): Fotoğraf çekme süresi (saniye cinsinden).
        unknown_counter (int): Bilinmeyen yüzler için sayaç.
    
    Returns:
        Tuple[dict, int]: Güncellenmiş ilk algılanma zamanı sözlüğü ve bilinmeyen sayaç.
    """
    
    valid_encodings = [enc for enc in known_encodings if isinstance(enc, (np.ndarray, list))]
    known_encodings = np.array(valid_encodings)

    
    if len(known_encodings) == 0:
        print("Hiçbir geçerli yüz kodlaması bulunamadı. Yüz tanıma işlemi sonlandırılıyor.")
        return first_seen_time, unknown_counter

    # Yüz kodlamaları ve tanıma işlemleri
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    if not face_encodings:
        print("Bu karede yüz kodlaması bulunamadı.")
        return first_seen_time, unknown_counter

    for face_encoding, face_location in zip(face_encodings, face_locations):
        face_encoding = np.array(face_encoding)

    
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Bilinmiyor"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

        
        if name == "Bilinmiyor":
            person_folder = os.path.join("captured_faces", f"Bilinmeyen{unknown_counter}")
            os.makedirs(person_folder, exist_ok=True)
            unknown_counter += 1
        else:
            person_folder = os.path.join("captured_faces", name)
            os.makedirs(person_folder, exist_ok=True)

        if name not in first_seen_time:
            first_seen_time[name] = time.time()

        current_time = time.time()
        if current_time - first_seen_time[name] <= capture_duration:
            timestamp = int(current_time)
            photo_path = os.path.join(person_folder, f"face_{timestamp}.jpg")
            cv2.imwrite(photo_path, frame)
            print(f"{name} için fotoğraf çekildi ve kaydedildi: {photo_path}")


        top, right, bottom, left = face_location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        
   
     
    return first_seen_time, unknown_counter

