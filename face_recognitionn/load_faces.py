import os
import face_recognition

def load_known_faces(known_faces_dir):
    known_encodings = []
    known_names = []

    if not os.path.exists(known_faces_dir):
        print(f"Dizin bulunamadı: {known_faces_dir}")
        return known_encodings, known_names  # Boş listelerle dön
    else:
        for item in os.listdir(known_faces_dir):
            person_path = os.path.join(known_faces_dir, item)
            if os.path.isdir(person_path):
                for filename in os.listdir(person_path):
                    image_path = os.path.join(person_path, filename)
                    
                    # Yalnızca resim dosyalarını işleyin
                    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        print(f"Geçersiz dosya formatı atlandı: {filename}")
                        continue
                    
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    
                    if encodings:
                        encoding = encodings[0]
                        known_encodings.append(encoding)
                        known_names.append(item)
                    else:
                        print(f"Yüz bulunamadı veya kodlama yapılamadı: {image_path}")

    return known_encodings, known_names
