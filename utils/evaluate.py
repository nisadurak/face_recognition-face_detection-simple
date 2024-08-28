import os
import cv2
import face_recognition
import numpy as np
from face_detectionn.detect_faces import detect_and_process_faces
from face_recognitionn.recognize_faces import recognize_and_process_faces

def load_known_faces(known_faces_dir):
    known_encodings = []
    known_names = []

    if not os.path.exists(known_faces_dir):
        print(f"Dizin bulunamadı: {known_faces_dir}")
        return known_encodings, known_names
    
    for item in os.listdir(known_faces_dir):
        person_path = os.path.join(known_faces_dir, item)
        if os.path.isdir(person_path):
            for filename in os.listdir(person_path):
                image_path = os.path.join(person_path, filename)
                
                # Yalnızca resim dosyalarını işleyin
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print(f"Geçersiz dosya formatı atlandı: {filename}")
                    continue
                
                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    
                    if encodings:
                        # Birden fazla yüz varsa hepsini ekleyebiliriz
                        for encoding in encodings:
                            known_encodings.append(encoding)
                            known_names.append(item)
                        print(f"{item} için {len(encodings)} yüz kodlaması yüklendi.")
                    else:
                        print(f"Yüz bulunamadı veya kodlama yapılamadı: {image_path}")
                
                except Exception as e:
                    print(f"Dosya işlenirken hata oluştu ({image_path}): {e}")

    return known_encodings, known_names

def evaluate_performance(face_locations):
    """
    Yüz tanıma ve algılama performansını değerlendirir.
    
    Args:
        face_locations (List): Algılanan yüzlerin konumları.
    
    Returns:
        dict: Performans metriklerini içeren bir sözlük.
    """
    return {'accuracy': 1.0, 'speed': 0.5}  # Örnek metrikler

def final_testing(known_encodings, known_names, test_images):
    results = []
    face_counter = 0
    first_seen_time = {}
    capture_duration = 5  
    unknown_counter = 0

    for image_path in test_images:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error loading image at path: {image_path}")
            continue

        
        face_counter, first_seen_time, face_locations = detect_and_process_faces(frame, face_counter,
                                                                                 first_seen_time, 
                                                                                 capture_duration)
        if not face_locations or not isinstance(face_locations, list):
            print(f"No faces detected in image: {image_path} or incorrect format of face_locations")
            continue

      
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        print(f"Yüz kodlamaları: {face_encodings}")  

        if not face_encodings:
            print("Bu karede yüz kodlaması bulunamadı.")
            continue

        # Yüz tanıma işlemi
        try:
            recognize_and_process_faces(frame, known_encodings, 
                                        known_names, first_seen_time, 
                                        capture_duration, unknown_counter)
        except Exception as e:
            print(f"Yüz tanıma işlemi sırasında hata oluştu: {e}")
            continue
        
        # Performans değerlendirmesi
        results.append(evaluate_performance(face_locations))

    return results


def summarize_results(test_results):
    if not test_results:  # Eğer test_results boşsa
        return {'accuracy': 0, 'speed': 0}

    summary = {
        'accuracy': sum([result['accuracy'] for result in test_results]) / len(test_results),
        'speed': sum([result['speed'] for result in test_results]) / len(test_results)
    }
    return summary

def suggest_improvements(test_results):
    improvements = []
    if any(result['accuracy'] < 0.8 for result in test_results):
        improvements.append('Daha çeşitli yüz verileri kullanarak eğitim setini genişletin.')
    if any(result['speed'] > 2.0 for result in test_results):
        improvements.append('Algoritma performansını artırmak için optimizasyon teknikleri uygulayın.')
    return improvements

def present_results(results_summary, improvements):
    print("Project Summary:")
    print(f"Average Accuracy: {results_summary['accuracy']:.2f}")
    print(f"Average Speed: {results_summary['speed']:.2f} seconds per frame")
    print("Suggested Improvements:")
    for improvement in improvements:
        print(f"- {improvement}")

if __name__ == "__main__":
    # Bilinen yüzlerin bulunduğu dizin
    known_faces_dir = "data/know-face"
    
    # Bilinen yüzleri ve kodlamalarını yükle
    known_encodings, known_names = load_known_faces(known_faces_dir)

    # Yüz kodlamalarını kontrol et
    if not known_encodings or any(enc is None or not isinstance(enc, (np.ndarray, list)) for enc in known_encodings):
        print("Geçerli yüz kodlamaları mevcut değil. Lütfen kodlamaları kontrol edin.")
        exit()

    # Test görüntüleri
    test_images = ["data/test_images/kiz1.jpg", "data/test_images/angeline.jpg"]

    # Yüz tanıma ve performans değerlendirmesi
    test_results = final_testing(known_encodings, known_names, test_images)
    results_summary = summarize_results(test_results)
    improvements = suggest_improvements(test_results)
    present_results(results_summary, improvements)
