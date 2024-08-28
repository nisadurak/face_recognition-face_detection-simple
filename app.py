from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
from face_detectionn.detect_faces import detect_and_process_faces
import os
from werkzeug.utils import secure_filename
import face_recognition
import time


app = Flask(__name__)

# Yükleme ve işlenmiş dosyalar için klasörler
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['DETECTED_FOLDER'] = 'static/detected_faces/'

# Kamera erişimi için OpenCV video yakalama nesnesi
video_capture = cv2.VideoCapture(0)

# Yüklenecek dosya türlerini kontrol etmek için izin verilen uzantılar
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Bilinen yüzlerin kodlamalarını ve isimlerini yükle
def load_known_faces(known_face_dir='data\know-face'):
    known_encodings = []
    known_names = []
    for name in os.listdir(known_face_dir):
        person_dir = os.path.join(known_face_dir, name)
        for filename in os.listdir(person_dir):
            image_path = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_encodings.append(encoding[0])
                known_names.append(name)
    return known_encodings, known_names

known_encodings, known_names = load_known_faces()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_frames():
    face_counter = 0
    first_seen_time = {}
    
    while True:
        # Kameradan bir kare yakala
        success, frame = video_capture.read()
        if not success:
            break
        else:
            # Yüz tanıma işlemleri için BGR'den RGB'ye çevir
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Yüz konumlarını tespit et ve işle
            face_counter, first_seen_time, face_locations = detect_and_process_faces(frame, face_counter, first_seen_time)

            # `face_recognition` kullanarak yüz tanıma işlemi
            face_encodings = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left) for (top, right, bottom, left) in face_locations])

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Yüzleri bilinenlerle karşılaştır
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                name = "Bilinmiyor"

                # Eğer bir eşleşme bulunursa, bilinen isimleri kullan
                if True in matches:
                    match_index = matches.index(True)
                    name = known_names[match_index]

                # Yüzün etrafına kare çiz
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            # Kareyi JPEG formatına dönüştür
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Yüz tanıma ve algılama işlemi burada yapılacak
        process_image(file_path, filename)
        
        return redirect(url_for('index', filename=filename))

    return redirect(request.url)

def process_image(file_path, filename):
    # Yüz algılama ve tanıma işlemi
    frame = cv2.imread(file_path)
    
    # OpenCV görüntüsü RGB formatına dönüştürülmelidir.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Yüz konumlarını tespit et
    face_locations = detect_and_process_faces(rgb_frame)  # `detect_faces` fonksiyonu burada kullanılıyor
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Yüzleri bilinenlerle karşılaştır
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Bilinmiyor"

        # Eğer bir eşleşme bulunursa, bilinen isimleri kullan
        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

        # Yüzün etrafına kare çiz
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
    
    # İşlenmiş resmi kaydet
    output_path = os.path.join(app.config['DETECTED_FOLDER'], filename)
    cv2.imwrite(output_path, frame)
    
@app.route('/video_feed')
def video_feed():
    # Video akışını döndürür
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
    
    
    
 
