# Yüz Tanıma ve Algılama Web Uygulaması

Bu proje, bir kamera aracılığıyla gerçek zamanlı olarak yüz tanıma ve yüz algılama işlemleri gerçekleştiren bir web uygulamasıdır. Python dilinde yazılmıştır ve `Flask`, `face_recognition`, `opencv-python` kütüphanelerini kullanmaktadır. Proje, yüz tanıma ve yüz algılama işlevlerini iki ana modülde sunmaktadır. İki modulü birleştirerek bir web app ortaya çıkmıştır. 

## Özellikler

### Yüz Tanıma (Face Recognition)
- **Bilinen Yüzlerin Yüklenmesi:** Bilinen yüzler bir veritabanından yüklenir ve kodlanır.
- **Gerçek Zamanlı Yüz Tanıma:** Kameradan alınan görüntüler üzerinde yüz tanıma işlemleri yapılır.
- **Tanınan Yüzler İçin Fotoğraf Çekme:** Tanınan yüzlerin fotoğrafları çekilir ve saklanır.
- **Tanınmayan Yüzler İçin Fotoğraf Çekme:** Tanınmayan yüzler ayrı klasörlerde saklanır.
- **İlk Algılama Zamanı İzleme:** Her yüz için ilk algılama zamanı izlenir.

### Yüz Algılama (Face Detection)
- **Gerçek Zamanlı Yüz Algılama:** Kameradan alınan görüntüler üzerinde yüz algılama yapılır.
- **İlk 5 Saniye Fotoğraf Çekme:** Algılanan yüzlerin ilk 5 saniye boyunca fotoğrafları çekilir ve saklanır.
- **Yüzlerin Kare İçine Alınması:** Algılanan yüzler kare içine alınır ve işlenir.
- **Fotoğrafların Kaydedilmesi:** Her yüz için ayrı ayrı fotoğraflar kaydedilir.

## Gereksinimler

Bu proje aşağıdaki Python kütüphanelerini gerektirir:

- `Flask`
- `face_recognition`
- `opencv-python`
- `numpy`

Gereksinimlerinizi kurmak için aşağıdaki komutu çalıştırabilirsiniz:

```bash
pip install Flask face_recognition opencv-python numpy
Kullanım
Web Uygulamasını Çalıştırma
Yüz Veritabanını Hazırlama:

Bilinen yüzlerin bulunduğu bir dizin oluşturun (data/know-face).
Her kişi için bir alt klasör oluşturun ve o kişinin fotoğraflarını bu klasöre koyun.
Flask Uygulamasını Başlatma:

Web uygulamasını başlatmak için aşağıdaki komutu çalıştırın:

python app.py
Uygulama http://127.0.0.1:5000/ adresinde çalışacaktır.
Yüz Tanıma ve Yüz Algılama:

Uygulama arayüzüne gidin (http://127.0.0.1:5000/).
Kamera akışını görüntüleyin ve bilinen yüzleri tanıyın.
Yüz fotoğraflarını yükleyin veya gerçek zamanlı olarak kameradan görüntü alın.
Yüz Tanıma (Face Recognition) Kullanımı
Bilinen Yüzlerin Yüklenmesi:

known_faces_dir değişkenini, bilinen yüzlerin bulunduğu dizin ile güncelleyin. Bu dizin içinde her kişi için ayrı bir alt klasör olmalı ve her alt klasörde o kişinin fotoğrafları bulunmalıdır.
Ana Fonksiyon:

app.py dosyasındaki Flask uygulamasını çalıştırarak yüz tanıma uygulamasını başlatabilirsiniz.
Fotoğraf Çekme Süresi:

Fotoğraf çekme süresi, capture_duration değişkeniyle saniye cinsinden ayarlanabilir.
Yüz Algılama (Face Detection) Kullanımı
Ana Fonksiyon:

app.py dosyasındaki Flask uygulamasını çalıştırarak yüz algılama uygulamasını başlatabilirsiniz.
Fotoğraf Çekme Süresi:

Her algılanan yüz için ilk 5 saniye boyunca fotoğraf çekilir. Bu süre capture_duration değişkeniyle ayarlanabilir.

Kodun Yapısı
load_known_faces(known_faces_dir): Bilinen yüzlerin bulunduğu dizinden yüz kodlamalarını yükler.
recognize_and_process_faces(frame, known_encodings, known_names, first_seen_time, capture_duration, unknown_counter): Görüntüdeki yüzleri tanır, fotoğraf çeker ve kaydeder.
detect_and_process_faces(frame, face_counter, first_seen_time, capture_duration): Görüntüdeki yüzleri algılar ve ilk 5 saniye boyunca fotoğraf çeker ve kaydeder.
main(): Ana döngüyü çalıştırır ve uygulamanın başlangıç noktasıdır.


Katkıda Bulunanlar
[Nisa DURAK]






