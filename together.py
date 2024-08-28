from face_recognitionn.load_faces import load_known_faces
from face_recognitionn.recognize_faces import recognize_and_process_faces
from face_detectionn.detect_faces import detect_and_process_faces
from utils.optimize import optimize_performance
from utils.evaluate import final_testing, summarize_results
import cv2


def integrated_system(frame, known_encodings, known_names, first_seen_time, capture_duration, unknown_counter, face_counter):
  
    face_counter, first_seen_time, face_locations = detect_and_process_faces(frame, face_counter, first_seen_time, capture_duration)
    first_seen_time, unknown_counter = recognize_and_process_faces(frame, known_encodings, known_names, first_seen_time, capture_duration, unknown_counter)
    return first_seen_time, unknown_counter, face_counter, face_locations 

def main():
    known_faces_dir = "data/test_images"
    known_encodings, known_names = load_known_faces(known_faces_dir)

    video_capture = cv2.VideoCapture(0)
    first_seen_time = {}
    capture_duration = 15
    unknown_counter = 1
    face_counter = 1

    while True:
        ret, frame = video_capture.read()
        first_seen_time, unknown_counter, face_counter, face_locations = integrated_system(
            frame, known_encodings, known_names, first_seen_time, capture_duration, unknown_counter, face_counter
        )
        
        cv2.imshow("Face Detection and Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
