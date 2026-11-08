import os
import cv2
import joblib
import numpy as np

# 1. Load model, scaler, dan daftar fitur
try:
    model = joblib.load('best_distance_model.pkl')
    scaler = joblib.load('best_scaler.pkl')
    features_used = joblib.load('model_features.pkl')
    print("========================================")
    print(" Model Terbaik Berhasil Dimuat!")
    print(f" Fitur yang digunakan: {features_used}")
    print("========================================")
except FileNotFoundError:
    print("Error: File .pkl belum ditemukan. Pastikan file model sudah diunduh dari Colab!")
    exit()

# 2. Inisialisasi Haar Cascade
face_xml = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
eye_xml = os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')

face_cascade = cv2.CascadeClassifier(face_xml)
eye_cascade = cv2.CascadeClassifier(eye_xml)

cap = cv2.VideoCapture(0)

print("\nTekan tombol 'q' di jendela video untuk keluar.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        f_pixels = w * h

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
        valid_eyes = [e for e in eyes if e[1] < h / 2]

        le_pixels, re_pixels = 0, 0
        eyes_detected = False

        if len(valid_eyes) >= 2:
            valid_eyes = sorted(valid_eyes, key=lambda e: e[0])[:2]
            (ex1, ey1, ew1, eh1) = valid_eyes[0]
            (ex2, ey2, ew2, eh2) = valid_eyes[1]

            cv2.rectangle(roi_color, (ex1, ey1), (ex1 + ew1, ey1 + eh1), (255, 0, 0), 2)
            cv2.rectangle(roi_color, (ex2, ey2), (ex2 + ew2, ey2 + eh2), (255, 0, 0), 2)

            le_pixels = ew1 * eh1
            re_pixels = ew2 * eh2
            eyes_detected = True

        # Validasi: Model All Features butuh Wajah DAN Kedua Mata
        if eyes_detected and f_pixels > 0:
            # Susun urutan input sesuai saat training: ['LE', 'RE', 'F']
            input_dict = {'LE': le_pixels, 'RE': re_pixels, 'F': f_pixels}
            input_vector = [input_dict[feat] for feat in features_used]

            input_scaled = scaler.transform([input_vector])
            pred_distance = model.predict(input_scaled)[0]

            # Tampilkan Prediksi Jarak (Warna Hijau)
            cv2.putText(frame, f"Jarak: {pred_distance:.1f} cm", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
            cv2.putText(frame, f"F:{f_pixels} | LE:{le_pixels} | RE:{re_pixels}", (x, y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            # Beri panduan visual jika mata belum terdeteksi (Warna Oranye)
            cv2.putText(frame, "Posisikan wajah & kedua mata terlihat lurus", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        break

    cv2.imshow("DEMO Real-time Distance Estimation (ITS - EE)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()