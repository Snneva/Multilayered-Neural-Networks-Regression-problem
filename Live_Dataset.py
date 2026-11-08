import os
import cv2
import csv
import time

# Inisialisasi Haar Cascade (Viola-Jones) dengan path yang aman
face_xml = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
eye_xml = os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')

face_cascade = cv2.CascadeClassifier(face_xml)
eye_cascade = cv2.CascadeClassifier(eye_xml)

# Validasi cascade berhasil dimuat
if face_cascade.empty() or eye_cascade.empty():
    raise IOError("Gagal memuat file Haar Cascade XML. Pastikan instalasi OpenCV lengkap.")

nama_file_csv = 'dataset_jarak.csv'

# Jika file CSV belum ada, buat file baru beserta kolom headernya
if not os.path.exists(nama_file_csv):
    with open(nama_file_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['LE', 'RE', 'F', 'Distance_cm'])
    print(f"File '{nama_file_csv}' baru berhasil dibuat!")

cap = cv2.VideoCapture(0)

print("\n=== CARA PENGAMBILAN DATA ===")
print("1. Mundur sesuai jarak yang Anda ukur dengan meteran (misal 30 cm).")
print("2. Ketik angka jarak di terminal VSCode, lalu tekan ENTER.")
print("3. Di jendela kamera, tekan SPASI untuk merekam 15 sampel secara otomatis.")
print("4. Ulangi untuk jarak lain (40, 50, 60, dst.).")
print("5. Tekan tombol 'q' di jendela kamera jika sudah selesai.\n")

while True:
    input_jarak = input("Masukkan jarak saat ini dalam cm (atau ketik 'selesai' untuk stop): ")
    if input_jarak.lower() == 'selesai':
        break
    
    try:
        jarak_cm = float(input_jarak)
    except ValueError:
        print("Masukkan angka yang valid!\n")
        continue

    print(f">> Bersiap merekam untuk jarak {jarak_cm} cm. Arahkan kursor ke jendela video dan tekan SPASI...")
    
    rekam_data = False
    sampel_terkumpul = 0
    target_sampel = 15  # Sekali tekan spasi, merekam 15 data

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        le_pixels, re_pixels, f_pixels = 0, 0, 0
        valid = False

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
            f_pixels = w * h
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
            valid_eyes = [e for e in eyes if e[1] < h / 2]
            
            if len(valid_eyes) >= 2:
                valid_eyes = sorted(valid_eyes, key=lambda e: e[0])[:2]
                (ex1, ey1, ew1, eh1) = valid_eyes[0]
                (ex2, ey2, ew2, eh2) = valid_eyes[1]
                
                cv2.rectangle(roi_color, (ex1, ey1), (ex1+ew1, ey1+eh1), (255, 0, 0), 2)
                cv2.rectangle(roi_color, (ex2, ey2), (ex2+ew2, ey2+eh2), (255, 0, 0), 2)
                
                le_pixels = ew1 * eh1
                re_pixels = ew2 * eh2
                valid = True
                
                cv2.putText(frame, f"L:{le_pixels} R:{re_pixels} F:{f_pixels}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            break
        
        # Tampilkan status di jendela video
        status = f"Target: {jarak_cm} cm | Rekam: {sampel_terkumpul}/{target_sampel}" if rekam_data else f"Target: {jarak_cm} cm (Tekan SPASI)"
        cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow('Pengumpul Dataset Live Cam', frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == 32:  # Tombol SPASI
            if valid:
                rekam_data = True
            else:
                print("Wajah dan 2 mata belum terdeteksi sempurna!")
        elif key == ord('q'):
            break

        # Proses penyimpanan otomatis
        if rekam_data and valid:
            with open(nama_file_csv, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([le_pixels, re_pixels, f_pixels, jarak_cm])
            sampel_terkumpul += 1
            time.sleep(0.05)
            
            if sampel_terkumpul >= target_sampel:
                print(f"[OK] {target_sampel} data untuk jarak {jarak_cm} cm berhasil disimpan ke CSV!\n")
                rekam_data = False
                break

cap.release()
cv2.destroyAllWindows()