# # run_inference.py

# import cv2
# import mediapipe as mp
# import numpy as np
# import joblib
# from scipy.spatial import distance as dist
# import warnings
# import os
# import time
# import argparse # Library untuk command-line

# warnings.filterwarnings('ignore')

# # --- KONFIGURASI DAN PEMUATAN MODEL ---
# MODEL_PATH = "models/boredom_geometric_model.joblib"
# print(f"Memuat model dari: {MODEL_PATH}")
# try:
#     model = joblib.load(MODEL_PATH)
#     print("✅ Model berhasil dimuat.")
# except FileNotFoundError:
#     print(f"❌ ERROR: File model tidak ditemukan di '{MODEL_PATH}'.")
#     print("Pastikan Anda sudah menjalankan skrip training (2_train_model.py) terlebih dahulu.")
#     exit() # Keluar dari skrip jika model tidak ada

# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# # --- FUNGSI PEMBANTU ---
# def calculate_ear(eye):
#     A = dist.euclidean(eye[1], eye[5])
#     B = dist.euclidean(eye[2], eye[4])
#     C = dist.euclidean(eye[0], eye[3])
#     return (A + B) / (2.0 * C) if C > 0 else 0.0

# def calculate_mar(mouth):
#     A = dist.euclidean(mouth[2], mouth[3])
#     B = dist.euclidean(mouth[0], mouth[1])
#     return A / B if B > 0 else 0.0

# # --- FUNGSI UTAMA ---
# def analyze_video(input_path, output_path):
#     """
#     Fungsi utama yang menerima path video input dan output.
#     """
#     cap = cv2.VideoCapture(input_path)
#     if not cap.isOpened():
#         print(f"❌ Error: Gagal membuka file video: {input_path}")
#         print("Coba konversi format video ke MP4 dengan codec H.264.")
#         return
        
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#     print(f"Memulai analisis video... Hasil akan disimpan di '{output_path}'")

#     # Indeks landmark
#     EYE_LANDMARKS_LEFT = [362, 385, 387, 263, 373, 380]
#     EYE_LANDMARKS_RIGHT = [33, 160, 158, 133, 153, 144]
#     MOUTH_LANDMARKS = [61, 291, 0, 17]
#     HEAD_POSE_LANDMARKS = [33, 263, 1, 61, 291, 199]

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break

#         image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(image_rgb)
        
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 # Proses ekstraksi fitur (sama seperti sebelumnya)
#                 landmarks_3d = np.array([(lm.x * width, lm.y * height, lm.z * width) for lm in face_landmarks.landmark])
#                 landmarks_2d = landmarks_3d[:, :2]
#                 avg_ear = (calculate_ear(landmarks_2d[EYE_LANDMARKS_LEFT]) + calculate_ear(landmarks_2d[EYE_LANDMARKS_RIGHT])) / 2.0
#                 mar = calculate_mar(landmarks_2d[MOUTH_LANDMARKS])
#                 img_pts = np.array([landmarks_2d[i] for i in HEAD_POSE_LANDMARKS], dtype="double")
#                 obj_pts = np.array([landmarks_3d[i] for i in HEAD_POSE_LANDMARKS], dtype="double")
#                 cam_matrix = np.array([[width, 0, width/2], [0, width, height/2], [0, 0, 1]])
#                 _, rot_vec, _ = cv2.solvePnP(obj_pts, img_pts, cam_matrix, np.zeros((4,1)))
#                 rot_mat, _ = cv2.Rodrigues(rot_vec)
#                 _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(cv2.hconcat((rot_mat, rot_vec)))
#                 pitch, yaw, roll = euler_angles.flatten()[:3]
#                 eyebrow_eye_dist = dist.euclidean(landmarks_2d[159], landmarks_2d[105])
#                 nose_mouth_dist = dist.euclidean(landmarks_2d[1], landmarks_2d[0])
#                 current_features = [avg_ear, mar, pitch, yaw, roll, eyebrow_eye_dist, nose_mouth_dist]
                
#                 # Prediksi dan gambar teks
#                 prediction = model.predict([current_features])[0]
#                 label_text = "BOSAN" if prediction == 1 else "TIDAK BOSAN"
#                 color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
#                 cv2.putText(frame, label_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
        
#         video_writer.write(frame)

#     cap.release()
#     video_writer.release()
#     print("✅ Analisis selesai.")

# if __name__ == '__main__':
#     # Membuat parser untuk argumen command-line
#     parser = argparse.ArgumentParser(description="Analisis Deteksi Kebosanan dari File Video.")
#     parser.add_argument("--input", required=True, help="Path ke file video input.")
#     parser.add_argument("--output", default=None, help="(Opsional) Path untuk menyimpan file video output.")
    
#     args = parser.parse_args()

#     # Menentukan nama file output
#     if args.output:
#         output_path = args.output
#     else:
#         # Buat nama file output otomatis jika tidak disediakan
#         output_folder = "output_videos"
#         os.makedirs(output_folder, exist_ok=True)
#         timestamp = int(time.time())
#         output_path = os.path.join(output_folder, f"result_{timestamp}.mp4")

#     # Jalankan fungsi analisis
#     analyze_video(args.input, output_path)