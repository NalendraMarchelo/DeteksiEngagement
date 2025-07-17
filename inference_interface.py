# import cv2
# import mediapipe as mp
# import numpy as np
# import joblib
# import gradio as gr
# from scipy.spatial import distance as dist
# import warnings
# import os
# import time
# import asyncio
# import sys

# # --- Patch Event Loop Windows (Wajib untuk Gradio Stabil) ---
# if sys.platform.startswith("win"):
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# warnings.filterwarnings('ignore')

# # --- KONFIGURASI DAN PEMUATAN MODEL ---
# MODEL_PATH = "models/boredom_geometric_model.joblib"
# print(f"Memuat model dari: {MODEL_PATH}")
# try:
#     model = joblib.load(MODEL_PATH)
#     print("✅ Model berhasil dimuat.")
# except FileNotFoundError:
#     print(f"❌ ERROR: File model tidak ditemukan di '{MODEL_PATH}'.")
#     model = None

# # --- FaceMesh Init ---
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
#                                   min_detection_confidence=0.5,
#                                   min_tracking_confidence=0.5)

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
# def analyze_video_for_boredom(video_path):
#     if model is None:
#         raise gr.Error("Model tidak dapat dimuat. Periksa terminal untuk detailnya.")

#     output_folder = "output_videos"
#     os.makedirs(output_folder, exist_ok=True)
#     timestamp = int(time.time())
#     output_path = os.path.join(output_folder, f"result_{timestamp}.mp4")

#     cap = cv2.VideoCapture(video_path)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     if not fps or fps <= 1:
#         fps = 25  # fallback default

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     EYE_LEFT = [362, 385, 387, 263, 373, 380]
#     EYE_RIGHT = [33, 160, 158, 133, 153, 144]
#     MOUTH = [61, 291, 0, 17]
#     HEAD = [33, 263, 1, 61, 291, 199]

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break

#         image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(image_rgb)

#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 lm3d = np.array([(lm.x * width, lm.y * height, lm.z * width) for lm in face_landmarks.landmark])
#                 lm2d = lm3d[:, :2]

#                 ear = (calculate_ear(lm2d[EYE_LEFT]) + calculate_ear(lm2d[EYE_RIGHT])) / 2.0
#                 mar = calculate_mar(lm2d[MOUTH])

#                 img_pts = np.array([lm2d[i] for i in HEAD], dtype="double")
#                 obj_pts = np.array([lm3d[i] for i in HEAD], dtype="double")
#                 cam_matrix = np.array([[width, 0, width/2], [0, width, height/2], [0, 0, 1]])
#                 dist_coeffs = np.zeros((4, 1))

#                 try:
#                     _, rvec, _ = cv2.solvePnP(obj_pts, img_pts, cam_matrix, dist_coeffs)
#                     rmat, _ = cv2.Rodrigues(rvec)
#                     _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(cv2.hconcat((rmat, rvec)))
#                     pitch, yaw, roll = euler.flatten()[:3]
#                 except:
#                     pitch, yaw, roll = 0, 0, 0

#                 eyebrow_eye = dist.euclidean(lm2d[159], lm2d[105])
#                 nose_mouth = dist.euclidean(lm2d[1], lm2d[0])

#                 features = [ear, mar, pitch, yaw, roll, eyebrow_eye, nose_mouth]
#                 pred = model.predict([features])[0]
#                 label = "BOSAN" if pred == 1 else "TIDAK BOSAN"
#                 color = (0, 0, 255) if pred == 1 else (0, 255, 0)

#                 cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

#         writer.write(frame)

#     cap.release()
#     writer.release()

#     # Tunggu sampai file benar-benar ada dan selesai ditulis
#     for _ in range(10):
#         if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
#             break
#         time.sleep(0.5)

#     return output_path

# # --- GRADIO INTERFACE ---
# def launch_interface():
#     gr.Interface(
#         fn=analyze_video_for_boredom,
#         inputs=gr.Video(label="Unggah Video Belajar"),
#         outputs=gr.Video(label="Hasil Prediksi"),
#         title="Deteksi Kebosanan Siswa",
#         description="Sistem akan memproses video dan memberi label BOSAN/TIDAK BOSAN pada tiap frame.",
#         allow_flagging="never"
#     ).launch(share=False)

# if __name__ == '__main__':
#     launch_interface()
