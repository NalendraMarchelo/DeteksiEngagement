# 1_extract_features.py

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from scipy.spatial import distance as dist
from tqdm import tqdm

def calculate_ear(eye):
    #Menghitung Eye Aspect Ratio (EAR)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0

def calculate_mar(mouth):
    #Menghitung Mouth Aspect Ratio (MAR)
    A = dist.euclidean(mouth[2], mouth[3])
    B = dist.euclidean(mouth[0], mouth[1])
    return A / B if B > 0 else 0.0

# --- LOGIKA UTAMA ---
def main():
    print("Memulai proses EKSTRAKSI FITUR LANJUTAN...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    data_dir = "data/DAiSEE/Frames/Train"
    labels_file = "data/DAiSEE/Labels/TrainLabels.csv"
    output_csv_path = os.path.join("data", "features_rich.csv")

    labels_df = pd.read_csv(labels_file)
    labels_df.columns = [col.strip() for col in labels_df.columns]
    labels_df["ClipID_clean"] = labels_df["ClipID"].str.replace(r"\.\w+$", "", regex=True)

    # Indeks landmark
    EYE_LANDMARKS_LEFT = [362, 385, 387, 263, 373, 380]
    EYE_LANDMARKS_RIGHT = [33, 160, 158, 133, 153, 144]
    MOUTH_LANDMARKS = [61, 291, 0, 17]
    HEAD_POSE_LANDMARKS = [33, 263, 1, 61, 291, 199]

    all_features_data = []

    for _, row in tqdm(labels_df.iterrows(), total=labels_df.shape[0], desc="Processing Images"):
        frame_path = os.path.join(data_dir, row["ClipID_clean"], "frame000.jpg")
        if not os.path.exists(frame_path):
            continue
        image = cv2.imread(frame_path)
        if image is None:
            continue
        
        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks_3d = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in face_landmarks.landmark])
                landmarks_2d = landmarks_3d[:, :2]

                # Kalkulasi 7 Fitur
                avg_ear = (calculate_ear(landmarks_2d[EYE_LANDMARKS_LEFT]) + calculate_ear(landmarks_2d[EYE_LANDMARKS_RIGHT])) / 2.0
                mar = calculate_mar(landmarks_2d[MOUTH_LANDMARKS])
                
                img_pts = np.array([landmarks_2d[i] for i in HEAD_POSE_LANDMARKS], dtype="double")
                obj_pts = np.array([landmarks_3d[i] for i in HEAD_POSE_LANDMARKS], dtype="double")
                cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]])
                dist_coeffs = np.zeros((4, 1))
                _, rot_vec, _ = cv2.solvePnP(obj_pts, img_pts, cam_matrix, dist_coeffs)
                rot_mat, _ = cv2.Rodrigues(rot_vec)
                _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(cv2.hconcat((rot_mat, rot_vec)))
                pitch, yaw, roll = euler_angles.flatten()[:3]
                
                eyebrow_eye_dist = dist.euclidean(landmarks_2d[159], landmarks_2d[105])
                nose_mouth_dist = dist.euclidean(landmarks_2d[1], landmarks_2d[0])
                
                label = 1 if row["Confusion"] > 0 else 0

                all_features_data.append([avg_ear, mar, pitch, yaw, roll, eyebrow_eye_dist, nose_mouth_dist, label])

    face_mesh.close()
    
    feature_df = pd.DataFrame(all_features_data, columns=['ear', 'mar', 'pitch', 'yaw', 'roll', 'eyebrow_dist', 'nose_mouth_dist', 'label'])
    feature_df.to_csv(output_csv_path, index=False)
    print(f"\n✅ Proses selesai. Fitur LENGKAP berhasil diekstrak dan disimpan di: {output_csv_path}")

if __name__ == '__main__':
    main()