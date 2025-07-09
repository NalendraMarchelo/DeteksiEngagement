import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from scipy.spatial import distance as dist
from tqdm import tqdm

# --- FUNGSI PEMBANTU UNTUK MENGHITUNG FITUR GEOMETRIS ---

def calculate_ear(eye):
    """Menghitung Eye Aspect Ratio (EAR) dari koordinat landmark mata."""
    # Jarak vertikal antara kelopak mata
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Jarak horizontal antara sudut mata
    C = dist.euclidean(eye[0], eye[3])
    # Rumus EAR
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_mar(mouth):
    """Menghitung Mouth Aspect Ratio (MAR) dari koordinat landmark mulut."""
    # Jarak vertikal bibir
    A = dist.euclidean(mouth[2], mouth[3]) # Jarak antara titik bibir atas dan bawah
    # Jarak horizontal bibir
    B = dist.euclidean(mouth[0], mouth[1]) # Jarak antara sudut bibir
    # Rumus MAR
    mar = A / B
    return mar

# --- LOGIKA UTAMA ---

def main():
    print("Memulai proses ekstraksi fitur geometris dari dataset DAiSEE...")

    # Inisialisasi MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    # Path ke dataset dan file label Anda
    data_dir = "data/DAiSEE/Frames/Train"
    labels_file = "data/DAiSEE/Labels/TrainLabels.csv"
    output_csv_path = "data/features_geometric.csv"

    # Muat label
    labels_df = pd.read_csv(labels_file)
    labels_df.columns = [col.strip() for col in labels_df.columns]
    labels_df["ClipID_clean"] = labels_df["ClipID"].str.replace(r"\.\w+$", "", regex=True)
    
    # Indeks landmark untuk setiap fitur wajah (sesuai dokumentasi MediaPipe)
    EYE_LANDMARKS_LEFT = [362, 385, 387, 263, 373, 380]
    EYE_LANDMARKS_RIGHT = [33, 160, 158, 133, 153, 144]
    MOUTH_LANDMARKS = [61, 291, 0, 17] # Sudut kiri, sudut kanan, bibir atas, bibir bawah

    all_features_data = []

    # Loop melalui setiap entri di file label dengan progress bar
    for _, row in tqdm(labels_df.iterrows(), total=labels_df.shape[0], desc="Processing Images"):
        clip_folder = os.path.join(data_dir, row["ClipID_clean"])
        frame_path = os.path.join(clip_folder, "frame000.jpg")

        if not os.path.exists(frame_path):
            continue

        image = cv2.imread(frame_path)
        if image is None:
            continue
        
        # Proses gambar dengan MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])

                # Hitung fitur geometris
                left_ear = calculate_ear(landmarks[EYE_LANDMARKS_LEFT])
                right_ear = calculate_ear(landmarks[EYE_LANDMARKS_RIGHT])
                avg_ear = (left_ear + right_ear) / 2.0
                mar = calculate_mar(landmarks[MOUTH_LANDMARKS])

                # Dapatkan label biner (0 = Tidak Bosan, 1 = Bosan)
                boredom_level = row["Confusion"]
                label = 1 if boredom_level > 0 else 0

                # Simpan fitur dan label ke list
                all_features_data.append([avg_ear, mar, label])

    face_mesh.close()

    # Buat DataFrame dari hasil ekstraksi dan simpan ke CSV
    feature_df = pd.DataFrame(all_features_data, columns=['ear', 'mar', 'label'])
    feature_df.to_csv(output_csv_path, index=False)

    print(f"\n✅ Proses selesai. Fitur berhasil diekstrak dan disimpan di: {output_csv_path}")
    print(f"Total fitur yang diekstrak: {len(feature_df)}")
    print("Distribusi label:")
    print(feature_df['label'].value_counts())

if __name__ == '__main__':
    main()