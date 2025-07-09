# engagement_project/predict.py

import os
import tensorflow as tf
import numpy as np
from utils.video_utils import extract_frames, detect_faces_in_video
from utils.preprocessing import preprocess_frames

# --- Konfigurasi ---
MODEL_PATH = "models/engagement_model.keras" # Sesuaikan dengan nama model dari train.py
LABELS = {0: "Tidak Bosan", 1: "Sedikit Bosan", 2: "Bosan", 3: "Sangat Bosan"} # Sesuaikan dengan kelas Anda

# --- Muat Model (dilakukan sekali) ---
print("Memuat model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"File model tidak ditemukan di {MODEL_PATH}. Jalankan train.py terlebih dahulu.")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model berhasil dimuat.")

def predict_boredom_from_video(video_path: str):
    """
    Menerima path video, memprosesnya, dan mengembalikan hasil prediksi.
    """
    temp_output_dir = "data/temp_inference"
    
    try:
        # 1. Ekstrak Frame & Deteksi Wajah
        print(f"Memproses video: {video_path}")
        faces_dir = os.path.join(temp_output_dir, "faces")
        os.makedirs(faces_dir, exist_ok=True)
        # Hapus file lama jika ada
        for f in os.listdir(faces_dir): os.remove(os.path.join(faces_dir, f))

        detect_faces_in_video(video_path, faces_dir)
        
        # 2. Pra-pemrosesan Frame Wajah
        print("Melakukan pra-pemrosesan wajah...")
        face_files = sorted([f for f in os.listdir(faces_dir) if f.endswith('.jpg')])
        if not face_files:
            return "Tidak ada wajah yang terdeteksi di dalam video."
        
        processed_frames = preprocess_frames(faces_dir)

        # 3. Lakukan Prediksi dengan Model
        print("Melakukan prediksi...")
        predictions = model.predict(processed_frames)
        predicted_classes = np.argmax(predictions, axis=1)

        # 4. Format Hasil Prediksi
        results = {
            "total_wajah_terdeteksi": len(predicted_classes),
            "prediksi_per_frame": {},
            "ringkasan": {}
        }
        
        # Prediksi detail per frame
        for i, class_idx in enumerate(predicted_classes):
            frame_name = face_files[i]
            results["prediksi_per_frame"][frame_name] = LABELS.get(class_idx, "Tidak Diketahui")

        # Ringkasan distribusi
        unique, counts = np.unique(predicted_classes, return_counts=True)
        summary_dict = dict(zip(unique, counts))
        
        for class_idx, count in summary_dict.items():
            label = LABELS.get(class_idx, "Tidak Diketahui")
            results["ringkasan"][label] = int(count)

        print("✅ Prediksi selesai.")
        return results

    except Exception as e:
        return f"Terjadi kesalahan: {e}"