import os
import pandas as pd
import cv2
from tqdm import tqdm

# --- KONFIGURASI ---
VIDEO_DIR_BASE = "data/DAISEE/DataSet/Train" 
LABELS_FILE = "data/DAISEE/Labels/TrainLabels.csv"
OUTPUT_DIR = os.path.join("data", "cnn_dataset_oversampling")

# Interval pengambilan frame
BINGUNG_INTERVAL = 15      # Ambil 1 frame setiap ~0.5 detik
TIDAK_BINGUNG_INTERVAL = 60 # Ambil 1 frame setiap ~2 detik

def process_videos(df, label_name, frame_interval, base_video_path, base_output_path):
    print(f"\nMemproses {len(df)} video untuk kelas '{label_name}'...")
    
    output_path_label = os.path.join(base_output_path, label_name)
    os.makedirs(output_path_label, exist_ok=True)
    
    image_counter = 0
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        video_filename = row['ClipID']
        
        # --- PERBAIKAN FINAL DI SINI ---
        # Membangun path yang benar sesuai struktur folder berlapis
<<<<<<< HEAD
=======
        # Contoh: .../Train/110001/1100011002/1100011002.avi
>>>>>>> 091a565f374eea8ebd84bed906d72b7670a033f5
        person_id_folder = video_filename[:6] # Ambil 6 karakter pertama untuk nama folder orang
        video_id_folder = os.path.splitext(video_filename)[0] # Ambil nama file tanpa ekstensi
        video_path = os.path.join(base_video_path, person_id_folder, video_id_folder, video_filename)
        # --------------------------------

        if not os.path.exists(video_path):
            continue

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            if frame_idx % frame_interval == 0:
                output_filename = f"{video_id_folder}_frame{frame_idx}.jpg"
                cv2.imwrite(os.path.join(output_path_label, output_filename), frame)
                image_counter += 1
            
            frame_idx += 1
        cap.release()
        
    print(f"✅ Selesai. Total {image_counter} gambar disimpan untuk kelas '{label_name}'.")

def main():
    """Fungsi utama untuk menjalankan seluruh pipeline persiapan data."""
    print("Memulai pipeline persiapan dataset untuk CNN...")

    labels_df = pd.read_csv(LABELS_FILE)
    labels_df.columns = [col.strip() for col in labels_df.columns]
    labels_df['is_confused'] = labels_df['Confusion'].apply(lambda x: 1 if x > 0 else 0)

    df_bingung = labels_df[labels_df['is_confused'] == 1]
    df_tidak_bingung = labels_df[labels_df['is_confused'] == 0]

    train_output_dir = os.path.join(OUTPUT_DIR, "train")

    process_videos(df_bingung, "1_bingung", BINGUNG_INTERVAL, VIDEO_DIR_BASE, train_output_dir)
    process_videos(df_tidak_bingung, "0_tidak_bingung", TIDAK_BINGUNG_INTERVAL, VIDEO_DIR_BASE, train_output_dir)

    print("\nPipeline persiapan data selesai.")

if __name__ == '__main__':
    main()