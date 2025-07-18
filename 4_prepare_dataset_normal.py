# 4_prepare_dataset_normal.py

import os
import pandas as pd
import shutil
from tqdm import tqdm

FRAMES_DIR = "data/DAISEE/Frames/Train" 
LABELS_FILE = "data/DAISEE/Labels/TrainLabels.csv"
OUTPUT_DIR = os.path.join("data", "cnn_dataset_normal", "train") 

def main():
    """Menyalin 1 frame per video untuk membuat dataset baseline (normal)."""
    print("Memulai persiapan dataset untuk Skenario Normal (Baseline)...")

    os.makedirs(os.path.join(OUTPUT_DIR, "0_tidak_bingung"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "1_bingung"), exist_ok=True)

    labels_df = pd.read_csv(LABELS_FILE)
    labels_df.columns = [col.strip() for col in labels_df.columns]
    
    copied_files = {'0_tidak_bingung': 0, '1_bingung': 0}

    for _, row in tqdm(labels_df.iterrows(), total=labels_df.shape[0]):
        clip_id_clean = row['ClipID'].replace('.avi', '').replace('.mp4', '')
        
        source_frame_path = os.path.join(FRAMES_DIR, clip_id_clean, "frame000.jpg")
        
        if os.path.exists(source_frame_path):
            label = 1 if row['Confusion'] > 0 else 0
            
            if label == 1:
                dest_folder = os.path.join(OUTPUT_DIR, "1_bingung")
                copied_files['1_bingung'] += 1
            else:
                dest_folder = os.path.join(OUTPUT_DIR, "0_tidak_bingung")
                copied_files['0_tidak_bingung'] += 1
            
            dest_file_name = f"{clip_id_clean}.jpg"
            dest_path = os.path.join(dest_folder, dest_file_name)
            shutil.copy(source_frame_path, dest_path)

    print("\n✅ Proses persiapan data Normal selesai.")
    print("Hasil penyalinan:")
    print(f"  - Gambar 'Tidak Bingung': {copied_files['0_tidak_bingung']}")
    print(f"  - Gambar 'Bingung': {copied_files['1_bingung']}")

if __name__ == '__main__':
    main()