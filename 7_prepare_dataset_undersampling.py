# 7_prepare_dataset_undersampling.py

import os
import pandas as pd
import shutil
from tqdm import tqdm
import random

FRAMES_DIR = "data/DAISEE/Frames/Train" 
LABELS_FILE = "data/DAISEE/Labels/TrainLabels.csv"
OUTPUT_DIR = os.path.join("data", "cnn_dataset_undersampling", "train") 

def main():
    """Membuat dataset yang seimbang dengan metode undersampling."""
    print("Memulai persiapan dataset untuk Skenario Undersampling...")

    output_bingung_dir = os.path.join(OUTPUT_DIR, "1_bingung")
    output_tidak_bingung_dir = os.path.join(OUTPUT_DIR, "0_tidak_bingung")
    os.makedirs(output_bingung_dir, exist_ok=True)
    os.makedirs(output_tidak_bingung_dir, exist_ok=True)

    labels_df = pd.read_csv(LABELS_FILE)
    labels_df.columns = [col.strip() for col in labels_df.columns]
    labels_df['is_confused'] = labels_df['Confusion'].apply(lambda x: 1 if x > 0 else 0)

    # Pisahkan daftar file berdasarkan label
    bingung_df = labels_df[labels_df['is_confused'] == 1]
    tidak_bingung_df = labels_df[labels_df['is_confused'] == 0]

    # Ambil semua sampel dari kelas minoritas (bingung)
    print(f"\nMenyalin semua {len(bingung_df)} gambar dari kelas 'Bingung'...")
    for _, row in tqdm(bingung_df.iterrows(), total=bingung_df.shape[0]):
        clip_id_clean = row['ClipID'].replace('.avi', '').replace('.mp4', '')
        source_path = os.path.join(FRAMES_DIR, clip_id_clean, "frame000.jpg")
        if os.path.exists(source_path):
            shutil.copy(source_path, os.path.join(output_bingung_dir, f"{clip_id_clean}.jpg"))

    # Ambil sampel acak dari kelas mayoritas (tidak bingung) sejumlah kelas minoritas
    num_samples_minority = len(bingung_df)
    tidak_bingung_sampled_df = tidak_bingung_df.sample(n=num_samples_minority, random_state=42)
    
    print(f"\nMenyalin {len(tidak_bingung_sampled_df)} gambar secara acak dari kelas 'Tidak Bingung'...")
    for _, row in tqdm(tidak_bingung_sampled_df.iterrows(), total=tidak_bingung_sampled_df.shape[0]):
        clip_id_clean = row['ClipID'].replace('.avi', '').replace('.mp4', '')
        source_path = os.path.join(FRAMES_DIR, clip_id_clean, "frame000.jpg")
        if os.path.exists(source_path):
            shutil.copy(source_path, os.path.join(output_tidak_bingung_dir, f"{clip_id_clean}.jpg"))

    print("\n✅ Proses persiapan data Undersampling selesai.")

if __name__ == '__main__':
    main()