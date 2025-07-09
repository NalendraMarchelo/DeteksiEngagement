# utils/preprocessing.py

import numpy as np
import pandas as pd
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def load_daisee_dataset(data_dir, labels_file, target_size=(224, 224)):
    """
    Memuat frame dataset DAiSEE dan mengubah label 'Confusion' menjadi format biner.
    """
    labels_df = pd.read_csv(labels_file)
    labels_df.columns = labels_df.columns.str.strip()
    if 'Confusion' not in labels_df.columns:
        raise ValueError("File label tidak mengandung kolom 'Confusion'.")

    print(f"Total entri di file label: {len(labels_df)}")
    
    df = labels_df.copy()
    df["ClipID_clean"] = df["ClipID"].astype(str).str.replace(r"\.\w+$", "", regex=True)
    
    existing_clip_folders = {name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))}
    df_valid = df[df["ClipID_clean"].isin(existing_clip_folders)].reset_index(drop=True)
    
    print(f"Entri valid dengan folder klip yang ada: {len(df_valid)}")

    images, labels = [], []
    skipped = 0
    for _, row in df_valid.iterrows():
        clip_folder = os.path.join(data_dir, row["ClipID_clean"])
        frame_path = os.path.join(clip_folder, "frame000.jpg") # Mengambil frame pertama saja
        if os.path.exists(frame_path):
            img = cv2.imread(frame_path)
            if img is not None:
                img = cv2.resize(img, target_size)
                images.append(img.astype("float32") / 255.0)
                labels.append(int(row["Confusion"]))
            else:
                skipped += 1
        else:
            skipped += 1

    print(f"Berhasil memuat {len(images)} gambar.")
    print(f"Melewatkan {skipped} entri karena frame hilang atau tidak bisa dibaca.")

    # --- PERUBAHAN UNTUK KLASIFIKASI BINER ---
    print("Distribusi label asli (0-3):", pd.Series(labels).value_counts().sort_index().to_dict())
    
    # Mengubah label multi-level menjadi label biner (0 vs 1)
    # Aturan: Level 0 -> Kelas 0 (Tidak Bosan)
    # Level 1, 2, 3 -> Kelas 1 (Bosan)
    labels_array = np.array(labels)
    labels_binary = np.where(labels_array > 0, 1, 0)
    
    print("Distribusi label setelah diubah ke biner:", pd.Series(labels_binary).value_counts().to_dict())
    # -----------------------------------------------

    if len(images) == 0:
        raise RuntimeError("Tidak ada gambar valid yang berhasil dimuat dari dataset.")

    return np.array(images), labels_binary.astype(int)


def preprocess_frames(frame_dir, target_size=(224, 224)):
    """
    Mengubah ukuran dan normalisasi semua frame JPG dalam sebuah direktori.
    Fungsi ini tidak berubah.
    """
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    frames = []
    for file in frame_files:
        path = os.path.join(frame_dir, file)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, target_size)
            frames.append(img.astype("float32") / 255.0)
        else:
            print(f"[PERINGATAN] Gagal membaca {file}")
    return np.array(frames)


def prepare_data_for_training(images, labels, test_size=0.2):
    """
    Membagi dataset dan mengembalikan generator training dengan augmentasi.
    Fungsi ini tidak berubah.
    """
    if len(images) == 0 or len(labels) == 0:
        raise ValueError("Gambar atau label kosong, tidak bisa melanjutkan.")

    X_train, X_val, y_train, y_val = train_test_split(
        images, labels,
        test_size=test_size,
        stratify=labels, # Stratify penting untuk data tidak seimbang
        random_state=42
    )

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    return X_train, X_val, y_train, y_val, train_datagen