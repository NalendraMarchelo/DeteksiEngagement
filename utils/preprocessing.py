import numpy as np
import pandas as pd
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def load_daisee_dataset(data_dir, labels_file, target_size=(224, 224)):
    """
    Load DAiSEE dataset frames and corresponding 'Confusion' labels only.

    Parameters:
        data_dir (str): Path to directory containing frames per video.
        labels_file (str): CSV path containing DAiSEE labels.
        target_size (tuple): Image size (height, width).

    Returns:
        images (np.ndarray): Normalized image tensors.
        labels (np.ndarray): Integer-encoded 'Confusion' labels.
    """
    labels_df = pd.read_csv(labels_file)
    labels_df.columns = labels_df.columns.str.strip()
    if 'Confusion' not in labels_df.columns:
        raise ValueError("Label CSV does not contain 'Confusion' column.")

    print(f"Total entries in label file: {len(labels_df)}")

    df = labels_df.copy()
    df["ClipID_clean"] = df["ClipID"].astype(str).str.replace(r"\.\w+$", "", regex=True)

    existing_clip_folders = {
        name for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
    }

    df_valid = df[df["ClipID_clean"].isin(existing_clip_folders)].reset_index(drop=True)
    print(f"Valid entries (with existing clip folders): {len(df_valid)}")

    images, labels = [], []
    skipped = 0

    for _, row in df_valid.iterrows():
        clip_folder = os.path.join(data_dir, row["ClipID_clean"])
        frame_path = os.path.join(clip_folder, "frame000.jpg")

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

    print(f"Loaded {len(images)} images with confusion labels.")
    print(f"Skipped {skipped} entries due to missing or unreadable frames.")
    print("Label distribution:", pd.Series(labels).value_counts().sort_index().to_dict())

    if len(images) == 0:
        raise RuntimeError("No valid images were loaded from dataset.")

    return np.array(images), np.array(labels).astype(int)


def preprocess_frames(frame_dir, target_size=(224, 224)):
    """
    Resize and normalize all JPG frames in a directory.

    Parameters:
        frame_dir (str): Directory with .jpg user video frames.
        target_size (tuple): Image resize target.

    Returns:
        np.ndarray: Preprocessed image tensors.
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
            print(f"[WARNING] Failed to read {file}")

    return np.array(frames)


def prepare_data_for_training(images, labels, test_size=0.2):
    """
    Split dataset and return augmented training generator.

    Parameters:
        images (np.ndarray): Input images.
        labels (np.ndarray): Class labels (0–3).
        test_size (float): Split ratio for validation set.

    Returns:
        tuple: X_train, X_val, y_train, y_val, train_datagen
    """
    if len(images) == 0 or len(labels) == 0:
        raise ValueError("Images or labels are empty. Cannot proceed.")

    X_train, X_val, y_train, y_val = train_test_split(
        images, labels,
        test_size=test_size,
        stratify=labels,
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
