# 5_train_cnn_normal.py (Versi Perbaikan)

import tensorflow as tf
import os
import argparse
# Pastikan file utils/models_utils.py ada dan berisi build_engagement_model
from utils.models_utils import build_engagement_model, train_model 

# --- KONFIGURASI ---
DATA_DIR = os.path.join("data", "cnn_dataset_normal", "train")
MODEL_SAVE_NAME = "cnn_model_normal.keras"
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

def main(args):
    """Melatih model CNN pada dataset normal (tidak seimbang)."""
    
    print("--- Memulai Training CNN Skenario Normal ---")
    
    # 1. Memuat Dataset
    print(f"Memuat dataset dari: {DATA_DIR}")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training",
        seed=123, image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE
    )
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation",
        seed=123, image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE
    )

    # --- PERBAIKAN URUTAN DI SINI ---
    # Ambil class_names SEBELUM dataset dioptimasi
    class_names = train_dataset.class_names
    print("Kelas yang ditemukan:", class_names)
    
    # Baru lakukan optimasi performa pada dataset
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    # --------------------------------

    # 2. Membangun Model
    # Bangun model dengan jumlah kelas yang sudah disimpan
    model = build_engagement_model(num_classes=len(class_names))
    model.summary()

    # 3. Melatih Model (tanpa class_weight)
    print(f"\nMemulai training untuk {args.epochs} epoch...")
    history = train_model(
        model,
        train_dataset,
        validation_dataset,
        epochs=args.epochs,
        class_weight=None # tidak ada pembobotan untuk skenario normal
    )

    # 4. Menyimpan Model Final
    model_save_path = os.path.join("models", MODEL_SAVE_NAME)
    os.makedirs("models", exist_ok=True)
    model.save(model_save_path)
    print(f"\n✅ Training selesai. Model berhasil disimpan di: {model_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training CNN Skenario Normal")
    parser.add_argument("--epochs", type=int, default=10, help="Jumlah epoch")
    args = parser.parse_args()
    main(args)