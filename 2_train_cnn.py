import tensorflow as tf
import os
import argparse

# Pastikan Anda memiliki file ini dari pekerjaan kita sebelumnya
from utils.models_utils import build_engagement_model

# --- KONFIGURASI ---
TRAIN_DIR = os.path.join("data", "cnn_dataset", "train")
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

def main(args):
    print("--- Memulai Tahap Training Model CNN ---")
    
    # 1. Memuat Dataset dari Direktori
    print(f"Memuat dataset dari: {TRAIN_DIR}")
    
    # Membuat dataset training (80%)
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    # Membuat dataset validasi (20%)
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_dataset.class_names
    print("Kelas yang ditemukan:", class_names)

    # Konfigurasi dataset untuk performa
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    
    # 2. Membangun Model
    # Kita panggil kembali fungsi dari utils yang sudah kita perbaiki untuk biner
    print("\nMembangun arsitektur model CNN...")
    model = build_engagement_model(num_classes=len(class_names)) # Otomatis mendeteksi jumlah kelas (2)
    model.summary()

    # 3. Melatih Model
    print(f"\nMemulai training untuk {args.epochs} epoch...")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs
    )

    # 4. Menyimpan Model Final
    model_save_path = os.path.join("models", "cnn_model_oversampling.keras")
    os.makedirs("models", exist_ok=True)
    model.save(model_save_path)
    print(f"\n✅ Training selesai. Model berhasil disimpan di: {model_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Pipeline for CNN-based Boredom Detection")
    parser.add_argument("--epochs", type=int, default=10, help="Jumlah epoch untuk training")
    args = parser.parse_args()
    main(args)