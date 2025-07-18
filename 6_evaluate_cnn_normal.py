# 6_evaluate_cnn_normal.py

import tensorflow as tf
import os
from utils.models_utils import evaluate_and_plot_cm

MODEL_PATH = os.path.join("models", "cnn_model_normal.keras")
DATA_DIR = os.path.join("data", "cnn_dataset_normal", "train")
OUTPUT_DIR = os.path.join("output", "normal")
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

def main():
    print(f"--- Memuat Model dari: {MODEL_PATH} ---")
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: File model tidak ditemukan.")
        return
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print(f"--- Memuat Data Validasi dari: {DATA_DIR} ---")
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation",
        seed=123, image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE
    )
    
    class_names = ['0_tidak_bingung', '1_bingung']
    print(f"Kelas yang akan dievaluasi: {class_names}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_save_path = os.path.join(OUTPUT_DIR, "evaluation_normal.txt")
    
    evaluate_and_plot_cm(model, validation_dataset, report_save_path, class_names)
    print(f"\n✅ Evaluasi selesai. Laporan disimpan di folder '{OUTPUT_DIR}'.")

if __name__ == '__main__':
    main()