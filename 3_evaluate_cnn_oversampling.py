import tensorflow as tf
import os
from utils.models_utils import evaluate_and_plot_cm

MODEL_PATH = os.path.join("models", "cnn_model_oversampling.keras")
DATA_DIR = os.path.join("data", "cnn_dataset", "train")
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

def main():
    print(f"--- Memuat Model dari: {MODEL_PATH} ---")
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: File model tidak ditemukan. Jalankan train_cnn.py terlebih dahulu.")
        return
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print(f"--- Memuat Data Uji/Validasi dari: {DATA_DIR} ---")
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
    
    class_names = validation_dataset.class_names
    print(f"Kelas yang akan dievaluasi: {class_names}")

    scenario_name = "oversampling"
    output_dir = os.path.join("output", scenario_name)
    os.makedirs(output_dir, exist_ok=True)
    report_save_path = os.path.join(output_dir, f"evaluation_{scenario_name}.txt")
    
    evaluate_and_plot_cm(model, validation_dataset, report_save_path, class_names)
    print(f"\n✅ Evaluasi selesai. Laporan lengkap dan confusion matrix disimpan di folder '{output_dir}'.")

if __name__ == '__main__':
    main()