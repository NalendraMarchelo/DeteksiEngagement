# train.py

import os
import argparse
import pandas as pd
from utils.preprocessing import load_daisee_dataset, prepare_data_for_training
from utils.models_utils import (
    build_engagement_model,
    train_model,
    plot_training_history,
    evaluate_model_metrics
)

def main(args):
    print("=== TAHAP 1: Mempersiapkan Dataset Training (DAiSEE) ===")
    daisee_frames_dir = "data/DAiSEE/Frames/Train"
    labels_file = "data/DAiSEE/Labels/TrainLabels.csv"
    
    X, y = load_daisee_dataset(daisee_frames_dir, labels_file)
    X_train, X_val, y_train, y_val, train_datagen = prepare_data_for_training(X, y)
    
    print("\n=== TAHAP 2: Membangun dan Melatih Model BINER ===")
    model = build_engagement_model(num_classes=2, learning_rate=1e-4)
    model.summary() # Tampilkan ringkasan model
    
    history = train_model(model, X_train, y_train, X_val, y_val, train_datagen, epochs=args.epochs)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    plot_training_history(history, save_path=os.path.join(output_dir, "binary_training_history.png"))
    
    print("\n=== TAHAP 3: Mengevaluasi Model BINER ===")
    evaluate_model_metrics(model, X_val, y_val, save_path=os.path.join(output_dir, "binary_evaluation_metrics.txt"))

    print("\n=== TAHAP 4: Menyimpan Model BINER Final ===")
    os.makedirs("models", exist_ok=True)
    model_save_path = os.path.join("models", args.model_name)
    model.save(model_save_path)
    print(f"✅ Model BINER berhasil disimpan di: {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Pipeline for BINARY Engagement Detection")
    parser.add_argument("--epochs", type=int, default=15, help="Jumlah epoch untuk training")
    parser.add_argument("--model_name", type=str, default="boredom_binary_classifier.keras", help="Nama file untuk model biner yang disimpan")
    args = parser.parse_args()
    main(args)