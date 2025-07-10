# 2_train_model.py

import pandas as pd
import joblib
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Abaikan peringatan yang mungkin muncul agar output lebih bersih
warnings.filterwarnings('ignore')

# --- KONFIGURASI ---
# Path sekarang mengarah ke dalam folder 'data'
DATA_PATH = os.path.join("data", "features_rich.csv")
MODEL_OUTPUT_NAME = "boredom_geometric_model.joblib"

def main():
    """Fungsi utama untuk melatih, mengevaluasi, dan menyimpan model."""
    
    print("Memulai tahap training model dengan fitur geometris...")
    
    # 1. Muat Dataset Fitur
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: File '{DATA_PATH}' tidak ditemukan.")
        print("Pastikan Anda sudah menjalankan skrip '1_extract_features.py' terlebih dahulu.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"\nDataset berhasil dimuat dari '{DATA_PATH}'. Jumlah data: {len(df)}")
    print("Distribusi label awal:")
    print(df['label'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

    # 2. Persiapan Data
    X = df[['ear', 'mar', 'pitch', 'yaw', 'roll', 'eyebrow_dist', 'nose_mouth_dist']] # Gunakan semua fitur
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Seimbangkan Data Training dengan SMOTE
    print("\nMenyeimbangkan data training dengan SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("Distribusi label setelah SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())

    # 4. Inisialisasi Model
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    best_model = None
    best_model_name = ""
    best_f1_score = 0.0

    # 5. Latih dan Evaluasi Setiap Model
    for name, model in models.items():
        print(f"\n{'='*20}\n--- Melatih Model: {name} ---\n{'='*20}")
        
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
        
        print(f"Hasil Evaluasi untuk {name}:")
        print(classification_report(y_test, y_pred, target_names=['Tidak Bosan (0)', 'Bosan (1)']))
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        report_dict = classification_report(y_test, y_pred, output_dict=True)
        current_f1 = report_dict['1']['f1-score'] # Menggunakan kunci '1' untuk kelas "Bosan"
        
        if current_f1 > best_f1_score:
            best_f1_score = current_f1
            best_model = model
            best_model_name = name
            print(f"✨ Model terbaik sementara: {name} (F1-score 'Bosan' = {best_f1_score:.4f})")

    # 6. Simpan Model Terbaik
    if best_model:
        model_save_path = os.path.join("models", MODEL_OUTPUT_NAME)
        joblib.dump(best_model, model_save_path)
        print(f"\n{'='*50}")
        print(f"✅ Training selesai. Model terbaik adalah '{best_model_name}'.")
        print(f"Model berhasil disimpan sebagai: '{model_save_path}'")
        print(f"{'='*50}")

if __name__ == '__main__':
    main()