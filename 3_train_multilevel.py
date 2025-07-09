import pandas as pd
import joblib
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler # Menggunakan Undersampler
import xgboost as xgb

warnings.filterwarnings('ignore')

# --- KONFIGURASI ---
FEATURES_PATH = os.path.join("data", "features_geometric.csv")
MODEL_OUTPUT_NAME = "boredom_multilevel_model.joblib"

def main():
    """Fungsi utama untuk melatih model klasifikasi multi-level."""
    
    print("Memulai tahap training model MULTI-LEVEL dengan fitur geometris...")
    
    # 1. Muat Dataset Fitur (Fitur tetap sama, kita hanya akan mengubah cara kita menggunakan label)
    if not os.path.exists(FEATURES_PATH):
        print(f"❌ Error: File '{FEATURES_PATH}' tidak ditemukan.")
        return

    df = pd.read_csv(FEATURES_PATH)
    
    # Ambil data asli dari file label untuk mendapatkan 4 level
    labels_df = pd.read_csv("data/DAiSEE/Labels/TrainLabels.csv")
    labels_df.columns = [col.strip() for col in labels_df.columns]
    
    # Ganti kolom 'label' di dataframe fitur dengan label 4-level
    df['label'] = labels_df['Confusion']

    print(f"\nDataset dimuat. Jumlah data: {len(df)}")
    print("Distribusi label awal (0-3):")
    print(df['label'].value_counts().sort_index())

    # 2. Persiapan Data
    X = df[['ear', 'mar']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Seimbangkan Data Training dengan Random Undersampling
    print("\nMenyeimbangkan data training dengan Random Undersampling...")
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
    print("Distribusi label setelah Undersampling:")
    print(pd.Series(y_train_resampled).value_counts().sort_index())

    # 4. Inisialisasi dan Latih Model
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    }

    best_model = None
    best_model_name = ""
    best_f1_score = 0.0

    for name, model in models.items():
        print(f"\n{'='*20}\n--- Melatih Model: {name} ---\n{'='*20}")
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
        
        print(f"Hasil Evaluasi untuk {name}:")
        print(classification_report(y_test, y_pred))
        
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        current_f1 = report_dict['macro avg']['f1-score']
        
        if current_f1 > best_f1_score:
            best_f1_score = current_f1
            best_model = model
            best_model_name = name
            print(f"✨ Model terbaik sementara: {name} (Macro F1-score = {best_f1_score:.4f})")

    # 5. Simpan Model Terbaik
    if best_model:
        model_save_path = os.path.join("models", MODEL_OUTPUT_NAME)
        joblib.dump(best_model, model_save_path)
        print(f"\n{'='*50}")
        print(f"✅ Training selesai. Model terbaik adalah '{best_model_name}'.")
        print(f"Model berhasil disimpan sebagai: '{model_save_path}'")
        print(f"{'='*50}")

if __name__ == '__main__':
    main()