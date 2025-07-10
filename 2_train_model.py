import pandas as pd
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb

warnings.filterwarnings('ignore')
DATA_PATH = os.path.join("data", "features_rich.csv")
MODEL_OUTPUT_NAME = "boredom_geometric_model.joblib"
PLOT_OUTPUT_NAME = "model_performance_comparison.png"
OUTPUT_DIR = "output"

def plot_and_save_cm(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Tidak Bosan', 'Bosan'], 
                yticklabels=['Tidak Bosan', 'Bosan'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_filename = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    save_path = os.path.join(OUTPUT_DIR, cm_filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Plot Confusion Matrix untuk {model_name} disimpan di: {save_path}")
# -------------------------------------------------------------

def main():
    print("Memulai tahap training model dengan fitur geometris...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Muat Dataset Fitur
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: File '{DATA_PATH}' tidak ditemukan.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"\nDataset berhasil dimuat dari '{DATA_PATH}'. Jumlah data: {len(df)}")
    print("Distribusi label awal:")
    print(df['label'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

    # 2. Persiapan Data
    X = df[['ear', 'mar', 'pitch', 'yaw', 'roll', 'eyebrow_dist', 'nose_mouth_dist']]
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

    # 4. Inisialisasi Model dan Penyimpanan Hasil
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    best_model, best_model_name, best_f1_score = None, "", 0.0
    results_for_plot = {}

    # 5. Latih dan Evaluasi Setiap Model
    for name, model in models.items():
        print(f"\n{'='*20}\n--- Melatih Model: {name} ---\n{'='*20}")
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
        print(f"Hasil Evaluasi untuk {name}:")
        print(classification_report(y_test, y_pred, target_names=['Tidak Bosan (0)', 'Bosan (1)']))
        cm = confusion_matrix(y_test, y_pred)
        plot_and_save_cm(cm, name)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        current_f1 = report_dict['1']['f1-score']
        macro_avg = report_dict['macro avg']
        results_for_plot[name] = {
            'precision': macro_avg['precision'],
            'recall': macro_avg['recall'],
            'f1-score': macro_avg['f1-score'],
            'accuracy': report_dict['accuracy']
        }
        if current_f1 > best_f1_score:
            best_f1_score = current_f1
            best_model = model
            best_model_name = name
            print(f"✨ Model terbaik sementara: {name} (F1-score 'Bosan' = {best_f1_score:.4f})")

    # 6. Buat dan Simpan Plot Perbandingan
    print(f"\n{'='*20}\n--- Membuat Plot Perbandingan Model ---\n{'='*20}")
    plot_save_path = os.path.join(OUTPUT_DIR, PLOT_OUTPUT_NAME)
    labels, metrics = list(results_for_plot.keys()), list(list(results_for_plot.values())[0].keys())
    x, width = np.arange(len(labels)), 0.2
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, metric in enumerate(metrics):
        metric_values = [results_for_plot[model][metric] for model in labels]
        offset = width * (i - (len(metrics) - 1) / 2)
        rects = ax.bar(x + offset, metric_values, width, label=metric.capitalize())
        ax.bar_label(rects, padding=3, fmt='%.2f')
    ax.set_ylabel('Scores'); ax.set_title('Perbandingan Performa Model (Macro Average & Accuracy)'); ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(loc='upper right'); ax.set_ylim(0, 1)
    fig.tight_layout(); plt.savefig(plot_save_path)
    print(f"📊 Plot perbandingan berhasil disimpan di: '{plot_save_path}'")
    
    # 7. Simpan Model Terbaik
    if best_model:
        model_save_path = os.path.join("models", MODEL_OUTPUT_NAME)
        joblib.dump(best_model, model_save_path)
        print(f"\n{'='*50}")
        print(f"✅ Training selesai. Model terbaik adalah '{best_model_name}'.")
        print(f"Model berhasil disimpan sebagai: '{model_save_path}'")
        print(f"{'='*50}")

if __name__ == '__main__':
    main()