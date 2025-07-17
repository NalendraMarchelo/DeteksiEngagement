import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- KONFIGURASI ---
OUTPUT_DIR = "output"
SCENARIOS = ["normal", "oversampling", "undersampling"]

def parse_report(filepath):
    """Fungsi untuk membaca file laporan dan mengekstrak metrik kunci."""
    results = {}
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
            # --- PERBAIKAN LOGIKA PARSING DI SINI ---
            # Mencari baris yang mengandung metrik 'accuracy'
            accuracy_line = next((line for line in lines if 'accuracy' in line), None)
            if accuracy_line:
                results['accuracy'] = float(accuracy_line.split()[1])

            # Mencari baris yang mengandung metrik untuk kelas '1_bingung'
            bingung_line = next((line for line in lines if '1_bingung' in line), None)
            if bingung_line:
                parts = bingung_line.split()
                # Indeks yang benar: [0]Label, [1]Precision, [2]Recall, [3]F1-Score
                results['precision_bingung'] = float(parts[1])
                results['recall_bingung'] = float(parts[2])
                results['f1_score_bingung'] = float(parts[3])
            # ----------------------------------------
            
        return results
    except (FileNotFoundError, IndexError, ValueError):
        print(f"Peringatan: File evaluasi tidak valid atau tidak ditemukan di {filepath}")
        return None

def main():
    """Fungsi utama untuk membandingkan semua skenario."""
    all_results = []
    for scenario in SCENARIOS:
        report_path = os.path.join(OUTPUT_DIR, scenario, f"evaluation_{scenario}.txt")
        result = parse_report(report_path)
        if result:
            result['scenario'] = scenario
            all_results.append(result)

    if not all_results:
        print("Tidak ada hasil evaluasi yang bisa dibandingkan. Jalankan skrip evaluasi terlebih dahulu.")
        return

    # Tampilkan tabel perbandingan
    df_results = pd.DataFrame(all_results)
    # Ganti nama kolom untuk kejelasan
    df_results.rename(columns={
        'scenario': 'Skenario',
        'accuracy': 'Akurasi Total',
        'recall_bingung': 'Recall (Bingung)',
        'f1_score_bingung': 'F1-Score (Bingung)'
    }, inplace=True)

    # Pilih kolom yang relevan untuk ditampilkan
    display_columns = ['Skenario', 'Akurasi Total', 'Recall (Bingung)', 'F1-Score (Bingung)']
    df_results = df_results[display_columns]

    print("\n--- Tabel Perbandingan Hasil Eksperimen ---")
    print(df_results.to_string(index=False))

    # Buat dan simpan plot perbandingan
    labels = df_results['Skenario']
    metrics_to_plot = {
        'Akurasi Total': df_results['Akurasi Total'],
        'Recall (Bingung)': df_results['Recall (Bingung)'],
        'F1-Score (Bingung)': df_results['F1-Score (Bingung)']
    }
    
    x = np.arange(len(labels))
    width = 0.25
    n_metrics = len(metrics_to_plot)

    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, (metric_name, metric_values) in enumerate(metrics_to_plot.items()):
        offset = width * (i - (n_metrics - 1) / 2)
        rects = ax.bar(x + offset, metric_values, width, label=metric_name)
        ax.bar_label(rects, padding=3, fmt='%.2f')

    ax.set_ylabel('Scores')
    ax.set_title('Perbandingan Performa Skenario Penanganan Data')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)

    fig.tight_layout()
    plot_save_path = os.path.join(OUTPUT_DIR, "final_scenario_comparison.png")
    plt.savefig(plot_save_path)
    print(f"\n📊 Plot perbandingan akhir berhasil disimpan di: '{plot_save_path}'")

if __name__ == '__main__':
    main()