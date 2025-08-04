import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --- Konfigurasi ---
# Tentukan path ke direktori data latih dari skenario Normal
# Ini digunakan untuk memvalidasi asumsi pelabelan awal
BASE_DIR = "data/cnn_dataset_undersampling/train"
TIDAK_BINGUNG_DIR = os.path.join(BASE_DIR, "0_tidak_bingung")
BINGUNG_DIR = os.path.join(BASE_DIR, "1_bingung")

# Jumlah sampel gambar yang ingin ditampilkan untuk setiap kelas
NUM_SAMPLES = 5

# Nama file untuk menyimpan output plot
OUTPUT_FILE = "justifikasi_label_visual.png"

# --- Fungsi Utama ---

def visualize_and_save_samples():
    """
    Membuat dan menyimpan plot yang menampilkan sampel gambar acak
    untuk kelas 'bingung' dan 'tidak bingung' sebagai justifikasi visual.
    """
    print("Memulai proses visualisasi untuk justifikasi label...")

    try:
        # Mengambil daftar file gambar dari masing-masing direktori
        tidak_bingung_images = [f for f in os.listdir(TIDAK_BINGUNG_DIR) if f.endswith(('.jpg', '.png'))]
        bingung_images = [f for f in os.listdir(BINGUNG_DIR) if f.endswith(('.jpg', '.png'))]

        # Memastikan ada cukup gambar untuk dijadikan sampel
        if len(tidak_bingung_images) < NUM_SAMPLES or len(bingung_images) < NUM_SAMPLES:
            print("Error: Jumlah gambar di direktori kurang dari jumlah sampel yang dibutuhkan.")
            return

        # Mengambil sampel file secara acak
        random_tidak_bingung = random.sample(tidak_bingung_images, NUM_SAMPLES)
        random_bingung = random.sample(bingung_images, NUM_SAMPLES)

    except FileNotFoundError as e:
        print(f"Error: Direktori tidak ditemukan. Pastikan path sudah benar.")
        print(f"Detail: {e}")
        return
        
    # Membuat subplot dengan 2 baris (untuk 2 kelas) dan NUM_SAMPLES kolom
    fig, axes = plt.subplots(2, NUM_SAMPLES, figsize=(15, 6))
    
    # Menambahkan judul utama untuk keseluruhan gambar
    fig.suptitle("Justifikasi Visual: Perbandingan Sampel Gambar", fontsize=16)

    # Menampilkan gambar untuk kelas 'Tidak Bingung'
    for i, filename in enumerate(random_tidak_bingung):
        img_path = os.path.join(TIDAK_BINGUNG_DIR, filename)
        img = mpimg.imread(img_path)
        axes[0, i].imshow(img)
        axes[0, i].axis('off') # Menghilangkan sumbu x dan y
        if i == 0:
            axes[0, i].set_title("Kelas 0: Tidak Bingung", loc='left', fontsize=12)

    # Menampilkan gambar untuk kelas 'Bingung'
    for i, filename in enumerate(random_bingung):
        img_path = os.path.join(BINGUNG_DIR, filename)
        img = mpimg.imread(img_path)
        axes[1, i].imshow(img)
        axes[1, i].axis('off') # Menghilangkan sumbu x dan y
        if i == 0:
            axes[1, i].set_title("Kelas 1: Bingung", loc='left', fontsize=12)

    # Merapikan layout dan menyimpan gambar
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
    plt.savefig(OUTPUT_FILE)
    
    print(f"\nProses selesai.")
    print(f"Visualisasi berhasil disimpan sebagai '{OUTPUT_FILE}'.")
    print("Anda bisa memasukkan file gambar ini ke dalam laporan Anda.")
    
    # Menampilkan plot (opsional, bisa di-comment jika hanya ingin menyimpan)
    plt.show()


if __name__ == "__main__":
    visualize_and_save_samples()