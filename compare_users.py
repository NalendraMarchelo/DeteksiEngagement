import os
import json
import matplotlib.pyplot as plt

def load_confusion_scores(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data["user_id"], data["confusion"]

def main():
    base_dir = "data/results"
    user_scores = {}

    # Telusuri semua folder user
    for user_folder in os.listdir(base_dir):
        user_path = os.path.join(base_dir, user_folder, "results.json")
        if os.path.exists(user_path):
            try:
                user_id, score = load_confusion_scores(user_path)
                user_scores[user_id] = score
            except Exception as e:
                print(f"❌ Gagal membaca {user_path}: {e}")

    # Tampilkan hasil
    print("\n=== Hasil Confusion Score per User ===")
    for user, score in user_scores.items():
        print(f"{user}: {score:.4f}")

    # Visualisasi
    if user_scores:
        plt.figure(figsize=(8, 4))
        plt.bar(user_scores.keys(), user_scores.values(), color='skyblue')
        plt.ylim(0, 1)
        plt.title("Perbandingan Tingkat Confusion antar User")
        plt.ylabel("Confusion Score")
        plt.xlabel("User ID")
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ Tidak ada skor yang bisa divisualisasikan.")

if __name__ == "__main__":
    main()
