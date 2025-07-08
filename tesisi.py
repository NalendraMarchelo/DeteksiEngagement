import os
import pandas as pd

FRAMES_FOLDER = "data/DAiSEE/Frames/Train"
LABEL_CSV = "data/DAiSEE/Labels/TrainLabels.csv"

def get_clipids_from_folder(frames_folder):
    clipids = set()
    for f in os.listdir(frames_folder):
        path = os.path.join(frames_folder, f)
        if os.path.isdir(path):
            clipids.add(f)
    return clipids

def get_clipids_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    if 'ClipID' not in df.columns:
        raise ValueError("Kolom 'ClipID' tidak ditemukan di CSV label")
    # Hapus ekstensi .avi, .mp4, dll dari nama file
    clipids = set(df['ClipID'].astype(str).str.replace(r'\.\w+$', '', regex=True))
    return clipids

def main():
    print("Mengambil ClipID dari folder Frames/Train...")
    folder_clipids = get_clipids_from_folder(FRAMES_FOLDER)
    print(f"Jumlah ClipID (folder) di Frames/Train: {len(folder_clipids)}")

    print("Mengambil ClipID dari CSV label...")
    csv_clipids = get_clipids_from_csv(LABEL_CSV)
    print(f"Jumlah ClipID di CSV label: {len(csv_clipids)}")

    missing_clipids = csv_clipids - folder_clipids
    print(f"\nJumlah ClipID yang ada di CSV tapi tidak ditemukan folder di Frames/Train: {len(missing_clipids)}")
    if missing_clipids:
        print("Contoh ClipID yang tidak ditemukan:")
        for i, clipid in enumerate(sorted(missing_clipids)):
            print(f" - {clipid}")
            if i >= 4:
                break

    extra_clipids = folder_clipids - csv_clipids
    print(f"\nJumlah ClipID yang ada di folder Frames/Train tapi tidak ada di CSV label: {len(extra_clipids)}")
    if extra_clipids:
        print("Contoh ClipID ekstra di folder:")
        for i, clipid in enumerate(sorted(extra_clipids)):
            print(f" - {clipid}")
            if i >= 4:
                break

if __name__ == "__main__":
    main()
