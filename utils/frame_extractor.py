import os
import cv2

def extract_frames_from_daisee_folder(input_dir, output_dir, interval=30):
    """
    Mengekstrak frame dari seluruh file .avi dalam folder input_dir dan menyimpannya ke output_dir.
    Frame disimpan dalam subfolder sesuai clip_id: <output_dir>/<clip_id>/frame000.jpg
    """
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".avi"):
                video_path = os.path.join(root, file)
                clip_id = os.path.splitext(file)[0]  # Misal: 1100011002

                # Buat folder per clip
                clip_output_dir = os.path.join(output_dir, clip_id)
                os.makedirs(clip_output_dir, exist_ok=True)

                cap = cv2.VideoCapture(video_path)
                frame_num = 0
                saved_frame = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_num % interval == 0:
                        frame_filename = f"frame{saved_frame:03d}.jpg"
                        frame_path = os.path.join(clip_output_dir, frame_filename)
                        cv2.imwrite(frame_path, frame)
                        saved_frame += 1

                    frame_num += 1

                cap.release()
                print(f"[✓] Extracted {saved_frame} frames from {file}")


def extract_all_splits(base_input_dir="data/DAiSEE/DataSet", base_output_dir="data/DAiSEE/Frames", interval=30):
    """
    Mengekstrak frame untuk semua split: Train, Validation, Test.
    """
    splits = ["Train", "Validation", "Test"]
    for split in splits:
        input_dir = os.path.join(base_input_dir, split)
        output_dir = os.path.join(base_output_dir, split)
        print(f"\n=== Processing {split} split ===")
        extract_frames_from_daisee_folder(input_dir, output_dir, interval=interval)


# Jalankan hanya jika file ini dijalankan langsung
if __name__ == "__main__":
    extract_all_splits(
        base_input_dir="data/DAiSEE/DataSet",
        base_output_dir="data/DAiSEE/Frames",
        interval=30  # ekstrak 1 frame tiap 30 frame (1 detik jika fps 30)
    )
