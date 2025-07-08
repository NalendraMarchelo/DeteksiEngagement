# inference_interface.py

import os
import matplotlib.pyplot as plt
import gradio as gr
from utils.video_utils import extract_frames, detect_faces_in_video
from utils.preprocessing import preprocess_frames
from utils.models_utils import predict_engagement

def analyze_confusion(model, video_path, user_id):
    print(f"\nReceived video for user: {user_id}")
    frames_dir = f"data/frames/{user_id}"
    faces_dir = f"data/processed/{user_id}"

    # Bersihkan direktori lama jika ada
    for dir_path in [frames_dir, faces_dir]:
        if os.path.exists(dir_path):
            for f in os.listdir(dir_path):
                os.remove(os.path.join(dir_path, f))

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(faces_dir, exist_ok=True)

    extract_frames(video_path, frames_dir, frame_interval=10)
    detect_faces_in_video(video_path, faces_dir)
    user_frames = preprocess_frames(faces_dir)
    predictions = predict_engagement(model, user_frames)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Confusion"], [predictions["confusion"]])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Predicted Confusion Level")

    return fig


def launch_gradio_interface(model):
    iface = gr.Interface(
        fn=lambda video, user_id: analyze_confusion(model, video, user_id),
        inputs=[
            gr.Video(label="Upload your video"),
            gr.Textbox(label="User ID", placeholder="e.g. user_001")
        ],
        outputs=gr.Plot(label="Confusion Score"),
        title="Confusion Detector",
        description="Upload a video and provide a user ID to get predicted confusion score."
    )
    iface.launch()
