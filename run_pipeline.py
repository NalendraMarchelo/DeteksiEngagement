#!/usr/bin/env python
# engagement_project/run_pipeline.py

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.video_utils import extract_frames, detect_faces_in_video
from utils.preprocessing import load_daisee_dataset, preprocess_frames, prepare_data_for_training
from utils.models_utils import (
    build_engagement_model,
    train_model,
    plot_training_history,
    predict_and_save_results,
    evaluate_model_metrics
)
from inference_interface import launch_gradio_interface


def process_user_video(user_id, video_path, frame_interval=10):
    print(f"\n=== Processing Video for {user_id} ===")
    frames_dir = f"data/frames/{user_id}"
    faces_dir = f"data/processed/{user_id}"

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(faces_dir, exist_ok=True)

    print(f"Extracting frames from {video_path} (every {frame_interval} frames)...")
    extract_frames(video_path, frames_dir, frame_interval=frame_interval)

    print("Detecting faces in video...")
    detect_faces_in_video(video_path, faces_dir)

    return faces_dir


def prepare_dataset(faces_dir=None):
    print("\n=== Preparing Dataset ===")
    daisee_frames_dir = "data/DAiSEE/Frames/Train"
    labels_file = "data/DAiSEE/Labels/TrainLabels.csv"

    print("Loading DAiSEE dataset...")
    X, y = load_daisee_dataset(daisee_frames_dir, labels_file)

    print("Label distribution:", pd.Series(y).value_counts().sort_index().to_dict())

    X_train, X_val, y_train, y_val, train_datagen = prepare_data_for_training(X, y)

    user_frames = None
    if faces_dir:
        print("Preprocessing user frames...")
        user_frames = preprocess_frames(faces_dir)

    return X_train, X_val, y_train, y_val, user_frames, train_datagen


def train_and_evaluate_model(X_train, y_train, X_val, y_val, train_datagen, epochs, user_id, num_classes):
    print("\n=== Training Model ===")
    model = build_engagement_model(num_classes=num_classes)
    history = train_model(model, X_train, y_train, X_val, y_val, train_datagen, epochs=epochs, user_id=user_id)
    plot_training_history(history, user_id=user_id)

    print("\n=== Evaluating Model Performance ===")
    evaluate_model_metrics(model, X_val, y_val, user_id=user_id)
    return model


def main():
    parser = argparse.ArgumentParser(description="Engagement Detection Pipeline")
    parser.add_argument("--user_id", default="user_001", help="User ID (used for saving models/results)")
    parser.add_argument("--frame_interval", type=int, default=10, help="Interval for frame extraction (for user video)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--gradio", action="store_true", help="Launch Gradio interface after training")
    parser.add_argument("--skip_user_video_processing", action="store_true", 
                        help="Skip processing user video and direct prediction. Only train and evaluate.")
    args = parser.parse_args()

    try:
        user_frames = None
        faces_dir = None
        num_classes = 4  # Target label: confusion (0–3)

        if not args.skip_user_video_processing:
            video_path = f"data/raw_videos/{args.user_id}/recording.mp4"
            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"User video not found at: {video_path}")
            faces_dir = process_user_video(args.user_id, video_path, args.frame_interval)

        X_train, X_val, y_train, y_val, user_frames_processed, train_datagen = prepare_dataset(faces_dir)

        if user_frames_processed is not None:
            user_frames = user_frames_processed

        model = train_and_evaluate_model(X_train, y_train, X_val, y_val, train_datagen,
                                         args.epochs, args.user_id, num_classes=num_classes)

        if user_frames is not None:
            predict_and_save_results(model, user_frames, args.user_id)  # tidak perlu `num_classes` lagi
        else:
            print("\nSkipping user video prediction as no user video was processed.")

        if args.gradio:
            launch_gradio_interface(model)

    except Exception as e:
        print(f"\n❌ Error in pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
