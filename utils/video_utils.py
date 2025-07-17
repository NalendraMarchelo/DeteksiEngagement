import cv2
import os
from tqdm import tqdm
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def extract_frames(video_path, output_dir, frame_interval=10):
    """
    Extract frames from video at specified intervals
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_file = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_file, frame)
                saved_count += 1
                
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    return saved_count

def detect_faces_in_video(video_path, output_dir):
    """
    Detect faces in video and save cropped face images
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        
        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(frame_rgb)
                
                if results.detections:
                    for detection in results.detections:
                        # Get face bounding box
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                     int(bboxC.width * iw), int(bboxC.height * ih)
                        
                        # Crop and save face
                        face = frame[y:y+h, x:x+w]
                        if face.size > 0:
                            face_file = os.path.join(output_dir, f"face_{frame_count:06d}.jpg")
                            cv2.imwrite(face_file, face)
                
                frame_count += 1
                pbar.update(1)
    
    cap.release()
    return frame_count