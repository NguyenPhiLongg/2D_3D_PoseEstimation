import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE2_ROOT = os.path.join(PROJECT_ROOT, "phase2_3d_upgrade")

if PHASE2_ROOT not in sys.path:
    sys.path.insert(0, PHASE2_ROOT)

from inference.infer_3d_pose import PoseFormerV2Inferencer
from inference.visualize_3d import save_3d_pose_image


YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "yolov8m-pose.pt")

OUTPUT_IMAGE_PATH = os.path.join(
    PHASE2_ROOT,
    "outputs",
    "demo_3d_frames",
    "real_video_pose3d.png"
)


def find_first_video():
    """
    Find one video from the existing Phase 1 dataset.
    """

    raw_video_dir = os.path.join(PROJECT_ROOT, "data", "1_raw_videos")

    for root, dirs, files in os.walk(raw_video_dir):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                return os.path.join(root, file)

    return None


def extract_3d_from_video(video_path=None, max_frames=300):
    """
    Read a real video, extract YOLO 2D keypoints, run PoseFormerV2,
    and save one 3D skeleton image.
    """

    if video_path is None:
        video_path = find_first_video()

    if video_path is None:
        raise FileNotFoundError("No video found in data/1_raw_videos")

    print("Input video:", video_path)

    yolo_model = YOLO(YOLO_MODEL_PATH)
    inferencer = PoseFormerV2Inferencer()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = 0
    detected_pose_count = 0
    pose_3d = None

    while cap.isOpened() and frame_count < max_frames:
        success, frame = cap.read()

        if not success:
            break

        results = yolo_model(frame, verbose=False)

        if len(results) > 0 and results[0].keypoints is not None:
            keypoints_xy = results[0].keypoints.xy

            if keypoints_xy is not None and len(keypoints_xy) > 0:
                coco_keypoints = keypoints_xy[0].detach().cpu().numpy()

                pose_3d = inferencer.add_and_predict(coco_keypoints)

                detected_pose_count += 1

                if pose_3d is not None:
                    break

        frame_count += 1

    cap.release()

    print("Processed frames:", frame_count)
    print("Detected 2D pose frames:", detected_pose_count)

    if pose_3d is None:
        print("Not enough valid poses to generate 3D skeleton.")
        print("Try another video with a clearly visible person.")
        return

    save_3d_pose_image(
        pose_3d,
        OUTPUT_IMAGE_PATH,
        title="PoseFormerV2 3D Pose from Real Video"
    )

    print("Saved 3D pose image to:", OUTPUT_IMAGE_PATH)


if __name__ == "__main__":
    video_path = r"C:\Users\ASUS\Downloads\3D_Human_Pose_NCKH\data\1_raw_videos\Not_Fall\Walking\Walking While Using Phone (82).mp4"
    extract_3d_from_video(video_path=video_path)