import os
import sys
import argparse
import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE2_ROOT = os.path.join(PROJECT_ROOT, "phase2_3d_upgrade")

if PHASE2_ROOT not in sys.path:
    sys.path.insert(0, PHASE2_ROOT)

from inference.infer_3d_pose import PoseFormerV2Inferencer


RAW_VIDEO_DIR = os.path.join(PROJECT_ROOT, "data", "1_raw_videos")
OUTPUT_3D_DIR = os.path.join(PROJECT_ROOT, "data", "3_extracted_3d")
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "yolov8m-pose.pt")


ACTION_LABELS = {
    "Fall": 0,
    "Sit": 1,
    "Sleep": 2,
    "Stand": 3,
    "Walking": 4
}


def get_video_label_info(video_path):
    normalized_path = video_path.replace("\\", "/")

    if "/Fall/" in normalized_path:
        binary_label = 1
        action_label = 0
        action_name = "Fall"
        return binary_label, action_label, action_name

    if "/Not_Fall/Sit/" in normalized_path:
        return 0, 1, "Sitting"

    if "/Not_Fall/Sleep/" in normalized_path:
        return 0, 2, "Sleeping"

    if "/Not_Fall/Stand/" in normalized_path:
        return 0, 3, "Standing"

    if "/Not_Fall/Walking/" in normalized_path:
        return 0, 4, "Walking"

    return None, None, None


def safe_output_name(video_path):
    relative_path = os.path.relpath(video_path, RAW_VIDEO_DIR)
    name = relative_path.replace("\\", "_").replace("/", "_")
    name = os.path.splitext(name)[0]
    return name + ".csv"


def find_all_videos():
    video_paths = []

    for root, dirs, files in os.walk(RAW_VIDEO_DIR):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_paths.append(os.path.join(root, file))

    return sorted(video_paths)


def pose3d_to_row(frame_idx, pose_3d, source_file, binary_label, action_label, action_name):
    row = {
        "frame": frame_idx
    }

    for joint_idx in range(17):
        row[f"x{joint_idx}"] = float(pose_3d[joint_idx, 0])
        row[f"y{joint_idx}"] = float(pose_3d[joint_idx, 1])
        row[f"z{joint_idx}"] = float(pose_3d[joint_idx, 2])

    row["source_file"] = source_file
    row["label"] = binary_label
    row["action_label"] = action_label
    row["action_name"] = action_name

    return row


def extract_3d_from_one_video(video_path, yolo_model, inferencer, max_frames=None):
    binary_label, action_label, action_name = get_video_label_info(video_path)

    if binary_label is None:
        print("Skipped unknown label video:", video_path)
        return None

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot open video:", video_path)
        return None

    inferencer.reset()

    rows = []
    frame_idx = 0
    valid_2d_count = 0
    generated_3d_count = 0

    source_file = os.path.relpath(video_path, RAW_VIDEO_DIR)

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        frame_idx += 1

        if max_frames is not None and frame_idx > max_frames:
            break

        results = yolo_model(frame, verbose=False)

        if len(results) == 0 or results[0].keypoints is None:
            continue

        keypoints_xy = results[0].keypoints.xy

        if keypoints_xy is None or len(keypoints_xy) == 0:
            continue

        coco_keypoints = keypoints_xy[0].detach().cpu().numpy()
        valid_2d_count += 1

        pose_3d = inferencer.add_and_predict(coco_keypoints)

        if pose_3d is None:
            continue

        generated_3d_count += 1

        row = pose3d_to_row(
            frame_idx=frame_idx,
            pose_3d=pose_3d,
            source_file=source_file,
            binary_label=binary_label,
            action_label=action_label,
            action_name=action_name
        )

        rows.append(row)

    cap.release()

    if len(rows) == 0:
        print(
            f"No 3D generated: {source_file} | "
            f"frames={frame_idx}, valid_2d={valid_2d_count}"
        )
        return None

    df = pd.DataFrame(rows)

    print(
        f"Done: {source_file} | "
        f"frames={frame_idx}, valid_2d={valid_2d_count}, generated_3d={generated_3d_count}"
    )

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos for testing"
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit frames per video for quick testing"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files"
    )

    args = parser.parse_args()

    os.makedirs(OUTPUT_3D_DIR, exist_ok=True)

    video_paths = find_all_videos()

    if args.limit is not None:
        video_paths = video_paths[:args.limit]

    print("Total videos found:", len(video_paths))
    print("Output dir:", OUTPUT_3D_DIR)

    yolo_model = YOLO(YOLO_MODEL_PATH)
    inferencer = PoseFormerV2Inferencer()

    processed = 0
    skipped = 0
    failed = 0

    for video_path in video_paths:
        output_name = safe_output_name(video_path)
        output_path = os.path.join(OUTPUT_3D_DIR, output_name)

        if os.path.exists(output_path) and not args.overwrite:
            skipped += 1
            print("Skipped existing:", output_name)
            continue

        try:
            df = extract_3d_from_one_video(
                video_path=video_path,
                yolo_model=yolo_model,
                inferencer=inferencer,
                max_frames=args.max_frames
            )

            if df is None:
                failed += 1
                continue

            df.to_csv(output_path, index=False)
            processed += 1

        except Exception as e:
            failed += 1
            print("Error video:", video_path)
            print("Error:", e)

    print()
    print("Finished extracting 3D keypoints.")
    print("Processed:", processed)
    print("Skipped existing:", skipped)
    print("Failed:", failed)
    print("Saved to:", OUTPUT_3D_DIR)


if __name__ == "__main__":
    main()