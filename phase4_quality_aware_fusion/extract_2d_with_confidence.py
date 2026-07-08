import os
import cv2
import json
import time
import argparse
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


"""
Phase 4 - Extract 2D pose with confidence.

Purpose:
    This script re-runs YOLOv8-Pose on raw videos and saves:
        - 2D keypoint coordinates: x, y
        - keypoint confidence: c
        - bounding box: bbox_x1, bbox_y1, bbox_x2, bbox_y2
        - bounding box confidence: bbox_conf

Why this is needed:
    Phase 1 only saved x/y keypoints. It did not save YOLO keypoint confidence.
    For Phase 4 quality-aware fusion, confidence is needed to compute quality features:
        - mean_confidence
        - min_confidence
        - missing_joint_ratio
        - pose reliability indicators

Default input:
    data/1_raw_videos

Default output:
    data/5_extracted_2d_confidence

Default behavior:
    - Keeps the same folder-structure-based file naming style as Phase 1.
    - Skips CSV files that already exist.
    - Selects the first detected person by default to stay close to the old extractor behavior.
"""


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "1_raw_videos")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "5_extracted_2d_confidence")

DEFAULT_PHASE4_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "phase4_quality_aware_fusion",
    "outputs",
    "confidence_extraction",
)

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "yolov8m-pose.pt")

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_device() -> int | str:
    if torch.cuda.is_available():
        print("GPU check: CUDA is available")
        return 0

    print("GPU check: CUDA is not available, using CPU")
    return "cpu"


def scan_videos(input_dir: str) -> List[str]:
    video_files = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(VIDEO_EXTENSIONS):
                video_files.append(os.path.join(root, file))

    video_files.sort()
    return video_files


def make_output_csv_path(video_path: str, input_dir: str, output_dir: str) -> str:
    relative_path = os.path.relpath(video_path, input_dir)
    relative_name = relative_path.replace("\\", "_").replace("/", "_")
    output_csv_name = os.path.splitext(relative_name)[0] + ".csv"
    return os.path.join(output_dir, output_csv_name)


def safe_float(value, default: float = np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def select_person_index(
    keypoints_xy: np.ndarray,
    keypoints_conf: Optional[np.ndarray],
    boxes_xyxy: Optional[np.ndarray],
    boxes_conf: Optional[np.ndarray],
    strategy: str,
) -> int:
    """
    Select which detected person to save.

    Available strategies:
        first:
            Use index 0. This is closest to the old Phase 1 extractor.

        highest_box_conf:
            Use person with the highest YOLO bounding-box confidence.

        largest_box:
            Use person with the largest bounding-box area.

        highest_keypoint_conf:
            Use person with the highest mean keypoint confidence.
    """
    num_persons = keypoints_xy.shape[0]

    if num_persons <= 0:
        return -1

    if strategy == "first":
        return 0

    if strategy == "highest_box_conf" and boxes_conf is not None and len(boxes_conf) >= num_persons:
        return int(np.nanargmax(boxes_conf[:num_persons]))

    if strategy == "largest_box" and boxes_xyxy is not None and len(boxes_xyxy) >= num_persons:
        widths = boxes_xyxy[:num_persons, 2] - boxes_xyxy[:num_persons, 0]
        heights = boxes_xyxy[:num_persons, 3] - boxes_xyxy[:num_persons, 1]
        areas = widths * heights
        return int(np.nanargmax(areas))

    if strategy == "highest_keypoint_conf" and keypoints_conf is not None and len(keypoints_conf) >= num_persons:
        mean_conf = np.nanmean(keypoints_conf[:num_persons], axis=1)
        return int(np.nanargmax(mean_conf))

    # Fallback keeps compatibility with old behavior.
    return 0


def build_columns() -> List[str]:
    columns = [
        "frame",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_conf",
        "person_index",
        "num_persons",
    ]

    for i in range(17):
        columns.extend([f"x{i}", f"y{i}", f"c{i}"])

    return columns


def extract_frame_pose_row(
    result,
    frame_idx: int,
    person_strategy: str,
) -> Optional[List[float]]:
    """
    Extract one row for one video frame.

    Return:
        list of values if a person is detected
        None if no valid person/keypoints detected
    """
    if result.keypoints is None:
        return None

    if result.keypoints.xy is None:
        return None

    if len(result.keypoints.xy) <= 0:
        return None

    keypoints_xy = result.keypoints.xy.cpu().numpy()

    if keypoints_xy.ndim != 3:
        return None

    if keypoints_xy.shape[0] <= 0 or keypoints_xy.shape[1] < 17:
        return None

    # Keypoint confidence may be unavailable in some cases.
    keypoints_conf = None
    if getattr(result.keypoints, "conf", None) is not None:
        keypoints_conf = result.keypoints.conf.cpu().numpy()

    boxes_xyxy = None
    boxes_conf = None

    if result.boxes is not None:
        if getattr(result.boxes, "xyxy", None) is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()

        if getattr(result.boxes, "conf", None) is not None:
            boxes_conf = result.boxes.conf.cpu().numpy()

    person_idx = select_person_index(
        keypoints_xy=keypoints_xy,
        keypoints_conf=keypoints_conf,
        boxes_xyxy=boxes_xyxy,
        boxes_conf=boxes_conf,
        strategy=person_strategy,
    )

    if person_idx < 0:
        return None

    # Keep only first 17 COCO keypoints.
    selected_xy = keypoints_xy[person_idx, :17, :2]

    if keypoints_conf is not None and person_idx < len(keypoints_conf):
        selected_conf = keypoints_conf[person_idx, :17]
    else:
        selected_conf = np.full((17,), np.nan, dtype=np.float32)

    if boxes_xyxy is not None and person_idx < len(boxes_xyxy):
        bbox = boxes_xyxy[person_idx].astype(float).tolist()
    else:
        bbox = [np.nan, np.nan, np.nan, np.nan]

    if boxes_conf is not None and person_idx < len(boxes_conf):
        bbox_conf = safe_float(boxes_conf[person_idx])
    else:
        bbox_conf = np.nan

    num_persons = int(keypoints_xy.shape[0])

    row = [
        int(frame_idx),
        safe_float(bbox[0]),
        safe_float(bbox[1]),
        safe_float(bbox[2]),
        safe_float(bbox[3]),
        bbox_conf,
        int(person_idx),
        num_persons,
    ]

    for i in range(17):
        row.append(safe_float(selected_xy[i, 0]))
        row.append(safe_float(selected_xy[i, 1]))
        row.append(safe_float(selected_conf[i]))

    return row


def process_single_video(
    model: YOLO,
    video_path: str,
    output_csv_path: str,
    device: int | str,
    person_strategy: str,
    progress_interval: int,
) -> Dict:
    video_name = os.path.basename(video_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {
            "video_path": video_path,
            "output_csv_path": output_csv_path,
            "status": "cannot_open",
            "total_frames_read": 0,
            "detected_frames": 0,
            "error": "Cannot open video",
        }

    rows = []
    frame_idx = 0
    yolo_error = None

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        try:
            results = model(frame, verbose=False, device=device)
        except Exception as exc:
            yolo_error = f"YOLO failed at frame {frame_idx}: {exc}"
            break

        # Ultralytics usually returns a list-like Results object with one result per image.
        for result in results:
            row = extract_frame_pose_row(
                result=result,
                frame_idx=frame_idx,
                person_strategy=person_strategy,
            )

            if row is not None:
                rows.append(row)

            # One input frame only has one result object, so break is safe.
            break

        frame_idx += 1

        if progress_interval > 0 and frame_idx % progress_interval == 0:
            print(f"Processed {frame_idx} frames in {video_name}")

    cap.release()

    if yolo_error is not None:
        return {
            "video_path": video_path,
            "output_csv_path": output_csv_path,
            "status": "yolo_error",
            "total_frames_read": frame_idx,
            "detected_frames": len(rows),
            "error": yolo_error,
        }

    if not rows:
        return {
            "video_path": video_path,
            "output_csv_path": output_csv_path,
            "status": "no_person_detected",
            "total_frames_read": frame_idx,
            "detected_frames": 0,
            "error": "No person detected",
        }

    df = pd.DataFrame(rows, columns=build_columns())
    df.to_csv(output_csv_path, index=False)

    return {
        "video_path": video_path,
        "output_csv_path": output_csv_path,
        "status": "processed",
        "total_frames_read": frame_idx,
        "detected_frames": len(df),
        "error": "",
    }


def save_summary(
    summary_records: List[Dict],
    output_report_dir: str,
    started_at: float,
    args: argparse.Namespace,
) -> None:
    ensure_dir(output_report_dir)

    summary_csv_path = os.path.join(output_report_dir, "confidence_extraction_summary.csv")
    summary_json_path = os.path.join(output_report_dir, "extraction_summary.json")
    failed_videos_path = os.path.join(output_report_dir, "failed_videos.txt")

    df = pd.DataFrame(summary_records)
    df.to_csv(summary_csv_path, index=False)

    status_counts = df["status"].value_counts().to_dict() if not df.empty else {}

    failed_df = df[df["status"].isin(["cannot_open", "yolo_error", "no_person_detected"])]

    with open(failed_videos_path, "w", encoding="utf-8") as f:
        for _, row in failed_df.iterrows():
            f.write(str(row["video_path"]) + "\n")

    elapsed_seconds = time.time() - started_at

    summary = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "model_path": args.model_path,
        "person_strategy": args.person_strategy,
        "reset": bool(args.reset),
        "num_videos_total": int(len(summary_records)),
        "status_counts": status_counts,
        "elapsed_seconds": elapsed_seconds,
        "summary_csv_path": summary_csv_path,
        "failed_videos_path": failed_videos_path,
        "columns": build_columns(),
        "note": (
            "This extraction saves 2D keypoint coordinates, keypoint confidence, "
            "bounding boxes, and bounding-box confidence for Phase 4 quality-aware fusion."
        ),
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print("\nSummary saved:")
    print(f"- {summary_csv_path}")
    print(f"- {summary_json_path}")
    print(f"- {failed_videos_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract YOLOv8-Pose 2D keypoints with confidence for Phase 4."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Input raw video directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for 2D keypoints with confidence CSV files.",
    )

    parser.add_argument(
        "--report-dir",
        type=str,
        default=DEFAULT_PHASE4_OUTPUT_DIR,
        help="Output directory for extraction summary files.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="YOLO pose model path.",
    )

    parser.add_argument(
        "--person-strategy",
        type=str,
        default="first",
        choices=["first", "highest_box_conf", "largest_box", "highest_keypoint_conf"],
        help=(
            "Person selection strategy. Use 'first' to stay closest to the old Phase 1 extractor."
        ),
    )

    parser.add_argument(
        "--progress-interval",
        type=int,
        default=30,
        help="Print progress every N frames. Set 0 to disable.",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reprocess and overwrite existing CSV files.",
    )

    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()

    ensure_dir(args.output_dir)
    ensure_dir(args.report_dir)

    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"YOLO model file not found: {args.model_path}")

    device = get_device()

    print("\nPhase 4 - 2D Pose Confidence Extraction")
    print("=" * 80)
    print(f"Input directory : {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Report directory: {args.report_dir}")
    print(f"Model path      : {args.model_path}")
    print(f"Person strategy : {args.person_strategy}")
    print(f"Reset           : {args.reset}")
    print("=" * 80)

    model = YOLO(args.model_path)

    video_files = scan_videos(args.input_dir)

    if args.max_videos is not None:
        video_files = video_files[: args.max_videos]

    if not video_files:
        print("No videos found.")
        return

    print(f"Found {len(video_files)} video(s).")

    summary_records = []

    for index, video_path in enumerate(video_files, start=1):
        output_csv_path = make_output_csv_path(
            video_path=video_path,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )

        if os.path.exists(output_csv_path) and not args.reset:
            print(f"\n[{index}/{len(video_files)}] Skipping existing CSV: {output_csv_path}")

            summary_records.append(
                {
                    "video_path": video_path,
                    "output_csv_path": output_csv_path,
                    "status": "skipped_existing",
                    "total_frames_read": np.nan,
                    "detected_frames": np.nan,
                    "error": "",
                }
            )
            continue

        print(f"\n[{index}/{len(video_files)}] Processing: {video_path}")

        try:
            record = process_single_video(
                model=model,
                video_path=video_path,
                output_csv_path=output_csv_path,
                device=device,
                person_strategy=args.person_strategy,
                progress_interval=args.progress_interval,
            )

            summary_records.append(record)

            if record["status"] == "processed":
                print(
                    f"Saved: {output_csv_path} "
                    f"({record['detected_frames']} detected frames)"
                )
            else:
                print(f"WARNING: {record['status']} - {record['error']}")

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break

        except Exception as exc:
            print(f"ERROR while processing video: {video_path}")
            print(f"Error detail: {exc}")
            traceback.print_exc()

            summary_records.append(
                {
                    "video_path": video_path,
                    "output_csv_path": output_csv_path,
                    "status": "exception",
                    "total_frames_read": np.nan,
                    "detected_frames": np.nan,
                    "error": str(exc),
                }
            )

    save_summary(
        summary_records=summary_records,
        output_report_dir=args.report_dir,
        started_at=started_at,
        args=args,
    )

    df = pd.DataFrame(summary_records)
    status_counts = df["status"].value_counts().to_dict() if not df.empty else {}

    print("\nDone.")
    print("=" * 80)
    print(f"Total videos handled: {len(summary_records)}")
    print(f"Status counts: {status_counts}")
    print("=" * 80)


if __name__ == "__main__":
    main()