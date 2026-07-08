import os
import sys
import json
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


# ============================================================
# PATH SETUP
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE5_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PHASE5_DIR.parent

if str(PHASE5_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE5_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT PHASE 5 UTILS
# ============================================================

from phase5_utils import (
    load_config,
    cfg_path,
    ensure_dir,
    save_csv,
    save_json,
    read_csv,
    bool_from_value,
    get_device,
    get_2d_feature_path,
    make_safe_filename,
    print_dict,
    print_dataframe_summary,
)


# ============================================================
# OPTIONAL TQDM
# ============================================================

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ============================================================
# YOLO IMPORT
# ============================================================

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "Ultralytics is required for YOLOv8 pose extraction.\n"
        "Install it with:\n"
        "    pip install ultralytics"
    ) from exc


# ============================================================
# CONSTANTS
# ============================================================

NUM_KEYPOINTS = 17

OUTPUT_COLUMNS = [
    "frame",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_conf",
    "person_index",
    "num_persons",
]

for i in range(NUM_KEYPOINTS):
    OUTPUT_COLUMNS.extend([f"x{i}", f"y{i}", f"c{i}"])


# ============================================================
# BASIC HELPERS
# ============================================================

def iter_progress(items, desc="Processing"):
    if tqdm is not None:
        return tqdm(items, desc=desc)

    return items


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        value = float(value)

        if np.isnan(value) or np.isinf(value):
            return default

        return value
    except Exception:
        return default


def empty_pose_row(frame_idx: int, num_persons: int = 0) -> dict:
    """
    Return one row when YOLO does not detect a person.

    Important:
        We use 0 instead of NaN so later model input does not break.
        Missing joints are represented by confidence c_i = 0.
    """
    row = {
        "frame": int(frame_idx),
        "bbox_x1": 0.0,
        "bbox_y1": 0.0,
        "bbox_x2": 0.0,
        "bbox_y2": 0.0,
        "bbox_conf": 0.0,
        "person_index": -1,
        "num_persons": int(num_persons),
    }

    for i in range(NUM_KEYPOINTS):
        row[f"x{i}"] = 0.0
        row[f"y{i}"] = 0.0
        row[f"c{i}"] = 0.0

    return row


def select_largest_bbox(boxes_xyxy: np.ndarray) -> int:
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return -1

    widths = np.maximum(0.0, boxes_xyxy[:, 2] - boxes_xyxy[:, 0])
    heights = np.maximum(0.0, boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    areas = widths * heights

    return int(np.argmax(areas))


def result_to_pose_row(result, frame_idx: int, select_person_strategy: str = "largest_bbox") -> dict:
    """
    Convert one YOLO result into one row.

    Output format:
        frame,
        bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_conf,
        person_index, num_persons,
        x0, y0, c0, ..., x16, y16, c16
    """
    boxes = result.boxes
    keypoints = result.keypoints

    if boxes is None or keypoints is None:
        return empty_pose_row(frame_idx, num_persons=0)

    if boxes.xyxy is None:
        return empty_pose_row(frame_idx, num_persons=0)

    boxes_xyxy = boxes.xyxy.detach().cpu().numpy()

    if boxes_xyxy.ndim == 1:
        boxes_xyxy = boxes_xyxy.reshape(1, -1)

    num_persons = int(len(boxes_xyxy))

    if num_persons <= 0:
        return empty_pose_row(frame_idx, num_persons=0)

    if boxes.conf is not None:
        boxes_conf = boxes.conf.detach().cpu().numpy()
    else:
        boxes_conf = np.ones((num_persons,), dtype=np.float32)

    if keypoints.xy is None:
        return empty_pose_row(frame_idx, num_persons=num_persons)

    kpts_xy = keypoints.xy.detach().cpu().numpy()

    if kpts_xy.ndim == 2:
        kpts_xy = kpts_xy.reshape(1, kpts_xy.shape[0], kpts_xy.shape[1])

    if keypoints.conf is not None:
        kpts_conf = keypoints.conf.detach().cpu().numpy()

        if kpts_conf.ndim == 1:
            kpts_conf = kpts_conf.reshape(1, -1)
    else:
        kpts_conf = np.ones((kpts_xy.shape[0], NUM_KEYPOINTS), dtype=np.float32)

    usable_persons = min(num_persons, kpts_xy.shape[0], kpts_conf.shape[0])

    if usable_persons <= 0:
        return empty_pose_row(frame_idx, num_persons=num_persons)

    boxes_xyxy = boxes_xyxy[:usable_persons]
    boxes_conf = boxes_conf[:usable_persons]
    kpts_xy = kpts_xy[:usable_persons]
    kpts_conf = kpts_conf[:usable_persons]

    if select_person_strategy == "largest_bbox":
        selected_idx = select_largest_bbox(boxes_xyxy)
    else:
        selected_idx = 0

    if selected_idx < 0:
        return empty_pose_row(frame_idx, num_persons=num_persons)

    selected_box = boxes_xyxy[selected_idx]
    selected_box_conf = boxes_conf[selected_idx]

    row = {
        "frame": int(frame_idx),
        "bbox_x1": safe_float(selected_box[0]),
        "bbox_y1": safe_float(selected_box[1]),
        "bbox_x2": safe_float(selected_box[2]),
        "bbox_y2": safe_float(selected_box[3]),
        "bbox_conf": safe_float(selected_box_conf),
        "person_index": int(selected_idx),
        "num_persons": int(num_persons),
    }

    for i in range(NUM_KEYPOINTS):
        if i < kpts_xy.shape[1]:
            row[f"x{i}"] = safe_float(kpts_xy[selected_idx, i, 0])
            row[f"y{i}"] = safe_float(kpts_xy[selected_idx, i, 1])
        else:
            row[f"x{i}"] = 0.0
            row[f"y{i}"] = 0.0

        if i < kpts_conf.shape[1]:
            row[f"c{i}"] = safe_float(kpts_conf[selected_idx, i])
        else:
            row[f"c{i}"] = 0.0

    return row


# ============================================================
# VIDEO EXTRACTION
# ============================================================

def extract_2d_confidence_for_video(
    video_path: Path,
    output_path: Path,
    model,
    device,
    image_size: int = 640,
    conf_threshold: float = 0.25,
    select_person_strategy: str = "largest_bbox",
    overwrite: bool = False,
    verbose_every: int = 500,
) -> dict:
    """
    Extract YOLOv8 pose with confidence for one video.

    This writes exactly one CSV row per video frame.

    Fairness note:
        We extract features per video once.
        Later every model will read from the same feature files and the same
        sequence manifest, not from separate model-specific sample sets.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        try:
            existing_df = pd.read_csv(output_path)
            return {
                "video_path": str(video_path),
                "output_path": str(output_path),
                "status": "skipped_existing",
                "frames_total": int(len(existing_df)),
                "frames_written": int(len(existing_df)),
                "no_person_frames": int((existing_df["num_persons"] == 0).sum()) if "num_persons" in existing_df.columns else -1,
                "multi_person_frames": int((existing_df["num_persons"] > 1).sum()) if "num_persons" in existing_df.columns else -1,
                "elapsed_sec": 0.0,
                "error": "",
            }
        except Exception:
            pass

    if not video_path.exists():
        return {
            "video_path": str(video_path),
            "output_path": str(output_path),
            "status": "failed",
            "frames_total": 0,
            "frames_written": 0,
            "no_person_frames": 0,
            "multi_person_frames": 0,
            "elapsed_sec": 0.0,
            "error": "video_file_not_found",
        }

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "video_path": str(video_path),
            "output_path": str(output_path),
            "status": "failed",
            "frames_total": 0,
            "frames_written": 0,
            "no_person_frames": 0,
            "multi_person_frames": 0,
            "elapsed_sec": 0.0,
            "error": "cannot_open_video",
        }

    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rows = []
    frame_idx = 0
    no_person_frames = 0
    multi_person_frames = 0

    start_time = time.time()

    yolo_device = 0 if device.type == "cuda" else "cpu"

    while True:
        ok, frame = cap.read()

        if not ok or frame is None:
            break

        try:
            results = model.predict(
                source=frame,
                imgsz=image_size,
                conf=conf_threshold,
                device=yolo_device,
                verbose=False,
            )

            if results is None or len(results) == 0:
                row = empty_pose_row(frame_idx, num_persons=0)
            else:
                row = result_to_pose_row(
                    result=results[0],
                    frame_idx=frame_idx,
                    select_person_strategy=select_person_strategy,
                )

        except Exception as exc:
            row = empty_pose_row(frame_idx, num_persons=0)
            row["error"] = str(exc)

        if row.get("num_persons", 0) == 0:
            no_person_frames += 1

        if row.get("num_persons", 0) > 1:
            multi_person_frames += 1

        rows.append(row)

        frame_idx += 1

        if verbose_every and frame_idx % verbose_every == 0:
            print(f"    processed {frame_idx}/{frame_total} frames")

    cap.release()

    elapsed = time.time() - start_time

    if not rows:
        return {
            "video_path": str(video_path),
            "output_path": str(output_path),
            "status": "failed",
            "frames_total": int(frame_total),
            "frames_written": 0,
            "no_person_frames": 0,
            "multi_person_frames": 0,
            "elapsed_sec": float(elapsed),
            "error": "no_frames_extracted",
        }

    df = pd.DataFrame(rows)

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    df = df[OUTPUT_COLUMNS]
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return {
        "video_path": str(video_path),
        "output_path": str(output_path),
        "status": "success",
        "frames_total": int(frame_total),
        "frames_written": int(len(df)),
        "no_person_frames": int(no_person_frames),
        "multi_person_frames": int(multi_person_frames),
        "elapsed_sec": float(elapsed),
        "error": "",
    }


# ============================================================
# METADATA / VIDEO LIST
# ============================================================

def load_unique_external_videos(config: dict) -> pd.DataFrame:
    metadata_path = cfg_path(config, config["outputs"]["all_metadata_csv"])

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            "Please run Step 01 first:\n"
            "python phase5_external_generalization/scripts/01_prepare_external_dataset.py"
        )

    df = pd.read_csv(metadata_path)

    if "include_eval" in df.columns:
        df = df[df["include_eval"].apply(bool_from_value)].copy()

    if "video_path" not in df.columns:
        raise ValueError("Metadata must contain column: video_path")

    df["video_path"] = df["video_path"].astype(str)

    df = df[df["video_path"].str.len() > 0].copy()

    required_cols = [
        "dataset",
        "video_id",
        "video_path",
        "frame_count",
        "fps",
        "width",
        "height",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    unique_df = df[required_cols].drop_duplicates(
        subset=["dataset", "video_id", "video_path"]
    ).reset_index(drop=True)

    return unique_df


def filter_video_df(video_df: pd.DataFrame, datasets=None, max_videos=None, start_index: int = 0) -> pd.DataFrame:
    df = video_df.copy()

    if datasets:
        datasets = set(datasets)
        df = df[df["dataset"].isin(datasets)].copy()

    df = df.reset_index(drop=True)

    if start_index > 0:
        df = df.iloc[start_index:].copy()

    if max_videos is not None:
        df = df.head(int(max_videos)).copy()

    return df.reset_index(drop=True)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 Step 02 - Extract external 2D pose with confidence using YOLOv8."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(PHASE5_DIR / "phase5_config.yaml"),
        help="Path to phase5_config.yaml",
    )

    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset filter, e.g. --datasets Le2i MulCamFall",
    )

    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Process only first N videos for quick testing.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start processing from this index in the filtered unique video list.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted 2D CSV files.",
    )

    args = parser.parse_args()

    print("\nPhase 5 - Step 02: Extract 2D Pose + Confidence")
    print("=" * 100)

    config = load_config(args.config)

    runtime_cfg = config.get("runtime", {})
    seed = int(runtime_cfg.get("seed", 42))

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_device(config)

    pose_cfg = config.get("pose_extraction", {})
    yolo_model_path = pose_cfg.get("yolo_model", "yolov8m-pose.pt")
    image_size = int(pose_cfg.get("image_size", 640))
    conf_threshold = float(pose_cfg.get("confidence_threshold", 0.25))
    select_person_strategy = str(pose_cfg.get("select_person_strategy", "largest_bbox"))

    extraction_output_dir = cfg_path(config, config["outputs"]["extraction_2d_dir"])
    extracted_2d_data_dir = cfg_path(config, config["outputs"]["extracted_2d_dir"])

    ensure_dir(extraction_output_dir)
    ensure_dir(extracted_2d_data_dir)

    print(f"Device                 : {device}")
    print(f"YOLO model             : {yolo_model_path}")
    print(f"Image size             : {image_size}")
    print(f"Confidence threshold   : {conf_threshold}")
    print(f"Person strategy        : {select_person_strategy}")
    print(f"Overwrite              : {args.overwrite}")

    print("\n[1/4] Loading external video list...")
    unique_videos = load_unique_external_videos(config)

    video_df = filter_video_df(
        video_df=unique_videos,
        datasets=args.datasets,
        max_videos=args.max_videos,
        start_index=args.start_index,
    )

    if video_df.empty:
        raise RuntimeError("No videos selected for extraction.")

    print_dataframe_summary("Selected videos", video_df, max_rows=10)

    print("\n[2/4] Loading YOLO pose model...")
    model = YOLO(yolo_model_path)

    print("\n[3/4] Extracting 2D pose confidence...")
    status_rows = []

    iterator = iter_progress(
        list(video_df.iterrows()),
        desc="Extracting 2D pose",
    )

    for local_idx, (_, row) in enumerate(iterator):
        dataset = str(row["dataset"])
        video_id = str(row["video_id"])
        video_path = Path(str(row["video_path"]))

        output_path = get_2d_feature_path(
            config=config,
            dataset=dataset,
            video_id=video_id,
        )

        print("\n" + "-" * 100)
        print(f"[{local_idx + 1}/{len(video_df)}] {dataset} | {video_id}")
        print(f"Video : {video_path}")
        print(f"Output: {output_path}")

        result = extract_2d_confidence_for_video(
            video_path=video_path,
            output_path=output_path,
            model=model,
            device=device,
            image_size=image_size,
            conf_threshold=conf_threshold,
            select_person_strategy=select_person_strategy,
            overwrite=args.overwrite,
        )

        result.update(
            {
                "dataset": dataset,
                "video_id": video_id,
                "expected_frame_count": int(row["frame_count"]) if not pd.isna(row["frame_count"]) else -1,
                "expected_fps": float(row["fps"]) if not pd.isna(row["fps"]) else 0.0,
                "expected_width": int(row["width"]) if not pd.isna(row["width"]) else 0,
                "expected_height": int(row["height"]) if not pd.isna(row["height"]) else 0,
            }
        )

        status_rows.append(result)

        print(
            f"Status: {result['status']} | "
            f"frames_written={result['frames_written']} | "
            f"no_person={result['no_person_frames']} | "
            f"multi_person={result['multi_person_frames']} | "
            f"time={result['elapsed_sec']:.2f}s"
        )

        if result["error"]:
            print(f"Error: {result['error']}")

    print("\n[4/4] Saving extraction summary...")

    status_df = pd.DataFrame(status_rows)

    summary_csv = extraction_output_dir / "02_extract_2d_confidence_summary.csv"
    save_csv(status_df, summary_csv)

    report = {
        "phase": "Phase 5 - External Dataset Generalization",
        "step": "02_extract_2d_confidence_external",
        "num_selected_videos": int(len(video_df)),
        "num_success": int((status_df["status"] == "success").sum()) if not status_df.empty else 0,
        "num_skipped_existing": int((status_df["status"] == "skipped_existing").sum()) if not status_df.empty else 0,
        "num_failed": int((status_df["status"] == "failed").sum()) if not status_df.empty else 0,
        "total_frames_written": int(status_df["frames_written"].sum()) if not status_df.empty else 0,
        "total_no_person_frames": int(status_df["no_person_frames"].sum()) if not status_df.empty else 0,
        "total_multi_person_frames": int(status_df["multi_person_frames"].sum()) if not status_df.empty else 0,
        "summary_csv": str(summary_csv),
        "output_data_dir": str(extracted_2d_data_dir),
        "fairness_note": (
            "This step only extracts video-level 2D pose features. "
            "Fair model comparison must still use the common external sequence manifest "
            "and later the common fair manifest after 2D, 3D, and quality feature availability checks."
        ),
    }

    report_json = extraction_output_dir / "02_extract_2d_confidence_report.json"
    save_json(report, report_json)

    print_dict("2D extraction report", report)

    print("\nDONE: Phase 5 Step 02 completed.")
    print("=" * 100)
    print(f"Summary CSV: {summary_csv}")
    print(f"Report JSON: {report_json}")
    print(f"2D feature dir: {extracted_2d_data_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()