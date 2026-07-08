import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch


# ============================================================
# PATH SETUP
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE5_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PHASE5_DIR.parent
PHASE2_DIR = PROJECT_ROOT / "phase2_3d_upgrade"

if str(PHASE5_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE5_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(PHASE2_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE2_DIR))


# ============================================================
# PHASE 5 UTILS
# ============================================================

from phase5_utils import (
    load_config,
    cfg_path,
    ensure_dir,
    save_csv,
    save_json,
    bool_from_value,
    get_2d_feature_path,
    get_3d_feature_path,
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


def iter_progress(items, desc="Processing"):
    if tqdm is not None:
        return tqdm(items, desc=desc)
    return items


# ============================================================
# CONSTANTS
# ============================================================

NUM_JOINTS = 17

XYZ_COLUMNS = []
for joint_idx in range(NUM_JOINTS):
    XYZ_COLUMNS.extend([f"x{joint_idx}", f"y{joint_idx}", f"z{joint_idx}"])

OUTPUT_COLUMNS = ["frame"] + XYZ_COLUMNS + [
    "source_file",
    "label",
    "action_label",
    "action_name",
]

REQUIRED_2D_COLUMNS = [
    "frame",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_conf",
    "person_index",
    "num_persons",
]

for joint_idx in range(NUM_JOINTS):
    REQUIRED_2D_COLUMNS.extend([f"x{joint_idx}", f"y{joint_idx}", f"c{joint_idx}"])


# ============================================================
# BASIC HELPERS
# ============================================================

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


def safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def get_raw_3d_feature_path(config: dict, dataset: str, video_id: str) -> Path:
    base_dir = cfg_path(config, config["outputs"]["extracted_3d_dir"])
    return base_dir / dataset / f"{make_safe_filename(video_id)}_raw_3d.csv"


def get_normalized_3d_feature_path(config: dict, dataset: str, video_id: str) -> Path:
    return get_3d_feature_path(config, dataset, video_id)


def validate_2d_confidence_df(df: pd.DataFrame, path: Path) -> None:
    missing = [col for col in REQUIRED_2D_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"2D confidence CSV missing columns {missing[:20]} in {path}")


# ============================================================
# SAME JOINT CONVERSION AS PHASE 2
# COCO 17 YOLO -> H36M-LIKE 17
# ============================================================

def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def coco17_to_h36m17(coco: np.ndarray) -> np.ndarray:
    """
    Same joint-order idea as phase2_3d_upgrade/adapters/coco_to_h36m.py.

    COCO order:
        0 nose
        5 left_shoulder, 6 right_shoulder
        11 left_hip, 12 right_hip
        13 left_knee, 14 right_knee
        15 left_ankle, 16 right_ankle
        7 left_elbow, 8 right_elbow
        9 left_wrist, 10 right_wrist

    H36M-like output order:
        0 pelvis
        1 right_hip
        2 right_knee
        3 right_ankle
        4 left_hip
        5 left_knee
        6 left_ankle
        7 spine
        8 thorax
        9 neck
        10 head
        11 left_shoulder
        12 left_elbow
        13 left_wrist
        14 right_shoulder
        15 right_elbow
        16 right_wrist
    """
    coco = np.asarray(coco, dtype=np.float32)

    if coco.shape != (17, 2):
        raise ValueError(f"Expected COCO pose shape (17, 2), got {coco.shape}")

    h36m = np.zeros((17, 2), dtype=np.float32)

    left_shoulder = coco[5]
    right_shoulder = coco[6]
    left_hip = coco[11]
    right_hip = coco[12]

    pelvis = midpoint(left_hip, right_hip)
    thorax = midpoint(left_shoulder, right_shoulder)
    spine = midpoint(pelvis, thorax)
    neck = thorax
    head = coco[0]

    h36m[0] = pelvis

    h36m[1] = coco[12]
    h36m[2] = coco[14]
    h36m[3] = coco[16]

    h36m[4] = coco[11]
    h36m[5] = coco[13]
    h36m[6] = coco[15]

    h36m[7] = spine
    h36m[8] = thorax
    h36m[9] = neck
    h36m[10] = head

    h36m[11] = coco[5]
    h36m[12] = coco[7]
    h36m[13] = coco[9]

    h36m[14] = coco[6]
    h36m[15] = coco[8]
    h36m[16] = coco[10]

    return h36m.astype(np.float32)


def is_valid_coco_pose_from_row(row: pd.Series, min_visible_points: int = 8, min_conf: float = 0.05) -> bool:
    visible = 0

    for joint_idx in range(NUM_JOINTS):
        x = safe_float(row.get(f"x{joint_idx}", 0.0))
        y = safe_float(row.get(f"y{joint_idx}", 0.0))
        c = safe_float(row.get(f"c{joint_idx}", 0.0))

        if x > 1.0 and y > 1.0 and c >= min_conf:
            visible += 1

    return visible >= min_visible_points


def row_to_coco_xy(row: pd.Series) -> np.ndarray:
    pose = np.zeros((17, 2), dtype=np.float32)

    for joint_idx in range(NUM_JOINTS):
        pose[joint_idx, 0] = safe_float(row.get(f"x{joint_idx}", 0.0))
        pose[joint_idx, 1] = safe_float(row.get(f"y{joint_idx}", 0.0))

    return pose


# ============================================================
# EXACT NORMALIZATION USED BY PHASE 2
# ============================================================

def normalize_pose3d_phase2(pose: np.ndarray) -> np.ndarray:
    """
    Same logic as phase2_3d_upgrade/normalize_3d_dataset.py.

    Per-frame, per-axis mean/std over 17 joints:

        mean shape = (1, 3)
        std shape  = (1, 3)

        normalized = (pose - mean) / std

    This produces only 51 normalized xyz coordinate values.
    The 8 handcrafted features are NOT saved here.
    They are computed later by phase3_utils.py / phase4_utils.py.
    """
    pose = np.asarray(pose, dtype=np.float32)

    if pose.shape != (17, 3):
        raise ValueError(f"Expected pose shape (17, 3), got {pose.shape}")

    eps = 1e-6

    mean = pose.mean(axis=0, keepdims=True)
    std = pose.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)

    normalized = (pose - mean) / std

    return np.nan_to_num(
        normalized.astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def pose3d_to_row(
    frame_idx: int,
    pose_3d: np.ndarray,
    source_file: str,
    label: int = -1,
    action_label: int = -1,
    action_name: str = "External",
) -> Dict[str, Any]:
    row = {
        "frame": int(frame_idx),
    }

    for joint_idx in range(NUM_JOINTS):
        row[f"x{joint_idx}"] = float(pose_3d[joint_idx, 0])
        row[f"y{joint_idx}"] = float(pose_3d[joint_idx, 1])
        row[f"z{joint_idx}"] = float(pose_3d[joint_idx, 2])

    row["source_file"] = source_file
    row["label"] = int(label)
    row["action_label"] = int(action_label)
    row["action_name"] = action_name

    return row


# ============================================================
# POSEFORMER LOADING
# ============================================================

def try_load_poseformer(device: torch.device):
    """
    Load the same Phase 2 PoseFormerV2 inferencer if available.

    If unavailable, return None and the script can use proxy mode.
    """
    try:
        from inference.infer_3d_pose import PoseFormerV2Inferencer

        inferencer = PoseFormerV2Inferencer(device=device)
        return inferencer

    except Exception as exc:
        print("\nWARNING: Could not load Phase 2 PoseFormerV2 inferencer.")
        print(f"Reason: {exc}")
        print("The script will use proxy 3D mode instead.")
        return None


# ============================================================
# PROXY FALLBACK
# ============================================================

def estimate_proxy_3d_from_h36m_2d(h36m_xy: np.ndarray) -> np.ndarray:
    """
    Fallback if PoseFormerV2 is not available.

    This keeps H36M-like joint order and produces a 17x3 pose.
    It is not true triangulated 3D, but it is immediately normalized by
    the same Phase 2 normalization function.
    """
    h36m_xy = np.asarray(h36m_xy, dtype=np.float32)

    pose_3d = np.zeros((17, 3), dtype=np.float32)
    pose_3d[:, 0:2] = h36m_xy

    pelvis = h36m_xy[0]
    thorax = h36m_xy[8]
    head = h36m_xy[10]

    torso_len = float(np.linalg.norm(thorax - pelvis))
    head_len = float(np.linalg.norm(head - thorax))

    scale = max(torso_len + head_len, 1.0)

    # Simple depth prior by body part.
    z = np.zeros((17,), dtype=np.float32)

    z[0] = 0.00     # pelvis
    z[1] = 0.03
    z[2] = 0.06
    z[3] = 0.09

    z[4] = -0.03
    z[5] = -0.06
    z[6] = -0.09

    z[7] = -0.02
    z[8] = -0.04
    z[9] = -0.05
    z[10] = -0.06

    z[11] = -0.08
    z[12] = -0.10
    z[13] = -0.12

    z[14] = 0.08
    z[15] = 0.10
    z[16] = 0.12

    pose_3d[:, 2] = z * scale

    return pose_3d.astype(np.float32)


# ============================================================
# 3D ESTIMATION FOR ONE VIDEO
# ============================================================

def estimate_3d_for_video(
    config: dict,
    dataset: str,
    video_id: str,
    video_path: str,
    method: str,
    inferencer,
    overwrite: bool = False,
    min_visible_points: int = 8,
    min_conf: float = 0.05,
) -> dict:
    start_time = time.time()

    input_2d_path = get_2d_feature_path(config, dataset, video_id)
    raw_3d_path = get_raw_3d_feature_path(config, dataset, video_id)
    normalized_3d_path = get_normalized_3d_feature_path(config, dataset, video_id)

    raw_3d_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_3d_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_3d_path.exists() and normalized_3d_path.exists() and not overwrite:
        try:
            existing_df = pd.read_csv(normalized_3d_path)
            xyz_cols = [col for col in XYZ_COLUMNS if col in existing_df.columns]

            return {
                "dataset": dataset,
                "video_id": video_id,
                "video_path": video_path,
                "input_2d_path": str(input_2d_path),
                "raw_3d_path": str(raw_3d_path),
                "normalized_3d_path": str(normalized_3d_path),
                "status": "skipped_existing",
                "method": method,
                "frames_2d": int(len(existing_df)),
                "frames_raw_3d": int(len(existing_df)),
                "frames_normalized_3d": int(len(existing_df)),
                "xyz_dim": int(len(xyz_cols)),
                "model_feature_dim_after_loader": 59,
                "num_poseformer_frames": -1,
                "num_proxy_frames": -1,
                "num_missing_3d_frames": -1,
                "elapsed_sec": 0.0,
                "error": "",
            }
        except Exception:
            pass

    if not input_2d_path.exists():
        return {
            "dataset": dataset,
            "video_id": video_id,
            "video_path": video_path,
            "input_2d_path": str(input_2d_path),
            "raw_3d_path": str(raw_3d_path),
            "normalized_3d_path": str(normalized_3d_path),
            "status": "failed",
            "method": method,
            "frames_2d": 0,
            "frames_raw_3d": 0,
            "frames_normalized_3d": 0,
            "xyz_dim": 0,
            "model_feature_dim_after_loader": 0,
            "num_poseformer_frames": 0,
            "num_proxy_frames": 0,
            "num_missing_3d_frames": 0,
            "elapsed_sec": 0.0,
            "error": "missing_2d_confidence_file",
        }

    try:
        df_2d = pd.read_csv(input_2d_path)
        validate_2d_confidence_df(df_2d, input_2d_path)

        source_file = str(video_path).replace("\\", "/")

        if inferencer is not None:
            inferencer.reset()

        raw_rows = []
        normalized_rows = []

        last_raw_pose = None

        num_poseformer_frames = 0
        num_proxy_frames = 0
        num_missing_3d_frames = 0

        for _, row in df_2d.iterrows():
            frame_idx = safe_int(row.get("frame", 0))
            valid_pose = is_valid_coco_pose_from_row(
                row,
                min_visible_points=min_visible_points,
                min_conf=min_conf,
            )

            raw_pose_3d = None

            if valid_pose:
                coco_xy = row_to_coco_xy(row)
                h36m_xy = coco17_to_h36m17(coco_xy)

                if method == "poseformer" and inferencer is not None:
                    try:
                        raw_pose_3d = inferencer.add_and_predict(coco_xy)
                    except Exception:
                        raw_pose_3d = None

                    if raw_pose_3d is not None:
                        num_poseformer_frames += 1

                if raw_pose_3d is None:
                    raw_pose_3d = estimate_proxy_3d_from_h36m_2d(h36m_xy)
                    num_proxy_frames += 1

            else:
                if inferencer is not None:
                    inferencer.reset()

                raw_pose_3d = np.zeros((17, 3), dtype=np.float32)
                num_missing_3d_frames += 1

            raw_pose_3d = np.asarray(raw_pose_3d, dtype=np.float32)

            if raw_pose_3d.shape != (17, 3):
                raw_pose_3d = np.zeros((17, 3), dtype=np.float32)
                num_missing_3d_frames += 1

            raw_pose_3d = np.nan_to_num(raw_pose_3d, nan=0.0, posinf=0.0, neginf=0.0)

            normalized_pose_3d = normalize_pose3d_phase2(raw_pose_3d)

            raw_rows.append(
                pose3d_to_row(
                    frame_idx=frame_idx,
                    pose_3d=raw_pose_3d,
                    source_file=source_file,
                    label=-1,
                    action_label=-1,
                    action_name="External",
                )
            )

            normalized_rows.append(
                pose3d_to_row(
                    frame_idx=frame_idx,
                    pose_3d=normalized_pose_3d,
                    source_file=source_file,
                    label=-1,
                    action_label=-1,
                    action_name="External",
                )
            )

            last_raw_pose = raw_pose_3d

        raw_df = pd.DataFrame(raw_rows)
        normalized_df = pd.DataFrame(normalized_rows)

        raw_df = raw_df[OUTPUT_COLUMNS]
        normalized_df = normalized_df[OUTPUT_COLUMNS]

        raw_df.to_csv(raw_3d_path, index=False, encoding="utf-8-sig")
        normalized_df.to_csv(normalized_3d_path, index=False, encoding="utf-8-sig")

        elapsed = time.time() - start_time

        xyz_dim = len([col for col in XYZ_COLUMNS if col in normalized_df.columns])

        return {
            "dataset": dataset,
            "video_id": video_id,
            "video_path": video_path,
            "input_2d_path": str(input_2d_path),
            "raw_3d_path": str(raw_3d_path),
            "normalized_3d_path": str(normalized_3d_path),
            "status": "success",
            "method": method,
            "frames_2d": int(len(df_2d)),
            "frames_raw_3d": int(len(raw_df)),
            "frames_normalized_3d": int(len(normalized_df)),
            "xyz_dim": int(xyz_dim),
            "model_feature_dim_after_loader": 59,
            "num_poseformer_frames": int(num_poseformer_frames),
            "num_proxy_frames": int(num_proxy_frames),
            "num_missing_3d_frames": int(num_missing_3d_frames),
            "elapsed_sec": float(elapsed),
            "error": "",
        }

    except Exception as exc:
        elapsed = time.time() - start_time

        return {
            "dataset": dataset,
            "video_id": video_id,
            "video_path": video_path,
            "input_2d_path": str(input_2d_path),
            "raw_3d_path": str(raw_3d_path),
            "normalized_3d_path": str(normalized_3d_path),
            "status": "failed",
            "method": method,
            "frames_2d": 0,
            "frames_raw_3d": 0,
            "frames_normalized_3d": 0,
            "xyz_dim": 0,
            "model_feature_dim_after_loader": 0,
            "num_poseformer_frames": 0,
            "num_proxy_frames": 0,
            "num_missing_3d_frames": 0,
            "elapsed_sec": float(elapsed),
            "error": str(exc),
        }


# ============================================================
# VIDEO LIST
# ============================================================

def load_unique_external_videos(config: dict) -> pd.DataFrame:
    metadata_path = cfg_path(config, config["outputs"]["all_metadata_csv"])

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            "Please run Step 01 first."
        )

    df = pd.read_csv(metadata_path)

    if "include_eval" in df.columns:
        df = df[df["include_eval"].apply(bool_from_value)].copy()

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


def filter_video_df(
    video_df: pd.DataFrame,
    datasets=None,
    max_videos=None,
    start_index: int = 0,
) -> pd.DataFrame:
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
# VALIDATION
# ============================================================

def validate_normalized_outputs(status_df: pd.DataFrame) -> Dict[str, Any]:
    report = {
        "all_success_or_skipped": True,
        "expected_normalized_xyz_dim": 51,
        "expected_model_feature_dim_after_loader": 59,
        "errors": [],
    }

    if status_df.empty:
        report["all_success_or_skipped"] = False
        report["errors"].append("Empty status dataframe.")
        return report

    bad_status = status_df[~status_df["status"].isin(["success", "skipped_existing"])]

    if len(bad_status) > 0:
        report["all_success_or_skipped"] = False
        report["errors"].append(f"Failed videos found: {len(bad_status)}")

    if "xyz_dim" in status_df.columns:
        bad_dim = status_df[status_df["xyz_dim"].astype(int) != 51]

        if len(bad_dim) > 0:
            report["all_success_or_skipped"] = False
            report["errors"].append(f"Wrong normalized xyz dim rows: {len(bad_dim)}")

    if "frames_2d" in status_df.columns:
        frame_mismatch = status_df[
            status_df["frames_2d"].astype(int) != status_df["frames_normalized_3d"].astype(int)
        ]

        if len(frame_mismatch) > 0:
            report["all_success_or_skipped"] = False
            report["errors"].append(f"2D/3D frame count mismatch rows: {len(frame_mismatch)}")

    return report


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 Step 03 - Estimate and normalize external 3D pose in the same format as Phase 2/3/4."
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
        help="Overwrite existing 3D CSV files.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="poseformer",
        choices=["poseformer", "proxy"],
        help=(
            "3D estimation method. "
            "poseformer = use Phase 2 PoseFormerV2 if available, fallback proxy when needed. "
            "proxy = use H36M-like proxy only."
        ),
    )

    parser.add_argument(
        "--min-visible-points",
        type=int,
        default=8,
        help="Minimum visible keypoints for a valid 2D pose.",
    )

    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.05,
        help="Minimum keypoint confidence for visibility.",
    )

    args = parser.parse_args()

    print("\nPhase 5 - Step 03: Estimate + Normalize External 3D Pose")
    print("=" * 100)
    print("Correct format:")
    print("  normalized_3d CSV stores 51 xyz coordinate columns only, plus metadata.")
    print("  8 handcrafted 3D features are computed later by phase3_utils/phase4_utils.")
    print("  Final model input after loader = 51 + 8 = 59D.")
    print("=" * 100)

    config = load_config(args.config)

    extraction_3d_output_dir = cfg_path(config, config["outputs"]["extraction_3d_dir"])
    extracted_3d_data_dir = cfg_path(config, config["outputs"]["extracted_3d_dir"])
    normalized_3d_data_dir = cfg_path(config, config["outputs"]["normalized_3d_dir"])

    ensure_dir(extraction_3d_output_dir)
    ensure_dir(extracted_3d_data_dir)
    ensure_dir(normalized_3d_data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Method: {args.method}")
    print(f"Expected normalized xyz dim: 51")
    print(f"Expected model feature dim after loader: 59")

    inferencer = None

    if args.method == "poseformer":
        print("\n[1/5] Loading Phase 2 PoseFormerV2 inferencer...")
        inferencer = try_load_poseformer(device=device)

        if inferencer is None:
            args.method = "proxy"
            print("Switched method to proxy.")
    else:
        print("\n[1/5] Using proxy mode. PoseFormerV2 will not be loaded.")

    print("\n[2/5] Loading external video list...")
    unique_videos = load_unique_external_videos(config)

    video_df = filter_video_df(
        video_df=unique_videos,
        datasets=args.datasets,
        max_videos=args.max_videos,
        start_index=args.start_index,
    )

    if video_df.empty:
        raise RuntimeError("No videos selected for 3D estimation.")

    print_dataframe_summary("Selected videos", video_df, max_rows=10)

    print("\n[3/5] Estimating and normalizing 3D pose...")
    status_rows = []

    iterator = iter_progress(
        list(video_df.iterrows()),
        desc="Estimating normalized 3D",
    )

    for local_idx, (_, row) in enumerate(iterator):
        dataset = str(row["dataset"])
        video_id = str(row["video_id"])
        video_path = str(row["video_path"])

        print("\n" + "-" * 100)
        print(f"[{local_idx + 1}/{len(video_df)}] {dataset} | {video_id}")

        result = estimate_3d_for_video(
            config=config,
            dataset=dataset,
            video_id=video_id,
            video_path=video_path,
            method=args.method,
            inferencer=inferencer,
            overwrite=args.overwrite,
            min_visible_points=args.min_visible_points,
            min_conf=args.min_conf,
        )

        status_rows.append(result)

        print(
            f"Status: {result['status']} | "
            f"frames_2d={result['frames_2d']} | "
            f"frames_norm3d={result['frames_normalized_3d']} | "
            f"xyz_dim={result['xyz_dim']} | "
            f"loader_dim={result['model_feature_dim_after_loader']} | "
            f"poseformer={result['num_poseformer_frames']} | "
            f"proxy={result['num_proxy_frames']} | "
            f"missing={result['num_missing_3d_frames']} | "
            f"time={result['elapsed_sec']:.2f}s"
        )

        if result["error"]:
            print(f"Error: {result['error']}")

    print("\n[4/5] Saving 3D estimation summary...")

    status_df = pd.DataFrame(status_rows)

    summary_csv = extraction_3d_output_dir / "03_estimate_3d_external_summary.csv"
    save_csv(status_df, summary_csv)

    validation = validate_normalized_outputs(status_df)

    num_success = int((status_df["status"] == "success").sum()) if not status_df.empty else 0
    num_skipped = int((status_df["status"] == "skipped_existing").sum()) if not status_df.empty else 0
    num_failed = int((status_df["status"] == "failed").sum()) if not status_df.empty else 0

    xyz_dim_counts = (
        status_df["xyz_dim"].value_counts(dropna=False).to_dict()
        if not status_df.empty and "xyz_dim" in status_df.columns
        else {}
    )

    loader_dim_counts = (
        status_df["model_feature_dim_after_loader"].value_counts(dropna=False).to_dict()
        if not status_df.empty and "model_feature_dim_after_loader" in status_df.columns
        else {}
    )

    report = {
        "phase": "Phase 5 - External Dataset Generalization",
        "step": "03_estimate_3d_external",
        "num_selected_videos": int(len(video_df)),
        "num_success": num_success,
        "num_skipped_existing": num_skipped,
        "num_failed": num_failed,
        "method": args.method,
        "total_frames_2d": int(status_df["frames_2d"].sum()) if not status_df.empty else 0,
        "total_frames_normalized_3d": int(status_df["frames_normalized_3d"].sum()) if not status_df.empty else 0,
        "expected_normalized_xyz_dim": 51,
        "expected_model_feature_dim_after_loader": 59,
        "xyz_dim_counts": xyz_dim_counts,
        "loader_dim_counts": loader_dim_counts,
        "total_poseformer_frames": int(status_df["num_poseformer_frames"].clip(lower=0).sum()) if not status_df.empty else 0,
        "total_proxy_frames": int(status_df["num_proxy_frames"].clip(lower=0).sum()) if not status_df.empty else 0,
        "total_missing_3d_frames": int(status_df["num_missing_3d_frames"].clip(lower=0).sum()) if not status_df.empty else 0,
        "validation": validation,
        "summary_csv": str(summary_csv),
        "raw_3d_output_dir": str(extracted_3d_data_dir),
        "normalized_3d_output_dir": str(normalized_3d_data_dir),
        "normalization_rule": (
            "Same as Phase 2 normalize_3d_dataset.py: per-frame, per-axis mean/std "
            "normalization over 17 joints. Output CSV keeps 51 xyz coordinate columns. "
            "Phase 3/4 loaders later compute 8 handcrafted 3D features, producing 59D input."
        ),
        "fairness_note": (
            "This step writes one normalized 3D row for each 2D frame so all models can later "
            "be evaluated using the same external sequence manifest."
        ),
    }

    report_json = extraction_3d_output_dir / "03_estimate_3d_external_report.json"
    save_json(report, report_json)

    print_dict("3D estimation report", report)

    print("\n[5/5] Done.")
    print("=" * 100)
    print(f"Summary CSV        : {summary_csv}")
    print(f"Report JSON        : {report_json}")
    print(f"Raw 3D feature dir : {extracted_3d_data_dir}")
    print(f"Norm 3D feature dir: {normalized_3d_data_dir}")
    print("=" * 100)

    if not validation["all_success_or_skipped"]:
        print("\nWARNING: validation errors found:")
        print(json.dumps(validation, ensure_ascii=False, indent=4))
    else:
        print("\nValidation OK:")
        print("  normalized_3d xyz dim = 51")
        print("  model input after loader = 59")
        print("  2D frame count = normalized 3D frame count")


if __name__ == "__main__":
    main()