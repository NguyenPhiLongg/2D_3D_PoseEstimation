import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd


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
# PHASE 5 UTILS
# ============================================================

from phase5_utils import (
    load_config,
    cfg_path,
    ensure_dir,
    save_csv,
    save_json,
    read_csv,
    bool_from_value,
    get_2d_feature_path,
    get_3d_feature_path,
    get_quality_feature_path,
    sequence_manifest_summary,
    print_dict,
    print_dataframe_summary,
)


# ============================================================
# CONSTANTS - MUST MATCH PHASE 4
# ============================================================

NUM_JOINTS = 17

COCO_BONES = [
    (5, 6),     # left_shoulder - right_shoulder
    (5, 7),     # left_shoulder - left_elbow
    (7, 9),     # left_elbow - left_wrist
    (6, 8),     # right_shoulder - right_elbow
    (8, 10),    # right_elbow - right_wrist
    (11, 12),   # left_hip - right_hip
    (11, 13),   # left_hip - left_knee
    (13, 15),   # left_knee - left_ankle
    (12, 14),   # right_hip - right_knee
    (14, 16),   # right_knee - right_ankle
    (5, 11),    # left_shoulder - left_hip
    (6, 12),    # right_shoulder - right_hip
]

QUALITY_FEATURE_COLUMNS = [
    "mean_confidence",
    "min_confidence",
    "std_confidence",
    "missing_joint_ratio",
    "low_conf_01_ratio",
    "low_conf_02_ratio",
    "low_conf_03_ratio",
    "low_conf_05_ratio",
    "mean_bbox_conf",
    "min_bbox_conf",
    "multi_person_ratio",
    "bbox_aspect_mean",
    "bbox_aspect_change",
    "bbox_area_mean",
    "bbox_area_change",
    "bbox_center_velocity_mean",
    "bbox_center_velocity_std",
    "temporal_jitter",
    "velocity_mean",
    "velocity_std",
    "bone_length_mean",
    "bone_length_std",
    "bone_length_cv",
    "3d_velocity_mean",
    "3d_velocity_std",
    "3d_bone_length_mean",
    "3d_bone_length_std",
    "3d_bone_length_cv",
    "3d_z_mean",
    "3d_z_std",
    "3d_z_velocity_mean",
    "3d_z_velocity_std",
    "3d_z_instability",
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

for i in range(NUM_JOINTS):
    REQUIRED_2D_COLUMNS.extend([f"x{i}", f"y{i}", f"c{i}"])

REQUIRED_3D_COLUMNS = ["frame"]

for i in range(NUM_JOINTS):
    REQUIRED_3D_COLUMNS.extend([f"x{i}", f"y{i}", f"z{i}"])


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
# BASIC NUMERIC HELPERS
# ============================================================

def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        value = float(value)

        if np.isnan(value) or np.isinf(value):
            return default

        return value
    except Exception:
        return default


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default

        return int(float(value))
    except Exception:
        return default


def safe_mean(values: np.ndarray, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=np.float32)

    if values.size == 0:
        return default

    if np.all(np.isnan(values)):
        return default

    return safe_float(np.nanmean(values), default)


def safe_std(values: np.ndarray, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=np.float32)

    if values.size == 0:
        return default

    if np.all(np.isnan(values)):
        return default

    return safe_float(np.nanstd(values), default)


def safe_min(values: np.ndarray, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=np.float32)

    if values.size == 0:
        return default

    if np.all(np.isnan(values)):
        return default

    return safe_float(np.nanmin(values), default)


def coefficient_of_variation(values: np.ndarray, eps: float = 1e-6) -> float:
    mean_value = safe_mean(values, default=0.0)
    std_value = safe_std(values, default=0.0)

    return float(std_value / (abs(mean_value) + eps))


# ============================================================
# COLUMN HELPERS
# ============================================================

def xy_columns() -> List[str]:
    cols = []

    for i in range(NUM_JOINTS):
        cols.extend([f"x{i}", f"y{i}"])

    return cols


def conf_columns() -> List[str]:
    return [f"c{i}" for i in range(NUM_JOINTS)]


def xyz_columns() -> List[str]:
    cols = []

    for i in range(NUM_JOINTS):
        cols.extend([f"x{i}", f"y{i}", f"z{i}"])

    return cols


def validate_2d_df(df: pd.DataFrame, path: Path) -> None:
    missing = [col for col in REQUIRED_2D_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            f"2D confidence CSV missing required columns: {missing[:20]} | path={path}"
        )


def validate_3d_df(df: pd.DataFrame, path: Path) -> None:
    missing = [col for col in REQUIRED_3D_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            f"Normalized 3D CSV missing required columns: {missing[:20]} | path={path}"
        )


# ============================================================
# ARRAY EXTRACTION
# ============================================================

def extract_2d_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
        frames      : [T]
        xy          : [T, 17, 2]
        conf        : [T, 17]
        bbox        : [T, 4]
        bbox_conf   : [T]
        num_persons : [T]
    """
    frames = df["frame"].to_numpy(dtype=np.int64)

    xy_values = df[xy_columns()].to_numpy(dtype=np.float32)
    xy = xy_values.reshape(-1, NUM_JOINTS, 2)

    conf = df[conf_columns()].to_numpy(dtype=np.float32)

    bbox = df[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(dtype=np.float32)
    bbox_conf = df["bbox_conf"].to_numpy(dtype=np.float32)
    num_persons = df["num_persons"].to_numpy(dtype=np.float32)

    return frames, xy, conf, bbox, bbox_conf, num_persons


def extract_3d_xyz(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
        frames : [T]
        xyz    : [T, 17, 3]

    Important:
        normalized_3d in Phase 5 must match Phase 2 format:
            frame + 51 xyz columns + metadata.
        Only 51 xyz columns are used here.
    """
    frames = df["frame"].to_numpy(dtype=np.int64)

    values = df[xyz_columns()].to_numpy(dtype=np.float32)
    xyz = values.reshape(-1, NUM_JOINTS, 3)

    return frames, xyz


def slice_by_frame_range(
    frames: np.ndarray,
    arrays: List[np.ndarray],
    start_frame: int,
    end_frame: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Slice arrays using inclusive frame range.

    The sequence manifest stores:
        start_frame
        end_frame

    We must respect that manifest to keep Phase 5 fair.
    """
    mask = (frames >= int(start_frame)) & (frames <= int(end_frame))

    sliced_frames = frames[mask]
    sliced_arrays = [arr[mask] for arr in arrays]

    return sliced_frames, sliced_arrays


def align_2d_3d_sequence(
    frames_2d: np.ndarray,
    xy: np.ndarray,
    conf: np.ndarray,
    bbox: np.ndarray,
    bbox_conf: np.ndarray,
    num_persons: np.ndarray,
    frames_3d: np.ndarray,
    xyz: np.ndarray,
    start_frame: int,
    end_frame: int,
) -> Optional[Dict[str, Any]]:
    """
    Slice 2D and 3D to the same sequence range and align by frame index.

    Returns None if no aligned frame is found.
    """
    seq_frames_2d, seq_2d_arrays = slice_by_frame_range(
        frames=frames_2d,
        arrays=[xy, conf, bbox, bbox_conf, num_persons],
        start_frame=start_frame,
        end_frame=end_frame,
    )

    seq_frames_3d, seq_3d_arrays = slice_by_frame_range(
        frames=frames_3d,
        arrays=[xyz],
        start_frame=start_frame,
        end_frame=end_frame,
    )

    if len(seq_frames_2d) == 0 or len(seq_frames_3d) == 0:
        return None

    common_frames = np.intersect1d(seq_frames_2d, seq_frames_3d)

    if len(common_frames) == 0:
        return None

    idx_2d = np.nonzero(np.isin(seq_frames_2d, common_frames))[0]
    idx_3d = np.nonzero(np.isin(seq_frames_3d, common_frames))[0]

    xy_seq = seq_2d_arrays[0][idx_2d]
    conf_seq = seq_2d_arrays[1][idx_2d]
    bbox_seq = seq_2d_arrays[2][idx_2d]
    bbox_conf_seq = seq_2d_arrays[3][idx_2d]
    num_persons_seq = seq_2d_arrays[4][idx_2d]
    xyz_seq = seq_3d_arrays[0][idx_3d]

    min_len = min(len(xy_seq), len(xyz_seq))

    if min_len <= 0:
        return None

    return {
        "frames": common_frames[:min_len],
        "xy": xy_seq[:min_len],
        "conf": conf_seq[:min_len],
        "bbox": bbox_seq[:min_len],
        "bbox_conf": bbox_conf_seq[:min_len],
        "num_persons": num_persons_seq[:min_len],
        "xyz": xyz_seq[:min_len],
        "aligned_length": int(min_len),
    }


# ============================================================
# QUALITY FEATURE COMPUTATION - SAME AS PHASE 4
# ============================================================

def normalize_xy_by_bbox(xy: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """
    Normalize 2D keypoints using per-frame bbox center and diagonal.
    This matches the Phase 4 quality feature calculation.
    """
    eps = 1e-6

    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    width = np.maximum(x2 - x1, eps)
    height = np.maximum(y2 - y1, eps)
    diag = np.sqrt(width ** 2 + height ** 2)

    center = np.stack([cx, cy], axis=1)[:, None, :]
    scale = diag[:, None, None]

    return (xy - center) / (scale + eps)


def compute_bone_lengths(points: np.ndarray, bones: List[Tuple[int, int]]) -> np.ndarray:
    """
    points:
        [T, 17, 2] or [T, 17, 3]

    Return:
        [T, num_bones]
    """
    lengths = []

    for a, b in bones:
        dist = np.linalg.norm(points[:, a, :] - points[:, b, :], axis=-1)
        lengths.append(dist)

    if not lengths:
        return np.zeros((points.shape[0], 0), dtype=np.float32)

    return np.stack(lengths, axis=1).astype(np.float32)


def compute_2d_quality_features(
    xy: np.ndarray,
    conf: np.ndarray,
    bbox: np.ndarray,
    bbox_conf: np.ndarray,
    num_persons: np.ndarray,
    confidence_threshold: float,
) -> Dict[str, float]:
    eps = 1e-6
    features = {}

    features["mean_confidence"] = safe_mean(conf)
    features["min_confidence"] = safe_min(conf)
    features["std_confidence"] = safe_std(conf)

    features["missing_joint_ratio"] = safe_mean((conf < confidence_threshold).astype(np.float32))
    features["low_conf_01_ratio"] = safe_mean((conf < 0.1).astype(np.float32))
    features["low_conf_02_ratio"] = safe_mean((conf < 0.2).astype(np.float32))
    features["low_conf_03_ratio"] = safe_mean((conf < 0.3).astype(np.float32))
    features["low_conf_05_ratio"] = safe_mean((conf < 0.5).astype(np.float32))

    features["mean_bbox_conf"] = safe_mean(bbox_conf)
    features["min_bbox_conf"] = safe_min(bbox_conf)

    features["multi_person_ratio"] = safe_mean((num_persons > 1).astype(np.float32))

    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]

    width = np.maximum(x2 - x1, eps)
    height = np.maximum(y2 - y1, eps)
    area = width * height
    aspect = width / (height + eps)

    features["bbox_aspect_mean"] = safe_mean(aspect)
    features["bbox_aspect_change"] = safe_std(aspect)
    features["bbox_area_mean"] = safe_mean(area)
    features["bbox_area_change"] = safe_std(area)

    bbox_center = np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0], axis=1)

    if len(bbox_center) >= 2:
        center_velocity = bbox_center[1:] - bbox_center[:-1]
        center_velocity_norm = np.linalg.norm(center_velocity, axis=-1)
        diag_median = np.nanmedian(np.sqrt(width ** 2 + height ** 2)) + eps
        center_velocity_norm = center_velocity_norm / diag_median

        features["bbox_center_velocity_mean"] = safe_mean(center_velocity_norm)
        features["bbox_center_velocity_std"] = safe_std(center_velocity_norm)
    else:
        features["bbox_center_velocity_mean"] = 0.0
        features["bbox_center_velocity_std"] = 0.0

    xy_norm = normalize_xy_by_bbox(xy, bbox)

    if xy_norm.shape[0] >= 3:
        acceleration = xy_norm[2:] - 2.0 * xy_norm[1:-1] + xy_norm[:-2]
        acceleration_norm = np.linalg.norm(acceleration, axis=-1)
        features["temporal_jitter"] = safe_mean(acceleration_norm)
    else:
        features["temporal_jitter"] = 0.0

    if xy_norm.shape[0] >= 2:
        velocity = xy_norm[1:] - xy_norm[:-1]
        velocity_norm = np.linalg.norm(velocity, axis=-1)

        features["velocity_mean"] = safe_mean(velocity_norm)
        features["velocity_std"] = safe_std(velocity_norm)
    else:
        features["velocity_mean"] = 0.0
        features["velocity_std"] = 0.0

    bone_lengths = compute_bone_lengths(xy_norm, COCO_BONES)

    features["bone_length_mean"] = safe_mean(bone_lengths)
    features["bone_length_std"] = safe_std(bone_lengths)
    features["bone_length_cv"] = coefficient_of_variation(bone_lengths)

    return features


def compute_3d_quality_features(xyz: np.ndarray) -> Dict[str, float]:
    features = {}

    if xyz.shape[0] >= 2:
        velocity = xyz[1:] - xyz[:-1]
        velocity_norm = np.linalg.norm(velocity, axis=-1)

        features["3d_velocity_mean"] = safe_mean(velocity_norm)
        features["3d_velocity_std"] = safe_std(velocity_norm)
    else:
        features["3d_velocity_mean"] = 0.0
        features["3d_velocity_std"] = 0.0

    bone_lengths = compute_bone_lengths(xyz, COCO_BONES)

    features["3d_bone_length_mean"] = safe_mean(bone_lengths)
    features["3d_bone_length_std"] = safe_std(bone_lengths)
    features["3d_bone_length_cv"] = coefficient_of_variation(bone_lengths)

    z = xyz[:, :, 2]

    features["3d_z_mean"] = safe_mean(z)
    features["3d_z_std"] = safe_std(z)

    if z.shape[0] >= 2:
        z_velocity = z[1:] - z[:-1]

        features["3d_z_velocity_mean"] = safe_mean(np.abs(z_velocity))
        features["3d_z_velocity_std"] = safe_std(z_velocity)
        features["3d_z_instability"] = safe_std(z_velocity)
    else:
        features["3d_z_velocity_mean"] = 0.0
        features["3d_z_velocity_std"] = 0.0
        features["3d_z_instability"] = 0.0

    return features


def compute_quality_for_sequence(
    xy_seq: np.ndarray,
    conf_seq: np.ndarray,
    bbox_seq: np.ndarray,
    bbox_conf_seq: np.ndarray,
    num_persons_seq: np.ndarray,
    xyz_seq: np.ndarray,
    confidence_threshold: float,
) -> Dict[str, float]:
    features = {}

    features.update(
        compute_2d_quality_features(
            xy=xy_seq,
            conf=conf_seq,
            bbox=bbox_seq,
            bbox_conf=bbox_conf_seq,
            num_persons=num_persons_seq,
            confidence_threshold=confidence_threshold,
        )
    )

    features.update(compute_3d_quality_features(xyz_seq))

    for col in QUALITY_FEATURE_COLUMNS:
        if col not in features:
            features[col] = 0.0

        features[col] = safe_float(features[col], default=0.0)

    return features


# ============================================================
# DATA LOADING
# ============================================================

def load_sequence_manifest(config: dict) -> pd.DataFrame:
    sequence_dir = cfg_path(config, config["outputs"]["external_sequences_dir"])
    sequence_path = sequence_dir / "all_external_sequences.csv"

    if not sequence_path.exists():
        raise FileNotFoundError(
            f"Sequence manifest not found: {sequence_path}\n"
            "Please run Step 01 first."
        )

    df = pd.read_csv(sequence_path)

    required = [
        "dataset",
        "video_id",
        "sequence_key",
        "video_path",
        "label",
        "label_name",
        "start_frame",
        "end_frame",
        "sequence_length",
        "valid_frame_count",
        "include_eval",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Sequence manifest missing columns: {missing}")

    if "include_eval" in df.columns:
        df = df[df["include_eval"].apply(bool_from_value)].copy()

    return df.reset_index(drop=True)


def filter_manifest(
    manifest_df: pd.DataFrame,
    datasets: Optional[List[str]] = None,
    max_sequences: Optional[int] = None,
) -> pd.DataFrame:
    df = manifest_df.copy()

    if datasets:
        datasets = set(datasets)
        df = df[df["dataset"].isin(datasets)].copy()

    df = df.reset_index(drop=True)

    if max_sequences is not None:
        df = df.head(int(max_sequences)).copy()

    return df.reset_index(drop=True)


def load_video_feature_cache(config: dict, dataset: str, video_id: str) -> Dict[str, Any]:
    path_2d = get_2d_feature_path(config, dataset, video_id)
    path_3d = get_3d_feature_path(config, dataset, video_id)

    if not path_2d.exists():
        raise FileNotFoundError(f"Missing 2D confidence file: {path_2d}")

    if not path_3d.exists():
        raise FileNotFoundError(f"Missing normalized 3D file: {path_3d}")

    df2d = pd.read_csv(path_2d)
    df3d = pd.read_csv(path_3d)

    validate_2d_df(df2d, path_2d)
    validate_3d_df(df3d, path_3d)

    frames_2d, xy, conf, bbox, bbox_conf, num_persons = extract_2d_arrays(df2d)
    frames_3d, xyz = extract_3d_xyz(df3d)

    return {
        "path_2d": str(path_2d),
        "path_3d": str(path_3d),
        "frames_2d": frames_2d,
        "xy": xy,
        "conf": conf,
        "bbox": bbox,
        "bbox_conf": bbox_conf,
        "num_persons": num_persons,
        "frames_3d": frames_3d,
        "xyz": xyz,
        "num_2d_frames": int(len(df2d)),
        "num_3d_frames": int(len(df3d)),
    }


# ============================================================
# MAIN PROCESSING
# ============================================================

def build_quality_features_for_manifest(
    config: dict,
    manifest_df: pd.DataFrame,
    confidence_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    skipped_rows = []

    cache = {}

    grouped = manifest_df.groupby(["dataset", "video_id"], sort=False)

    total_videos = len(grouped)

    for video_idx, ((dataset, video_id), group_df) in enumerate(grouped, start=1):
        print("\n" + "-" * 100)
        print(f"[{video_idx}/{total_videos}] {dataset} | {video_id} | sequences={len(group_df)}")

        try:
            cache_key = (dataset, video_id)

            if cache_key not in cache:
                cache[cache_key] = load_video_feature_cache(config, dataset, video_id)

            video_data = cache[cache_key]

        except Exception as exc:
            for _, seq_row in group_df.iterrows():
                skipped_rows.append(
                    {
                        "dataset": dataset,
                        "video_id": video_id,
                        "sequence_key": seq_row.get("sequence_key", ""),
                        "status": "missing_video_features",
                        "error": str(exc),
                    }
                )
            print(f"Skipped video due to error: {exc}")
            continue

        video_quality_rows = []

        for _, seq_row in group_df.iterrows():
            sequence_key = str(seq_row["sequence_key"])
            start_frame = safe_int(seq_row["start_frame"])
            end_frame = safe_int(seq_row["end_frame"])

            aligned = align_2d_3d_sequence(
                frames_2d=video_data["frames_2d"],
                xy=video_data["xy"],
                conf=video_data["conf"],
                bbox=video_data["bbox"],
                bbox_conf=video_data["bbox_conf"],
                num_persons=video_data["num_persons"],
                frames_3d=video_data["frames_3d"],
                xyz=video_data["xyz"],
                start_frame=start_frame,
                end_frame=end_frame,
            )

            if aligned is None:
                skipped_rows.append(
                    {
                        "dataset": dataset,
                        "video_id": video_id,
                        "sequence_key": sequence_key,
                        "status": "empty_aligned_sequence",
                        "error": f"No aligned 2D/3D frames for range {start_frame}-{end_frame}",
                    }
                )
                continue

            features = compute_quality_for_sequence(
                xy_seq=aligned["xy"],
                conf_seq=aligned["conf"],
                bbox_seq=aligned["bbox"],
                bbox_conf_seq=aligned["bbox_conf"],
                num_persons_seq=aligned["num_persons"],
                xyz_seq=aligned["xyz"],
                confidence_threshold=confidence_threshold,
            )

            out_row = {
                "dataset": dataset,
                "scene": seq_row.get("scene", ""),
                "scenario": seq_row.get("scenario", ""),
                "camera": seq_row.get("camera", ""),
                "video_id": video_id,
                "sequence_key": sequence_key,
                "video_path": seq_row.get("video_path", ""),
                "label": int(seq_row["label"]),
                "label_name": seq_row.get("label_name", ""),
                "segment_id": seq_row.get("segment_id", ""),
                "segment_start_frame": safe_int(seq_row.get("segment_start_frame", -1), -1),
                "segment_end_frame": safe_int(seq_row.get("segment_end_frame", -1), -1),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "sequence_length": safe_int(seq_row.get("sequence_length", 60), 60),
                "valid_frame_count": safe_int(seq_row.get("valid_frame_count", aligned["aligned_length"]), aligned["aligned_length"]),
                "aligned_frame_count": int(aligned["aligned_length"]),
                "path_2d_conf": video_data["path_2d"],
                "path_3d": video_data["path_3d"],
            }

            out_row.update(features)

            all_rows.append(out_row)
            video_quality_rows.append(out_row)

        # Save one quality file per video.
        if video_quality_rows:
            video_quality_df = pd.DataFrame(video_quality_rows)
            video_quality_path = get_quality_feature_path(config, dataset, video_id)
            video_quality_path.parent.mkdir(parents=True, exist_ok=True)
            video_quality_df.to_csv(video_quality_path, index=False, encoding="utf-8-sig")

        print(f"Built quality rows: {len(video_quality_rows)}")

    quality_df = pd.DataFrame(all_rows)
    skipped_df = pd.DataFrame(skipped_rows)

    return quality_df, skipped_df


# ============================================================
# VALIDATION AND OUTPUT
# ============================================================

def validate_quality_df(quality_df: pd.DataFrame, manifest_df: pd.DataFrame) -> Dict[str, Any]:
    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "num_quality_sequences": int(len(quality_df)),
        "num_manifest_sequences": int(len(manifest_df)),
        "expected_quality_dim": 33,
    }

    if quality_df.empty:
        report["valid"] = False
        report["errors"].append("quality_df is empty.")
        return report

    missing_features = [col for col in QUALITY_FEATURE_COLUMNS if col not in quality_df.columns]

    if missing_features:
        report["valid"] = False
        report["errors"].append(f"Missing quality feature columns: {missing_features}")

    quality_dim = len([col for col in QUALITY_FEATURE_COLUMNS if col in quality_df.columns])
    report["quality_dim"] = int(quality_dim)

    if quality_dim != 33:
        report["valid"] = False
        report["errors"].append(f"Wrong quality_dim={quality_dim}, expected 33.")

    if "sequence_key" not in quality_df.columns:
        report["valid"] = False
        report["errors"].append("Missing sequence_key column.")
        return report

    duplicated = quality_df[quality_df["sequence_key"].duplicated(keep=False)]

    if len(duplicated) > 0:
        report["valid"] = False
        report["errors"].append(f"Duplicated sequence_key in quality_df: {len(duplicated)}")

    manifest_keys = set(manifest_df["sequence_key"].astype(str).tolist())
    quality_keys = set(quality_df["sequence_key"].astype(str).tolist())

    missing_quality = manifest_keys - quality_keys
    extra_quality = quality_keys - manifest_keys

    report["num_missing_quality_sequences"] = int(len(missing_quality))
    report["num_extra_quality_sequences"] = int(len(extra_quality))

    if missing_quality:
        report["valid"] = False
        report["errors"].append(f"Missing quality rows for manifest sequences: {len(missing_quality)}")

    if extra_quality:
        report["valid"] = False
        report["errors"].append(f"Quality rows not in manifest: {len(extra_quality)}")

    feature_values = quality_df[QUALITY_FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    if np.isnan(feature_values).any():
        report["valid"] = False
        report["errors"].append("NaN found in quality features.")

    if np.isinf(feature_values).any():
        report["valid"] = False
        report["errors"].append("Inf found in quality features.")

    report["label_counts"] = (
        quality_df["label_name"].value_counts(dropna=False).to_dict()
        if "label_name" in quality_df.columns
        else {}
    )

    report["dataset_counts"] = (
        quality_df["dataset"].value_counts(dropna=False).to_dict()
        if "dataset" in quality_df.columns
        else {}
    )

    return report


def save_quality_outputs(
    config: dict,
    quality_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    validation_report: Dict[str, Any],
) -> Dict[str, str]:
    quality_data_dir = cfg_path(config, config["outputs"]["quality_features_dir"])
    quality_report_dir = cfg_path(config, config["outputs"]["quality_output_dir"])

    ensure_dir(quality_data_dir)
    ensure_dir(quality_report_dir)

    all_quality_path = quality_data_dir / "all_external_quality_sequences.csv"
    report_quality_path = quality_report_dir / "04_quality_feature_summary.csv"
    skipped_path = quality_report_dir / "04_quality_feature_skipped.csv"
    stats_csv_path = quality_report_dir / "04_quality_feature_distribution.csv"
    stats_json_path = quality_report_dir / "04_quality_feature_stats.json"
    report_json_path = quality_report_dir / "04_build_external_quality_features_report.json"

    quality_df.to_csv(all_quality_path, index=False, encoding="utf-8-sig")
    quality_df.to_csv(report_quality_path, index=False, encoding="utf-8-sig")
    skipped_df.to_csv(skipped_path, index=False, encoding="utf-8-sig")

    if not quality_df.empty:
        distribution_df = quality_df[QUALITY_FEATURE_COLUMNS].describe(
            percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
        ).T

        distribution_df.to_csv(stats_csv_path, encoding="utf-8-sig")

        stats_json = distribution_df.reset_index().rename(
            columns={"index": "feature"}
        ).to_dict(orient="records")
    else:
        distribution_df = pd.DataFrame()
        distribution_df.to_csv(stats_csv_path, encoding="utf-8-sig")
        stats_json = []

    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats_json, f, ensure_ascii=False, indent=4)

    summary_report = {
        "phase": "Phase 5 - External Dataset Generalization",
        "step": "04_build_external_quality_features",
        "num_manifest_sequences": int(len(manifest_df)),
        "num_quality_sequences": int(len(quality_df)),
        "num_skipped_sequences": int(len(skipped_df)),
        "quality_dim": 33,
        "quality_feature_columns": QUALITY_FEATURE_COLUMNS,
        "validation": validation_report,
        "manifest_summary": sequence_manifest_summary(manifest_df),
        "outputs": {
            "all_quality_path": str(all_quality_path),
            "report_quality_path": str(report_quality_path),
            "skipped_path": str(skipped_path),
            "stats_csv_path": str(stats_csv_path),
            "stats_json_path": str(stats_json_path),
            "report_json_path": str(report_json_path),
        },
        "fairness_note": (
            "Quality features are computed using all_external_sequences.csv. "
            "Therefore each quality row has the same sequence_key as the common external sequence manifest. "
            "This avoids comparing models on different sequence sets."
        ),
        "phase4_consistency_note": (
            "The 33 quality features and formulas match Phase 4 build_quality_features.py. "
            "External normalized 3D stores 51 xyz coordinates. "
            "Quality features are sequence-level reliability indicators used by Phase 4 models."
        ),
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=4)

    return {
        "all_quality_path": str(all_quality_path),
        "report_quality_path": str(report_quality_path),
        "skipped_path": str(skipped_path),
        "stats_csv_path": str(stats_csv_path),
        "stats_json_path": str(stats_json_path),
        "report_json_path": str(report_json_path),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 Step 04 - Build external quality features consistent with Phase 4."
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
        "--max-sequences",
        type=int,
        default=None,
        help="Optional quick test limit.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Confidence threshold used for missing_joint_ratio. Phase 4 default = 0.3.",
    )

    args = parser.parse_args()

    print("\nPhase 5 - Step 04: Build External Quality Features")
    print("=" * 100)
    print("Goal:")
    print("  Build 33 sequence-level quality features exactly like Phase 4.")
    print("  Use all_external_sequences.csv to preserve fair sequence_key alignment.")
    print("=" * 100)

    config = load_config(args.config)

    print("\n[1/5] Loading external sequence manifest...")
    manifest_df = load_sequence_manifest(config)

    manifest_df = filter_manifest(
        manifest_df=manifest_df,
        datasets=args.datasets,
        max_sequences=args.max_sequences,
    )

    if manifest_df.empty:
        raise RuntimeError("No sequences selected for quality feature building.")

    print_dataframe_summary("Selected sequence manifest", manifest_df, max_rows=10)
    print_dict("Manifest summary", sequence_manifest_summary(manifest_df))

    print("\n[2/5] Building quality features...")
    start_time = time.time()

    quality_df, skipped_df = build_quality_features_for_manifest(
        config=config,
        manifest_df=manifest_df,
        confidence_threshold=args.confidence_threshold,
    )

    elapsed = time.time() - start_time

    print("\n[3/5] Ordering columns...")

    metadata_cols = [
        "dataset",
        "scene",
        "scenario",
        "camera",
        "video_id",
        "sequence_key",
        "video_path",
        "label",
        "label_name",
        "segment_id",
        "segment_start_frame",
        "segment_end_frame",
        "start_frame",
        "end_frame",
        "sequence_length",
        "valid_frame_count",
        "aligned_frame_count",
    ]

    path_cols = [
        "path_2d_conf",
        "path_3d",
    ]

    if not quality_df.empty:
        ordered_cols = (
            [col for col in metadata_cols if col in quality_df.columns]
            + [col for col in QUALITY_FEATURE_COLUMNS if col in quality_df.columns]
            + [col for col in path_cols if col in quality_df.columns]
        )

        remaining_cols = [col for col in quality_df.columns if col not in ordered_cols]
        quality_df = quality_df[ordered_cols + remaining_cols]

    print("\n[4/5] Validating quality output...")
    validation_report = validate_quality_df(quality_df, manifest_df)

    print_dict("Quality validation", validation_report)

    if not validation_report["valid"]:
        print("\nWARNING: Quality validation failed. See errors above.")
        print("The CSV will still be saved for debugging, but Step 05 should not proceed until fixed.")

    print("\n[5/5] Saving quality outputs...")
    outputs = save_quality_outputs(
        config=config,
        quality_df=quality_df,
        skipped_df=skipped_df,
        manifest_df=manifest_df,
        validation_report=validation_report,
    )

    report = {
        "elapsed_sec": float(elapsed),
        "outputs": outputs,
    }

    print_dict("Saved outputs", report)

    print("\nDONE: Phase 5 Step 04 completed.")
    print("=" * 100)
    print(f"Quality sequences : {len(quality_df)}")
    print(f"Skipped sequences : {len(skipped_df)}")
    print(f"Quality dim       : 33")
    print(f"Elapsed seconds   : {elapsed:.2f}")

    if not quality_df.empty:
        print("\nMain quality feature averages:")
        for col in [
            "mean_confidence",
            "min_confidence",
            "missing_joint_ratio",
            "multi_person_ratio",
            "temporal_jitter",
            "velocity_std",
            "bone_length_std",
            "3d_z_instability",
        ]:
            if col in quality_df.columns:
                print(f"- {col}: {quality_df[col].mean():.6f}")

    print("=" * 100)


if __name__ == "__main__":
    main()