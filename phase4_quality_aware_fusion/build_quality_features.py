import os
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


"""
Phase 4 - Build pose quality features.

Purpose:
    This script builds sequence-level quality features for Phase 4 quality-aware fusion.

Inputs:
    1. data/5_extracted_2d_confidence/
        New 2D CSV files with:
            frame,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_conf,
            person_index, num_persons,
            x0, y0, c0, ..., x16, y16, c16

    2. data/4_normalized_3d/
        Normalized estimated 3D pose CSV files from Phase 2.

    3. phase3_common_set_gated_fusion/outputs/common_split/common_metadata.csv
        Common-set metadata used in Phase 3.

Outputs:
    1. data/6_pose_quality_features/quality_sequences.csv
        One row per generated sequence.

    2. phase4_quality_aware_fusion/outputs/quality_features/
        Summary CSV/JSON files.

Why:
    Quality-aware fusion needs extra reliability indicators, such as:
        - mean_confidence
        - min_confidence
        - missing_joint_ratio
        - temporal_jitter
        - bone_length_std
        - velocity_std
        - bbox_aspect_change
        - 3d_z_instability

    These features help the model decide when to trust 2D pose or estimated 3D pose.
"""


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_COMMON_METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "phase3_common_set_gated_fusion",
    "outputs",
    "common_split",
    "common_metadata.csv",
)

DEFAULT_2D_CONF_DIR = os.path.join(PROJECT_ROOT, "data", "5_extracted_2d_confidence")
DEFAULT_NORMALIZED_3D_DIR = os.path.join(PROJECT_ROOT, "data", "4_normalized_3d")
DEFAULT_QUALITY_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "6_pose_quality_features")

DEFAULT_REPORT_DIR = os.path.join(
    PROJECT_ROOT,
    "phase4_quality_aware_fusion",
    "outputs",
    "quality_features",
)


COCO_BONES = [
    (5, 6),    # left_shoulder - right_shoulder
    (5, 7),    # left_shoulder - left_elbow
    (7, 9),    # left_elbow - left_wrist
    (6, 8),    # right_shoulder - right_elbow
    (8, 10),   # right_elbow - right_wrist
    (11, 12),  # left_hip - right_hip
    (11, 13),  # left_hip - left_knee
    (13, 15),  # left_knee - left_ankle
    (12, 14),  # right_hip - right_knee
    (14, 16),  # right_knee - right_ankle
    (5, 11),   # left_shoulder - left_hip
    (6, 12),   # right_shoulder - right_hip
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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def xy_columns() -> List[str]:
    cols = []

    for i in range(17):
        cols.extend([f"x{i}", f"y{i}"])

    return cols


def conf_columns() -> List[str]:
    return [f"c{i}" for i in range(17)]


def required_2d_conf_columns() -> List[str]:
    cols = [
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
        cols.extend([f"x{i}", f"y{i}", f"c{i}"])

    return cols


def safe_float(value, default: float = np.nan) -> float:
    try:
        if value is None:
            return default

        value = float(value)

        if np.isnan(value) or np.isinf(value):
            return default

        return value
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


def safe_max(values: np.ndarray, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=np.float32)

    if values.size == 0:
        return default

    if np.all(np.isnan(values)):
        return default

    return safe_float(np.nanmax(values), default)


def coefficient_of_variation(values: np.ndarray, eps: float = 1e-6) -> float:
    mean_value = safe_mean(values, default=0.0)
    std_value = safe_std(values, default=0.0)

    return float(std_value / (abs(mean_value) + eps))


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def list_csv_by_name(folder: str) -> Dict[str, str]:
    mapping = {}

    if not os.path.exists(folder):
        return mapping

    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".csv"):
                mapping[file] = os.path.join(root, file)

    return mapping


def resolve_2d_conf_path(
    record: pd.Series,
    two_d_conf_dir: str,
    two_d_conf_files: Dict[str, str],
) -> Optional[str]:
    candidates = []

    if "path_2d" in record.index:
        candidates.append(os.path.basename(str(record["path_2d"])))

    if "video_key" in record.index:
        key = str(record["video_key"])
        if key.lower().endswith(".csv"):
            candidates.append(os.path.basename(key))
        else:
            candidates.append(os.path.basename(key) + ".csv")

    for name in candidates:
        if name in two_d_conf_files:
            return two_d_conf_files[name]

        path = os.path.join(two_d_conf_dir, name)
        if os.path.exists(path):
            return path

    return None


def resolve_3d_path(
    record: pd.Series,
    normalized_3d_dir: str,
    normalized_3d_files: Dict[str, str],
) -> Optional[str]:
    candidates = []

    if "path_3d" in record.index:
        path_3d = str(record["path_3d"])
        base_name = os.path.basename(path_3d)
        candidates.append(base_name)

        if os.path.isabs(path_3d) and os.path.exists(path_3d):
            candidates.append(path_3d)

        relative_path = os.path.join(PROJECT_ROOT, path_3d)
        if os.path.exists(relative_path):
            candidates.append(relative_path)

    if "path_2d" in record.index:
        candidates.append(os.path.basename(str(record["path_2d"])))

    if "video_key" in record.index:
        key = str(record["video_key"])
        if key.lower().endswith(".csv"):
            candidates.append(os.path.basename(key))
        else:
            candidates.append(os.path.basename(key) + ".csv")

    for item in candidates:
        if item is None:
            continue

        item = str(item)

        if os.path.exists(item):
            return item

        base_name = os.path.basename(item)

        if base_name in normalized_3d_files:
            return normalized_3d_files[base_name]

        path = os.path.join(normalized_3d_dir, base_name)
        if os.path.exists(path):
            return path

    return None


def get_video_key(record: pd.Series, two_d_conf_path: str) -> str:
    if "video_key" in record.index:
        return str(record["video_key"])

    if "path_2d" in record.index:
        return os.path.splitext(os.path.basename(str(record["path_2d"])))[0]

    return os.path.splitext(os.path.basename(two_d_conf_path))[0]


def copy_optional_metadata(record: pd.Series) -> Dict:
    output = {}

    for col in [
        "video_key",
        "source",
        "class_name",
        "label",
        "binary_label",
        "action_label",
        "fall_label",
        "action_name",
        "category",
        "split",
    ]:
        if col in record.index:
            value = record[col]

            if pd.isna(value):
                continue

            output[col] = value

    return output


def validate_2d_conf_df(df: pd.DataFrame, path: str) -> None:
    missing = [col for col in required_2d_conf_columns() if col not in df.columns]

    if missing:
        raise ValueError(
            f"2D confidence CSV missing required columns: {missing[:20]} | path={path}"
        )


def extract_2d_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
        frames: [T]
        xy: [T, 17, 2]
        conf: [T, 17]
        bbox: [T, 4]
        bbox_conf: [T]
        num_persons: [T]
    """
    frames = df["frame"].to_numpy(dtype=np.int64)

    xy_values = df[xy_columns()].to_numpy(dtype=np.float32)
    xy = xy_values.reshape(-1, 17, 2)

    conf = df[conf_columns()].to_numpy(dtype=np.float32)

    bbox = df[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(dtype=np.float32)
    bbox_conf = df["bbox_conf"].to_numpy(dtype=np.float32)
    num_persons = df["num_persons"].to_numpy(dtype=np.float32)

    return frames, xy, conf, bbox, bbox_conf, num_persons


def find_3d_columns(df: pd.DataFrame) -> Optional[List[str]]:
    """
    Try to find 17 * 3 = 51 coordinate columns for estimated 3D pose.

    Supported patterns:
        x0, y0, z0, ..., x16, y16, z16
        x_0, y_0, z_0, ..., x_16, y_16, z_16
        joint0_x, joint0_y, joint0_z, ..., joint16_x, joint16_y, joint16_z

    Fallback:
        Use first 51 numeric columns excluding frame.
    """
    patterns = []

    pattern_1 = []
    for i in range(17):
        pattern_1.extend([f"x{i}", f"y{i}", f"z{i}"])
    patterns.append(pattern_1)

    pattern_2 = []
    for i in range(17):
        pattern_2.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
    patterns.append(pattern_2)

    pattern_3 = []
    for i in range(17):
        pattern_3.extend([f"joint{i}_x", f"joint{i}_y", f"joint{i}_z"])
    patterns.append(pattern_3)

    for pattern in patterns:
        if all(col in df.columns for col in pattern):
            return pattern

    numeric_cols = []

    for col in df.columns:
        if col == "frame":
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    if len(numeric_cols) >= 51:
        return numeric_cols[:51]

    return None


def extract_3d_xyz(df: pd.DataFrame, path: str) -> np.ndarray:
    cols = find_3d_columns(df)

    if cols is None:
        raise ValueError(f"Cannot find 3D coordinate columns in: {path}")

    values = df[cols].to_numpy(dtype=np.float32)
    xyz = values.reshape(-1, 17, 3)

    return xyz


def sequence_windows(
    length: int,
    sequence_length: int,
    stride: int,
    include_short: bool,
) -> List[Tuple[int, int]]:
    if length <= 0:
        return []

    if length < sequence_length:
        if include_short:
            return [(0, length)]
        return []

    windows = []

    start = 0

    while start + sequence_length <= length:
        windows.append((start, start + sequence_length))
        start += stride

    return windows


def normalize_xy_by_bbox(xy: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """
    Normalize 2D keypoints using per-frame bounding box center and diagonal.

    This makes jitter, velocity, and bone-length features less dependent on image size.
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


def process_single_record(
    record: pd.Series,
    two_d_conf_files: Dict[str, str],
    normalized_3d_files: Dict[str, str],
    args: argparse.Namespace,
) -> Tuple[List[Dict], Optional[Dict]]:
    two_d_conf_path = resolve_2d_conf_path(
        record=record,
        two_d_conf_dir=args.two_d_conf_dir,
        two_d_conf_files=two_d_conf_files,
    )

    three_d_path = resolve_3d_path(
        record=record,
        normalized_3d_dir=args.normalized_3d_dir,
        normalized_3d_files=normalized_3d_files,
    )

    video_key = str(record["video_key"]) if "video_key" in record.index else "UNKNOWN"

    if two_d_conf_path is None:
        return [], {
            "video_key": video_key,
            "status": "missing_2d_conf",
            "error": "Cannot resolve 2D confidence CSV path.",
        }

    if three_d_path is None:
        return [], {
            "video_key": video_key,
            "status": "missing_3d",
            "error": "Cannot resolve normalized 3D CSV path.",
        }

    try:
        df2d = load_csv(two_d_conf_path)
        df3d = load_csv(three_d_path)

        validate_2d_conf_df(df2d, two_d_conf_path)

        frames, xy, conf, bbox, bbox_conf, num_persons = extract_2d_arrays(df2d)
        xyz = extract_3d_xyz(df3d, three_d_path)

        min_len = min(xy.shape[0], xyz.shape[0])

        if min_len <= 0:
            return [], {
                "video_key": video_key,
                "status": "empty_aligned_sequence",
                "error": "2D or 3D has zero usable frames.",
            }

        frames = frames[:min_len]
        xy = xy[:min_len]
        conf = conf[:min_len]
        bbox = bbox[:min_len]
        bbox_conf = bbox_conf[:min_len]
        num_persons = num_persons[:min_len]
        xyz = xyz[:min_len]

        windows = sequence_windows(
            length=min_len,
            sequence_length=args.sequence_length,
            stride=args.stride,
            include_short=args.include_short,
        )

        if not windows:
            return [], {
                "video_key": video_key,
                "status": "no_valid_window",
                "error": f"Aligned length {min_len} is shorter than sequence length {args.sequence_length}.",
            }

        rows = []
        base_metadata = copy_optional_metadata(record)
        video_key = get_video_key(record, two_d_conf_path)
        csv_file_name = os.path.basename(two_d_conf_path)

        for seq_idx, (start, end) in enumerate(windows):
            xy_seq = xy[start:end]
            conf_seq = conf[start:end]
            bbox_seq = bbox[start:end]
            bbox_conf_seq = bbox_conf[start:end]
            num_persons_seq = num_persons[start:end]
            xyz_seq = xyz[start:end]

            feature_values = compute_quality_for_sequence(
                xy_seq=xy_seq,
                conf_seq=conf_seq,
                bbox_seq=bbox_seq,
                bbox_conf_seq=bbox_conf_seq,
                num_persons_seq=num_persons_seq,
                xyz_seq=xyz_seq,
                confidence_threshold=args.confidence_threshold,
            )

            row = {
                "video_key": video_key,
                "csv_file": csv_file_name,
                "sequence_index": int(seq_idx),
                "start_index": int(start),
                "end_index": int(end),
                "start_frame": int(frames[start]),
                "end_frame": int(frames[end - 1]),
                "sequence_length": int(end - start),
                "aligned_num_frames": int(min_len),
                "path_2d_conf": two_d_conf_path,
                "path_3d": three_d_path,
            }

            row.update(base_metadata)
            row.update(feature_values)

            rows.append(row)

        return rows, None

    except Exception as exc:
        return [], {
            "video_key": video_key,
            "status": "exception",
            "error": str(exc),
            "path_2d_conf": two_d_conf_path,
            "path_3d": three_d_path,
        }


def save_outputs(
    quality_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    ensure_dir(args.quality_output_dir)
    ensure_dir(args.report_dir)

    quality_csv_path = os.path.join(args.quality_output_dir, "quality_sequences.csv")
    quality_report_csv_path = os.path.join(args.report_dir, "quality_feature_summary.csv")
    skipped_csv_path = os.path.join(args.report_dir, "quality_feature_skipped.csv")
    stats_csv_path = os.path.join(args.report_dir, "quality_feature_distribution.csv")
    stats_json_path = os.path.join(args.report_dir, "quality_feature_stats.json")
    summary_json_path = os.path.join(args.report_dir, "quality_build_summary.json")

    quality_df.to_csv(quality_csv_path, index=False)
    quality_df.to_csv(quality_report_csv_path, index=False)
    skipped_df.to_csv(skipped_csv_path, index=False)

    if not quality_df.empty:
        existing_quality_cols = [
            col for col in QUALITY_FEATURE_COLUMNS if col in quality_df.columns
        ]

        distribution_df = quality_df[existing_quality_cols].describe(
            percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
        ).T

        distribution_df.to_csv(stats_csv_path)

        stats_json = distribution_df.reset_index().rename(
            columns={"index": "feature"}
        ).to_dict(orient="records")
    else:
        distribution_df = pd.DataFrame()
        distribution_df.to_csv(stats_csv_path)
        stats_json = []

    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats_json, f, ensure_ascii=False, indent=4)

    skipped_status_counts = (
        skipped_df["status"].value_counts().to_dict()
        if not skipped_df.empty and "status" in skipped_df.columns
        else {}
    )

    summary = {
        "common_metadata_path": args.common_metadata,
        "two_d_conf_dir": args.two_d_conf_dir,
        "normalized_3d_dir": args.normalized_3d_dir,
        "quality_output_dir": args.quality_output_dir,
        "report_dir": args.report_dir,
        "sequence_length": args.sequence_length,
        "stride": args.stride,
        "include_short": bool(args.include_short),
        "confidence_threshold": args.confidence_threshold,
        "num_quality_sequences": int(len(quality_df)),
        "num_unique_videos": int(quality_df["video_key"].nunique()) if not quality_df.empty else 0,
        "num_skipped_records": int(len(skipped_df)),
        "skipped_status_counts": skipped_status_counts,
        "quality_feature_columns": QUALITY_FEATURE_COLUMNS,
        "quality_csv_path": quality_csv_path,
        "quality_report_csv_path": quality_report_csv_path,
        "skipped_csv_path": skipped_csv_path,
        "stats_csv_path": stats_csv_path,
        "stats_json_path": stats_json_path,
        "note": (
            "quality_sequences.csv stores sequence-level quality features. "
            "Phase 4 training can join this file with 2D confidence sequences "
            "and existing 3D sequences using video_key + sequence_index."
        ),
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print("\nSaved outputs:")
    print(f"- {quality_csv_path}")
    print(f"- {quality_report_csv_path}")
    print(f"- {skipped_csv_path}")
    print(f"- {stats_csv_path}")
    print(f"- {stats_json_path}")
    print(f"- {summary_json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sequence-level pose quality features for Phase 4."
    )

    parser.add_argument(
        "--common-metadata",
        type=str,
        default=DEFAULT_COMMON_METADATA_PATH,
        help="Path to Phase 3 common_metadata.csv.",
    )

    parser.add_argument(
        "--two-d-conf-dir",
        type=str,
        default=DEFAULT_2D_CONF_DIR,
        help="Directory containing 2D keypoints with confidence.",
    )

    parser.add_argument(
        "--normalized-3d-dir",
        type=str,
        default=DEFAULT_NORMALIZED_3D_DIR,
        help="Directory containing normalized estimated 3D pose CSV files.",
    )

    parser.add_argument(
        "--quality-output-dir",
        type=str,
        default=DEFAULT_QUALITY_OUTPUT_DIR,
        help="Directory to save quality_sequences.csv.",
    )

    parser.add_argument(
        "--report-dir",
        type=str,
        default=DEFAULT_REPORT_DIR,
        help="Directory to save quality feature reports.",
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=60,
        help="Sequence length, should match Phase 3.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=15,
        help="Sliding window stride, should match Phase 3.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Confidence threshold used for missing_joint_ratio.",
    )

    parser.add_argument(
        "--include-short",
        action="store_true",
        default=True,
        help=(
            "Include videos shorter than sequence length as short sequences. "
            "Default is True to keep the same test videos as Phase 3."
        ),
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

    if not os.path.exists(args.common_metadata):
        raise FileNotFoundError(f"common_metadata.csv not found: {args.common_metadata}")

    if not os.path.exists(args.two_d_conf_dir):
        raise FileNotFoundError(f"2D confidence directory not found: {args.two_d_conf_dir}")

    if not os.path.exists(args.normalized_3d_dir):
        raise FileNotFoundError(f"Normalized 3D directory not found: {args.normalized_3d_dir}")

    ensure_dir(args.quality_output_dir)
    ensure_dir(args.report_dir)

    metadata = pd.read_csv(args.common_metadata)

    if args.max_videos is not None:
        metadata = metadata.head(args.max_videos).copy()

    two_d_conf_files = list_csv_by_name(args.two_d_conf_dir)
    normalized_3d_files = list_csv_by_name(args.normalized_3d_dir)

    print("\nPhase 4 - Build Quality Features")
    print("=" * 80)
    print(f"Common metadata       : {args.common_metadata}")
    print(f"2D confidence dir     : {args.two_d_conf_dir}")
    print(f"Normalized 3D dir     : {args.normalized_3d_dir}")
    print(f"Quality output dir    : {args.quality_output_dir}")
    print(f"Report dir            : {args.report_dir}")
    print(f"Sequence length       : {args.sequence_length}")
    print(f"Stride                : {args.stride}")
    print(f"Confidence threshold  : {args.confidence_threshold}")
    print(f"Metadata records      : {len(metadata)}")
    print(f"2D confidence files   : {len(two_d_conf_files)}")
    print(f"Normalized 3D files   : {len(normalized_3d_files)}")
    print("=" * 80)

    all_rows = []
    skipped_rows = []

    for idx, record in metadata.iterrows():
        if idx == 0 or (idx + 1) % 200 == 0 or idx + 1 == len(metadata):
            print(f"Processing metadata record [{idx + 1}/{len(metadata)}]")

        rows, skipped = process_single_record(
            record=record,
            two_d_conf_files=two_d_conf_files,
            normalized_3d_files=normalized_3d_files,
            args=args,
        )

        all_rows.extend(rows)

        if skipped is not None:
            skipped_rows.append(skipped)

    quality_df = pd.DataFrame(all_rows)
    skipped_df = pd.DataFrame(skipped_rows)

    # Keep a stable and readable column order.
    metadata_cols = [
        "video_key",
        "csv_file",
        "sequence_index",
        "start_index",
        "end_index",
        "start_frame",
        "end_frame",
        "sequence_length",
        "aligned_num_frames",
    ]

    optional_cols = [
        col
        for col in [
            "source",
            "class_name",
            "label",
            "binary_label",
            "action_label",
            "fall_label",
            "action_name",
            "category",
            "split",
        ]
        if col in quality_df.columns
    ]

    path_cols = ["path_2d_conf", "path_3d"]

    ordered_cols = (
        [col for col in metadata_cols if col in quality_df.columns]
        + optional_cols
        + [col for col in QUALITY_FEATURE_COLUMNS if col in quality_df.columns]
        + [col for col in path_cols if col in quality_df.columns]
    )

    remaining_cols = [col for col in quality_df.columns if col not in ordered_cols]

    if not quality_df.empty:
        quality_df = quality_df[ordered_cols + remaining_cols]

    save_outputs(
        quality_df=quality_df,
        skipped_df=skipped_df,
        args=args,
    )

    print("\nQuality feature building finished.")
    print("=" * 80)
    print(f"Quality sequences: {len(quality_df)}")
    print(f"Unique videos     : {quality_df['video_key'].nunique() if not quality_df.empty else 0}")
    print(f"Skipped records   : {len(skipped_df)}")

    if not skipped_df.empty and "status" in skipped_df.columns:
        print(f"Skipped status    : {skipped_df['status'].value_counts().to_dict()}")

    if not quality_df.empty:
        print("\nMain quality feature averages:")
        for col in [
            "mean_confidence",
            "min_confidence",
            "missing_joint_ratio",
            "temporal_jitter",
            "bone_length_std",
            "velocity_std",
            "bbox_aspect_change",
            "3d_z_instability",
        ]:
            if col in quality_df.columns:
                print(f"- {col}: {quality_df[col].mean():.6f}")

    print("=" * 80)


if __name__ == "__main__":
    main()