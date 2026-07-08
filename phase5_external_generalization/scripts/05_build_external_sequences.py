import os
import sys
import json
import time
import argparse
import importlib.util
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

PHASE3_DIR = PROJECT_ROOT / "phase3_common_set_gated_fusion"
PHASE4_DIR = PROJECT_ROOT / "phase4_quality_aware_fusion"

if str(PHASE5_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE5_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(PHASE3_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE3_DIR))

if str(PHASE4_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE4_DIR))


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
    sequence_manifest_summary,
    build_fair_common_manifest,
    save_feature_availability,
    print_dict,
    print_dataframe_summary,
)


# ============================================================
# LOAD EXACT PHASE 3 / PHASE 4 MODULES
# ============================================================

def load_module_from_file(module_name: str, file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required module not found: {file_path}\n"
            "This file is required to keep Phase 5 preprocessing consistent with previous phases."
        )

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


PHASE3_UTILS = load_module_from_file(
    "phase3_utils_exact",
    PHASE3_DIR / "phase3_utils.py",
)

PHASE4_UTILS = load_module_from_file(
    "phase4_utils_exact",
    PHASE4_DIR / "phase4_utils.py",
)


# Exact preprocessing functions from earlier phases.
extract_phase3_2d_features_from_df = PHASE3_UTILS.extract_2d_features_from_df
extract_phase3_3d_features_from_df = PHASE3_UTILS.extract_3d_features_from_df

extract_phase4_2d_features_from_conf_df = PHASE4_UTILS.extract_2d_features_from_conf_df
extract_phase4_3d_features_from_df = PHASE4_UTILS.extract_3d_features_from_df

try:
    QUALITY_FEATURE_COLUMNS = list(PHASE4_UTILS.QUALITY_FEATURE_COLUMNS)
except Exception:
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


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_2D_DIM = 40
EXPECTED_3D_DIM = 59
EXPECTED_CONCAT_DIM = 99
EXPECTED_QUALITY_DIM = 33


# ============================================================
# BASIC HELPERS
# ============================================================

def read_csv_sorted(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    if "frame" in df.columns:
        df = df.sort_values("frame").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default

        value = float(value)

        if np.isnan(value) or np.isinf(value):
            return default

        return value
    except Exception:
        return default


def ensure_2d_array(name: str, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D array [T,D], got shape {arr.shape}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    return arr.astype(np.float32)


def pad_or_trim_sequence(
    features: np.ndarray,
    sequence_length: int,
    padding_strategy: str = "repeat_last_frame",
) -> Tuple[np.ndarray, int]:
    """
    Convert variable-length sequence into fixed [sequence_length, D].

    Returns:
        fixed_sequence
        valid_length_before_padding
    """
    features = ensure_2d_array("features", features)

    t, d = features.shape
    valid_length = min(t, sequence_length)

    if t == sequence_length:
        return features.astype(np.float32), valid_length

    if t > sequence_length:
        return features[:sequence_length].astype(np.float32), valid_length

    if t <= 0:
        return np.zeros((sequence_length, d), dtype=np.float32), 0

    pad_count = sequence_length - t

    if padding_strategy == "zero":
        pad = np.zeros((pad_count, d), dtype=np.float32)
    else:
        pad = np.repeat(features[-1:, :], pad_count, axis=0).astype(np.float32)

    fixed = np.concatenate([features, pad], axis=0).astype(np.float32)

    return fixed, valid_length


def get_frame_array(df: pd.DataFrame) -> np.ndarray:
    if "frame" not in df.columns:
        return np.arange(len(df), dtype=np.int64)

    return df["frame"].to_numpy(dtype=np.int64)


def indices_for_frame_range(frames: np.ndarray, start_frame: int, end_frame: int) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.int64)
    mask = (frames >= int(start_frame)) & (frames <= int(end_frame))
    return np.nonzero(mask)[0]


def get_quality_vector(row: pd.Series) -> np.ndarray:
    values = []

    for col in QUALITY_FEATURE_COLUMNS:
        if col not in row.index:
            values.append(0.0)
            continue

        values.append(safe_float(row[col], 0.0))

    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.shape[0] != EXPECTED_QUALITY_DIM:
        raise ValueError(
            f"Quality vector must be {EXPECTED_QUALITY_DIM}D, got {arr.shape[0]}"
        )

    return arr.astype(np.float32)


# ============================================================
# DATA LOADERS
# ============================================================

def load_sequence_manifest(config: dict) -> pd.DataFrame:
    sequence_dir = cfg_path(config, config["outputs"]["external_sequences_dir"])
    path = sequence_dir / "all_external_sequences.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"External sequence manifest not found: {path}\n"
            "Run Step 01 first."
        )

    df = pd.read_csv(path)

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
        raise ValueError(f"all_external_sequences.csv missing columns: {missing}")

    df = df[df["include_eval"].apply(bool_from_value)].copy()
    df = df.reset_index(drop=True)

    return df


def load_quality_dataframe(config: dict) -> pd.DataFrame:
    quality_dir = cfg_path(config, config["outputs"]["quality_features_dir"])
    path = quality_dir / "all_external_quality_sequences.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"External quality CSV not found: {path}\n"
            "Run Step 04 first."
        )

    df = pd.read_csv(path)

    if "sequence_key" not in df.columns:
        raise ValueError("Quality CSV missing sequence_key column.")

    missing_features = [col for col in QUALITY_FEATURE_COLUMNS if col not in df.columns]

    if missing_features:
        raise ValueError(
            f"Quality CSV missing quality feature columns: {missing_features}"
        )

    return df


def filter_manifest(
    df: pd.DataFrame,
    datasets: Optional[List[str]] = None,
    max_sequences: Optional[int] = None,
) -> pd.DataFrame:
    out = df.copy()

    if datasets:
        datasets = set(datasets)
        out = out[out["dataset"].isin(datasets)].copy()

    out = out.reset_index(drop=True)

    if max_sequences is not None:
        out = out.head(int(max_sequences)).copy()

    return out.reset_index(drop=True)


# ============================================================
# VIDEO FEATURE CACHE
# ============================================================

def build_video_feature_cache(
    config: dict,
    dataset: str,
    video_id: str,
) -> Dict[str, Any]:
    """
    Load one video and compute both old-phase and phase4-specific features.

    Important:
        Phase 1/2/3 models must use Phase 3 preprocessing.
        Phase 4 models must use Phase 4 preprocessing.

    So we keep both:
        x2d_common  : from phase3_utils.extract_2d_features_from_df
        x3d_common  : from phase3_utils.extract_3d_features_from_df
        x2d_quality : from phase4_utils.extract_2d_features_from_conf_df
        x3d_quality : from phase4_utils.extract_3d_features_from_df
    """
    path_2d = get_2d_feature_path(config, dataset, video_id)
    path_3d = get_3d_feature_path(config, dataset, video_id)

    df2d = read_csv_sorted(path_2d)
    df3d = read_csv_sorted(path_3d)

    frames_2d = get_frame_array(df2d)
    frames_3d = get_frame_array(df3d)

    x2d_common = extract_phase3_2d_features_from_df(df2d)
    x3d_common = extract_phase3_3d_features_from_df(df3d)

    x2d_quality = extract_phase4_2d_features_from_conf_df(df2d)
    x3d_quality = extract_phase4_3d_features_from_df(df3d, path=str(path_3d))

    x2d_common = ensure_2d_array("x2d_common", x2d_common)
    x3d_common = ensure_2d_array("x3d_common", x3d_common)
    x2d_quality = ensure_2d_array("x2d_quality", x2d_quality)
    x3d_quality = ensure_2d_array("x3d_quality", x3d_quality)

    if x2d_common.shape[1] != EXPECTED_2D_DIM:
        raise ValueError(f"x2d_common dim must be 40, got {x2d_common.shape}")

    if x3d_common.shape[1] != EXPECTED_3D_DIM:
        raise ValueError(f"x3d_common dim must be 59, got {x3d_common.shape}")

    if x2d_quality.shape[1] != EXPECTED_2D_DIM:
        raise ValueError(f"x2d_quality dim must be 40, got {x2d_quality.shape}")

    if x3d_quality.shape[1] != EXPECTED_3D_DIM:
        raise ValueError(f"x3d_quality dim must be 59, got {x3d_quality.shape}")

    min_len = min(
        len(frames_2d),
        len(frames_3d),
        x2d_common.shape[0],
        x3d_common.shape[0],
        x2d_quality.shape[0],
        x3d_quality.shape[0],
    )

    if min_len <= 0:
        raise ValueError(f"No aligned frames for {dataset} | {video_id}")

    return {
        "path_2d": str(path_2d),
        "path_3d": str(path_3d),
        "frames_2d": frames_2d[:min_len],
        "frames_3d": frames_3d[:min_len],
        "x2d_common": x2d_common[:min_len],
        "x3d_common": x3d_common[:min_len],
        "x2d_quality": x2d_quality[:min_len],
        "x3d_quality": x3d_quality[:min_len],
        "min_len": int(min_len),
    }


def slice_sequence_from_cache(
    cache: Dict[str, Any],
    start_frame: int,
    end_frame: int,
    sequence_length: int,
    padding_strategy: str,
) -> Optional[Dict[str, Any]]:
    frames_2d = cache["frames_2d"]
    frames_3d = cache["frames_3d"]

    common_frames = np.intersect1d(frames_2d, frames_3d)

    common_frames = common_frames[
        (common_frames >= int(start_frame)) & (common_frames <= int(end_frame))
    ]

    if len(common_frames) <= 0:
        return None

    idx_2d = np.nonzero(np.isin(frames_2d, common_frames))[0]
    idx_3d = np.nonzero(np.isin(frames_3d, common_frames))[0]

    min_len = min(len(idx_2d), len(idx_3d))

    if min_len <= 0:
        return None

    idx_2d = idx_2d[:min_len]
    idx_3d = idx_3d[:min_len]

    x2d_common = cache["x2d_common"][idx_2d]
    x3d_common = cache["x3d_common"][idx_3d]

    x2d_quality = cache["x2d_quality"][idx_2d]
    x3d_quality = cache["x3d_quality"][idx_3d]

    x2d_common, valid_len_2d_common = pad_or_trim_sequence(
        x2d_common,
        sequence_length,
        padding_strategy,
    )

    x3d_common, valid_len_3d_common = pad_or_trim_sequence(
        x3d_common,
        sequence_length,
        padding_strategy,
    )

    x2d_quality, valid_len_2d_quality = pad_or_trim_sequence(
        x2d_quality,
        sequence_length,
        padding_strategy,
    )

    x3d_quality, valid_len_3d_quality = pad_or_trim_sequence(
        x3d_quality,
        sequence_length,
        padding_strategy,
    )

    valid_lengths = [
        valid_len_2d_common,
        valid_len_3d_common,
        valid_len_2d_quality,
        valid_len_3d_quality,
    ]

    valid_length = int(min(valid_lengths))

    return {
        "x2d_common": x2d_common,
        "x3d_common": x3d_common,
        "x2d_quality": x2d_quality,
        "x3d_quality": x3d_quality,
        "valid_length": valid_length,
        "aligned_frame_count": int(min_len),
    }


# ============================================================
# BUILD SEQUENCE INPUTS
# ============================================================

def build_sequence_inputs(
    config: dict,
    manifest_df: pd.DataFrame,
    quality_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    sequence_cfg = config.get("sequence", {})
    sequence_length = int(sequence_cfg.get("sequence_length", 60))
    padding_strategy = str(sequence_cfg.get("padding_strategy", "repeat_last_frame"))

    quality_map = {
        str(row["sequence_key"]): row
        for _, row in quality_df.iterrows()
    }

    cache_by_video = {}

    fair_rows = []
    skipped_rows = []

    x2d_common_list = []
    x3d_common_list = []
    xconcat_common_list = []

    x2d_quality_list = []
    x3d_quality_list = []
    quality_raw_list = []

    y_binary_list = []
    valid_length_list = []
    sequence_key_list = []
    dataset_list = []
    video_id_list = []

    grouped = manifest_df.groupby(["dataset", "video_id"], sort=False)

    total_videos = len(grouped)

    for video_idx, ((dataset, video_id), group_df) in enumerate(grouped, start=1):
        print("\n" + "-" * 100)
        print(f"[{video_idx}/{total_videos}] {dataset} | {video_id} | sequences={len(group_df)}")

        try:
            cache_key = (dataset, video_id)

            if cache_key not in cache_by_video:
                cache_by_video[cache_key] = build_video_feature_cache(
                    config=config,
                    dataset=dataset,
                    video_id=video_id,
                )

            cache = cache_by_video[cache_key]

        except Exception as exc:
            for _, row in group_df.iterrows():
                skipped_rows.append(
                    {
                        "dataset": dataset,
                        "video_id": video_id,
                        "sequence_key": row.get("sequence_key", ""),
                        "reason": "video_feature_load_failed",
                        "error": str(exc),
                    }
                )
            print(f"Skipped whole video: {exc}")
            continue

        built_for_video = 0

        for _, row in group_df.iterrows():
            sequence_key = str(row["sequence_key"])

            if sequence_key not in quality_map:
                skipped_rows.append(
                    {
                        "dataset": dataset,
                        "video_id": video_id,
                        "sequence_key": sequence_key,
                        "reason": "missing_quality_row",
                        "error": "sequence_key not found in all_external_quality_sequences.csv",
                    }
                )
                continue

            start_frame = safe_int(row["start_frame"])
            end_frame = safe_int(row["end_frame"])

            seq = slice_sequence_from_cache(
                cache=cache,
                start_frame=start_frame,
                end_frame=end_frame,
                sequence_length=sequence_length,
                padding_strategy=padding_strategy,
            )

            if seq is None:
                skipped_rows.append(
                    {
                        "dataset": dataset,
                        "video_id": video_id,
                        "sequence_key": sequence_key,
                        "reason": "empty_aligned_sequence",
                        "error": f"No aligned 2D/3D frames for {start_frame}-{end_frame}",
                    }
                )
                continue

            quality_vector = get_quality_vector(quality_map[sequence_key])

            x2d_common = seq["x2d_common"]
            x3d_common = seq["x3d_common"]
            xconcat_common = np.concatenate([x2d_common, x3d_common], axis=1).astype(np.float32)

            x2d_quality = seq["x2d_quality"]
            x3d_quality = seq["x3d_quality"]

            if x2d_common.shape != (sequence_length, EXPECTED_2D_DIM):
                raise ValueError(f"x2d_common wrong shape: {x2d_common.shape}")

            if x3d_common.shape != (sequence_length, EXPECTED_3D_DIM):
                raise ValueError(f"x3d_common wrong shape: {x3d_common.shape}")

            if xconcat_common.shape != (sequence_length, EXPECTED_CONCAT_DIM):
                raise ValueError(f"xconcat_common wrong shape: {xconcat_common.shape}")

            if x2d_quality.shape != (sequence_length, EXPECTED_2D_DIM):
                raise ValueError(f"x2d_quality wrong shape: {x2d_quality.shape}")

            if x3d_quality.shape != (sequence_length, EXPECTED_3D_DIM):
                raise ValueError(f"x3d_quality wrong shape: {x3d_quality.shape}")

            out_row = row.to_dict()
            out_row["fair_include"] = True
            out_row["valid_length_built"] = int(seq["valid_length"])
            out_row["aligned_frame_count_built"] = int(seq["aligned_frame_count"])
            out_row["path_2d_conf"] = cache["path_2d"]
            out_row["path_3d"] = cache["path_3d"]

            fair_rows.append(out_row)

            x2d_common_list.append(x2d_common)
            x3d_common_list.append(x3d_common)
            xconcat_common_list.append(xconcat_common)

            x2d_quality_list.append(x2d_quality)
            x3d_quality_list.append(x3d_quality)
            quality_raw_list.append(quality_vector)

            y_binary_list.append(int(row["label"]))
            valid_length_list.append(int(seq["valid_length"]))
            sequence_key_list.append(sequence_key)
            dataset_list.append(str(dataset))
            video_id_list.append(str(video_id))

            built_for_video += 1

        print(f"Built sequences: {built_for_video}")

    fair_df = pd.DataFrame(fair_rows)
    skipped_df = pd.DataFrame(skipped_rows)

    if fair_df.empty:
        raise RuntimeError("No fair external sequences were built.")

    arrays = {
        "x2d_common": np.stack(x2d_common_list, axis=0).astype(np.float32),
        "x3d_common": np.stack(x3d_common_list, axis=0).astype(np.float32),
        "xconcat_common": np.stack(xconcat_common_list, axis=0).astype(np.float32),
        "x2d_quality": np.stack(x2d_quality_list, axis=0).astype(np.float32),
        "x3d_quality": np.stack(x3d_quality_list, axis=0).astype(np.float32),
        "quality_raw": np.stack(quality_raw_list, axis=0).astype(np.float32),
        "y_binary": np.asarray(y_binary_list, dtype=np.int64),
        "valid_lengths": np.asarray(valid_length_list, dtype=np.int64),
        "sequence_keys": np.asarray(sequence_key_list, dtype=object),
        "datasets": np.asarray(dataset_list, dtype=object),
        "video_ids": np.asarray(video_id_list, dtype=object),
    }

    return fair_df, skipped_df, arrays


# ============================================================
# VALIDATION
# ============================================================

def validate_arrays(arrays: Dict[str, np.ndarray], fair_df: pd.DataFrame) -> Dict[str, Any]:
    report = {
        "valid": True,
        "errors": [],
        "num_fair_sequences": int(len(fair_df)),
    }

    n = len(fair_df)

    expected_shapes = {
        "x2d_common": (n, 60, 40),
        "x3d_common": (n, 60, 59),
        "xconcat_common": (n, 60, 99),
        "x2d_quality": (n, 60, 40),
        "x3d_quality": (n, 60, 59),
        "quality_raw": (n, 33),
        "y_binary": (n,),
        "valid_lengths": (n,),
        "sequence_keys": (n,),
    }

    shape_report = {}

    for key, expected_shape in expected_shapes.items():
        if key not in arrays:
            report["valid"] = False
            report["errors"].append(f"Missing array: {key}")
            continue

        actual_shape = tuple(arrays[key].shape)
        shape_report[key] = actual_shape

        if actual_shape != expected_shape:
            report["valid"] = False
            report["errors"].append(
                f"{key} wrong shape. Expected {expected_shape}, got {actual_shape}"
            )

    for key in [
        "x2d_common",
        "x3d_common",
        "xconcat_common",
        "x2d_quality",
        "x3d_quality",
        "quality_raw",
    ]:
        if key not in arrays:
            continue

        arr = arrays[key]

        if np.isnan(arr).any():
            report["valid"] = False
            report["errors"].append(f"NaN found in {key}")

        if np.isinf(arr).any():
            report["valid"] = False
            report["errors"].append(f"Inf found in {key}")

    y = arrays.get("y_binary", np.asarray([]))

    if y.size:
        unique, counts = np.unique(y, return_counts=True)
        report["label_counts"] = {
            str(int(k)): int(v)
            for k, v in zip(unique, counts)
        }

    report["shapes"] = {
        key: list(value)
        for key, value in shape_report.items()
    }

    report["sequence_key_match"] = True

    if "sequence_keys" in arrays and "sequence_key" in fair_df.columns:
        keys_from_array = [str(x) for x in arrays["sequence_keys"].tolist()]
        keys_from_df = fair_df["sequence_key"].astype(str).tolist()

        if keys_from_array != keys_from_df:
            report["valid"] = False
            report["sequence_key_match"] = False
            report["errors"].append("sequence_keys array order does not match fair_df sequence_key order.")

    return report


def create_feature_availability(fair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in fair_df.iterrows():
        rows.append(
            {
                "sequence_key": row["sequence_key"],
                "dataset": row["dataset"],
                "video_id": row["video_id"],
                "has_2d": True,
                "has_3d": True,
                "has_quality": True,
                "path_2d_conf": row.get("path_2d_conf", ""),
                "path_3d": row.get("path_3d", ""),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    config: dict,
    fair_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    arrays: Dict[str, np.ndarray],
    validation_report: Dict[str, Any],
    elapsed_sec: float,
) -> Dict[str, str]:
    output_dir = cfg_path(config, config["outputs"]["external_sequences_dir"])
    report_dir = cfg_path(config, config["outputs"]["sequences_output_dir"])

    ensure_dir(output_dir)
    ensure_dir(report_dir)

    fair_manifest_path = output_dir / "all_external_sequences_common_fair.csv"
    skipped_path = report_dir / "05_build_external_sequences_skipped.csv"
    npz_path = output_dir / "external_sequence_inputs.npz"
    feature_availability_path = output_dir / "feature_availability.csv"
    report_json_path = report_dir / "05_build_external_sequences_report.json"

    fair_df.to_csv(fair_manifest_path, index=False, encoding="utf-8-sig")
    skipped_df.to_csv(skipped_path, index=False, encoding="utf-8-sig")

    feature_availability_df = create_feature_availability(fair_df)
    feature_availability_df.to_csv(feature_availability_path, index=False, encoding="utf-8-sig")

    np.savez_compressed(
        npz_path,
        x2d_common=arrays["x2d_common"],
        x3d_common=arrays["x3d_common"],
        xconcat_common=arrays["xconcat_common"],
        x2d_quality=arrays["x2d_quality"],
        x3d_quality=arrays["x3d_quality"],
        quality_raw=arrays["quality_raw"],
        y_binary=arrays["y_binary"],
        valid_lengths=arrays["valid_lengths"],
        sequence_keys=arrays["sequence_keys"],
        datasets=arrays["datasets"],
        video_ids=arrays["video_ids"],
        quality_feature_columns=np.asarray(QUALITY_FEATURE_COLUMNS, dtype=object),
    )

    report = {
        "phase": "Phase 5 - External Dataset Generalization",
        "step": "05_build_external_sequences",
        "elapsed_sec": float(elapsed_sec),
        "num_fair_sequences": int(len(fair_df)),
        "num_skipped_sequences": int(len(skipped_df)),
        "sequence_length": 60,
        "input_dimensions": {
            "x2d_common": [60, 40],
            "x3d_common": [60, 59],
            "xconcat_common": [60, 99],
            "x2d_quality": [60, 40],
            "x3d_quality": [60, 59],
            "quality_raw": [33],
        },
        "preprocessing_alignment": {
            "phase1_2d_common": "uses x2d_common from phase3_utils.extract_2d_features_from_df",
            "phase2_3d_common": "uses x3d_common from phase3_utils.extract_3d_features_from_df",
            "phase2_concat_common": "uses xconcat_common = x2d_common + x3d_common",
            "phase3_gated_fusion": "uses x2d_common and x3d_common",
            "phase4_quality_models": "uses x2d_quality, x3d_quality, and quality_raw from phase4_utils preprocessing",
        },
        "fairness_rule": (
            "Every model will be evaluated using the same sequence_key order from "
            "all_external_sequences_common_fair.csv. Different preprocessing variants are kept only "
            "because Phase 1/2/3 and Phase 4 were trained with different preprocessing code."
        ),
        "validation": validation_report,
        "label_counts": (
            fair_df["label_name"].value_counts(dropna=False).to_dict()
            if "label_name" in fair_df.columns
            else {}
        ),
        "dataset_counts": (
            fair_df["dataset"].value_counts(dropna=False).to_dict()
            if "dataset" in fair_df.columns
            else {}
        ),
        "outputs": {
            "fair_manifest_path": str(fair_manifest_path),
            "skipped_path": str(skipped_path),
            "npz_path": str(npz_path),
            "feature_availability_path": str(feature_availability_path),
            "report_json_path": str(report_json_path),
        },
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    return report["outputs"]


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 Step 05 - Build fair external model input sequences."
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

    args = parser.parse_args()

    print("\nPhase 5 - Step 05: Build External Fair Sequence Inputs")
    print("=" * 100)
    print("This step uses exact preprocessing from previous phases:")
    print("  Phase 1/2/3 models -> phase3_utils")
    print("  Phase 4 models     -> phase4_utils")
    print("All models will still share the same sequence_key set.")
    print("=" * 100)

    start_time = time.time()

    config = load_config(args.config)

    print("\n[1/5] Loading sequence manifest...")
    manifest_df = load_sequence_manifest(config)

    manifest_df = filter_manifest(
        manifest_df,
        datasets=args.datasets,
        max_sequences=args.max_sequences,
    )

    if manifest_df.empty:
        raise RuntimeError("No external sequences selected.")

    print_dataframe_summary("Selected manifest", manifest_df, max_rows=10)
    print_dict("Manifest summary", sequence_manifest_summary(manifest_df))

    print("\n[2/5] Loading quality features...")
    quality_df = load_quality_dataframe(config)

    if args.datasets:
        quality_df = quality_df[quality_df["dataset"].isin(set(args.datasets))].copy()

    if args.max_sequences is not None:
        selected_keys = set(manifest_df["sequence_key"].astype(str).tolist())
        quality_df = quality_df[quality_df["sequence_key"].astype(str).isin(selected_keys)].copy()

    print_dataframe_summary("Quality dataframe", quality_df, max_rows=10)

    print("\n[3/5] Building sequence tensors...")
    fair_df, skipped_df, arrays = build_sequence_inputs(
        config=config,
        manifest_df=manifest_df,
        quality_df=quality_df,
    )

    print("\n[4/5] Validating sequence tensors...")
    validation_report = validate_arrays(arrays, fair_df)

    print_dict("Validation report", validation_report)

    if not validation_report["valid"]:
        raise RuntimeError(
            "Step 05 validation failed. Fix errors before evaluating models:\n"
            + json.dumps(validation_report, ensure_ascii=False, indent=4)
        )

    elapsed_sec = time.time() - start_time

    print("\n[5/5] Saving outputs...")
    outputs = save_outputs(
        config=config,
        fair_df=fair_df,
        skipped_df=skipped_df,
        arrays=arrays,
        validation_report=validation_report,
        elapsed_sec=elapsed_sec,
    )

    print_dict("Saved outputs", outputs)

    print("\nDONE: Phase 5 Step 05 completed.")
    print("=" * 100)
    print(f"Fair sequences   : {len(fair_df)}")
    print(f"Skipped sequences: {len(skipped_df)}")
    print(f"x2d_common       : {arrays['x2d_common'].shape}")
    print(f"x3d_common       : {arrays['x3d_common'].shape}")
    print(f"xconcat_common   : {arrays['xconcat_common'].shape}")
    print(f"x2d_quality      : {arrays['x2d_quality'].shape}")
    print(f"x3d_quality      : {arrays['x3d_quality'].shape}")
    print(f"quality_raw      : {arrays['quality_raw'].shape}")
    print(f"y_binary         : {arrays['y_binary'].shape}")
    print(f"Elapsed seconds  : {elapsed_sec:.2f}")
    print("=" * 100)

    print("\nImportant:")
    print("  Step 06 must load external_sequence_inputs.npz and")
    print("  all_external_sequences_common_fair.csv.")
    print("  Do not rebuild sequences again inside Step 06.")


if __name__ == "__main__":
    main()