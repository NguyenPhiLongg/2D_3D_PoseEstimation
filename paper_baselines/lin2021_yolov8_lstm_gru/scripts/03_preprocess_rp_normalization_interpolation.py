from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
BASELINE_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = BASELINE_ROOT.parents[1]


# ======================================================================================
# FILE 03: PREPROCESS PAPER-STYLE SKELETON
# ======================================================================================
#
# This file follows the preprocessing part of the Lin et al. 2021-style pipeline.
#
# Input from File 02:
#   paper-style 15-joint skeleton with x, y, confidence:
#       joint_0_x, joint_0_y, joint_0_c, ..., joint_14_x, joint_14_y, joint_14_c
#
# Output for File 04:
#   normalized skeleton sequence without confidence:
#       joint_0_x, joint_0_y, ..., joint_14_x, joint_14_y
#
# Steps:
#   1. Read paper-style 15-joint skeleton CSV
#   2. Use confidence score to detect missing joints
#   3. Apply linear interpolation for missing x/y values over time
#   4. Apply relative-position normalization
#   5. Remove confidence score before RNN/LSTM/GRU training
#
# Important:
#   Confidence is used only for missing-value handling.
#   It is not included in the final model input, to stay close to the paper-style pipeline.
# ======================================================================================


PAPER15_NAMES = [
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "mid_hip",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
]


# Paper-style joint indices
NOSE = 0
NECK = 1
RIGHT_SHOULDER = 2
RIGHT_ELBOW = 3
RIGHT_WRIST = 4
LEFT_SHOULDER = 5
LEFT_ELBOW = 6
LEFT_WRIST = 7
MID_HIP = 8
RIGHT_HIP = 9
RIGHT_KNEE = 10
RIGHT_ANKLE = 11
LEFT_HIP = 12
LEFT_KNEE = 13
LEFT_ANKLE = 14


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def make_safe_name(text: str) -> str:
    safe = str(text)
    safe = safe.replace("\\", "__")
    safe = safe.replace("/", "__")
    safe = safe.replace(":", "")
    safe = safe.replace(" ", "_")
    safe = safe.replace("-", "_")
    safe = safe.replace(".", "_")
    safe = safe.replace("(", "")
    safe = safe.replace(")", "")
    safe = safe.replace("[", "")
    safe = safe.replace("]", "")
    safe = re.sub(r"_+", "_", safe)
    safe = safe.strip("_")

    if safe == "":
        safe = "unknown"

    return safe


def label_to_name(label: Optional[int]) -> str:
    if label == 1:
        return "Fall"
    if label == 0:
        return "Not_Fall"
    return "Unknown"


def read_csv_safe(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "latin1"]

    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e

    raise last_error


def get_required_columns() -> List[str]:
    cols = [
        "video_id",
        "frame_idx",
        "label",
        "label_name",
    ]

    for j in range(15):
        cols.extend([
            f"joint_{j}_x",
            f"joint_{j}_y",
            f"joint_{j}_c",
        ])

    return cols


def validate_input_df(df: pd.DataFrame, csv_path: Path):
    required = get_required_columns()

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required paper skeleton columns in {csv_path}. Missing examples: {missing[:20]}"
        )


def to_numeric_array(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    arr = df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    return arr


def extract_xyc(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        xy: shape (T, 15, 2)
        conf: shape (T, 15)
    """
    x_cols = [f"joint_{j}_x" for j in range(15)]
    y_cols = [f"joint_{j}_y" for j in range(15)]
    c_cols = [f"joint_{j}_c" for j in range(15)]

    x = to_numeric_array(df, x_cols)
    y = to_numeric_array(df, y_cols)
    c = to_numeric_array(df, c_cols)

    xy = np.stack([x, y], axis=-1).astype(np.float32)
    conf = c.astype(np.float32)

    return xy, conf


def build_missing_mask(xy: np.ndarray, conf: np.ndarray, conf_threshold: float) -> np.ndarray:
    """
    missing_mask shape: (T, 15)
    True means the joint is missing or unreliable.
    """
    x = xy[:, :, 0]
    y = xy[:, :, 1]

    coord_missing = (x == 0.0) & (y == 0.0)
    conf_missing = conf <= conf_threshold
    nan_missing = np.isnan(x) | np.isnan(y) | np.isnan(conf)

    missing = coord_missing | conf_missing | nan_missing

    return missing


def interpolate_missing_xy(xy: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
    """
    Linear interpolation over time for each joint coordinate.

    If a joint is missing in a frame:
        x/y are set to NaN
        pandas interpolate is used over time
        front/back missing values are filled by nearest valid value
        remaining NaN values are filled with 0
    """
    out = xy.copy().astype(np.float32)

    T, J, _ = out.shape

    for j in range(J):
        for axis in range(2):
            values = out[:, j, axis].astype(np.float32)

            values[missing_mask[:, j]] = np.nan

            series = pd.Series(values)

            series = series.interpolate(
                method="linear",
                limit_direction="both",
            )

            series = series.ffill().bfill().fillna(0.0)

            out[:, j, axis] = series.to_numpy(dtype=np.float32)

    return out


def point_valid(point: np.ndarray) -> bool:
    if point.shape[0] != 2:
        return False

    if np.isnan(point).any():
        return False

    if float(point[0]) == 0.0 and float(point[1]) == 0.0:
        return False

    return True


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    if not point_valid(p1) or not point_valid(p2):
        return 0.0

    return float(np.linalg.norm(p1 - p2))


def compute_body_scale(frame_xy: np.ndarray) -> float:
    """
    Compute scale for relative-position normalization.

    Priority:
        1. distance(neck, mid_hip)
        2. shoulder width
        3. hip width
        4. max body spread
        5. 1.0 fallback

    This keeps skeleton size comparable across videos.
    """
    neck_midhip = distance(frame_xy[NECK], frame_xy[MID_HIP])

    if neck_midhip > 1e-6:
        return neck_midhip

    shoulder_width = distance(frame_xy[LEFT_SHOULDER], frame_xy[RIGHT_SHOULDER])

    if shoulder_width > 1e-6:
        return shoulder_width

    hip_width = distance(frame_xy[LEFT_HIP], frame_xy[RIGHT_HIP])

    if hip_width > 1e-6:
        return hip_width

    valid_points = []

    for p in frame_xy:
        if point_valid(p):
            valid_points.append(p)

    if len(valid_points) >= 2:
        pts = np.stack(valid_points, axis=0)
        spread = np.max(pts, axis=0) - np.min(pts, axis=0)
        scale = float(np.linalg.norm(spread))

        if scale > 1e-6:
            return scale

    return 1.0


def compute_root_point(frame_xy: np.ndarray) -> np.ndarray:
    """
    Root point for relative-position normalization.

    Priority:
        1. mid_hip
        2. neck
        3. average of valid joints
        4. zero fallback
    """
    if point_valid(frame_xy[MID_HIP]):
        return frame_xy[MID_HIP].astype(np.float32)

    if point_valid(frame_xy[NECK]):
        return frame_xy[NECK].astype(np.float32)

    valid_points = []

    for p in frame_xy:
        if point_valid(p):
            valid_points.append(p)

    if len(valid_points) > 0:
        return np.mean(np.stack(valid_points, axis=0), axis=0).astype(np.float32)

    return np.array([0.0, 0.0], dtype=np.float32)


def rp_normalize_xy(xy: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Relative-position normalization.

    For each frame:
        normalized_joint = (joint_xy - root_xy) / body_scale

    This removes absolute image location and reduces scale differences.
    """
    out = np.zeros_like(xy, dtype=np.float32)

    scales = []
    root_xs = []
    root_ys = []

    T = xy.shape[0]

    for t in range(T):
        frame = xy[t]

        root = compute_root_point(frame)
        scale = compute_body_scale(frame)

        if scale <= 1e-6:
            scale = 1.0

        out[t] = (frame - root.reshape(1, 2)) / float(scale)

        scales.append(float(scale))
        root_xs.append(float(root[0]))
        root_ys.append(float(root[1]))

    stats = {
        "scale_min": float(np.min(scales)) if len(scales) > 0 else 0.0,
        "scale_max": float(np.max(scales)) if len(scales) > 0 else 0.0,
        "scale_mean": float(np.mean(scales)) if len(scales) > 0 else 0.0,
        "root_x_mean": float(np.mean(root_xs)) if len(root_xs) > 0 else 0.0,
        "root_y_mean": float(np.mean(root_ys)) if len(root_ys) > 0 else 0.0,
    }

    return out, stats


def flatten_normalized_xy(normalized_xy: np.ndarray) -> np.ndarray:
    """
    Convert shape (T, 15, 2) to (T, 30) with order:
        joint_0_x, joint_0_y, joint_1_x, joint_1_y, ..., joint_14_x, joint_14_y
    """
    return normalized_xy.reshape(normalized_xy.shape[0], 30).astype(np.float32)


def build_output_dataframe(
    input_df: pd.DataFrame,
    normalized_xy: np.ndarray,
    missing_mask: np.ndarray,
    interpolation_xy: np.ndarray,
    csv_path: Path,
) -> pd.DataFrame:
    T = normalized_xy.shape[0]

    if "video_id" in input_df.columns:
        video_id = str(input_df["video_id"].iloc[0])
    else:
        video_id = csv_path.stem

    label = int(pd.to_numeric(input_df["label"].iloc[0], errors="coerce"))

    if "label_name" in input_df.columns:
        label_name = str(input_df["label_name"].iloc[0])
    else:
        label_name = label_to_name(label)

    if "frame_idx" in input_df.columns:
        frame_idx = pd.to_numeric(input_df["frame_idx"], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        frame_idx = np.arange(T, dtype=np.int64)

    feature_matrix = flatten_normalized_xy(normalized_xy)

    out = pd.DataFrame({
        "video_id": [video_id] * T,
        "frame_idx": frame_idx,
        "source_file": [str(csv_path)] * T,
        "label": [label] * T,
        "label_name": [label_name] * T,
    })

    col_idx = 0

    for j in range(15):
        out[f"joint_{j}_name"] = PAPER15_NAMES[j]
        out[f"joint_{j}_x"] = feature_matrix[:, col_idx]
        out[f"joint_{j}_y"] = feature_matrix[:, col_idx + 1]

        col_idx += 2

    # Extra diagnostic columns, not used by File 04 as features.
    out["num_missing_joints_before_interpolation"] = missing_mask.sum(axis=1).astype(int)
    out["all_joints_missing_before_interpolation"] = (missing_mask.sum(axis=1) == 15).astype(int)

    interpolated_zero_mask = (interpolation_xy[:, :, 0] == 0.0) & (interpolation_xy[:, :, 1] == 0.0)
    out["num_zero_joints_after_interpolation"] = interpolated_zero_mask.sum(axis=1).astype(int)

    return out


def preprocess_one_dataframe(
    df: pd.DataFrame,
    csv_path: Path,
    conf_threshold: float,
) -> Tuple[pd.DataFrame, Dict]:
    validate_input_df(df, csv_path)

    if "frame_idx" in df.columns:
        df = df.sort_values("frame_idx").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    xy, conf = extract_xyc(df)

    missing_mask = build_missing_mask(
        xy=xy,
        conf=conf,
        conf_threshold=conf_threshold,
    )

    missing_values_before = int(missing_mask.sum())
    total_joint_values = int(missing_mask.size)

    interpolated_xy = interpolate_missing_xy(
        xy=xy,
        missing_mask=missing_mask,
    )

    normalized_xy, norm_stats = rp_normalize_xy(interpolated_xy)

    out_df = build_output_dataframe(
        input_df=df,
        normalized_xy=normalized_xy,
        missing_mask=missing_mask,
        interpolation_xy=interpolated_xy,
        csv_path=csv_path,
    )

    feature_cols = []

    for j in range(15):
        feature_cols.extend([
            f"joint_{j}_x",
            f"joint_{j}_y",
        ])

    feature_matrix = out_df[feature_cols].to_numpy(dtype=np.float32)

    report = {
        "input_csv": str(csv_path),
        "num_frames": int(len(df)),
        "raw_shape": list(xy.shape),
        "processed_shape": list(feature_matrix.shape),
        "missing_joint_values_before_interpolation": missing_values_before,
        "total_joint_values": total_joint_values,
        "missing_joint_ratio_before_interpolation": float(missing_values_before / max(total_joint_values, 1)),
        "feature_min": float(np.min(feature_matrix)) if feature_matrix.size > 0 else 0.0,
        "feature_max": float(np.max(feature_matrix)) if feature_matrix.size > 0 else 0.0,
        "feature_mean": float(np.mean(feature_matrix)) if feature_matrix.size > 0 else 0.0,
        "feature_std": float(np.std(feature_matrix)) if feature_matrix.size > 0 else 0.0,
        "normalization_stats": norm_stats,
    }

    return out_df, report


def find_input_csv_files(input_dir: Path) -> List[Path]:
    csv_files = sorted([
        p for p in input_dir.rglob("*.csv")
        if p.name != "all_paper_skeleton_csv_index.csv"
    ])

    return csv_files


def process_one_csv(
    csv_path: Path,
    output_dir: Path,
    conf_threshold: float,
) -> Dict:
    df = read_csv_safe(csv_path)

    out_df, item_report = preprocess_one_dataframe(
        df=df,
        csv_path=csv_path,
        conf_threshold=conf_threshold,
    )

    if len(out_df) == 0:
        raise ValueError(f"No rows after preprocessing: {csv_path}")

    video_id = str(out_df["video_id"].iloc[0])

    labels = sorted(out_df["label"].astype(int).unique().tolist())

    if len(labels) == 1:
        label = int(labels[0])
    else:
        label = int(out_df["label"].value_counts().idxmax())

    label_name = label_to_name(label)

    label_folder = label_name if label_name in ["Fall", "Not_Fall"] else "Unknown"

    video_output_dir = output_dir / label_folder
    ensure_dir(video_output_dir)

    safe_video_id = make_safe_name(video_id)

    output_csv = video_output_dir / f"{safe_video_id}_processed.csv"

    out_df.to_csv(output_csv, index=False)

    feature_cols = []

    for j in range(15):
        feature_cols.extend([
            f"joint_{j}_x",
            f"joint_{j}_y",
        ])

    feature_matrix = out_df[feature_cols].to_numpy(dtype=np.float32)

    result = {
        "video_id": safe_video_id,
        "label": int(label),
        "label_name": label_name,
        "num_frames": int(len(out_df)),
        "feature_dim": int(feature_matrix.shape[1]),
        "missing_joint_values_before_interpolation": int(item_report["missing_joint_values_before_interpolation"]),
        "missing_joint_ratio_before_interpolation": float(item_report["missing_joint_ratio_before_interpolation"]),
        "feature_min": float(item_report["feature_min"]),
        "feature_max": float(item_report["feature_max"]),
        "feature_mean": float(item_report["feature_mean"]),
        "feature_std": float(item_report["feature_std"]),
        "input_csv": str(csv_path),
        "output_csv": str(output_csv),
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Apply RP-normalization, interpolation, and confidence removal for Lin-style YOLOv8 skeleton baseline."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(BASELINE_ROOT / "data" / "paper_skeleton_csv"),
        help="Input folder from File 02.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BASELINE_ROOT / "data" / "processed"),
        help="Output folder for processed skeleton CSV files.",
    )

    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.0,
        help="Joint confidence <= this value is treated as missing.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    reports_dir = BASELINE_ROOT / "outputs" / "reports"

    ensure_dir(input_dir)
    ensure_dir(output_dir)
    ensure_dir(reports_dir)

    csv_files = find_input_csv_files(input_dir)

    print("=" * 100)
    print("FILE 03 - RP-normalization, interpolation, and confidence removal")
    print("=" * 100)
    print(f"Input dir:        {input_dir}")
    print(f"Output dir:       {output_dir}")
    print(f"Input CSV files:  {len(csv_files)}")
    print(f"Conf threshold:   {args.conf_threshold}")
    print("=" * 100)

    if len(csv_files) == 0:
        report = {
            "status": "no_input_csv",
            "input_dir": str(input_dir),
            "message": "No paper-style skeleton CSV files found. Run file 02 first.",
        }

        report_path = reports_dir / "03_preprocess_rp_normalization_interpolation_report.json"
        save_json(report, report_path)

        print("No input CSV files found.")
        print(f"Report: {report_path}")
        return

    results = []
    failed = []

    for idx, csv_path in enumerate(csv_files, start=1):
        print(f"[{idx}/{len(csv_files)}] Processing: {csv_path.name}")

        try:
            result = process_one_csv(
                csv_path=csv_path,
                output_dir=output_dir,
                conf_threshold=args.conf_threshold,
            )

            results.append(result)

            print(
                f"  OK -> frames={result['num_frames']}, "
                f"feature_dim={result['feature_dim']}, "
                f"missing_ratio={result['missing_joint_ratio_before_interpolation']:.4f}"
            )

        except Exception as e:
            failed.append({
                "input_csv": str(csv_path),
                "error": repr(e),
            })

            print(f"  FAILED: {repr(e)}")

    index_df = pd.DataFrame(results)
    index_path = output_dir / "all_processed_skeleton_index.csv"

    if len(index_df) > 0:
        index_df.to_csv(index_path, index=False)

    total_frames = int(index_df["num_frames"].sum()) if len(index_df) > 0 else 0

    if len(index_df) > 0:
        avg_missing_ratio = float(index_df["missing_joint_ratio_before_interpolation"].mean())
    else:
        avg_missing_ratio = 0.0

    report = {
        "status": "completed",
        "pipeline_note": "This step follows the paper-style preprocessing: confidence is used to detect missing joints, missing joints are compensated by linear interpolation, skeletons are normalized by relative position, and confidence is removed before sequence modeling.",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_input_csv": len(csv_files),
        "num_success": len(results),
        "num_failed": len(failed),
        "total_frames": total_frames,
        "average_missing_joint_ratio_before_interpolation": avg_missing_ratio,
        "feature_dim": 30,
        "index_csv": str(index_path),
        "failed": failed,
        "results": results,
    }

    report_path = reports_dir / "03_preprocess_rp_normalization_interpolation_report.json"
    save_json(report, report_path)

    print("=" * 100)
    print("FILE 03 completed.")
    print("=" * 100)
    print(f"Success videos:                 {len(results)}")
    print(f"Failed videos:                  {len(failed)}")
    print(f"Total frames:                   {total_frames}")
    print(f"Average missing joint ratio:    {avg_missing_ratio:.6f}")
    print(f"Feature dim:                    30")
    print(f"Index CSV:                      {index_path}")
    print(f"Report:                         {report_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
