from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
BASELINE_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = BASELINE_ROOT.parents[1]


# ======================================================================================
# FILE 03: COMPUTE CHEN 2020 RULE FEATURES
# ======================================================================================
#
# Chen et al. 2020 rule-based fall detection uses three key geometric conditions:
#
#   1. Speed of descent at the center of the hip joint
#   2. Angle between the human body centerline and the ground
#   3. Width-to-height ratio of the human body's external rectangle
#
# This file computes those three features frame-by-frame from Chen-style skeleton nodes.
#
# Input from File 02:
#   s0_x, s0_y, s0_c, ..., s13_x, s13_y, s13_c
#
# Output:
#   per-frame rule features:
#       hip_center_x
#       hip_center_y
#       hip_descent_velocity
#       body_centerline_angle_degrees
#       bbox_width
#       bbox_height
#       width_height_ratio
#
# Important:
#   This file does NOT apply the final rule yet.
#   Threshold tuning will be done in File 05.
#   Final three-condition prediction will be done in File 06.
# ======================================================================================


CHEN14_NAMES = [
    "head_or_nose",       # s0
    "shoulder_center",   # s1
    "right_shoulder",    # s2
    "right_elbow",       # s3
    "right_hand",        # s4
    "left_shoulder",     # s5
    "left_elbow",        # s6
    "left_hand",         # s7
    "right_hip",         # s8
    "right_knee",        # s9
    "right_ankle",       # s10
    "left_hip",          # s11
    "left_knee",         # s12
    "left_ankle",        # s13
]


# Chen-style node indices
S_HEAD = 0
S_SHOULDER_CENTER = 1
S_RIGHT_HIP = 8
S_RIGHT_ANKLE = 10
S_LEFT_HIP = 11
S_LEFT_ANKLE = 13


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
    return safe if safe else "unknown"


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

    for j in range(14):
        cols.extend([
            f"s{j}_x",
            f"s{j}_y",
            f"s{j}_c",
        ])

    return cols


def validate_chen_node_df(df: pd.DataFrame, csv_path: Path):
    required = get_required_columns()

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required Chen-node columns in {csv_path}. "
            f"Missing examples: {missing[:20]}"
        )


def to_float(value, default: float = 0.0) -> float:
    try:
        out = pd.to_numeric(value, errors="coerce")
        if pd.isna(out):
            return default
        return float(out)
    except Exception:
        return default


def extract_nodes(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        xy:   shape (T, 14, 2)
        conf: shape (T, 14)
    """
    x_cols = [f"s{j}_x" for j in range(14)]
    y_cols = [f"s{j}_y" for j in range(14)]
    c_cols = [f"s{j}_c" for j in range(14)]

    x = df[x_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y = df[y_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    c = df[c_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    xy = np.stack([x, y], axis=-1).astype(np.float32)
    conf = c.astype(np.float32)

    return xy, conf


def is_valid_node(xy: np.ndarray, conf: np.ndarray, node_idx: int, conf_threshold: float) -> np.ndarray:
    """
    Returns boolean mask shape (T,).
    """
    x = xy[:, node_idx, 0]
    y = xy[:, node_idx, 1]
    c = conf[:, node_idx]

    valid = (
        (c > conf_threshold)
        & ((x != 0.0) | (y != 0.0))
        & (~np.isnan(x))
        & (~np.isnan(y))
        & (~np.isnan(c))
    )

    return valid


def midpoint_nodes(
    xy: np.ndarray,
    conf: np.ndarray,
    idx_a: int,
    idx_b: int,
    conf_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute midpoint between two nodes.

    If both nodes are valid:
        midpoint = average(a, b)
    If only one node is valid:
        midpoint = that valid node
    If none is valid:
        midpoint = 0,0 and valid=False
    """
    T = xy.shape[0]

    out = np.zeros((T, 2), dtype=np.float32)
    valid = np.zeros((T,), dtype=bool)

    valid_a = is_valid_node(xy, conf, idx_a, conf_threshold)
    valid_b = is_valid_node(xy, conf, idx_b, conf_threshold)

    for t in range(T):
        if valid_a[t] and valid_b[t]:
            out[t] = (xy[t, idx_a] + xy[t, idx_b]) / 2.0
            valid[t] = True
        elif valid_a[t]:
            out[t] = xy[t, idx_a]
            valid[t] = True
        elif valid_b[t]:
            out[t] = xy[t, idx_b]
            valid[t] = True
        else:
            out[t] = np.array([0.0, 0.0], dtype=np.float32)
            valid[t] = False

    return out, valid


def compute_hip_center(xy: np.ndarray, conf: np.ndarray, conf_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Paper condition 1 uses the center of the hip joint.

    In Chen node model:
        s8  = right hip
        s11 = left hip

    hip_center = midpoint(s8, s11)
    """
    return midpoint_nodes(
        xy=xy,
        conf=conf,
        idx_a=S_RIGHT_HIP,
        idx_b=S_LEFT_HIP,
        conf_threshold=conf_threshold,
    )


def compute_ankle_center(xy: np.ndarray, conf: np.ndarray, conf_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    For body centerline, use midpoint of two ankles as lower body point.

    s10 = right ankle
    s13 = left ankle
    """
    return midpoint_nodes(
        xy=xy,
        conf=conf,
        idx_a=S_RIGHT_ANKLE,
        idx_b=S_LEFT_ANKLE,
        conf_threshold=conf_threshold,
    )


def compute_descent_velocity(
    hip_center: np.ndarray,
    hip_valid: np.ndarray,
    window_frames: int,
) -> np.ndarray:
    """
    Compute hip-center descent velocity over a fixed frame window.

    Image coordinate convention:
        y increases downward.

    Therefore:
        positive delta_y means the hip center moves downward.

    velocity[t] = hip_y[t] - hip_y[t - window_frames]

    This is a pixel-based velocity proxy because YOLOv8 keypoints are in image pixels.
    File 05 will tune the threshold on validation set.
    """
    T = hip_center.shape[0]
    velocity = np.zeros((T,), dtype=np.float32)

    y = hip_center[:, 1]

    for t in range(T):
        prev = max(0, t - window_frames)

        if t == prev:
            velocity[t] = 0.0
            continue

        if hip_valid[t] and hip_valid[prev]:
            velocity[t] = float(y[t] - y[prev])
        else:
            velocity[t] = 0.0

    return velocity


def compute_body_centerline_angle(
    xy: np.ndarray,
    conf: np.ndarray,
    conf_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute angle between human body centerline and the ground.

    Paper condition:
        if angle between body centerline and ground is less than 45 degrees,
        the body is considered close to horizontal, which supports fall detection.

    Implementation:
        upper point:
            s0 head/nose if valid, otherwise s1 shoulder center

        lower point:
            ankle center = midpoint(s10, s13)
            if ankles are invalid, fallback to hip center

        angle with ground:
            angle = atan2(abs(dy), abs(dx)) in degrees

    Meaning:
        standing body: angle close to 90 degrees
        lying body:    angle close to 0 degrees
    """
    T = xy.shape[0]

    angles = np.zeros((T,), dtype=np.float32)
    valid = np.zeros((T,), dtype=bool)

    head_valid = is_valid_node(xy, conf, S_HEAD, conf_threshold)
    shoulder_valid = is_valid_node(xy, conf, S_SHOULDER_CENTER, conf_threshold)

    ankle_center, ankle_valid = compute_ankle_center(xy, conf, conf_threshold)
    hip_center, hip_valid = compute_hip_center(xy, conf, conf_threshold)

    for t in range(T):
        if head_valid[t]:
            top = xy[t, S_HEAD]
            top_ok = True
        elif shoulder_valid[t]:
            top = xy[t, S_SHOULDER_CENTER]
            top_ok = True
        else:
            top = np.array([0.0, 0.0], dtype=np.float32)
            top_ok = False

        if ankle_valid[t]:
            bottom = ankle_center[t]
            bottom_ok = True
        elif hip_valid[t]:
            bottom = hip_center[t]
            bottom_ok = True
        else:
            bottom = np.array([0.0, 0.0], dtype=np.float32)
            bottom_ok = False

        if not top_ok or not bottom_ok:
            angles[t] = 0.0
            valid[t] = False
            continue

        dx = float(bottom[0] - top[0])
        dy = float(bottom[1] - top[1])

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            angles[t] = 0.0
            valid[t] = False
            continue

        angle = math.degrees(math.atan2(abs(dy), abs(dx)))

        angles[t] = float(angle)
        valid[t] = True

    return angles, valid


def compute_external_rectangle_features(
    xy: np.ndarray,
    conf: np.ndarray,
    conf_threshold: float,
    min_valid_nodes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute external rectangle around valid skeleton nodes.

    Paper condition:
        P = Width / Height
        Normal standing/walking: P < 1
        Falling/lying:           P > 1

    Returns:
        width, height, ratio, valid_mask
    """
    T = xy.shape[0]

    widths = np.zeros((T,), dtype=np.float32)
    heights = np.zeros((T,), dtype=np.float32)
    ratios = np.zeros((T,), dtype=np.float32)
    valid = np.zeros((T,), dtype=bool)

    for t in range(T):
        pts = []

        for j in range(14):
            x = float(xy[t, j, 0])
            y = float(xy[t, j, 1])
            c = float(conf[t, j])

            if c > conf_threshold and (x != 0.0 or y != 0.0):
                pts.append([x, y])

        if len(pts) < min_valid_nodes:
            widths[t] = 0.0
            heights[t] = 0.0
            ratios[t] = 0.0
            valid[t] = False
            continue

        pts_arr = np.asarray(pts, dtype=np.float32)

        min_x = float(np.min(pts_arr[:, 0]))
        max_x = float(np.max(pts_arr[:, 0]))
        min_y = float(np.min(pts_arr[:, 1]))
        max_y = float(np.max(pts_arr[:, 1]))

        width = max_x - min_x
        height = max_y - min_y

        if height <= 1e-6:
            ratio = 0.0
            valid[t] = False
        else:
            ratio = width / height
            valid[t] = True

        widths[t] = float(width)
        heights[t] = float(height)
        ratios[t] = float(ratio)

    return widths, heights, ratios, valid


def compute_features_for_dataframe(
    df: pd.DataFrame,
    csv_path: Path,
    velocity_window_frames: int,
    conf_threshold: float,
    min_valid_nodes: int,
) -> Tuple[pd.DataFrame, Dict]:
    validate_chen_node_df(df, csv_path)

    if "frame_idx" in df.columns:
        df = df.sort_values("frame_idx").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    xy, conf = extract_nodes(df)

    hip_center, hip_valid = compute_hip_center(
        xy=xy,
        conf=conf,
        conf_threshold=conf_threshold,
    )

    hip_velocity = compute_descent_velocity(
        hip_center=hip_center,
        hip_valid=hip_valid,
        window_frames=velocity_window_frames,
    )

    angle_deg, angle_valid = compute_body_centerline_angle(
        xy=xy,
        conf=conf,
        conf_threshold=conf_threshold,
    )

    bbox_width, bbox_height, width_height_ratio, bbox_valid = compute_external_rectangle_features(
        xy=xy,
        conf=conf,
        conf_threshold=conf_threshold,
        min_valid_nodes=min_valid_nodes,
    )

    T = len(df)

    video_id = str(df["video_id"].iloc[0])
    label = int(pd.to_numeric(df["label"].iloc[0], errors="coerce"))
    label_name = str(df["label_name"].iloc[0]) if "label_name" in df.columns else label_to_name(label)

    frame_idx = pd.to_numeric(df["frame_idx"], errors="coerce").fillna(0).astype(int).to_numpy()

    out = pd.DataFrame({
        "video_id": [video_id] * T,
        "frame_idx": frame_idx,
        "source_file": [str(csv_path)] * T,
        "label": [label] * T,
        "label_name": [label_name] * T,

        "hip_center_x": hip_center[:, 0],
        "hip_center_y": hip_center[:, 1],
        "hip_center_valid": hip_valid.astype(int),

        "hip_descent_velocity": hip_velocity,
        "velocity_window_frames": [velocity_window_frames] * T,

        "body_centerline_angle_degrees": angle_deg,
        "body_centerline_valid": angle_valid.astype(int),

        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "width_height_ratio": width_height_ratio,
        "bbox_valid": bbox_valid.astype(int),
    })

    # Paper-style diagnostic conditions with default thresholds.
    # These are not final tuned predictions yet.
    out["default_condition_velocity_positive"] = (out["hip_descent_velocity"] > 0).astype(int)
    out["default_condition_angle_lt_45"] = (out["body_centerline_angle_degrees"] < 45.0).astype(int)
    out["default_condition_ratio_gt_1"] = (out["width_height_ratio"] > 1.0).astype(int)

    valid_feature_mask = (
        (out["hip_center_valid"] == 1)
        & (out["body_centerline_valid"] == 1)
        & (out["bbox_valid"] == 1)
    )

    out["all_rule_features_valid"] = valid_feature_mask.astype(int)

    summary = {
        "input_csv": str(csv_path),
        "video_id": video_id,
        "label": int(label),
        "label_name": label_name,
        "num_frames": int(T),

        "hip_valid_frames": int(out["hip_center_valid"].sum()),
        "angle_valid_frames": int(out["body_centerline_valid"].sum()),
        "bbox_valid_frames": int(out["bbox_valid"].sum()),
        "all_rule_features_valid_frames": int(out["all_rule_features_valid"].sum()),

        "hip_descent_velocity_max": float(out["hip_descent_velocity"].max()) if T > 0 else 0.0,
        "hip_descent_velocity_mean": float(out["hip_descent_velocity"].mean()) if T > 0 else 0.0,
        "hip_descent_velocity_std": float(out["hip_descent_velocity"].std()) if T > 1 else 0.0,

        "body_angle_min": float(out["body_centerline_angle_degrees"].min()) if T > 0 else 0.0,
        "body_angle_mean": float(out["body_centerline_angle_degrees"].mean()) if T > 0 else 0.0,

        "width_height_ratio_max": float(out["width_height_ratio"].max()) if T > 0 else 0.0,
        "width_height_ratio_mean": float(out["width_height_ratio"].mean()) if T > 0 else 0.0,

        "default_angle_lt_45_frames": int(out["default_condition_angle_lt_45"].sum()),
        "default_ratio_gt_1_frames": int(out["default_condition_ratio_gt_1"].sum()),
    }

    return out, summary


def find_input_csv_files(input_dir: Path) -> List[Path]:
    csv_files = sorted([
        p for p in input_dir.rglob("*.csv")
        if p.name != "all_chen_nodes_index.csv"
    ])

    return csv_files


def process_one_csv(
    csv_path: Path,
    output_dir: Path,
    velocity_window_frames: int,
    conf_threshold: float,
    min_valid_nodes: int,
) -> Dict:
    df = read_csv_safe(csv_path)

    out_df, summary = compute_features_for_dataframe(
        df=df,
        csv_path=csv_path,
        velocity_window_frames=velocity_window_frames,
        conf_threshold=conf_threshold,
        min_valid_nodes=min_valid_nodes,
    )

    if len(out_df) == 0:
        raise ValueError(f"No rows after feature computation: {csv_path}")

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
    output_csv = video_output_dir / f"{safe_video_id}_rule_features.csv"

    out_df.to_csv(output_csv, index=False)

    result = {
        **summary,
        "output_csv": str(output_csv),
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute Chen et al. 2020 three rule features from Chen-style skeleton nodes."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(BASELINE_ROOT / "data" / "chen_nodes"),
        help="Input folder from File 02.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BASELINE_ROOT / "data" / "rule_features"),
        help="Output folder for per-video rule feature CSV files.",
    )

    parser.add_argument(
        "--velocity-window-frames",
        type=int,
        default=5,
        help="Number of frames used to compute hip descent velocity. Paper mentions checking every 5 adjacent frames.",
    )

    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.0,
        help="Node confidence <= this value is treated as invalid.",
    )

    parser.add_argument(
        "--min-valid-nodes",
        type=int,
        default=5,
        help="Minimum valid skeleton nodes needed to compute external rectangle.",
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
    print("FILE 03 - Compute Chen 2020 rule features")
    print("=" * 100)
    print(f"Input dir:               {input_dir}")
    print(f"Output dir:              {output_dir}")
    print(f"Input CSV files:         {len(csv_files)}")
    print(f"Velocity window frames:  {args.velocity_window_frames}")
    print(f"Confidence threshold:    {args.conf_threshold}")
    print(f"Minimum valid nodes:     {args.min_valid_nodes}")
    print("=" * 100)

    if len(csv_files) == 0:
        report = {
            "status": "no_input_csv",
            "input_dir": str(input_dir),
            "message": "No Chen node CSV files found. Run file 02 first.",
        }

        report_path = reports_dir / "03_compute_rule_features_report.json"
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
                velocity_window_frames=args.velocity_window_frames,
                conf_threshold=args.conf_threshold,
                min_valid_nodes=args.min_valid_nodes,
            )

            results.append(result)

            print(
                f"  OK -> frames={result['num_frames']}, "
                f"max_vel={result['hip_descent_velocity_max']:.2f}, "
                f"min_angle={result['body_angle_min']:.2f}, "
                f"max_ratio={result['width_height_ratio_max']:.2f}"
            )

        except Exception as e:
            failed.append({
                "input_csv": str(csv_path),
                "error": repr(e),
            })

            print(f"  FAILED: {repr(e)}")

    index_df = pd.DataFrame(results)
    index_path = output_dir / "all_rule_features_index.csv"

    if len(index_df) > 0:
        index_df.to_csv(index_path, index=False)

    total_frames = int(index_df["num_frames"].sum()) if len(index_df) > 0 else 0
    total_valid_feature_frames = int(index_df["all_rule_features_valid_frames"].sum()) if len(index_df) > 0 else 0

    if len(index_df) > 0:
        avg_max_velocity = float(index_df["hip_descent_velocity_max"].mean())
        avg_min_angle = float(index_df["body_angle_min"].mean())
        avg_max_ratio = float(index_df["width_height_ratio_max"].mean())
    else:
        avg_max_velocity = 0.0
        avg_min_angle = 0.0
        avg_max_ratio = 0.0

    report = {
        "status": "completed",
        "pipeline_note": "This step computes the three Chen et al. 2020 geometric rule features: hip-center descent velocity, body centerline angle with the ground, and external rectangle width-to-height ratio.",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_input_csv": len(csv_files),
        "num_success": len(results),
        "num_failed": len(failed),
        "velocity_window_frames": int(args.velocity_window_frames),
        "conf_threshold": float(args.conf_threshold),
        "min_valid_nodes": int(args.min_valid_nodes),
        "total_frames": total_frames,
        "total_valid_feature_frames": total_valid_feature_frames,
        "average_max_hip_descent_velocity": avg_max_velocity,
        "average_min_body_angle": avg_min_angle,
        "average_max_width_height_ratio": avg_max_ratio,
        "index_csv": str(index_path),
        "failed": failed,
        "results": results,
    }

    report_path = reports_dir / "03_compute_rule_features_report.json"
    save_json(report, report_path)

    print("=" * 100)
    print("FILE 03 completed.")
    print("=" * 100)
    print(f"Success videos:                   {len(results)}")
    print(f"Failed videos:                    {len(failed)}")
    print(f"Total frames:                     {total_frames}")
    print(f"Total valid feature frames:       {total_valid_feature_frames}")
    print(f"Average max hip descent velocity: {avg_max_velocity:.4f}")
    print(f"Average min body angle:           {avg_min_angle:.4f}")
    print(f"Average max width/height ratio:   {avg_max_ratio:.4f}")
    print(f"Index CSV:                        {index_path}")
    print(f"Report:                           {report_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
