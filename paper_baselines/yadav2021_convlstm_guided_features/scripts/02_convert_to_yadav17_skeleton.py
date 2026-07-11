from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
BASELINE_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = BASELINE_ROOT.parents[1]

sys.path.insert(0, str(BASELINE_ROOT))

from utils.io_utils import ensure_dir, read_csv_safe, save_json, make_safe_name


# ======================================================================================
# FILE 02: CONVERT RAW 3D SKELETON TO ADAPTED YADAV17 SKELETON
# ======================================================================================
#
# Original Yadav et al. pipeline:
#   Kinect-v2 3D skeleton coordinates
#   -> selected informative joints
#   -> 3D joint normalization
#   -> bounding box / guided features
#
# Adapted project pipeline:
#   PoseFormerV2 estimated 3D pose has 17 H36M-style joints:
#
#   0: pelvis
#   1: right hip
#   2: right knee
#   3: right ankle
#   4: left hip
#   5: left knee
#   6: left ankle
#   7: spine
#   8: thorax
#   9: neck
#   10: head
#   11: left shoulder
#   12: left elbow
#   13: left wrist
#   14: right shoulder
#   15: right elbow
#   16: right wrist
#
# This script maps the H36M-style 17 joints into an adapted Yadav-style 17-joint order.
# Then it applies min-max normalization to follow the normalization idea in the paper.
#
# Output:
#   data/yadav17_skeleton/Fall/*.csv
#   data/yadav17_skeleton/Not_Fall/*.csv
#   data/yadav17_skeleton/yadav17_skeleton_index.csv
#   outputs/reports/02_convert_to_yadav17_skeleton_report.json
# ======================================================================================


RAW_XYZ_COLS = [f"{axis}{idx}" for idx in range(17) for axis in ["x", "y", "z"]]


# H36M / PoseFormerV2 joint names in current project
H36M_JOINTS = {
    0: "pelvis",
    1: "right_hip",
    2: "right_knee",
    3: "right_ankle",
    4: "left_hip",
    5: "left_knee",
    6: "left_ankle",
    7: "spine",
    8: "thorax",
    9: "neck",
    10: "head",
    11: "left_shoulder",
    12: "left_elbow",
    13: "left_wrist",
    14: "right_shoulder",
    15: "right_elbow",
    16: "right_wrist",
}


# Adapted Yadav-style order based on Table 2 in the paper.
# Some Kinect joints do not exist in H36M/PoseFormer, so foot joints are approximated by ankle joints.
YADAV17_MAPPING = [
    ("hip_right", 1),
    ("foot_left_proxy", 6),
    ("knee_right", 2),
    ("knee_left", 5),
    ("hip_left", 4),
    ("foot_right_proxy", 3),
    ("ankle_right", 3),
    ("ankle_left", 6),
    ("head", 10),
    ("spine_mid", 7),
    ("spine_base", 0),
    ("wrist_left", 13),
    ("shoulder_right", 14),
    ("shoulder_left", 11),
    ("elbow_left", 12),
    ("wrist_right", 16),
    ("elbow_right", 15),
]


YADAV_XYZ_COLS = [f"{axis}{idx}" for idx in range(17) for axis in ["x", "y", "z"]]


def label_to_name(label: int) -> str:
    return "Fall" if int(label) == 1 else "Not_Fall"


def check_required_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [c for c in RAW_XYZ_COLS if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required 3D columns in {path}: {missing[:10]}")


def df_to_pose_array(df: pd.DataFrame) -> np.ndarray:
    """
    Convert dataframe with x0,y0,z0...x16,y16,z16 into:
        pose shape = (T, 17, 3)
    """
    values = df[RAW_XYZ_COLS].to_numpy(dtype=np.float32)
    pose = values.reshape(len(df), 17, 3)
    return pose


def convert_h36m_to_yadav17(pose_h36m: np.ndarray) -> np.ndarray:
    """
    Input:
        pose_h36m shape = (T, 17, 3)

    Output:
        pose_yadav shape = (T, 17, 3)
    """
    if pose_h36m.ndim != 3 or pose_h36m.shape[1:] != (17, 3):
        raise ValueError(f"Expected pose shape (T,17,3), got {pose_h36m.shape}")

    mapped_indices = [source_idx for _, source_idx in YADAV17_MAPPING]
    pose_yadav = pose_h36m[:, mapped_indices, :]

    return pose_yadav.astype(np.float32)


def minmax_normalize_per_video_per_axis(pose: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Paper-style min-max normalization.

    This implementation normalizes each video independently and separately for x, y, z:
        x' = (x - xmin) / (xmax - xmin)

    Why per-video?
    - It follows the min-max idea.
    - It avoids leaking global test-set statistics before train/val/test split.
    - It makes skeleton scale/location more comparable across videos.
    """
    eps = 1e-6

    flat = pose.reshape(-1, 3)

    mins = np.nanmin(flat, axis=0)
    maxs = np.nanmax(flat, axis=0)

    denom = maxs - mins
    denom = np.where(np.abs(denom) < eps, 1.0, denom)

    normalized = (pose - mins.reshape(1, 1, 3)) / denom.reshape(1, 1, 3)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    stats = {
        "x_min": float(mins[0]),
        "y_min": float(mins[1]),
        "z_min": float(mins[2]),
        "x_max": float(maxs[0]),
        "y_max": float(maxs[1]),
        "z_max": float(maxs[2]),
    }

    return normalized.astype(np.float32), stats


def pose_array_to_df(
    pose_yadav: np.ndarray,
    original_df: pd.DataFrame,
    video_id: str,
    label: int,
    label_name: str,
    normalization_stats: Dict,
) -> pd.DataFrame:
    """
    Convert pose array back into dataframe.

    The output columns x0,y0,z0...x16,y16,z16 now represent the adapted Yadav17 order,
    not the original H36M order.
    """
    values = pose_yadav.reshape(len(pose_yadav), 51)
    out_df = pd.DataFrame(values, columns=YADAV_XYZ_COLS)

    # Keep useful metadata
    meta_cols = [c for c in original_df.columns if c not in RAW_XYZ_COLS]

    for col in meta_cols:
        out_df[col] = original_df[col].values

    out_df["video_id"] = video_id
    out_df["label"] = int(label)
    out_df["label_name"] = label_name

    out_df["normalization"] = "per_video_minmax_per_axis"

    out_df["x_min_raw"] = normalization_stats["x_min"]
    out_df["y_min_raw"] = normalization_stats["y_min"]
    out_df["z_min_raw"] = normalization_stats["z_min"]
    out_df["x_max_raw"] = normalization_stats["x_max"]
    out_df["y_max_raw"] = normalization_stats["y_max"]
    out_df["z_max_raw"] = normalization_stats["z_max"]

    # Clean order
    fixed_meta = [
        "video_id",
        "frame",
        "source_file",
        "label",
        "label_name",
        "action_label",
        "action_name",
        "normalization",
        "x_min_raw",
        "y_min_raw",
        "z_min_raw",
        "x_max_raw",
        "y_max_raw",
        "z_max_raw",
    ]

    existing_fixed = [c for c in fixed_meta if c in out_df.columns]
    remaining = [c for c in out_df.columns if c not in YADAV_XYZ_COLS and c not in existing_fixed]

    out_df = out_df[YADAV_XYZ_COLS + existing_fixed + remaining]

    return out_df


def process_one_csv(
    source_csv: Path,
    output_csv: Path,
    video_id: str,
    label: int,
    label_name: str,
    overwrite: bool = False,
) -> Dict:
    if output_csv.exists() and not overwrite:
        return {
            "status": "skipped_existing",
            "video_id": video_id,
            "label": int(label),
            "label_name": label_name,
            "source_csv": str(source_csv),
            "output_csv": str(output_csv),
        }

    df = read_csv_safe(source_csv)

    check_required_columns(df, source_csv)

    pose_h36m = df_to_pose_array(df)
    pose_yadav = convert_h36m_to_yadav17(pose_h36m)
    pose_yadav_norm, stats = minmax_normalize_per_video_per_axis(pose_yadav)

    out_df = pose_array_to_df(
        pose_yadav=pose_yadav_norm,
        original_df=df,
        video_id=video_id,
        label=label,
        label_name=label_name,
        normalization_stats=stats,
    )

    ensure_dir(output_csv.parent)
    out_df.to_csv(output_csv, index=False)

    return {
        "status": "processed",
        "video_id": video_id,
        "label": int(label),
        "label_name": label_name,
        "source_csv": str(source_csv),
        "output_csv": str(output_csv),
        "num_frames": int(len(out_df)),
        "num_features": int(len(YADAV_XYZ_COLS)),
        "normalization": "per_video_minmax_per_axis",
        **stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw estimated 3D skeleton CSVs to adapted Yadav17 skeleton format."
    )

    parser.add_argument(
        "--index-csv",
        type=str,
        default=str(BASELINE_ROOT / "data" / "raw_3d_skeleton" / "raw_3d_skeleton_index.csv"),
        help="Index CSV generated by file 01.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BASELINE_ROOT / "data" / "yadav17_skeleton"),
        help="Output folder for adapted Yadav17 skeleton CSVs.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos for quick testing.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing converted CSVs.",
    )

    args = parser.parse_args()

    index_csv = Path(args.index_csv)
    output_dir = Path(args.output_dir)
    reports_dir = BASELINE_ROOT / "outputs" / "reports"

    ensure_dir(output_dir)
    ensure_dir(reports_dir)

    if not index_csv.exists():
        raise FileNotFoundError(f"File 01 index not found: {index_csv}")

    index_df = pd.read_csv(index_csv)

    if args.limit is not None:
        index_df = index_df.head(args.limit).copy()

    print("=" * 100)
    print("FILE 02 - Convert raw 3D skeleton to adapted Yadav17 skeleton + min-max normalization")
    print("=" * 100)
    print(f"Input index:     {index_csv}")
    print(f"Output dir:      {output_dir}")
    print(f"Videos in index: {len(index_df)}")
    print(f"Limit:           {args.limit}")
    print(f"Overwrite:       {args.overwrite}")
    print("=" * 100)

    rows = []
    failed_rows = []

    for i, row in index_df.iterrows():
        try:
            video_id = make_safe_name(str(row["video_id"]))
            label = int(row["label"])
            label_name = str(row["label_name"])

            source_csv = Path(str(row["collected_csv"]))

            if not source_csv.exists():
                # fallback to original source path
                source_csv = Path(str(row["source_csv"]))

            output_csv = output_dir / label_name / f"{video_id}.csv"

            result = process_one_csv(
                source_csv=source_csv,
                output_csv=output_csv,
                video_id=video_id,
                label=label,
                label_name=label_name,
                overwrite=args.overwrite,
            )

            rows.append(result)

        except Exception as e:
            failed_rows.append({
                "video_id": str(row.get("video_id", "")),
                "source_csv": str(row.get("collected_csv", row.get("source_csv", ""))),
                "error": repr(e),
            })

        done = i + 1

        if done % 500 == 0 or done == len(index_df):
            print(f"[{done}/{len(index_df)}] processed_or_skipped={len(rows)} | failed={len(failed_rows)}")

    out_index_df = pd.DataFrame(rows)
    failed_df = pd.DataFrame(failed_rows)

    out_index_csv = output_dir / "yadav17_skeleton_index.csv"
    failed_csv = output_dir / "yadav17_skeleton_failed_files.csv"

    if len(out_index_df) > 0:
        out_index_df.to_csv(out_index_csv, index=False)
    else:
        pd.DataFrame().to_csv(out_index_csv, index=False)

    if len(failed_df) > 0:
        failed_df.to_csv(failed_csv, index=False)
    else:
        pd.DataFrame(columns=["video_id", "source_csv", "error"]).to_csv(failed_csv, index=False)

    processed_df = out_index_df[out_index_df["status"].isin(["processed", "skipped_existing"])] if len(out_index_df) > 0 else pd.DataFrame()

    total_frames = int(processed_df["num_frames"].fillna(0).sum()) if "num_frames" in processed_df.columns else 0

    if len(processed_df) > 0:
        label_video_counts = processed_df.groupby("label_name")["video_id"].nunique().to_dict()
    else:
        label_video_counts = {}

    mapping_report = [
        {
            "yadav_index": idx,
            "yadav_joint": joint_name,
            "source_h36m_index": source_idx,
            "source_h36m_joint": H36M_JOINTS[source_idx],
        }
        for idx, (joint_name, source_idx) in enumerate(YADAV17_MAPPING)
    ]

    report = {
        "status": "completed",
        "pipeline_note": "Convert PoseFormer/H36M-style 17-joint 3D pose into adapted Yadav17 selected skeleton order and apply per-video min-max normalization.",
        "input_index_csv": str(index_csv),
        "output_dir": str(output_dir),
        "num_input_videos": int(len(index_df)),
        "num_success_or_skipped": int(len(rows)),
        "num_failed": int(len(failed_rows)),
        "total_frames_success": total_frames,
        "label_video_counts": {str(k): int(v) for k, v in label_video_counts.items()},
        "normalization": "per_video_minmax_per_axis",
        "output_index_csv": str(out_index_csv),
        "failed_csv": str(failed_csv),
        "yadav17_mapping": mapping_report,
    }

    report_path = reports_dir / "02_convert_to_yadav17_skeleton_report.json"
    save_json(report, report_path)

    print("=" * 100)
    print("FILE 02 completed.")
    print("=" * 100)
    print(f"Success/skipped videos: {len(rows)}")
    print(f"Failed videos:          {len(failed_rows)}")
    print(f"Total frames:           {total_frames}")
    print(f"Label videos:           {label_video_counts}")
    print(f"Output index CSV:       {out_index_csv}")
    print(f"Failed CSV:             {failed_csv}")
    print(f"Report:                 {report_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
