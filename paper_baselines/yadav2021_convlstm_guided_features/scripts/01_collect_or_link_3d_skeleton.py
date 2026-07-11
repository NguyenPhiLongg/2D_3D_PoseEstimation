from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
BASELINE_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = BASELINE_ROOT.parents[1]

sys.path.insert(0, str(BASELINE_ROOT))

from utils.io_utils import (
    ensure_dir,
    save_json,
    read_csv_safe,
    list_csv_files,
    make_safe_name,
    label_to_name,
    infer_label_from_df_or_path,
    get_video_id_from_df_or_path,
    resolve_project_path,
)


# ======================================================================================
# FILE 01: COLLECT OR LINK 3D SKELETON DATA
# ======================================================================================
#
# Original Yadav et al. pipeline:
#   Kinect-v2 video stream
#   -> acquire body frame
#   -> track 3D skeleton joint coordinates
#
# Adapted project pipeline:
#   We do not have Kinect-v2 skeletons.
#   Therefore, this script searches for estimated 3D pose CSV files already generated
#   by the project, then collects/copies them into this paper baseline folder.
#
# Expected 3D CSV format:
#   x0, y0, z0, x1, y1, z1, ..., x16, y16, z16
#
# Other columns such as:
#   label, action_label, video_id, frame_idx
# are preserved.
# ======================================================================================


def get_default_candidate_dirs() -> List[str]:
    return [
        "data/4_normalized_3d",
        "data/3_extracted_3d",
        "phase2_3d_upgrade",
        "phase4_quality_aware_fusion",
    ]


def detect_xyz_columns(df: pd.DataFrame) -> Tuple[List[str], str]:
    """
    Detect 3D keypoint columns.

    Preferred project format:
        x0, y0, z0, ..., x16, y16, z16

    Backup supported format:
        joint0_x, joint0_y, joint0_z, ...
    """
    cols = set(df.columns)

    direct_cols = []

    for i in range(17):
        direct_cols.extend([f"x{i}", f"y{i}", f"z{i}"])

    if all(c in cols for c in direct_cols):
        return direct_cols, "x0_y0_z0_to_x16_y16_z16"

    alt_cols = []

    for i in range(17):
        alt_cols.extend([f"joint{i}_x", f"joint{i}_y", f"joint{i}_z"])

    if all(c in cols for c in alt_cols):
        return alt_cols, "joint0_x_joint0_y_joint0_z_to_joint16"

    # More flexible fallback: detect all columns that look like x/y/z + number.
    fallback = []

    for i in range(17):
        candidates = [
            (f"x_{i}", f"y_{i}", f"z_{i}"),
            (f"kp{i}_x", f"kp{i}_y", f"kp{i}_z"),
            (f"keypoint_{i}_x", f"keypoint_{i}_y", f"keypoint_{i}_z"),
        ]

        found_triplet = None

        for triplet in candidates:
            if all(c in cols for c in triplet):
                found_triplet = triplet
                break

        if found_triplet is None:
            return [], "not_detected"

        fallback.extend(list(found_triplet))

    return fallback, "fallback_17x3"


def is_valid_3d_csv(path: Path, min_rows: int = 2) -> Tuple[bool, Dict]:
    try:
        df_head = read_csv_safe(path)

        if len(df_head) < min_rows:
            return False, {
                "reason": "too_few_rows",
                "num_rows": int(len(df_head)),
            }

        xyz_cols, fmt = detect_xyz_columns(df_head)

        if len(xyz_cols) != 51:
            return False, {
                "reason": "missing_3d_columns",
                "detected_format": fmt,
                "num_columns": int(len(df_head.columns)),
            }

        label = infer_label_from_df_or_path(df_head, path)
        label_name = label_to_name(label)

        # Use CSV stem as the video_id because data/3_extracted_3d filenames
        # are already generated from the original relative video path.
        # This avoids accidental duplicates such as different folders containing
        # videos with the same basename.
        video_id = make_safe_name(path.stem)

        return True, {
            "reason": "valid",
            "video_id": video_id,
            "label": int(label),
            "label_name": label_name,
            "num_rows": int(len(df_head)),
            "num_columns": int(len(df_head.columns)),
            "xyz_format": fmt,
            "xyz_columns": xyz_cols,
        }

    except Exception as e:
        return False, {
            "reason": "exception",
            "error": repr(e),
        }


def collect_source_files(candidate_dirs: List[Path]) -> List[Path]:
    files = []

    for folder in candidate_dirs:
        files.extend(list_csv_files(folder))

    # remove duplicates by resolved absolute path
    seen = set()
    unique = []

    for path in files:
        resolved = str(path.resolve()).lower()

        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)

    return sorted(unique)


def copy_or_index_file(
    source_csv: Path,
    output_root: Path,
    label_name: str,
    video_id: str,
    mode: str,
) -> str:
    if mode == "index_only":
        return str(source_csv)

    output_dir = output_root / label_name
    ensure_dir(output_dir)

    output_csv = output_dir / f"{make_safe_name(video_id)}.csv"

    if mode == "copy":
        shutil.copy2(source_csv, output_csv)
        return str(output_csv)

    raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect or link estimated 3D skeleton CSVs for Yadav2021-style ConvLSTM baseline."
    )

    parser.add_argument(
        "--candidate-dirs",
        type=str,
        nargs="*",
        default=get_default_candidate_dirs(),
        help="Candidate folders containing estimated 3D pose CSVs.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BASELINE_ROOT / "data" / "raw_3d_skeleton"),
        help="Output directory for collected raw 3D skeleton CSVs.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="copy",
        choices=["copy", "index_only"],
        help="copy = copy CSVs into this baseline folder; index_only = keep source paths only.",
    )

    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only scan and report valid 3D CSV files, do not copy.",
    )

    args = parser.parse_args()

    candidate_dirs = [
        resolve_project_path(PROJECT_ROOT, p)
        for p in args.candidate_dirs
    ]

    output_dir = Path(args.output_dir)
    reports_dir = BASELINE_ROOT / "outputs" / "reports"

    ensure_dir(output_dir)
    ensure_dir(reports_dir)

    print("=" * 100)
    print("FILE 01 - Collect or link estimated 3D skeleton CSV files")
    print("=" * 100)
    print("Candidate dirs:")

    for folder in candidate_dirs:
        print(f"  - {folder} | exists={folder.exists()}")

    print(f"Output dir: {output_dir}")
    print(f"Mode:       {args.mode}")
    print(f"Scan only:  {args.scan_only}")
    print("=" * 100)

    source_files = collect_source_files(candidate_dirs)

    print(f"CSV files discovered: {len(source_files)}")
    print("=" * 100)

    valid_rows = []
    invalid_rows = []

    seen_video_ids = set()

    for idx, source_csv in enumerate(source_files, start=1):
        valid, info = is_valid_3d_csv(source_csv)

        if valid:
            video_id = make_safe_name(str(info["video_id"]))

            # Avoid duplicate video IDs from multiple folders.
            if video_id in seen_video_ids:
                invalid_rows.append({
                    "source_csv": str(source_csv),
                    "valid": 0,
                    "reason": "duplicate_video_id",
                    "video_id": video_id,
                })
                continue

            seen_video_ids.add(video_id)

            label = int(info["label"])
            label_name = str(info["label_name"])

            if args.scan_only:
                collected_csv = str(source_csv)
            else:
                collected_csv = copy_or_index_file(
                    source_csv=source_csv,
                    output_root=output_dir,
                    label_name=label_name,
                    video_id=video_id,
                    mode=args.mode,
                )

            valid_rows.append({
                "video_id": video_id,
                "label": label,
                "label_name": label_name,
                "source_csv": str(source_csv),
                "collected_csv": collected_csv,
                "num_rows": int(info["num_rows"]),
                "num_columns": int(info["num_columns"]),
                "xyz_format": str(info["xyz_format"]),
            })

        else:
            invalid_rows.append({
                "source_csv": str(source_csv),
                "valid": 0,
                **info,
            })

        if idx % 500 == 0 or idx == len(source_files):
            print(f"[{idx}/{len(source_files)}] scanned | valid={len(valid_rows)} | invalid={len(invalid_rows)}")

    valid_df = pd.DataFrame(valid_rows)
    invalid_df = pd.DataFrame(invalid_rows)

    index_csv = output_dir / "raw_3d_skeleton_index.csv"
    invalid_csv = output_dir / "raw_3d_skeleton_invalid_files.csv"

    if len(valid_df) > 0:
        valid_df.to_csv(index_csv, index=False)
    else:
        pd.DataFrame(columns=[
            "video_id",
            "label",
            "label_name",
            "source_csv",
            "collected_csv",
            "num_rows",
            "num_columns",
            "xyz_format",
        ]).to_csv(index_csv, index=False)

    if len(invalid_df) > 0:
        invalid_df.to_csv(invalid_csv, index=False)
    else:
        pd.DataFrame(columns=["source_csv", "valid", "reason"]).to_csv(invalid_csv, index=False)

    total_frames = int(valid_df["num_rows"].sum()) if len(valid_df) > 0 else 0

    if len(valid_df) > 0:
        label_counts_video = valid_df.groupby("label_name")["video_id"].nunique().to_dict()
        label_counts_frames = valid_df.groupby("label_name")["num_rows"].sum().to_dict()
    else:
        label_counts_video = {}
        label_counts_frames = {}

    report = {
        "status": "completed",
        "pipeline_note": "Original Yadav et al. uses Kinect-v2 to acquire 3D skeleton coordinates. This adapted implementation collects estimated 3D pose CSVs already available in the project.",
        "candidate_dirs": [str(p) for p in candidate_dirs],
        "output_dir": str(output_dir),
        "mode": args.mode,
        "scan_only": bool(args.scan_only),
        "num_csv_discovered": int(len(source_files)),
        "num_valid_3d_csv": int(len(valid_rows)),
        "num_invalid_csv": int(len(invalid_rows)),
        "total_frames": total_frames,
        "label_counts_by_video": {str(k): int(v) for k, v in label_counts_video.items()},
        "label_counts_by_frames": {str(k): int(v) for k, v in label_counts_frames.items()},
        "index_csv": str(index_csv),
        "invalid_csv": str(invalid_csv),
    }

    report_path = reports_dir / "01_collect_or_link_3d_skeleton_report.json"
    save_json(report, report_path)

    print("=" * 100)
    print("FILE 01 completed.")
    print("=" * 100)
    print(f"Valid 3D videos:     {len(valid_rows)}")
    print(f"Invalid CSV files:   {len(invalid_rows)}")
    print(f"Total frames:        {total_frames}")
    print(f"Label videos:        {label_counts_video}")
    print(f"Label frames:        {label_counts_frames}")
    print(f"Index CSV:           {index_csv}")
    print(f"Invalid CSV:         {invalid_csv}")
    print(f"Report:              {report_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
