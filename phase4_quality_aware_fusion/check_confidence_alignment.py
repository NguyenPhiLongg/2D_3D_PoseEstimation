import os
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


"""
Phase 4 - Check alignment between old 2D keypoints and new 2D keypoints with confidence.

Purpose:
    The old Phase 1 / Phase 3 pipeline used:
        data/2_extracted_2d/
            frame, x0, y0, ..., x16, y16

    Phase 4 now extracts:
        data/5_extracted_2d_confidence/
            frame,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_conf,
            person_index, num_persons,
            x0, y0, c0, ..., x16, y16, c16

    This script checks whether the new confidence CSV files are aligned with the old 2D CSV files.

Why this matters:
    If the new 2D confidence files are aligned with the old 2D files, Phase 4 can reuse:
        data/4_normalized_3d/

    If they are not aligned, the old 3D files may no longer correspond to the new 2D confidence files.
"""


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_OLD_2D_DIR = os.path.join(PROJECT_ROOT, "data", "2_extracted_2d")
DEFAULT_NEW_2D_CONF_DIR = os.path.join(PROJECT_ROOT, "data", "5_extracted_2d_confidence")

DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "phase4_quality_aware_fusion",
    "outputs",
    "confidence_extraction",
)

DEFAULT_COMMON_METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "phase3_common_set_gated_fusion",
    "outputs",
    "common_split",
    "common_metadata.csv",
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def xy_columns() -> List[str]:
    cols = []

    for i in range(17):
        cols.extend([f"x{i}", f"y{i}"])

    return cols


def conf_columns() -> List[str]:
    return [f"c{i}" for i in range(17)]


def new_required_columns() -> List[str]:
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


def old_required_columns() -> List[str]:
    return ["frame"] + xy_columns()


def list_csv_files(folder: str) -> Dict[str, str]:
    """
    Return:
        {file_name: absolute_path}
    """
    files = {}

    if not os.path.exists(folder):
        return files

    for root, _, names in os.walk(folder):
        for name in names:
            if name.lower().endswith(".csv"):
                path = os.path.join(root, name)

                if name in files:
                    print(f"WARNING: Duplicate CSV file name detected: {name}")
                    print(f"Existing: {files[name]}")
                    print(f"New     : {path}")

                files[name] = path

    return files


def read_csv_safely(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def missing_columns(df: pd.DataFrame, required_cols: List[str]) -> List[str]:
    return [col for col in required_cols if col not in df.columns]


def load_common_file_names(common_metadata_path: str) -> Optional[set]:
    """
    Load Phase 3 common-set file names.

    This is optional. If --common-only is used, the script only checks videos
    that appear in common_metadata.csv.
    """
    if not os.path.exists(common_metadata_path):
        return None

    df = pd.read_csv(common_metadata_path)

    file_names = set()

    if "path_2d" in df.columns:
        for path in df["path_2d"].dropna().tolist():
            file_names.add(os.path.basename(str(path)))

    elif "video_key" in df.columns:
        for key in df["video_key"].dropna().tolist():
            key = str(key)

            if key.lower().endswith(".csv"):
                file_names.add(os.path.basename(key))
            else:
                file_names.add(os.path.basename(key) + ".csv")

    else:
        return None

    return file_names


def compute_confidence_stats(new_df: pd.DataFrame, confidence_threshold: float) -> Dict:
    c_cols = conf_columns()

    available = [col for col in c_cols if col in new_df.columns]

    if not available:
        return {
            "mean_confidence": np.nan,
            "min_confidence": np.nan,
            "max_confidence": np.nan,
            "missing_joint_ratio": np.nan,
            "low_conf_01_ratio": np.nan,
            "low_conf_02_ratio": np.nan,
            "low_conf_03_ratio": np.nan,
            "low_conf_05_ratio": np.nan,
        }

    conf_values = new_df[available].to_numpy(dtype=np.float32)

    return {
        "mean_confidence": float(np.nanmean(conf_values)),
        "min_confidence": float(np.nanmin(conf_values)),
        "max_confidence": float(np.nanmax(conf_values)),
        "missing_joint_ratio": float(np.mean(conf_values < confidence_threshold)),
        "low_conf_01_ratio": float(np.mean(conf_values < 0.1)),
        "low_conf_02_ratio": float(np.mean(conf_values < 0.2)),
        "low_conf_03_ratio": float(np.mean(conf_values < 0.3)),
        "low_conf_05_ratio": float(np.mean(conf_values < 0.5)),
    }


def compute_bbox_stats(new_df: pd.DataFrame) -> Dict:
    required = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_conf"]

    if any(col not in new_df.columns for col in required):
        return {
            "mean_bbox_conf": np.nan,
            "min_bbox_conf": np.nan,
            "bbox_aspect_std": np.nan,
            "bbox_area_mean": np.nan,
        }

    eps = 1e-6

    x1 = new_df["bbox_x1"].to_numpy(dtype=np.float32)
    y1 = new_df["bbox_y1"].to_numpy(dtype=np.float32)
    x2 = new_df["bbox_x2"].to_numpy(dtype=np.float32)
    y2 = new_df["bbox_y2"].to_numpy(dtype=np.float32)

    width = x2 - x1
    height = y2 - y1
    area = width * height
    aspect = width / (height + eps)

    return {
        "mean_bbox_conf": float(np.nanmean(new_df["bbox_conf"].to_numpy(dtype=np.float32))),
        "min_bbox_conf": float(np.nanmin(new_df["bbox_conf"].to_numpy(dtype=np.float32))),
        "bbox_aspect_std": float(np.nanstd(aspect)),
        "bbox_area_mean": float(np.nanmean(area)),
    }


def compare_single_file(
    old_path: str,
    new_path: str,
    file_name: str,
    confidence_threshold: float,
    min_frame_coverage: float,
    max_mean_xy_error: float,
    max_p95_xy_error: float,
) -> Dict:
    record = {
        "file_name": file_name,
        "old_path": old_path,
        "new_path": new_path,
        "status": "UNKNOWN",
        "use_existing_3d_recommended": False,
        "old_rows": np.nan,
        "new_rows": np.nan,
        "old_unique_frames": np.nan,
        "new_unique_frames": np.nan,
        "common_frames": np.nan,
        "old_frames_missing_in_new": np.nan,
        "new_frames_extra_vs_old": np.nan,
        "old_to_new_frame_coverage": np.nan,
        "new_to_old_frame_coverage": np.nan,
        "mean_abs_xy_error": np.nan,
        "median_abs_xy_error": np.nan,
        "p95_abs_xy_error": np.nan,
        "max_abs_xy_error": np.nan,
        "old_duplicate_frames": np.nan,
        "new_duplicate_frames": np.nan,
        "mean_confidence": np.nan,
        "min_confidence": np.nan,
        "max_confidence": np.nan,
        "missing_joint_ratio": np.nan,
        "low_conf_01_ratio": np.nan,
        "low_conf_02_ratio": np.nan,
        "low_conf_03_ratio": np.nan,
        "low_conf_05_ratio": np.nan,
        "mean_bbox_conf": np.nan,
        "min_bbox_conf": np.nan,
        "bbox_aspect_std": np.nan,
        "bbox_area_mean": np.nan,
        "error": "",
    }

    try:
        old_df = read_csv_safely(old_path)
        new_df = read_csv_safely(new_path)

        record["old_rows"] = int(len(old_df))
        record["new_rows"] = int(len(new_df))

        old_missing = missing_columns(old_df, old_required_columns())
        new_missing = missing_columns(new_df, new_required_columns())

        if old_missing:
            record["status"] = "FAIL_OLD_MISSING_COLUMNS"
            record["error"] = "Old CSV missing columns: " + ", ".join(old_missing[:20])
            return record

        if new_missing:
            record["status"] = "FAIL_NEW_MISSING_COLUMNS"
            record["error"] = "New CSV missing columns: " + ", ".join(new_missing[:20])
            return record

        if len(old_df) == 0:
            record["status"] = "FAIL_OLD_EMPTY"
            record["error"] = "Old CSV is empty."
            return record

        if len(new_df) == 0:
            record["status"] = "FAIL_NEW_EMPTY"
            record["error"] = "New CSV is empty."
            return record

        conf_stats = compute_confidence_stats(
            new_df=new_df,
            confidence_threshold=confidence_threshold,
        )
        record.update(conf_stats)

        bbox_stats = compute_bbox_stats(new_df)
        record.update(bbox_stats)

        old_frames = set(old_df["frame"].astype(int).tolist())
        new_frames = set(new_df["frame"].astype(int).tolist())

        common_frames = old_frames.intersection(new_frames)
        old_missing_in_new = old_frames.difference(new_frames)
        new_extra_vs_old = new_frames.difference(old_frames)

        record["old_unique_frames"] = int(len(old_frames))
        record["new_unique_frames"] = int(len(new_frames))
        record["common_frames"] = int(len(common_frames))
        record["old_frames_missing_in_new"] = int(len(old_missing_in_new))
        record["new_frames_extra_vs_old"] = int(len(new_extra_vs_old))

        record["old_duplicate_frames"] = int(old_df["frame"].duplicated().sum())
        record["new_duplicate_frames"] = int(new_df["frame"].duplicated().sum())

        if len(old_frames) > 0:
            record["old_to_new_frame_coverage"] = float(len(common_frames) / len(old_frames))

        if len(new_frames) > 0:
            record["new_to_old_frame_coverage"] = float(len(common_frames) / len(new_frames))

        if len(common_frames) == 0:
            record["status"] = "FAIL_NO_COMMON_FRAMES"
            record["error"] = "No common frames between old and new CSV."
            return record

        # Use only one row per frame for safe merge.
        old_sub = (
            old_df[["frame"] + xy_columns()]
            .drop_duplicates(subset=["frame"], keep="first")
            .copy()
        )
        new_sub = (
            new_df[["frame"] + xy_columns()]
            .drop_duplicates(subset=["frame"], keep="first")
            .copy()
        )

        merged = pd.merge(
            old_sub,
            new_sub,
            on="frame",
            suffixes=("_old", "_new"),
            how="inner",
        )

        if merged.empty:
            record["status"] = "FAIL_EMPTY_MERGE"
            record["error"] = "Merge produced no rows."
            return record

        diffs = []

        for col in xy_columns():
            old_col = f"{col}_old"
            new_col = f"{col}_new"

            diff = np.abs(
                merged[old_col].to_numpy(dtype=np.float32)
                - merged[new_col].to_numpy(dtype=np.float32)
            )
            diffs.append(diff)

        diffs = np.concatenate(diffs, axis=0)

        record["mean_abs_xy_error"] = float(np.nanmean(diffs))
        record["median_abs_xy_error"] = float(np.nanmedian(diffs))
        record["p95_abs_xy_error"] = float(np.nanpercentile(diffs, 95))
        record["max_abs_xy_error"] = float(np.nanmax(diffs))

        old_coverage_ok = record["old_to_new_frame_coverage"] >= min_frame_coverage
        mean_xy_ok = record["mean_abs_xy_error"] <= max_mean_xy_error
        p95_xy_ok = record["p95_abs_xy_error"] <= max_p95_xy_error

        if not old_coverage_ok:
            record["status"] = "FAIL_FRAME_COVERAGE"
            record["error"] = (
                f"Old-to-new frame coverage "
                f"{record['old_to_new_frame_coverage']:.4f} "
                f"is lower than threshold {min_frame_coverage:.4f}."
            )
            return record

        if not mean_xy_ok or not p95_xy_ok:
            record["status"] = "FAIL_XY_MISMATCH"
            record["error"] = (
                f"XY mismatch. mean_abs_xy_error={record['mean_abs_xy_error']:.4f}, "
                f"p95_abs_xy_error={record['p95_abs_xy_error']:.4f}. "
                f"Thresholds: mean<={max_mean_xy_error}, p95<={max_p95_xy_error}."
            )
            return record

        if record["new_to_old_frame_coverage"] < min_frame_coverage:
            # New extractor found many extra frames. This is not fatal because
            # Phase 4 can still use the frames that are common with old 2D/3D.
            record["status"] = "WARNING_EXTRA_NEW_FRAMES"
            record["use_existing_3d_recommended"] = True
            record["error"] = (
                "New CSV has many extra frames compared with old CSV, "
                "but old frames are sufficiently covered."
            )
            return record

        record["status"] = "OK"
        record["use_existing_3d_recommended"] = True
        return record

    except Exception as exc:
        record["status"] = "FAIL_EXCEPTION"
        record["error"] = str(exc)
        return record


def save_text_list(path: str, values: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for value in values:
            f.write(str(value) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check alignment between old 2D CSV and new 2D confidence CSV files."
    )

    parser.add_argument(
        "--old-2d-dir",
        type=str,
        default=DEFAULT_OLD_2D_DIR,
        help="Directory containing old 2D CSV files.",
    )

    parser.add_argument(
        "--new-2d-conf-dir",
        type=str,
        default=DEFAULT_NEW_2D_CONF_DIR,
        help="Directory containing new 2D CSV files with confidence.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for alignment reports.",
    )

    parser.add_argument(
        "--common-metadata",
        type=str,
        default=DEFAULT_COMMON_METADATA_PATH,
        help="Phase 3 common_metadata.csv path.",
    )

    parser.add_argument(
        "--common-only",
        action="store_true",
        help="Only check videos that appear in Phase 3 common_metadata.csv.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Threshold used to count low-confidence or missing keypoints.",
    )

    parser.add_argument(
        "--min-frame-coverage",
        type=float,
        default=0.95,
        help=(
            "Minimum ratio of old frames that must appear in the new confidence CSV. "
            "Default 0.95 means at least 95 percent of old frames should be covered."
        ),
    )

    parser.add_argument(
        "--max-mean-xy-error",
        type=float,
        default=5.0,
        help="Maximum allowed mean absolute XY error in pixels.",
    )

    parser.add_argument(
        "--max-p95-xy-error",
        type=float,
        default=25.0,
        help="Maximum allowed 95th percentile absolute XY error in pixels.",
    )

    parser.add_argument(
        "--min-recommended-ratio",
        type=float,
        default=0.95,
        help=(
            "If at least this ratio of checked files can reuse existing 3D, "
            "the overall recommendation is to reuse existing 3D."
        ),
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_dir(args.output_dir)

    print("\nPhase 4 - Confidence Alignment Check")
    print("=" * 80)
    print(f"Old 2D directory        : {args.old_2d_dir}")
    print(f"New 2D confidence dir   : {args.new_2d_conf_dir}")
    print(f"Output directory        : {args.output_dir}")
    print(f"Common metadata         : {args.common_metadata}")
    print(f"Common only             : {args.common_only}")
    print(f"Min frame coverage      : {args.min_frame_coverage}")
    print(f"Max mean XY error       : {args.max_mean_xy_error}")
    print(f"Max p95 XY error        : {args.max_p95_xy_error}")
    print("=" * 80)

    old_files = list_csv_files(args.old_2d_dir)
    new_files = list_csv_files(args.new_2d_conf_dir)

    if not old_files:
        raise FileNotFoundError(f"No old 2D CSV files found in: {args.old_2d_dir}")

    if not new_files:
        raise FileNotFoundError(
            f"No new 2D confidence CSV files found in: {args.new_2d_conf_dir}"
        )

    old_names = set(old_files.keys())
    new_names = set(new_files.keys())

    if args.common_only:
        common_names = load_common_file_names(args.common_metadata)

        if common_names is None:
            raise FileNotFoundError(
                "Could not load common-set file names from common_metadata.csv. "
                "Check --common-metadata path or run without --common-only."
            )

        old_names = old_names.intersection(common_names)
        new_names = new_names.intersection(common_names)

    matched_names = sorted(old_names.intersection(new_names))
    missing_new_names = sorted(old_names.difference(new_names))
    extra_new_names = sorted(new_names.difference(old_names))

    if args.max_files is not None:
        matched_names = matched_names[: args.max_files]

    print(f"Old CSV files        : {len(old_files)}")
    print(f"New confidence files : {len(new_files)}")
    print(f"Matched files        : {len(matched_names)}")
    print(f"Missing new files    : {len(missing_new_names)}")
    print(f"Extra new files      : {len(extra_new_names)}")

    records = []

    for idx, file_name in enumerate(matched_names, start=1):
        if idx % 100 == 0 or idx == 1 or idx == len(matched_names):
            print(f"Checking [{idx}/{len(matched_names)}]: {file_name}")

        record = compare_single_file(
            old_path=old_files[file_name],
            new_path=new_files[file_name],
            file_name=file_name,
            confidence_threshold=args.confidence_threshold,
            min_frame_coverage=args.min_frame_coverage,
            max_mean_xy_error=args.max_mean_xy_error,
            max_p95_xy_error=args.max_p95_xy_error,
        )

        records.append(record)

    # Add records for missing new files.
    for file_name in missing_new_names:
        records.append(
            {
                "file_name": file_name,
                "old_path": old_files.get(file_name, ""),
                "new_path": "",
                "status": "FAIL_MISSING_NEW_FILE",
                "use_existing_3d_recommended": False,
                "error": "Old 2D CSV exists but new 2D confidence CSV is missing.",
            }
        )

    df = pd.DataFrame(records)

    alignment_csv_path = os.path.join(args.output_dir, "alignment_check.csv")
    summary_json_path = os.path.join(args.output_dir, "alignment_summary.json")
    problem_videos_path = os.path.join(args.output_dir, "problem_alignment_videos.txt")
    missing_new_path = os.path.join(args.output_dir, "missing_new_confidence_files.txt")
    extra_new_path = os.path.join(args.output_dir, "extra_new_confidence_files.txt")

    df.to_csv(alignment_csv_path, index=False)

    problem_df = df[df["status"] != "OK"]

    save_text_list(
        problem_videos_path,
        problem_df["file_name"].astype(str).tolist() if not problem_df.empty else [],
    )
    save_text_list(missing_new_path, missing_new_names)
    save_text_list(extra_new_path, extra_new_names)

    status_counts = df["status"].value_counts().to_dict() if not df.empty else {}

    checked_count = int(len(df))
    recommended_count = int(df["use_existing_3d_recommended"].fillna(False).sum()) if not df.empty else 0

    recommended_ratio = (
        float(recommended_count / checked_count) if checked_count > 0 else 0.0
    )

    overall_reuse_existing_3d = recommended_ratio >= args.min_recommended_ratio

    mean_xy_error = (
        float(df["mean_abs_xy_error"].dropna().mean())
        if "mean_abs_xy_error" in df.columns and not df["mean_abs_xy_error"].dropna().empty
        else None
    )

    p95_xy_error = (
        float(df["p95_abs_xy_error"].dropna().mean())
        if "p95_abs_xy_error" in df.columns and not df["p95_abs_xy_error"].dropna().empty
        else None
    )

    mean_confidence = (
        float(df["mean_confidence"].dropna().mean())
        if "mean_confidence" in df.columns and not df["mean_confidence"].dropna().empty
        else None
    )

    missing_joint_ratio = (
        float(df["missing_joint_ratio"].dropna().mean())
        if "missing_joint_ratio" in df.columns and not df["missing_joint_ratio"].dropna().empty
        else None
    )

    summary = {
        "old_2d_dir": args.old_2d_dir,
        "new_2d_conf_dir": args.new_2d_conf_dir,
        "output_dir": args.output_dir,
        "common_only": bool(args.common_only),
        "num_old_files": int(len(old_files)),
        "num_new_files": int(len(new_files)),
        "num_matched_files_checked": int(len(matched_names)),
        "num_missing_new_files": int(len(missing_new_names)),
        "num_extra_new_files": int(len(extra_new_names)),
        "status_counts": status_counts,
        "checked_count": checked_count,
        "recommended_count": recommended_count,
        "recommended_ratio": recommended_ratio,
        "min_recommended_ratio": args.min_recommended_ratio,
        "overall_reuse_existing_3d_recommended": bool(overall_reuse_existing_3d),
        "mean_abs_xy_error_average": mean_xy_error,
        "p95_abs_xy_error_average": p95_xy_error,
        "mean_confidence_average": mean_confidence,
        "missing_joint_ratio_average": missing_joint_ratio,
        "alignment_csv_path": alignment_csv_path,
        "problem_videos_path": problem_videos_path,
        "missing_new_path": missing_new_path,
        "extra_new_path": extra_new_path,
        "interpretation": (
            "If overall_reuse_existing_3d_recommended is true, Phase 4 can reuse "
            "data/4_normalized_3d. If false, consider checking person selection or "
            "re-extracting 3D from the new 2D confidence files."
        ),
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print("\nAlignment check finished.")
    print("=" * 80)
    print(f"Status counts: {status_counts}")
    print(f"Recommended files: {recommended_count}/{checked_count}")
    print(f"Recommended ratio: {recommended_ratio:.4f}")

    if overall_reuse_existing_3d:
        print("OVERALL RESULT: OK - Existing 3D files can be reused for Phase 4.")
    else:
        print("OVERALL RESULT: WARNING - Existing 3D reuse is not recommended yet.")
        print("Check problem_alignment_videos.txt and alignment_check.csv.")

    print("\nSaved reports:")
    print(f"- {alignment_csv_path}")
    print(f"- {summary_json_path}")
    print(f"- {problem_videos_path}")
    print(f"- {missing_new_path}")
    print(f"- {extra_new_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()