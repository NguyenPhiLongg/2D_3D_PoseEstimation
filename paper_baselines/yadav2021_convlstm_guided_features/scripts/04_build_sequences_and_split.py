from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


SCRIPT_PATH = Path(__file__).resolve()
BASELINE_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = BASELINE_ROOT.parents[1]

sys.path.insert(0, str(BASELINE_ROOT))

from utils.io_utils import ensure_dir, read_csv_safe, save_json, make_safe_name


# ======================================================================================
# FILE 04: BUILD SEQUENCES AND VIDEO-LEVEL TRAIN/VAL/TEST SPLIT
# ======================================================================================
#
# Original Yadav et al. pipeline:
#   Raw skeleton coordinates + guided features
#   -> input deep learning models
#   -> train/validation/test split 60:20:20
#
# Adapted project:
#   Input features:
#       51 adapted Yadav17 skeleton coordinate features
#       14 guided geometric/kinematic features
#       = 65 total features per frame
#
# Output:
#   data/sequences/X_sequences.npy
#   data/sequences/y_sequences.npy
#   data/sequences/sequence_metadata.csv
#   data/sequences/feature_columns.json
#
#   data/splits/yadav2021_video_level_split.csv
#   data/splits/yadav2021_sequence_level_split.csv
#
# Notes:
#   - Split is video-level to avoid data leakage.
#   - Sequence windows are generated inside each video.
#   - Default sequence_length = 30, stride = 15.
# ======================================================================================


RAW_SKELETON_COLS = [f"{axis}{idx}" for idx in range(17) for axis in ["x", "y", "z"]]

GUIDED_FEATURE_COLS = [
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "floor_distance_proxy",
    "angle_left_standing_deg",
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "body_height",
    "angle_standing_deg",
    "angle_sitting_left_deg",
    "angle_sitting_right_deg",
    "angle_right_standing_deg",
    "body_width",
]

FEATURE_COLUMNS = RAW_SKELETON_COLS + GUIDED_FEATURE_COLS


def label_to_name(label: int) -> str:
    return "Fall" if int(label) == 1 else "Not_Fall"


def check_required_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(f"Missing feature columns in {path}: {missing[:10]}")


def pad_sequence(x: np.ndarray, target_len: int) -> np.ndarray:
    """
    Pad short sequence to target length by repeating the last valid frame.
    """
    current_len = len(x)

    if current_len >= target_len:
        return x[:target_len]

    if current_len == 0:
        return np.zeros((target_len, x.shape[1]), dtype=np.float32)

    pad_count = target_len - current_len
    pad_values = np.repeat(x[-1:, :], pad_count, axis=0)

    return np.concatenate([x, pad_values], axis=0).astype(np.float32)


def build_sequences_from_video(
    df: pd.DataFrame,
    video_id: str,
    label: int,
    label_name: str,
    source_csv: str,
    sequence_length: int,
    stride: int,
    min_valid_frames: int,
    allow_padding: bool,
) -> Tuple[List[np.ndarray], List[Dict]]:
    check_required_columns(df, Path(source_csv))

    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    num_frames = len(x)

    sequences = []
    metadata_rows = []

    if num_frames < min_valid_frames:
        return sequences, metadata_rows

    if num_frames < sequence_length:
        if allow_padding:
            seq = pad_sequence(x, sequence_length)

            sequences.append(seq)

            metadata_rows.append({
                "sequence_id": f"{video_id}__seq0000",
                "video_id": video_id,
                "label": int(label),
                "label_name": label_name,
                "source_csv": source_csv,
                "start_frame_index": 0,
                "end_frame_index": int(num_frames - 1),
                "original_num_frames": int(num_frames),
                "sequence_length": int(sequence_length),
                "is_padded": 1,
            })

        return sequences, metadata_rows

    seq_idx = 0

    for start in range(0, num_frames - sequence_length + 1, stride):
        end = start + sequence_length

        seq = x[start:end]

        if len(seq) != sequence_length:
            continue

        sequences.append(seq.astype(np.float32))

        metadata_rows.append({
            "sequence_id": f"{video_id}__seq{seq_idx:04d}",
            "video_id": video_id,
            "label": int(label),
            "label_name": label_name,
            "source_csv": source_csv,
            "start_frame_index": int(start),
            "end_frame_index": int(end - 1),
            "original_num_frames": int(num_frames),
            "sequence_length": int(sequence_length),
            "is_padded": 0,
        })

        seq_idx += 1

    # Ensure the tail of the video is represented.
    last_start = num_frames - sequence_length

    if len(metadata_rows) > 0:
        last_existing_start = int(metadata_rows[-1]["start_frame_index"])
    else:
        last_existing_start = -1

    if last_start > 0 and last_start != last_existing_start:
        seq = x[last_start:num_frames]

        if len(seq) == sequence_length:
            sequences.append(seq.astype(np.float32))

            metadata_rows.append({
                "sequence_id": f"{video_id}__seq{seq_idx:04d}",
                "video_id": video_id,
                "label": int(label),
                "label_name": label_name,
                "source_csv": source_csv,
                "start_frame_index": int(last_start),
                "end_frame_index": int(num_frames - 1),
                "original_num_frames": int(num_frames),
                "sequence_length": int(sequence_length),
                "is_padded": 0,
            })

    return sequences, metadata_rows


def make_video_level_split(
    video_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """
    Create video-level stratified split.

    Default:
        train 60%
        val   20%
        test  20%
    """
    ratio_sum = train_ratio + val_ratio + test_ratio

    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {ratio_sum}")

    video_df = video_df[["video_id", "label", "label_name"]].drop_duplicates("video_id").copy()

    train_df, temp_df = train_test_split(
        video_df,
        train_size=train_ratio,
        random_state=seed,
        stratify=video_df["label"],
    )

    val_relative = val_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_relative,
        random_state=seed,
        stratify=temp_df["label"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    split_df = split_df.sort_values(["split", "label", "video_id"]).reset_index(drop=True)

    return split_df


def summarize_split(split_df: pd.DataFrame, sequence_meta_df: pd.DataFrame) -> Dict:
    video_summary = (
        split_df
        .groupby(["split", "label_name"])
        .agg(num_videos=("video_id", "nunique"))
        .reset_index()
    )

    seq_with_split = sequence_meta_df.merge(
        split_df[["video_id", "split"]],
        on="video_id",
        how="left",
    )

    sequence_summary = (
        seq_with_split
        .groupby(["split", "label_name"])
        .agg(num_sequences=("sequence_id", "count"))
        .reset_index()
    )

    return {
        "video_summary": video_summary.to_dict(orient="records"),
        "sequence_summary": sequence_summary.to_dict(orient="records"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build sequence tensors and video-level train/val/test split for Yadav2021-style baseline."
    )

    parser.add_argument(
        "--index-csv",
        type=str,
        default=str(BASELINE_ROOT / "data" / "guided_features" / "guided_features_index.csv"),
        help="Index CSV generated by file 03.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BASELINE_ROOT / "data" / "sequences"),
        help="Output directory for sequence arrays.",
    )

    parser.add_argument(
        "--split-dir",
        type=str,
        default=str(BASELINE_ROOT / "data" / "splits"),
        help="Output directory for split CSV files.",
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=30,
        help="Number of frames per sequence.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=15,
        help="Sliding-window stride.",
    )

    parser.add_argument(
        "--min-valid-frames",
        type=int,
        default=10,
        help="Minimum number of frames required to keep a video.",
    )

    parser.add_argument(
        "--no-padding",
        action="store_true",
        help="Disable padding for videos shorter than sequence length.",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.60,
        help="Train video ratio.",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Validation video ratio.",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.20,
        help="Test video ratio.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for video-level split.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos for quick testing.",
    )

    args = parser.parse_args()

    index_csv = Path(args.index_csv)
    output_dir = Path(args.output_dir)
    split_dir = Path(args.split_dir)
    reports_dir = BASELINE_ROOT / "outputs" / "reports"

    ensure_dir(output_dir)
    ensure_dir(split_dir)
    ensure_dir(reports_dir)

    if not index_csv.exists():
        raise FileNotFoundError(f"File 03 index not found: {index_csv}")

    index_df = pd.read_csv(index_csv)

    if args.limit is not None:
        index_df = index_df.head(args.limit).copy()

    print("=" * 100)
    print("FILE 04 - Build sequence tensors and video-level train/val/test split")
    print("=" * 100)
    print(f"Input index:       {index_csv}")
    print(f"Output dir:        {output_dir}")
    print(f"Split dir:         {split_dir}")
    print(f"Videos in index:   {len(index_df)}")
    print(f"Sequence length:   {args.sequence_length}")
    print(f"Stride:            {args.stride}")
    print(f"Min valid frames:  {args.min_valid_frames}")
    print(f"Padding:           {not args.no_padding}")
    print(f"Split ratio:       train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"Seed:              {args.seed}")
    print("=" * 100)

    all_sequences: List[np.ndarray] = []
    all_meta_rows: List[Dict] = []
    failed_rows: List[Dict] = []
    skipped_too_short = 0

    for pos, (_, row) in enumerate(index_df.iterrows(), start=1):
        try:
            video_id = make_safe_name(str(row["video_id"]))
            label = int(row["label"])
            label_name = str(row["label_name"])

            input_csv = Path(str(row["output_csv"]))

            if not input_csv.exists():
                input_csv = Path(str(row["input_csv"]))

            df = read_csv_safe(input_csv)

            sequences, meta_rows = build_sequences_from_video(
                df=df,
                video_id=video_id,
                label=label,
                label_name=label_name,
                source_csv=str(input_csv),
                sequence_length=args.sequence_length,
                stride=args.stride,
                min_valid_frames=args.min_valid_frames,
                allow_padding=not args.no_padding,
            )

            if len(sequences) == 0:
                skipped_too_short += 1
            else:
                all_sequences.extend(sequences)
                all_meta_rows.extend(meta_rows)

        except Exception as e:
            failed_rows.append({
                "video_id": str(row.get("video_id", "")),
                "input_csv": str(row.get("output_csv", "")),
                "error": repr(e),
            })

        if pos % 500 == 0 or pos == len(index_df):
            print(
                f"[{pos}/{len(index_df)}] "
                f"sequences={len(all_sequences)} | "
                f"failed={len(failed_rows)} | "
                f"skipped_too_short={skipped_too_short}"
            )

    if len(all_sequences) == 0:
        raise RuntimeError("No sequences were generated. Check input guided feature CSV files.")

    X = np.stack(all_sequences).astype(np.float32)
    meta_df = pd.DataFrame(all_meta_rows)
    y = meta_df["label"].to_numpy(dtype=np.int64)

    # Stable sequence ordering
    meta_df["sequence_index"] = np.arange(len(meta_df), dtype=np.int64)
    meta_df = meta_df[
        [
            "sequence_index",
            "sequence_id",
            "video_id",
            "label",
            "label_name",
            "source_csv",
            "start_frame_index",
            "end_frame_index",
            "original_num_frames",
            "sequence_length",
            "is_padded",
        ]
    ]

    video_df_for_split = meta_df[["video_id", "label", "label_name"]].drop_duplicates("video_id")

    split_df = make_video_level_split(
        video_df=video_df_for_split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    sequence_split_df = meta_df.merge(
        split_df[["video_id", "split"]],
        on="video_id",
        how="left",
    )

    if sequence_split_df["split"].isna().any():
        missing = sequence_split_df[sequence_split_df["split"].isna()]["video_id"].unique()[:10]
        raise RuntimeError(f"Some sequences do not have split labels. Examples: {missing}")

    X_path = output_dir / "X_sequences.npy"
    y_path = output_dir / "y_sequences.npy"
    meta_path = output_dir / "sequence_metadata.csv"
    feature_columns_path = output_dir / "feature_columns.json"

    video_split_path = split_dir / "yadav2021_video_level_split.csv"
    sequence_split_path = split_dir / "yadav2021_sequence_level_split.csv"
    failed_path = output_dir / "sequence_build_failed_files.csv"

    np.save(X_path, X)
    np.save(y_path, y)

    meta_df.to_csv(meta_path, index=False)
    split_df.to_csv(video_split_path, index=False)
    sequence_split_df.to_csv(sequence_split_path, index=False)

    if len(failed_rows) > 0:
        pd.DataFrame(failed_rows).to_csv(failed_path, index=False)
    else:
        pd.DataFrame(columns=["video_id", "input_csv", "error"]).to_csv(failed_path, index=False)

    save_json(
        {
            "feature_columns": FEATURE_COLUMNS,
            "raw_skeleton_columns": RAW_SKELETON_COLS,
            "guided_feature_columns": GUIDED_FEATURE_COLS,
            "num_raw_skeleton_features": len(RAW_SKELETON_COLS),
            "num_guided_features": len(GUIDED_FEATURE_COLS),
            "num_total_features": len(FEATURE_COLUMNS),
        },
        feature_columns_path,
    )

    split_summary = summarize_split(split_df, meta_df)

    report = {
        "status": "completed",
        "pipeline_note": "Build fixed-length sequence tensors from raw skeleton coordinates plus guided features, then create a video-level 60:20:20 train/val/test split following the original paper's experimental setup.",
        "input_index_csv": str(index_csv),
        "output_dir": str(output_dir),
        "split_dir": str(split_dir),
        "sequence_length": int(args.sequence_length),
        "stride": int(args.stride),
        "min_valid_frames": int(args.min_valid_frames),
        "padding_enabled": bool(not args.no_padding),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "seed": int(args.seed),
        "num_input_videos": int(len(index_df)),
        "num_videos_with_sequences": int(meta_df["video_id"].nunique()),
        "num_sequences": int(len(meta_df)),
        "x_shape": list(X.shape),
        "y_shape": list(y.shape),
        "num_features_per_frame": int(X.shape[-1]),
        "num_failed_videos": int(len(failed_rows)),
        "num_skipped_too_short": int(skipped_too_short),
        "x_sequences_path": str(X_path),
        "y_sequences_path": str(y_path),
        "sequence_metadata_csv": str(meta_path),
        "feature_columns_json": str(feature_columns_path),
        "video_split_csv": str(video_split_path),
        "sequence_split_csv": str(sequence_split_path),
        "failed_csv": str(failed_path),
        "split_summary": split_summary,
    }

    report_path = reports_dir / "04_build_sequences_and_split_report.json"
    save_json(report, report_path)

    print("=" * 100)
    print("FILE 04 completed.")
    print("=" * 100)
    print(f"X shape:                 {X.shape}")
    print(f"y shape:                 {y.shape}")
    print(f"Videos with sequences:   {meta_df['video_id'].nunique()}")
    print(f"Total sequences:         {len(meta_df)}")
    print(f"Feature dim:             {X.shape[-1]}")
    print(f"Failed videos:           {len(failed_rows)}")
    print(f"Skipped too short:       {skipped_too_short}")
    print(f"X path:                  {X_path}")
    print(f"y path:                  {y_path}")
    print(f"Metadata CSV:            {meta_path}")
    print(f"Video split CSV:         {video_split_path}")
    print(f"Sequence split CSV:      {sequence_split_path}")
    print(f"Report:                  {report_path}")
    print("=" * 100)

    print("Video-level split summary:")
    print(split_df.groupby(["split", "label_name"])["video_id"].nunique())

    print("=" * 100)
    print("Sequence-level split summary:")
    print(sequence_split_df.groupby(["split", "label_name"])["sequence_id"].count())
    print("=" * 100)


if __name__ == "__main__":
    main()

