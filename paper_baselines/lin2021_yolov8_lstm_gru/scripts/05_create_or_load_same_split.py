from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SCRIPT_PATH = Path(__file__).resolve()
BASELINE_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = BASELINE_ROOT.parents[1]


# ======================================================================================
# FILE 05: CREATE OR LOAD TRAIN / VAL / TEST SPLIT
# ======================================================================================
#
# This file prepares the data split for the Lin-style YOLOv8 + RNN/LSTM/GRU baseline.
#
# Important:
#   Split must be done at VIDEO LEVEL, not sequence level.
#
# Why:
#   One video can produce multiple 100-frame sequences.
#   If sequences from the same video appear in both train and test, the evaluation leaks
#   video-specific information and becomes unfair.
#
# Output:
#   data/splits/lin2021_yolov8_train_sequences.npz
#   data/splits/lin2021_yolov8_val_sequences.npz
#   data/splits/lin2021_yolov8_test_sequences.npz
#   data/splits/lin2021_yolov8_sequence_manifest_with_split.csv
#   data/splits/lin2021_yolov8_video_level_split.csv
# ======================================================================================


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


def label_to_name(label: int) -> str:
    if int(label) == 1:
        return "Fall"
    if int(label) == 0:
        return "Not_Fall"
    return "Unknown"


def load_sequence_data(npz_path: Path, manifest_path: Path):
    if not npz_path.exists():
        raise FileNotFoundError(f"Sequence NPZ not found: {npz_path}")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Sequence manifest not found: {manifest_path}")

    data = np.load(npz_path, allow_pickle=True)

    required_npz_keys = ["X", "y", "sequence_ids", "video_ids"]

    for key in required_npz_keys:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in {npz_path}")

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    sequence_ids = data["sequence_ids"].astype(str)
    video_ids = data["video_ids"].astype(str)

    manifest_df = pd.read_csv(manifest_path)

    if len(manifest_df) != X.shape[0]:
        raise ValueError(
            f"Manifest rows do not match X. manifest={len(manifest_df)}, X={X.shape[0]}"
        )

    if len(y) != X.shape[0]:
        raise ValueError(
            f"y length does not match X. y={len(y)}, X={X.shape[0]}"
        )

    return X, y, sequence_ids, video_ids, manifest_df


def validate_manifest(manifest_df: pd.DataFrame):
    required_cols = [
        "sequence_id",
        "video_id",
        "label",
        "label_name",
        "sequence_index",
        "start_frame",
        "end_frame",
        "sequence_length",
        "feature_dim",
    ]

    missing = [
        col for col in required_cols
        if col not in manifest_df.columns
    ]

    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")


def build_video_level_table(manifest_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for video_id, group in manifest_df.groupby("video_id"):
        labels = sorted(
            pd.to_numeric(group["label"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        if len(labels) != 1:
            raise ValueError(f"Video {video_id} has multiple labels: {labels}")

        label = int(labels[0])

        if label not in [0, 1]:
            raise ValueError(f"Invalid label for video {video_id}: {label}")

        label_name = label_to_name(label)

        rows.append({
            "video_id": str(video_id),
            "label": int(label),
            "label_name": label_name,
            "num_sequences": int(len(group)),
            "num_original_frames_mean": float(group["num_original_frames"].mean()) if "num_original_frames" in group.columns else None,
            "num_padded_sequences": int(group["is_padded"].sum()) if "is_padded" in group.columns else None,
            "num_all_zero_sequences": int(group["is_all_zero"].sum()) if "is_all_zero" in group.columns else None,
        })

    video_df = pd.DataFrame(rows)
    video_df = video_df.sort_values(["label", "video_id"]).reset_index(drop=True)

    return video_df


def create_video_level_split(
    video_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    ratio_sum = train_ratio + val_ratio + test_ratio

    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must be 1. Got {ratio_sum}"
        )

    if video_df["label"].nunique() != 2:
        raise ValueError(
            f"Expected binary labels 0/1, got: {video_df['label'].unique().tolist()}"
        )

    train_df, temp_df = train_test_split(
        video_df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=video_df["label"],
    )

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temp_df["label"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat(
        [train_df, val_df, test_df],
        axis=0,
        ignore_index=True,
    )

    split_df = split_df.sort_values(["split", "label", "video_id"]).reset_index(drop=True)

    return split_df


def load_existing_split(split_csv: Path, video_df: pd.DataFrame) -> pd.DataFrame:
    if not split_csv.exists():
        raise FileNotFoundError(f"Existing split CSV not found: {split_csv}")

    split_df = pd.read_csv(split_csv)

    required = ["video_id", "split"]

    missing = [
        col for col in required
        if col not in split_df.columns
    ]

    if missing:
        raise ValueError(
            f"Existing split CSV must contain {required}. Missing: {missing}"
        )

    split_df["video_id"] = split_df["video_id"].astype(str).apply(make_safe_name)
    split_df["split"] = split_df["split"].astype(str).str.lower()

    allowed = {"train", "val", "test"}
    bad = sorted([
        x for x in split_df["split"].unique().tolist()
        if x not in allowed
    ])

    if bad:
        raise ValueError(f"Invalid split values: {bad}. Allowed: train, val, test.")

    # Add label metadata from current manifest.
    video_meta = video_df.drop_duplicates("video_id")

    split_df = split_df.merge(
        video_meta,
        on="video_id",
        how="left",
        suffixes=("", "_from_manifest"),
    )

    missing_video = split_df[split_df["label"].isna()]["video_id"].tolist()

    if len(missing_video) > 0:
        raise ValueError(
            f"Some video_id in existing split cannot be found in current manifest. Examples: {missing_video[:10]}"
        )

    return split_df


def apply_split_to_manifest(
    manifest_df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> pd.DataFrame:
    split_map = dict(
        zip(
            split_df["video_id"].astype(str),
            split_df["split"].astype(str),
        )
    )

    out = manifest_df.copy()
    out["video_id"] = out["video_id"].astype(str)
    out["split"] = out["video_id"].map(split_map)

    missing = out[out["split"].isna()]["video_id"].unique().tolist()

    if len(missing) > 0:
        raise ValueError(
            f"Some manifest video_id values do not exist in split map. Examples: {missing[:10]}"
        )

    return out


def verify_no_video_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict:
    train_videos = set(train_df["video_id"].astype(str).tolist())
    val_videos = set(val_df["video_id"].astype(str).tolist())
    test_videos = set(test_df["video_id"].astype(str).tolist())

    train_val = sorted(list(train_videos & val_videos))
    train_test = sorted(list(train_videos & test_videos))
    val_test = sorted(list(val_videos & test_videos))

    valid = (
        len(train_val) == 0
        and len(train_test) == 0
        and len(val_test) == 0
    )

    return {
        "valid": bool(valid),
        "train_val_overlap_count": int(len(train_val)),
        "train_test_overlap_count": int(len(train_test)),
        "val_test_overlap_count": int(len(val_test)),
        "train_val_overlap_examples": train_val[:10],
        "train_test_overlap_examples": train_test[:10],
        "val_test_overlap_examples": val_test[:10],
    }


def summarize_split(split_name: str, df: pd.DataFrame) -> Dict:
    label_counts_seq = df["label_name"].value_counts().to_dict()
    label_counts_video = df.groupby("label_name")["video_id"].nunique().to_dict()

    out = {
        "split": split_name,
        "num_sequences": int(len(df)),
        "num_videos": int(df["video_id"].nunique()),
        "label_counts_by_sequence": {str(k): int(v) for k, v in label_counts_seq.items()},
        "label_counts_by_video": {str(k): int(v) for k, v in label_counts_video.items()},
    }

    if "is_padded" in df.columns:
        out["padded_sequences"] = int(df["is_padded"].sum())

    if "is_all_zero" in df.columns:
        out["all_zero_sequences"] = int(df["is_all_zero"].sum())

    return out


def save_split_npz(
    X: np.ndarray,
    y: np.ndarray,
    sequence_ids: np.ndarray,
    video_ids: np.ndarray,
    indices: np.ndarray,
    output_path: Path,
):
    ensure_dir(output_path.parent)

    np.savez_compressed(
        output_path,
        X=X[indices].astype(np.float32),
        y=y[indices].astype(np.int64),
        sequence_ids=sequence_ids[indices].astype(str),
        video_ids=video_ids[indices].astype(str),
        indices=indices.astype(np.int64),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create or load video-level train/val/test split for Lin-style YOLOv8 RNN/LSTM/GRU baseline."
    )

    parser.add_argument(
        "--npz",
        type=str,
        default=str(BASELINE_ROOT / "data" / "sequences" / "lin2021_yolov8_lstm_gru_sequences.npz"),
        help="Input sequence NPZ from File 04.",
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default=str(BASELINE_ROOT / "data" / "sequences" / "lin2021_yolov8_sequence_manifest.csv"),
        help="Input sequence manifest from File 04.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BASELINE_ROOT / "data" / "splits"),
        help="Output split directory.",
    )

    parser.add_argument(
        "--existing-split-csv",
        type=str,
        default=None,
        help="Optional existing video-level split CSV with columns video_id, split.",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train video ratio.",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation video ratio.",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test video ratio.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    npz_path = Path(args.npz)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    reports_dir = BASELINE_ROOT / "outputs" / "reports"

    ensure_dir(output_dir)
    ensure_dir(reports_dir)

    X, y, sequence_ids, video_ids, manifest_df = load_sequence_data(
        npz_path=npz_path,
        manifest_path=manifest_path,
    )

    validate_manifest(manifest_df)

    print("=" * 100)
    print("FILE 05 - Create or load video-level train/val/test split")
    print("=" * 100)
    print(f"Input NPZ:        {npz_path}")
    print(f"Input manifest:   {manifest_path}")
    print(f"Output dir:       {output_dir}")
    print(f"X shape:          {X.shape}")
    print(f"Manifest rows:    {len(manifest_df)}")
    print(f"Videos:           {manifest_df['video_id'].nunique()}")
    print(f"Existing split:   {args.existing_split_csv}")
    print(f"Ratios:           train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"Seed:             {args.seed}")
    print("=" * 100)

    video_df = build_video_level_table(manifest_df)

    if args.existing_split_csv is not None:
        split_df = load_existing_split(
            split_csv=Path(args.existing_split_csv),
            video_df=video_df,
        )
        split_mode = "loaded_existing_video_level_split"
        split_source = str(args.existing_split_csv)
    else:
        split_df = create_video_level_split(
            video_df=video_df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        split_mode = "created_new_stratified_video_level_split"
        split_source = "generated_by_file_05"

    video_split_path = output_dir / "lin2021_yolov8_video_level_split.csv"
    split_df.to_csv(video_split_path, index=False)

    manifest_with_split = apply_split_to_manifest(
        manifest_df=manifest_df,
        split_df=split_df,
    )

    train_indices = manifest_with_split.index[
        manifest_with_split["split"] == "train"
    ].to_numpy(dtype=np.int64)

    val_indices = manifest_with_split.index[
        manifest_with_split["split"] == "val"
    ].to_numpy(dtype=np.int64)

    test_indices = manifest_with_split.index[
        manifest_with_split["split"] == "test"
    ].to_numpy(dtype=np.int64)

    train_df = manifest_with_split.loc[train_indices].reset_index(drop=True)
    val_df = manifest_with_split.loc[val_indices].reset_index(drop=True)
    test_df = manifest_with_split.loc[test_indices].reset_index(drop=True)

    leakage_report = verify_no_video_leakage(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    if not leakage_report["valid"]:
        raise RuntimeError(f"Video leakage detected: {leakage_report}")

    full_manifest_path = output_dir / "lin2021_yolov8_sequence_manifest_with_split.csv"
    train_manifest_path = output_dir / "lin2021_yolov8_train_manifest.csv"
    val_manifest_path = output_dir / "lin2021_yolov8_val_manifest.csv"
    test_manifest_path = output_dir / "lin2021_yolov8_test_manifest.csv"

    manifest_with_split.to_csv(full_manifest_path, index=False)
    train_df.to_csv(train_manifest_path, index=False)
    val_df.to_csv(val_manifest_path, index=False)
    test_df.to_csv(test_manifest_path, index=False)

    train_npz_path = output_dir / "lin2021_yolov8_train_sequences.npz"
    val_npz_path = output_dir / "lin2021_yolov8_val_sequences.npz"
    test_npz_path = output_dir / "lin2021_yolov8_test_sequences.npz"

    save_split_npz(
        X=X,
        y=y,
        sequence_ids=sequence_ids,
        video_ids=video_ids,
        indices=train_indices,
        output_path=train_npz_path,
    )

    save_split_npz(
        X=X,
        y=y,
        sequence_ids=sequence_ids,
        video_ids=video_ids,
        indices=val_indices,
        output_path=val_npz_path,
    )

    save_split_npz(
        X=X,
        y=y,
        sequence_ids=sequence_ids,
        video_ids=video_ids,
        indices=test_indices,
        output_path=test_npz_path,
    )

    summary = {
        "train": summarize_split("train", train_df),
        "val": summarize_split("val", val_df),
        "test": summarize_split("test", test_df),
    }

    report = {
        "status": "completed",
        "pipeline_note": "This split is created at video level to avoid leakage between train, validation, and test sequences.",
        "split_mode": split_mode,
        "split_source": split_source,
        "seed": int(args.seed),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "input_npz": str(npz_path),
        "input_manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "X_shape": list(X.shape),
        "total_sequences": int(len(manifest_with_split)),
        "total_videos": int(manifest_with_split["video_id"].nunique()),
        "video_level_split_csv": str(video_split_path),
        "sequence_manifest_with_split": str(full_manifest_path),
        "train_npz": str(train_npz_path),
        "val_npz": str(val_npz_path),
        "test_npz": str(test_npz_path),
        "train_manifest": str(train_manifest_path),
        "val_manifest": str(val_manifest_path),
        "test_manifest": str(test_manifest_path),
        "summary": summary,
        "leakage_report": leakage_report,
    }

    report_path = reports_dir / "05_create_or_load_same_split_report.json"
    save_json(report, report_path)

    print("=" * 100)
    print("FILE 05 completed.")
    print("=" * 100)
    print(f"Split mode:        {split_mode}")
    print(f"Leakage valid:     {leakage_report['valid']}")
    print("-" * 100)
    print(f"Train: {summary['train']}")
    print(f"Val:   {summary['val']}")
    print(f"Test:  {summary['test']}")
    print("-" * 100)
    print(f"Video split:       {video_split_path}")
    print(f"Full manifest:     {full_manifest_path}")
    print(f"Train NPZ:         {train_npz_path}")
    print(f"Val NPZ:           {val_npz_path}")
    print(f"Test NPZ:          {test_npz_path}")
    print(f"Report:            {report_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
