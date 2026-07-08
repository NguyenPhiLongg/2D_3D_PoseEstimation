"""
Build common-set split for Phase 3.

Purpose:
- Find videos that have both 2D keypoint CSV and 3D keypoint CSV.
- Create one shared common dataset for fair comparison.
- Save:
    outputs/common_split/common_metadata.csv
    outputs/common_split/train_videos.txt
    outputs/common_split/val_videos.txt
    outputs/common_split/test_videos.txt
    outputs/common_split/common_split_summary.json

Why this is important:
Phase 1 2D baseline used the full 2D dataset.
Phase 2 3D and fusion could only use videos where 3D pose was generated successfully.

Phase 3 fixes this by forcing all models to use the exact same common video set:
    2D Common
    3D Common
    Concat Fusion Common
    Gated Fusion
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from phase3_utils import (
    ACTION_CLASS_NAMES,
    ACTION_LABEL_TO_INDEX,
    BINARY_CLASS_NAMES,
    COMMON_SPLIT_DIR,
    DATA_2D_DIR,
    DATA_3D_DIR,
    DATA_3D_RAW_DIR,
    collect_csv_files,
    extract_label_info,
    read_csv_safely,
    set_seed,
)


def make_output_dirs() -> None:
    COMMON_SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def safe_train_test_split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    stratify_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run train_test_split with stratification when possible.
    If stratification fails because a class has too few samples, fallback to random split.
    """
    if len(df) == 0:
        return df.copy(), df.copy()

    if stratify_col not in df.columns:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

    try:
        counts = df[stratify_col].value_counts()

        if counts.min() < 2:
            print(
                "WARNING: Some strata have fewer than 2 samples. "
                "Using non-stratified split for this step."
            )
            return train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                shuffle=True,
            )

        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=df[stratify_col],
        )

    except Exception as exc:
        print("WARNING: Stratified split failed.")
        print("Reason:", exc)
        print("Using non-stratified split instead.")

        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )


def build_stratify_key(label: int, action_label: int) -> str:
    """
    Build a stratification key that preserves both fall and normal actions.

    label:
        0 = Not_Fall
        1 = Fall

    action_label:
        0 = Fall
        1 = Sitting
        2 = Sleeping
        3 = Standing
        4 = Walking

    Example keys:
        Fall
        Sitting
        Sleeping
        Standing
        Walking
        Unknown_Not_Fall
    """
    if int(label) == 1:
        return "Fall"

    if int(action_label) in ACTION_LABEL_TO_INDEX:
        return ACTION_CLASS_NAMES[ACTION_LABEL_TO_INDEX[int(action_label)]]

    return "Unknown_Not_Fall"


def build_common_metadata(
    use_raw_3d_if_normalized_missing: bool = False,
) -> pd.DataFrame:
    """
    Match 2D and 3D CSV files by normalized video key.
    """
    print("=" * 80)
    print("Building Phase 3 common metadata")
    print("=" * 80)

    print("2D directory:", DATA_2D_DIR)
    print("3D normalized directory:", DATA_3D_DIR)

    files_2d = collect_csv_files(DATA_2D_DIR)
    files_3d = collect_csv_files(DATA_3D_DIR)

    if len(files_3d) == 0 and use_raw_3d_if_normalized_missing:
        print("WARNING: No normalized 3D CSV files found.")
        print("Using raw 3D directory instead:", DATA_3D_RAW_DIR)
        files_3d = collect_csv_files(DATA_3D_RAW_DIR)

    print("Number of 2D CSV files:", len(files_2d))
    print("Number of 3D CSV files:", len(files_3d))

    common_keys = sorted(set(files_2d.keys()) & set(files_3d.keys()))

    print("Number of common videos:", len(common_keys))

    if len(common_keys) == 0:
        raise RuntimeError(
            "No common videos found. Please check data/2_extracted_2d and data/4_normalized_3d."
        )

    rows = []
    failed = []

    for key in common_keys:
        path_2d = files_2d[key]
        path_3d = files_3d[key]

        try:
            df_for_label = read_csv_safely(path_2d)
            label, action_label, action_name = extract_label_info(df_for_label, path_2d)

            # If 2D CSV does not contain good label information, try 3D CSV.
            if action_label < 0 or action_name == "Unknown":
                try:
                    df3d_for_label = read_csv_safely(path_3d)
                    label_3d, action_label_3d, action_name_3d = extract_label_info(
                        df3d_for_label,
                        path_3d,
                    )

                    label = label_3d
                    action_label = action_label_3d
                    action_name = action_name_3d

                except Exception:
                    pass

            stratify_key = build_stratify_key(label, action_label)

            rows.append(
                {
                    "video_key": key,
                    "path_2d": str(path_2d.resolve()),
                    "path_3d": str(path_3d.resolve()),
                    "label": int(label),
                    "binary_name": BINARY_CLASS_NAMES[int(label)] if int(label) in [0, 1] else "Unknown",
                    "action_label": int(action_label),
                    "action_name": str(action_name),
                    "stratify_key": stratify_key,
                }
            )

        except Exception as exc:
            failed.append(
                {
                    "video_key": key,
                    "path_2d": str(path_2d),
                    "path_3d": str(path_3d),
                    "error": str(exc),
                }
            )

    metadata = pd.DataFrame(rows)

    if len(metadata) == 0:
        raise RuntimeError("All common videos failed while reading metadata.")

    print("Valid common videos:", len(metadata))
    print("Failed videos:", len(failed))

    print("\nBinary distribution:")
    print(metadata["binary_name"].value_counts())

    print("\nStratify distribution:")
    print(metadata["stratify_key"].value_counts())

    if failed:
        failed_path = COMMON_SPLIT_DIR / "failed_common_metadata.json"

        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=4, ensure_ascii=False)

        print("\nFailed metadata saved to:", failed_path)

    return metadata


def assign_splits(
    metadata: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create train/val/test split.

    Default:
        train = 70%
        val   = 15%
        test  = 15%

    The split is video-level, not sequence-level.
    This prevents leakage where sequences from the same video appear in both train and test.
    """
    total_ratio = train_ratio + val_ratio + test_ratio

    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {total_ratio}"
        )

    metadata = metadata.copy().reset_index(drop=True)

    test_size = test_ratio
    train_val_df, test_df = safe_train_test_split(
        metadata,
        test_size=test_size,
        random_state=seed,
        stratify_col="stratify_key",
    )

    val_size_relative = val_ratio / (train_ratio + val_ratio)

    train_df, val_df = safe_train_test_split(
        train_val_df,
        test_size=val_size_relative,
        random_state=seed,
        stratify_col="stratify_key",
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    final_df = pd.concat([train_df, val_df, test_df], axis=0)
    final_df = final_df.sort_values(["split", "video_key"]).reset_index(drop=True)

    return final_df


def save_video_list(df: pd.DataFrame, split: str) -> None:
    split_df = df[df["split"] == split].copy()
    video_keys = split_df["video_key"].astype(str).tolist()

    path = COMMON_SPLIT_DIR / f"{split}_videos.txt"

    with open(path, "w", encoding="utf-8") as f:
        for key in video_keys:
            f.write(key + "\n")

    print(f"Saved {split} video list:", path, "| count:", len(video_keys))


def summarize_split(df: pd.DataFrame) -> dict:
    summary = {
        "total_videos": int(len(df)),
        "split_counts": df["split"].value_counts().to_dict(),
        "binary_distribution_by_split": {},
        "stratify_distribution_by_split": {},
        "action_distribution_for_action_task": {},
    }

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]

        summary["binary_distribution_by_split"][split] = (
            split_df["binary_name"].value_counts().to_dict()
        )

        summary["stratify_distribution_by_split"][split] = (
            split_df["stratify_key"].value_counts().to_dict()
        )

        action_df = split_df[split_df["action_label"].isin(list(ACTION_LABEL_TO_INDEX.keys()))]

        summary["action_distribution_for_action_task"][split] = (
            action_df["action_name"].value_counts().to_dict()
        )

    return summary


def save_outputs(df: pd.DataFrame) -> None:
    make_output_dirs()

    metadata_path = COMMON_SPLIT_DIR / "common_metadata.csv"
    df.to_csv(metadata_path, index=False, encoding="utf-8-sig")

    print("\nSaved common metadata:", metadata_path)

    save_video_list(df, "train")
    save_video_list(df, "val")
    save_video_list(df, "test")

    summary = summarize_split(df)
    summary_path = COMMON_SPLIT_DIR / "common_split_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("Saved summary:", summary_path)

    print("\n" + "=" * 80)
    print("COMMON SET SUMMARY")
    print("=" * 80)

    print("Total videos:", summary["total_videos"])
    print("Split counts:", summary["split_counts"])

    print("\nBinary distribution by split:")
    for split, counts in summary["binary_distribution_by_split"].items():
        print(split, counts)

    print("\nAction distribution by split:")
    for split, counts in summary["action_distribution_for_action_task"].items():
        print(split, counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase 3 common-set split for fair 2D/3D/Fusion comparison."
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train video ratio. Default: 0.70",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation video ratio. Default: 0.15",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test video ratio. Default: 0.15",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )

    parser.add_argument(
        "--use-raw-3d-if-normalized-missing",
        action="store_true",
        help=(
            "Use data/3_extracted_3d if data/4_normalized_3d has no CSV files. "
            "Recommended only when normalized 3D files are not available."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    make_output_dirs()

    metadata = build_common_metadata(
        use_raw_3d_if_normalized_missing=args.use_raw_3d_if_normalized_missing
    )

    final_df = assign_splits(
        metadata,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    save_outputs(final_df)

    print("\nDone.")
    print("Next step:")
    print("python phase3_common_set_gated_fusion/train_2d_common.py --task binary")
    print("python phase3_common_set_gated_fusion/train_2d_common.py --task action")


if __name__ == "__main__":
    main()