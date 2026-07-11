from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split


SCRIPT_PATH = Path(__file__).resolve()
BASELINE_ROOT = SCRIPT_PATH.parents[1]


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


def read_csv_safe(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e

    raise last_error


def validate_index_df(df: pd.DataFrame, index_path: Path):
    required_cols = [
        "video_id",
        "label",
        "label_name",
        "num_frames",
        "output_csv",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"Index CSV missing required columns in {index_path}: {missing}"
        )


def build_video_df(index_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in index_df.iterrows():
        video_id = make_safe_name(str(row["video_id"]))
        label = int(row["label"])
        label_name = str(row["label_name"])

        if label not in [0, 1]:
            raise ValueError(f"Invalid binary label for video {video_id}: {label}")

        rows.append({
            "video_id": video_id,
            "label": label,
            "label_name": label_name,
            "num_frames": int(row["num_frames"]),
            "output_csv": str(row["output_csv"]),
            "all_rule_features_valid_frames": int(row.get("all_rule_features_valid_frames", 0)),
        })

    video_df = pd.DataFrame(rows)
    video_df = video_df.drop_duplicates("video_id").reset_index(drop=True)
    video_df = video_df.sort_values(["label", "video_id"]).reset_index(drop=True)

    return video_df


def create_new_split(
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

    split_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    split_df = split_df.sort_values(["split", "label", "video_id"]).reset_index(drop=True)

    return split_df


def load_existing_split(existing_split_csv: Path, video_df: pd.DataFrame) -> pd.DataFrame:
    if not existing_split_csv.exists():
        raise FileNotFoundError(f"Existing split CSV not found: {existing_split_csv}")

    split_df = read_csv_safe(existing_split_csv)

    required_cols = ["video_id", "split"]
    missing = [col for col in required_cols if col not in split_df.columns]

    if missing:
        raise ValueError(
            f"Existing split CSV must contain {required_cols}. Missing: {missing}"
        )

    split_df = split_df.copy()
    split_df["video_id"] = split_df["video_id"].astype(str).apply(make_safe_name)
    split_df["split"] = split_df["split"].astype(str).str.lower()

    allowed = {"train", "val", "test"}
    bad = sorted([x for x in split_df["split"].unique().tolist() if x not in allowed])

    if bad:
        raise ValueError(f"Invalid split values in existing split: {bad}")

    current_videos = set(video_df["video_id"].astype(str).tolist())
    split_videos = set(split_df["video_id"].astype(str).tolist())

    missing_from_current = sorted(list(split_videos - current_videos))
    missing_from_split = sorted(list(current_videos - split_videos))

    if len(missing_from_current) > 0:
        raise ValueError(
            "Existing split contains videos not found in current Chen features. "
            f"Examples: {missing_from_current[:10]}"
        )

    if len(missing_from_split) > 0:
        raise ValueError(
            "Current Chen features contain videos not found in existing split. "
            f"Examples: {missing_from_split[:10]}"
        )

    split_keep = split_df[["video_id", "split"]].drop_duplicates("video_id")

    out = video_df.merge(split_keep, on="video_id", how="left")

    if out["split"].isna().any():
        bad_videos = out[out["split"].isna()]["video_id"].tolist()
        raise ValueError(f"Some videos have no split assigned. Examples: {bad_videos[:10]}")

    out = out.sort_values(["split", "label", "video_id"]).reset_index(drop=True)

    return out


def apply_split_to_index(index_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    split_map = dict(
        zip(
            split_df["video_id"].astype(str),
            split_df["split"].astype(str),
        )
    )

    out = index_df.copy()
    out["video_id"] = out["video_id"].astype(str).apply(make_safe_name)
    out["split"] = out["video_id"].map(split_map)

    missing = out[out["split"].isna()]["video_id"].unique().tolist()

    if len(missing) > 0:
        raise ValueError(
            f"Some videos in index do not have split assignment. Examples: {missing[:10]}"
        )

    return out


def verify_no_video_leakage(manifest_df: pd.DataFrame) -> Dict:
    train_videos = set(manifest_df[manifest_df["split"] == "train"]["video_id"].astype(str).tolist())
    val_videos = set(manifest_df[manifest_df["split"] == "val"]["video_id"].astype(str).tolist())
    test_videos = set(manifest_df[manifest_df["split"] == "test"]["video_id"].astype(str).tolist())

    train_val = sorted(list(train_videos & val_videos))
    train_test = sorted(list(train_videos & test_videos))
    val_test = sorted(list(val_videos & test_videos))

    valid = len(train_val) == 0 and len(train_test) == 0 and len(val_test) == 0

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
    label_video_counts = df.groupby("label_name")["video_id"].nunique().to_dict()
    label_frame_counts = df.groupby("label_name")["num_frames"].sum().to_dict()

    out = {
        "split": split_name,
        "num_videos": int(df["video_id"].nunique()),
        "num_frames": int(df["num_frames"].sum()),
        "label_counts_by_video": {str(k): int(v) for k, v in label_video_counts.items()},
        "label_counts_by_frame": {str(k): int(v) for k, v in label_frame_counts.items()},
    }

    if "all_rule_features_valid_frames" in df.columns:
        out["valid_rule_feature_frames"] = int(df["all_rule_features_valid_frames"].sum())

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Create or load video-level train/val/test split for Chen 2020-style threshold baseline."
    )

    parser.add_argument(
        "--index-csv",
        type=str,
        default=str(BASELINE_ROOT / "data" / "rule_features" / "all_rule_features_index.csv"),
        help="Input index CSV from File 03.",
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

    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    index_csv = Path(args.index_csv)
    output_dir = Path(args.output_dir)
    reports_dir = BASELINE_ROOT / "outputs" / "reports"

    ensure_dir(output_dir)
    ensure_dir(reports_dir)

    index_df = read_csv_safe(index_csv)
    validate_index_df(index_df, index_csv)

    video_df = build_video_df(index_df)

    print("=" * 100)
    print("FILE 04 - Create or load video-level split")
    print("=" * 100)
    print(f"Index CSV:        {index_csv}")
    print(f"Output dir:       {output_dir}")
    print(f"Videos:           {video_df['video_id'].nunique()}")
    print(f"Total frames:     {int(video_df['num_frames'].sum())}")
    print(f"Existing split:   {args.existing_split_csv}")
    print(f"Ratios:           train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"Seed:             {args.seed}")
    print("=" * 100)

    if args.existing_split_csv is not None:
        split_df = load_existing_split(
            existing_split_csv=Path(args.existing_split_csv),
            video_df=video_df,
        )
        split_mode = "loaded_existing_video_level_split"
        split_source = str(args.existing_split_csv)
    else:
        split_df = create_new_split(
            video_df=video_df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        split_mode = "created_new_stratified_video_level_split"
        split_source = "generated_by_file_04"

    manifest_with_split = apply_split_to_index(index_df=index_df, split_df=split_df)

    leakage_report = verify_no_video_leakage(manifest_with_split)

    if not leakage_report["valid"]:
        raise RuntimeError(f"Video leakage detected: {leakage_report}")

    video_split_path = output_dir / "chen2020_video_level_split.csv"
    full_manifest_path = output_dir / "chen2020_rule_features_manifest_with_split.csv"

    train_index_path = output_dir / "chen2020_train_rule_features_index.csv"
    val_index_path = output_dir / "chen2020_val_rule_features_index.csv"
    test_index_path = output_dir / "chen2020_test_rule_features_index.csv"

    split_df.to_csv(video_split_path, index=False)
    manifest_with_split.to_csv(full_manifest_path, index=False)

    train_df = manifest_with_split[manifest_with_split["split"] == "train"].reset_index(drop=True)
    val_df = manifest_with_split[manifest_with_split["split"] == "val"].reset_index(drop=True)
    test_df = manifest_with_split[manifest_with_split["split"] == "test"].reset_index(drop=True)

    train_df.to_csv(train_index_path, index=False)
    val_df.to_csv(val_index_path, index=False)
    test_df.to_csv(test_index_path, index=False)

    summary = {
        "train": summarize_split("train", train_df),
        "val": summarize_split("val", val_df),
        "test": summarize_split("test", test_df),
    }

    report = {
        "status": "completed",
        "pipeline_note": "This split is created at video level to avoid leakage. Validation split will be used to tune thresholds, and test split will be used for final evaluation.",
        "split_mode": split_mode,
        "split_source": split_source,
        "seed": int(args.seed),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "index_csv": str(index_csv),
        "output_dir": str(output_dir),
        "total_videos": int(manifest_with_split["video_id"].nunique()),
        "total_frames": int(manifest_with_split["num_frames"].sum()),
        "video_level_split_csv": str(video_split_path),
        "manifest_with_split_csv": str(full_manifest_path),
        "train_index_csv": str(train_index_path),
        "val_index_csv": str(val_index_path),
        "test_index_csv": str(test_index_path),
        "summary": summary,
        "leakage_report": leakage_report,
    }

    report_path = reports_dir / "04_create_or_load_video_split_report.json"
    save_json(report, report_path)

    print("=" * 100)
    print("FILE 04 completed.")
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
    print(f"Train index:       {train_index_path}")
    print(f"Val index:         {val_index_path}")
    print(f"Test index:        {test_index_path}")
    print(f"Report:            {report_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
