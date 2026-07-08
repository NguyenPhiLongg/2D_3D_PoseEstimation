import os
import re
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# PATH SETUP
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE5_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PHASE5_DIR.parent

if str(PHASE5_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE5_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PHASE 5 UTILS
# ============================================================

from phase5_utils import (
    load_config,
    cfg_path,
    ensure_dir,
    print_dict,
    print_dataframe_summary,
)


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_str(value, default: str = "") -> str:
    try:
        if value is None or pd.isna(value):
            return default
        return str(value)
    except Exception:
        return default


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def value_counts_dict(series: pd.Series) -> Dict[str, int]:
    return {
        str(k): int(v)
        for k, v in series.value_counts(dropna=False).to_dict().items()
    }


def label_counts_dict(df: pd.DataFrame) -> Dict[str, int]:
    if df.empty:
        return {}

    if "label_name" in df.columns:
        return value_counts_dict(df["label_name"])

    return value_counts_dict(df["label"])


def dataset_counts_dict(df: pd.DataFrame) -> Dict[str, int]:
    if df.empty or "dataset" not in df.columns:
        return {}

    return value_counts_dict(df["dataset"])


# ============================================================
# LOAD STEP 05 OUTPUTS
# ============================================================

def load_step05_outputs(config: Dict) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Path, Path]:
    external_sequences_dir = cfg_path(
        config,
        config["outputs"]["external_sequences_dir"],
    )

    npz_path = external_sequences_dir / "external_sequence_inputs.npz"
    fair_manifest_path = external_sequences_dir / "all_external_sequences_common_fair.csv"

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing Step 05 NPZ file:\n{npz_path}\n"
            "Run 05_build_external_sequences.py first."
        )

    if not fair_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing Step 05 fair manifest:\n{fair_manifest_path}\n"
            "Run 05_build_external_sequences.py first."
        )

    npz = np.load(npz_path, allow_pickle=True)
    arrays = {key: npz[key] for key in npz.files}

    manifest_df = pd.read_csv(fair_manifest_path)

    required_cols = [
        "dataset",
        "video_id",
        "sequence_key",
        "label",
        "label_name",
        "start_frame",
        "end_frame",
    ]

    missing_cols = [col for col in required_cols if col not in manifest_df.columns]

    if missing_cols:
        raise ValueError(
            f"Fair manifest missing required columns: {missing_cols}"
        )

    if "sequence_keys" not in arrays:
        raise ValueError("external_sequence_inputs.npz missing sequence_keys array.")

    array_keys = [str(x) for x in arrays["sequence_keys"].tolist()]
    manifest_keys = manifest_df["sequence_key"].astype(str).tolist()

    if array_keys != manifest_keys:
        raise ValueError(
            "sequence_keys in NPZ do not match all_external_sequences_common_fair.csv order. "
            "Do not continue because train/val/test split would be misaligned."
        )

    if "y_binary" not in arrays:
        raise ValueError("external_sequence_inputs.npz missing y_binary array.")

    y_npz = arrays["y_binary"].astype(int)
    y_manifest = manifest_df["label"].astype(int).to_numpy()

    if not np.array_equal(y_npz, y_manifest):
        raise ValueError(
            "y_binary in NPZ does not match label column in manifest."
        )

    return arrays, manifest_df, npz_path, fair_manifest_path


# ============================================================
# GROUP KEY FOR NO-LEAKAGE SPLIT
# ============================================================

def infer_mulcamfall_chute(text: str) -> Optional[str]:
    text = safe_str(text)

    match = re.search(r"chute[_\-\s]*0*([0-9]+)", text, flags=re.IGNORECASE)

    if match:
        chute_id = int(match.group(1))
        return f"chute{chute_id:02d}"

    return None


def infer_group_key(row: pd.Series) -> str:
    """
    Important:
        We do NOT split random sequences directly.

    Why:
        One video/chute is cut into many sequences.
        If sequences from the same video appear in both train and test,
        the result becomes data leakage.

    For MulCamFall:
        group by chute, not by sequence.
        Example:
            mulcamfall__chute22_cam1
            mulcamfall__chute22_cam2
        should stay in the same split.

    For other datasets:
        fallback group by video_id.
    """
    dataset = safe_str(row.get("dataset", "Unknown"))
    video_id = safe_str(row.get("video_id", ""))
    scenario = safe_str(row.get("scenario", ""))
    video_path = safe_str(row.get("video_path", ""))

    if dataset.lower() == "mulcamfall":
        for text in [scenario, video_id, video_path]:
            chute = infer_mulcamfall_chute(text)
            if chute:
                return f"MulCamFall__{chute}"

        # fallback: remove camera part if possible
        base = re.sub(r"_cam[0-9]+", "", video_id, flags=re.IGNORECASE)
        return f"MulCamFall__{base}"

    # Le2i hiện tại parse ra toàn Fall nên không nên dùng làm binary training chính.
    # Nếu vẫn include, group theo video để tránh leakage.
    if video_id:
        return f"{dataset}__{video_id}"

    return f"{dataset}__row_{row.name}"


def add_group_key(manifest_df: pd.DataFrame) -> pd.DataFrame:
    df = manifest_df.copy()
    df["group_key"] = df.apply(infer_group_key, axis=1)
    return df


# ============================================================
# SPLIT LOGIC
# ============================================================

def validate_binary_labels(df: pd.DataFrame) -> None:
    labels = sorted(df["label"].astype(int).unique().tolist())

    if labels != [0, 1]:
        raise ValueError(
            f"Selected dataset must contain both binary labels [0, 1], got labels={labels}.\n"
            "For current Phase 5, use MulCamFall as the main binary external adaptation dataset. "
            "Le2i currently has only Fall labels after parsing, so it should not be used alone for binary training."
        )


def choose_best_fold(
    df: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    target_ratio: float,
    name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pick the fold whose validation/test size is closest to target_ratio
    and contains both labels when possible.
    """
    n = len(df)
    best_score = None
    best_fold = None

    for train_idx, holdout_idx in folds:
        holdout_df = df.iloc[holdout_idx]
        train_df = df.iloc[train_idx]

        holdout_ratio = len(holdout_df) / max(n, 1)

        holdout_labels = set(holdout_df["label"].astype(int).unique().tolist())
        train_labels = set(train_df["label"].astype(int).unique().tolist())

        has_both_holdout = int(holdout_labels == {0, 1})
        has_both_train = int(train_labels == {0, 1})

        ratio_error = abs(holdout_ratio - target_ratio)

        # Lower score is better.
        # Strong penalty if a split misses a class.
        score = (
            ratio_error
            + (0 if has_both_holdout else 10)
            + (0 if has_both_train else 10)
        )

        if best_score is None or score < best_score:
            best_score = score
            best_fold = (train_idx, holdout_idx)

    if best_fold is None:
        raise RuntimeError(f"Could not choose a valid {name} fold.")

    return best_fold


def make_stratified_group_split(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
) -> pd.DataFrame:
    """
    Create train/val/test split using StratifiedGroupKFold.

    Split ratios are approximate because we preserve groups.
    """
    df = df.copy().reset_index(drop=True)

    y = df["label"].astype(int).to_numpy()
    groups = df["group_key"].astype(str).to_numpy()

    unique_groups = sorted(df["group_key"].unique().tolist())

    if len(unique_groups) < 3:
        raise ValueError(
            f"Need at least 3 groups for train/val/test split, got {len(unique_groups)} groups."
        )

    validate_binary_labels(df)

    # -----------------------------
    # Test split
    # -----------------------------
    n_splits_test = max(2, min(len(unique_groups), round(1.0 / test_size)))

    sgkf_test = StratifiedGroupKFold(
        n_splits=n_splits_test,
        shuffle=True,
        random_state=seed,
    )

    test_folds = list(
        sgkf_test.split(
            X=np.zeros(len(df)),
            y=y,
            groups=groups,
        )
    )

    trainval_idx, test_idx = choose_best_fold(
        df=df,
        folds=test_folds,
        target_ratio=test_size,
        name="test",
    )

    trainval_df = df.iloc[trainval_idx].copy().reset_index(drop=False)
    trainval_df = trainval_df.rename(columns={"index": "original_index"})

    # -----------------------------
    # Val split inside trainval
    # -----------------------------
    val_relative_size = val_size / max(1.0 - test_size, 1e-6)

    trainval_y = trainval_df["label"].astype(int).to_numpy()
    trainval_groups = trainval_df["group_key"].astype(str).to_numpy()
    trainval_unique_groups = sorted(trainval_df["group_key"].unique().tolist())

    n_splits_val = max(2, min(len(trainval_unique_groups), round(1.0 / val_relative_size)))

    sgkf_val = StratifiedGroupKFold(
        n_splits=n_splits_val,
        shuffle=True,
        random_state=seed + 13,
    )

    val_folds_local = list(
        sgkf_val.split(
            X=np.zeros(len(trainval_df)),
            y=trainval_y,
            groups=trainval_groups,
        )
    )

    train_local_idx, val_local_idx = choose_best_fold(
        df=trainval_df,
        folds=val_folds_local,
        target_ratio=val_relative_size,
        name="val",
    )

    train_original_idx = trainval_df.iloc[train_local_idx]["original_index"].to_numpy(dtype=int)
    val_original_idx = trainval_df.iloc[val_local_idx]["original_index"].to_numpy(dtype=int)

    split = np.array(["unused"] * len(df), dtype=object)
    split[train_original_idx] = "train"
    split[val_original_idx] = "val"
    split[test_idx] = "test"

    df["split"] = split

    if (df["split"] == "unused").any():
        raise RuntimeError("Some rows were not assigned to train/val/test.")

    return df


# ============================================================
# LEAKAGE CHECK
# ============================================================

def check_split_leakage(split_df: pd.DataFrame) -> Dict[str, Any]:
    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
    }

    split_names = ["train", "val", "test"]

    group_sets = {}
    seq_sets = {}

    for split_name in split_names:
        part = split_df[split_df["split"] == split_name]

        group_sets[split_name] = set(part["group_key"].astype(str).tolist())
        seq_sets[split_name] = set(part["sequence_key"].astype(str).tolist())

    for i, a in enumerate(split_names):
        for b in split_names[i + 1:]:
            group_overlap = group_sets[a] & group_sets[b]
            seq_overlap = seq_sets[a] & seq_sets[b]

            if group_overlap:
                report["valid"] = False
                report["errors"].append(
                    f"Group leakage between {a} and {b}: {len(group_overlap)} groups"
                )

            if seq_overlap:
                report["valid"] = False
                report["errors"].append(
                    f"Sequence leakage between {a} and {b}: {len(seq_overlap)} sequences"
                )

    for split_name in split_names:
        part = split_df[split_df["split"] == split_name]

        labels = set(part["label"].astype(int).unique().tolist())

        if labels != {0, 1}:
            report["valid"] = False
            report["errors"].append(
                f"Split {split_name} does not contain both labels. labels={sorted(labels)}"
            )

    return report


# ============================================================
# ARRAY SUBSETTING
# ============================================================

def subset_arrays(arrays: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Subset arrays that are sequence-level.

    Keep metadata arrays like quality_feature_columns unchanged.
    """
    out = {}
    n_total = arrays["y_binary"].shape[0]

    for key, value in arrays.items():
        arr = np.asarray(value)

        if arr.shape[0] == n_total:
            out[key] = arr[indices]
        else:
            out[key] = arr

    return out


def save_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


# ============================================================
# SUMMARY
# ============================================================

def split_summary(split_df: pd.DataFrame) -> Dict[str, Any]:
    summary = {}

    for split_name in ["train", "val", "test"]:
        part = split_df[split_df["split"] == split_name].copy()

        summary[split_name] = {
            "num_sequences": int(len(part)),
            "num_groups": int(part["group_key"].nunique()),
            "num_videos": int(part["video_id"].nunique()) if "video_id" in part.columns else 0,
            "dataset_counts": dataset_counts_dict(part),
            "label_counts": label_counts_dict(part),
            "group_keys": sorted(part["group_key"].astype(str).unique().tolist()),
        }

    return summary


# ============================================================
# SAVE OUTPUTS
# ============================================================

def get_split_output_dirs(config: Dict) -> Tuple[Path, Path]:
    external_sequences_dir = cfg_path(
        config,
        config["outputs"]["external_sequences_dir"],
    )

    split_data_dir = external_sequences_dir / "train_val_test_splits"
    split_report_dir = PHASE5_DIR / "outputs" / "splits"

    return split_data_dir, split_report_dir


def save_split_outputs(
    config: Dict,
    arrays: Dict[str, np.ndarray],
    split_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    leakage_report: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, str]:
    split_data_dir, split_report_dir = get_split_output_dirs(config)

    ensure_dir(split_data_dir)
    ensure_dir(split_report_dir)

    all_split_csv = split_data_dir / "external_train_val_test_split.csv"
    train_csv = split_data_dir / "external_train_manifest.csv"
    val_csv = split_data_dir / "external_val_manifest.csv"
    test_csv = split_data_dir / "external_test_manifest.csv"

    train_npz = split_data_dir / "external_train_inputs.npz"
    val_npz = split_data_dir / "external_val_inputs.npz"
    test_npz = split_data_dir / "external_test_inputs.npz"

    report_json = split_report_dir / "06_create_external_train_val_test_split_report.json"

    # Save manifests
    save_csv(split_df, all_split_csv)
    save_csv(split_df[split_df["split"] == "train"], train_csv)
    save_csv(split_df[split_df["split"] == "val"], val_csv)
    save_csv(split_df[split_df["split"] == "test"], test_csv)

    # Save arrays
    for split_name, npz_path in [
        ("train", train_npz),
        ("val", val_npz),
        ("test", test_npz),
    ]:
        part = split_df[split_df["split"] == split_name]
        original_indices = part["original_row_index"].to_numpy(dtype=int)

        sub_arrays = subset_arrays(arrays, original_indices)
        save_npz(npz_path, sub_arrays)

    report = {
        "phase": "Phase 5 - External Dataset Adaptation",
        "step": "06_create_external_train_val_test_split",
        "goal": (
            "Create train/val/test split from the external dataset for fine-tuning. "
            "This replaces zero-shot external evaluation because the model must be adapted to the new data domain."
        ),
        "selected_datasets": args.datasets,
        "default_note": (
            "MulCamFall is used as the main binary external adaptation dataset because it has both Fall and Not_Fall. "
            "Le2i is not recommended for binary training in the current parsed form because it contains only Fall labels."
        ),
        "test_size": float(args.test_size),
        "val_size": float(args.val_size),
        "seed": int(args.seed),
        "num_selected_sequences": int(len(selected_df)),
        "split_summary": split_summary(split_df),
        "leakage_report": leakage_report,
        "outputs": {
            "all_split_csv": str(all_split_csv),
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "test_csv": str(test_csv),
            "train_npz": str(train_npz),
            "val_npz": str(val_npz),
            "test_npz": str(test_npz),
            "report_json": str(report_json),
        },
        "important_rule": (
            "Train, validation, and test are separated by group_key, not random sequence. "
            "This avoids leakage from sequences cut from the same chute/video."
        ),
    }

    save_json(report, report_json)

    return report["outputs"]


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 Step 06 - Create external train/val/test split for fine-tuning."
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
        default=["MulCamFall"],
        help=(
            "Datasets to use for external adaptation. "
            "Default: MulCamFall. Le2i is not recommended for binary training because current parsed labels are Fall-only."
        ),
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Approximate test ratio. Group-preserving split may not match exactly.",
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.20,
        help="Approximate validation ratio. Group-preserving split may not match exactly.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for StratifiedGroupKFold.",
    )

    args = parser.parse_args()

    print("\nPhase 5 - Step 06: Create External Train/Val/Test Split")
    print("=" * 100)
    print("This replaces zero-shot Step 06.")
    print("Goal:")
    print("  Use the new dataset for fine-tuning/adaptation.")
    print("  Split by group_key to avoid leakage from the same chute/video.")
    print("=" * 100)

    config = load_config(args.config)

    print("\n[1/5] Loading Step 05 outputs...")
    arrays, manifest_df, npz_path, fair_manifest_path = load_step05_outputs(config)

    print(f"Loaded NPZ      : {npz_path}")
    print(f"Loaded manifest : {fair_manifest_path}")

    manifest_df = manifest_df.copy()
    manifest_df["original_row_index"] = np.arange(len(manifest_df), dtype=int)

    print_dataframe_summary("Full fair manifest", manifest_df, max_rows=5)

    print("\n[2/5] Selecting datasets for external adaptation...")

    selected_df = manifest_df[manifest_df["dataset"].isin(set(args.datasets))].copy()
    selected_df = selected_df.reset_index(drop=True)

    if selected_df.empty:
        raise RuntimeError(
            f"No sequences found for datasets={args.datasets}"
        )

    selected_df = add_group_key(selected_df)

    validate_binary_labels(selected_df)

    print_dataframe_summary("Selected external adaptation data", selected_df, max_rows=10)

    print("\nSelected dataset summary:")
    print(f"- sequences : {len(selected_df)}")
    print(f"- groups    : {selected_df['group_key'].nunique()}")
    print(f"- videos    : {selected_df['video_id'].nunique()}")
    print(f"- labels    : {label_counts_dict(selected_df)}")
    print(f"- datasets  : {dataset_counts_dict(selected_df)}")

    print("\n[3/5] Creating train/val/test split...")
    split_df = make_stratified_group_split(
        df=selected_df,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    print("\nSplit summary:")
    summary = split_summary(split_df)
    print(json.dumps(summary, ensure_ascii=False, indent=4))

    print("\n[4/5] Checking leakage...")
    leakage_report = check_split_leakage(split_df)

    print_dict("Leakage report", leakage_report)

    if not leakage_report["valid"]:
        raise RuntimeError(
            "Split leakage check failed:\n"
            + json.dumps(leakage_report, ensure_ascii=False, indent=4)
        )

    print("\n[5/5] Saving split files...")
    outputs = save_split_outputs(
        config=config,
        arrays=arrays,
        split_df=split_df,
        selected_df=selected_df,
        leakage_report=leakage_report,
        args=args,
    )

    print_dict("Saved outputs", outputs)

    print("\nDONE: Phase 5 Step 06 completed.")
    print("=" * 100)
    print("Created external train/val/test split for fine-tuning.")
    print("Next step:")
    print("  07_finetune_phase4_quality_concat_external.py")
    print("=" * 100)


if __name__ == "__main__":
    main()