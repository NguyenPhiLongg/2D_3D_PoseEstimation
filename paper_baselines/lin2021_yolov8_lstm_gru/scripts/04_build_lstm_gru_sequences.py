from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
BASELINE_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = BASELINE_ROOT.parents[1]


# ======================================================================================
# FILE 04: BUILD FIXED-LENGTH SKELETON SEQUENCES FOR RNN/LSTM/GRU
# ======================================================================================
#
# This file follows the temporal modeling input preparation of the Lin-style pipeline.
#
# Input from File 03:
#   normalized skeleton CSV
#   each frame has 30 features:
#       15 joints x 2 coordinates
#
# Output:
#   X: (num_sequences, sequence_length, 30)
#   y: (num_sequences,)
#
# Default:
#   sequence_length = 100 frames
#   stride = 30 frames
#
# This file does NOT train the model.
# It only builds fixed-length skeleton sequences for RNN/LSTM/GRU.
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


def read_csv_safe(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "latin1"]

    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e

    raise last_error


def get_feature_columns() -> List[str]:
    feature_cols = []

    for j in range(15):
        feature_cols.extend([
            f"joint_{j}_x",
            f"joint_{j}_y",
        ])

    return feature_cols


def validate_processed_df(df: pd.DataFrame, csv_path: Path):
    required = [
        "video_id",
        "frame_idx",
        "label",
        "label_name",
    ] + get_feature_columns()

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required processed columns in {csv_path}. Missing examples: {missing[:20]}"
        )


def get_label_from_df(df: pd.DataFrame, csv_path: Path) -> int:
    labels = sorted(
        pd.to_numeric(df["label"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if len(labels) == 0:
        raise ValueError(f"No valid label found in {csv_path}")

    if len(labels) > 1:
        raise ValueError(f"Multiple labels found in one video {csv_path}: {labels}")

    label = int(labels[0])

    if label not in [0, 1]:
        raise ValueError(f"Invalid binary label in {csv_path}: {label}")

    return label


def extract_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    feature_cols = get_feature_columns()

    X = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    return X


def pad_sequence_to_length(sequence: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, int]:
    """
    If the video has fewer than sequence_length frames, pad by repeating the last frame.
    This keeps short videos usable while avoiding all-zero padding.
    """
    T, D = sequence.shape

    if T >= sequence_length:
        return sequence[:sequence_length], sequence_length

    if T == 0:
        return np.zeros((sequence_length, D), dtype=np.float32), 0

    pad_count = sequence_length - T
    last_frame = sequence[-1:].copy()
    pad = np.repeat(last_frame, pad_count, axis=0)

    out = np.concatenate([sequence, pad], axis=0).astype(np.float32)

    return out, T


def build_windows(
    sequence: np.ndarray,
    sequence_length: int,
    stride: int,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Build fixed-length windows.

    For videos longer than sequence_length:
        sliding windows with stride
        final window is anchored at the last possible start to cover the video ending

    For videos shorter than sequence_length:
        one padded window
    """
    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.ndim != 2:
        raise ValueError(f"Expected sequence shape (T, D), got {sequence.shape}")

    T, D = sequence.shape

    windows = []
    meta = []

    if T == 0:
        window = np.zeros((sequence_length, D), dtype=np.float32)
        windows.append(window)

        meta.append({
            "start_frame_local": 0,
            "end_frame_local": 0,
            "valid_length": 0,
            "is_padded": 1,
        })

        return np.stack(windows, axis=0), meta

    if T <= sequence_length:
        window, valid_length = pad_sequence_to_length(sequence, sequence_length)

        windows.append(window)

        meta.append({
            "start_frame_local": 0,
            "end_frame_local": T - 1,
            "valid_length": int(valid_length),
            "is_padded": int(T < sequence_length),
        })

        return np.stack(windows, axis=0), meta

    starts = list(range(0, T - sequence_length + 1, stride))

    last_possible_start = T - sequence_length

    if starts[-1] != last_possible_start:
        starts.append(last_possible_start)

    for start in starts:
        end = start + sequence_length

        window = sequence[start:end]

        if window.shape[0] != sequence_length:
            raise RuntimeError(
                f"Unexpected window length {window.shape[0]} at start={start}, T={T}"
            )

        windows.append(window.astype(np.float32))

        meta.append({
            "start_frame_local": int(start),
            "end_frame_local": int(end - 1),
            "valid_length": int(sequence_length),
            "is_padded": 0,
        })

    return np.stack(windows, axis=0), meta


def process_one_csv(
    csv_path: Path,
    sequence_length: int,
    stride: int,
) -> Tuple[np.ndarray, List[Dict]]:
    df = read_csv_safe(csv_path)

    validate_processed_df(df, csv_path)

    if len(df) == 0:
        raise ValueError(f"Empty processed CSV: {csv_path}")

    df = df.sort_values("frame_idx").reset_index(drop=True)

    video_id = str(df["video_id"].iloc[0])
    label = get_label_from_df(df, csv_path)
    label_name = label_to_name(label)

    frame_indices = pd.to_numeric(df["frame_idx"], errors="coerce").fillna(0).astype(int).to_numpy()

    features = extract_feature_matrix(df)

    windows, window_meta = build_windows(
        sequence=features,
        sequence_length=sequence_length,
        stride=stride,
    )

    manifest_rows = []

    for seq_idx, meta in enumerate(window_meta):
        start_local = int(meta["start_frame_local"])
        end_local = int(meta["end_frame_local"])

        if len(frame_indices) > 0:
            start_frame = int(frame_indices[min(start_local, len(frame_indices) - 1)])
            end_frame = int(frame_indices[min(end_local, len(frame_indices) - 1)])
        else:
            start_frame = 0
            end_frame = 0

        window = windows[seq_idx]

        is_all_zero = int(np.all(window == 0.0))
        zero_ratio = float((window == 0.0).mean())

        sequence_id = f"{make_safe_name(video_id)}__seq_{seq_idx:04d}"

        manifest_rows.append({
            "sequence_id": sequence_id,
            "video_id": make_safe_name(video_id),
            "source_csv": str(csv_path),
            "label": int(label),
            "label_name": label_name,
            "sequence_index": int(seq_idx),
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "start_frame_local": int(start_local),
            "end_frame_local": int(end_local),
            "valid_length": int(meta["valid_length"]),
            "sequence_length": int(sequence_length),
            "feature_dim": int(features.shape[1]),
            "num_original_frames": int(features.shape[0]),
            "is_padded": int(meta["is_padded"]),
            "is_all_zero": int(is_all_zero),
            "zero_ratio": float(zero_ratio),
        })

    return windows, manifest_rows


def find_processed_csv_files(input_dir: Path) -> List[Path]:
    csv_files = sorted([
        p for p in input_dir.rglob("*_processed.csv")
    ])

    return csv_files


def main():
    parser = argparse.ArgumentParser(
        description="Build fixed-length skeleton sequences for Lin-style YOLOv8 RNN/LSTM/GRU baseline."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(BASELINE_ROOT / "data" / "processed"),
        help="Input folder from File 03.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BASELINE_ROOT / "data" / "sequences"),
        help="Output folder for sequence NPZ and manifest.",
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=100,
        help="Number of frames per sequence.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=30,
        help="Sliding-window stride.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    reports_dir = BASELINE_ROOT / "outputs" / "reports"

    ensure_dir(input_dir)
    ensure_dir(output_dir)
    ensure_dir(reports_dir)

    csv_files = find_processed_csv_files(input_dir)

    print("=" * 100)
    print("FILE 04 - Build fixed-length skeleton sequences")
    print("=" * 100)
    print(f"Input dir:        {input_dir}")
    print(f"Output dir:       {output_dir}")
    print(f"Input CSV files:  {len(csv_files)}")
    print(f"Sequence length:  {args.sequence_length}")
    print(f"Stride:           {args.stride}")
    print("=" * 100)

    if len(csv_files) == 0:
        report = {
            "status": "no_input_csv",
            "input_dir": str(input_dir),
            "message": "No *_processed.csv files found. Run file 03 first.",
        }

        report_path = reports_dir / "04_build_lstm_gru_sequences_report.json"
        save_json(report, report_path)

        print("No input CSV files found.")
        print(f"Report: {report_path}")
        return

    all_windows = []
    all_manifest_rows = []
    failed = []

    for idx, csv_path in enumerate(csv_files, start=1):
        print(f"[{idx}/{len(csv_files)}] Building sequences: {csv_path.name}")

        try:
            windows, manifest_rows = process_one_csv(
                csv_path=csv_path,
                sequence_length=args.sequence_length,
                stride=args.stride,
            )

            all_windows.append(windows)
            all_manifest_rows.extend(manifest_rows)

            print(f"  OK -> windows={windows.shape[0]}")

        except Exception as e:
            failed.append({
                "input_csv": str(csv_path),
                "error": repr(e),
            })

            print(f"  FAILED: {repr(e)}")

    if len(all_windows) == 0:
        raise RuntimeError("No valid sequence windows were created.")

    X = np.concatenate(all_windows, axis=0).astype(np.float32)

    manifest_df = pd.DataFrame(all_manifest_rows)

    if len(manifest_df) != X.shape[0]:
        raise RuntimeError(
            f"Manifest rows do not match X. manifest={len(manifest_df)}, X={X.shape[0]}"
        )

    y = manifest_df["label"].astype(int).to_numpy(dtype=np.int64)
    sequence_ids = manifest_df["sequence_id"].astype(str).to_numpy()
    video_ids = manifest_df["video_id"].astype(str).to_numpy()

    npz_path = output_dir / "lin2021_yolov8_lstm_gru_sequences.npz"
    manifest_path = output_dir / "lin2021_yolov8_sequence_manifest.csv"

    np.savez_compressed(
        npz_path,
        X=X,
        y=y,
        sequence_ids=sequence_ids,
        video_ids=video_ids,
    )

    manifest_df.to_csv(manifest_path, index=False)

    label_counts = manifest_df["label_name"].value_counts().to_dict()
    video_counts = manifest_df.groupby("label_name")["video_id"].nunique().to_dict()

    all_zero_count = int(manifest_df["is_all_zero"].sum())
    padded_count = int(manifest_df["is_padded"].sum())

    report = {
        "status": "completed",
        "pipeline_note": "This step builds fixed-length skeleton sequences for RNN/LSTM/GRU, following the temporal modeling stage of the Lin-style pipeline.",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_input_csv": len(csv_files),
        "num_success_csv": int(len(csv_files) - len(failed)),
        "num_failed_csv": len(failed),
        "sequence_length": int(args.sequence_length),
        "stride": int(args.stride),
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "feature_dim": int(X.shape[2]),
        "num_sequences": int(X.shape[0]),
        "num_videos": int(manifest_df["video_id"].nunique()),
        "label_counts_by_sequence": {str(k): int(v) for k, v in label_counts.items()},
        "label_counts_by_video": {str(k): int(v) for k, v in video_counts.items()},
        "all_zero_sequences": all_zero_count,
        "padded_sequences": padded_count,
        "npz_path": str(npz_path),
        "manifest_path": str(manifest_path),
        "failed": failed,
    }

    report_path = reports_dir / "04_build_lstm_gru_sequences_report.json"
    save_json(report, report_path)

    print("=" * 100)
    print("FILE 04 completed.")
    print("=" * 100)
    print(f"X shape:              {X.shape}")
    print(f"y shape:              {y.shape}")
    print(f"Feature dim:          {X.shape[2]}")
    print(f"Videos:               {manifest_df['video_id'].nunique()}")
    print(f"Sequences:            {len(manifest_df)}")
    print(f"Label counts seq:     {label_counts}")
    print(f"Label counts video:   {video_counts}")
    print(f"All-zero sequences:   {all_zero_count}")
    print(f"Padded sequences:     {padded_count}")
    print(f"Failed CSVs:          {len(failed)}")
    print(f"NPZ:                  {npz_path}")
    print(f"Manifest:             {manifest_path}")
    print(f"Report:               {report_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
