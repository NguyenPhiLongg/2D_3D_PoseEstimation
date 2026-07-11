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

from utils.io_utils import ensure_dir, save_json
from utils.metrics_utils import compute_binary_metrics


# ======================================================================================
# FILE 06: EVALUATE CNN, LSTM, AND CONVLSTM
# ======================================================================================
#
# This file evaluates the models trained in file 05.
#
# It reports:
#   1. sequence-level metrics
#      - each 30-frame sequence is one sample
#
#   2. video-level majority-vote metrics
#      - each video has many sequence predictions
#      - final video prediction = most common sequence prediction
#
#   3. video-level mean-probability metrics
#      - average prob_fall across all sequences of the same video
#      - final video prediction = 1 if mean_prob_fall >= threshold
#
# Why video-level?
#   Other baselines such as Chen2020-style threshold and Lin2021-style RNN/GRU
#   are mainly compared at video level. Therefore, this file makes the Yadav-style
#   baseline comparable with previous baselines.
#
# Output:
#   outputs/metrics/06_evaluate_cnn_lstm_convlstm_metrics.csv
#   outputs/predictions/06_video_predictions_*.csv
#   outputs/reports/06_evaluate_cnn_lstm_convlstm_report.json
# ======================================================================================


MODEL_NAMES = ["cnn", "lstm", "convlstm"]
SPLITS = ["train", "val", "test"]


def label_to_name(label: int) -> str:
    return "Fall" if int(label) == 1 else "Not_Fall"


def prediction_path(predictions_dir: Path, split: str, model_name: str) -> Path:
    return predictions_dir / f"05_predictions_{split}_{model_name}.csv"


def load_prediction_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")

    df = pd.read_csv(path)

    required = [
        "sequence_index",
        "video_id",
        "split",
        "y_true",
        "y_pred",
        "prob_fall",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = df["y_pred"].astype(int)
    df["prob_fall"] = df["prob_fall"].astype(float)

    return df


def sequence_level_metrics(df: pd.DataFrame) -> Dict:
    metrics = compute_binary_metrics(
        df["y_true"].to_numpy(dtype=int),
        df["y_pred"].to_numpy(dtype=int),
    )

    metrics["num_sequences"] = int(len(df))
    metrics["num_videos"] = int(df["video_id"].nunique())

    return metrics


def majority_vote(values: pd.Series) -> int:
    counts = values.astype(int).value_counts()

    if len(counts) == 0:
        return 0

    if len(counts) == 1:
        return int(counts.index[0])

    # Tie breaker:
    # If Fall and Not_Fall have equal votes, choose Fall for safety.
    if counts.loc[0] == counts.loc[1]:
        return 1

    return int(counts.idxmax())


def aggregate_video_majority_vote(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for video_id, g in df.groupby("video_id"):
        y_true_values = g["y_true"].astype(int).unique()

        if len(y_true_values) != 1:
            raise ValueError(f"Video {video_id} has inconsistent y_true values: {y_true_values}")

        y_true = int(y_true_values[0])
        y_pred_video = majority_vote(g["y_pred"])

        rows.append({
            "video_id": video_id,
            "y_true": y_true,
            "y_pred": int(y_pred_video),
            "prob_fall_mean": float(g["prob_fall"].mean()),
            "prob_fall_max": float(g["prob_fall"].max()),
            "num_sequences": int(len(g)),
            "aggregation": "majority_vote",
            "true_label_name": label_to_name(y_true),
            "pred_label_name": label_to_name(y_pred_video),
        })

    return pd.DataFrame(rows)


def aggregate_video_mean_prob(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    rows = []

    for video_id, g in df.groupby("video_id"):
        y_true_values = g["y_true"].astype(int).unique()

        if len(y_true_values) != 1:
            raise ValueError(f"Video {video_id} has inconsistent y_true values: {y_true_values}")

        y_true = int(y_true_values[0])
        prob_mean = float(g["prob_fall"].mean())
        prob_max = float(g["prob_fall"].max())

        y_pred_video = 1 if prob_mean >= threshold else 0

        rows.append({
            "video_id": video_id,
            "y_true": y_true,
            "y_pred": int(y_pred_video),
            "prob_fall_mean": prob_mean,
            "prob_fall_max": prob_max,
            "num_sequences": int(len(g)),
            "threshold": float(threshold),
            "aggregation": "mean_prob",
            "true_label_name": label_to_name(y_true),
            "pred_label_name": label_to_name(y_pred_video),
        })

    return pd.DataFrame(rows)


def video_level_metrics(video_df: pd.DataFrame) -> Dict:
    metrics = compute_binary_metrics(
        video_df["y_true"].to_numpy(dtype=int),
        video_df["y_pred"].to_numpy(dtype=int),
    )

    metrics["num_videos"] = int(len(video_df))
    metrics["num_sequences"] = int(video_df["num_sequences"].sum())

    return metrics


def find_best_threshold_on_val(
    val_df: pd.DataFrame,
    thresholds: List[float],
    metric_name: str = "macro_f1",
) -> Tuple[float, Dict]:
    best_threshold = 0.5
    best_metrics = None
    best_score = -1.0

    for threshold in thresholds:
        video_df = aggregate_video_mean_prob(val_df, threshold=threshold)
        metrics = video_level_metrics(video_df)

        score = float(metrics[metric_name])

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def add_metric_row(
    rows: List[Dict],
    model_name: str,
    split: str,
    level: str,
    aggregation: str,
    threshold: float | None,
    metrics: Dict,
) -> None:
    row = {
        "model_name": model_name,
        "split": split,
        "level": level,
        "aggregation": aggregation,
        "threshold": threshold,

        "accuracy": metrics.get("accuracy"),
        "macro_precision": metrics.get("macro_precision"),
        "macro_recall": metrics.get("macro_recall"),
        "macro_f1": metrics.get("macro_f1"),

        "fall_precision": metrics.get("fall_precision"),
        "fall_recall": metrics.get("fall_recall"),
        "fall_f1": metrics.get("fall_f1"),

        "not_fall_precision": metrics.get("not_fall_precision"),
        "not_fall_recall": metrics.get("not_fall_recall"),
        "not_fall_f1": metrics.get("not_fall_f1"),

        "sensitivity_fall_recall": metrics.get("sensitivity_fall_recall"),
        "specificity_not_fall_recall": metrics.get("specificity_not_fall_recall"),

        "tn": metrics.get("tn"),
        "fp": metrics.get("fp"),
        "fn": metrics.get("fn"),
        "tp": metrics.get("tp"),

        "num_sequences": metrics.get("num_sequences"),
        "num_videos": metrics.get("num_videos"),
    }

    rows.append(row)


def save_confusion_matrix_csv(metrics_row: Dict, output_path: Path) -> None:
    cm_df = pd.DataFrame(
        [
            {
                "actual": "Not_Fall",
                "pred_Not_Fall": int(metrics_row["tn"]),
                "pred_Fall": int(metrics_row["fp"]),
            },
            {
                "actual": "Fall",
                "pred_Not_Fall": int(metrics_row["fn"]),
                "pred_Fall": int(metrics_row["tp"]),
            },
        ]
    )

    ensure_dir(output_path.parent)
    cm_df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Yadav2021-style CNN/LSTM/ConvLSTM at sequence-level and video-level."
    )

    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "predictions"),
    )

    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "metrics"),
    )

    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "reports"),
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=MODEL_NAMES,
        choices=MODEL_NAMES,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Default mean-probability threshold.",
    )

    parser.add_argument(
        "--tune-threshold-on-val",
        action="store_true",
        help="Tune mean-probability threshold on validation split and apply it to train/val/test.",
    )

    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    metrics_dir = Path(args.metrics_dir)
    reports_dir = Path(args.reports_dir)

    ensure_dir(metrics_dir)
    ensure_dir(reports_dir)
    ensure_dir(predictions_dir)

    print("=" * 100)
    print("FILE 06 - Evaluate CNN, LSTM, and Sequential ConvLSTM")
    print("=" * 100)
    print(f"Predictions dir:       {predictions_dir}")
    print(f"Metrics dir:           {metrics_dir}")
    print(f"Reports dir:           {reports_dir}")
    print(f"Models:                {args.models}")
    print(f"Default threshold:     {args.threshold}")
    print(f"Tune threshold on val: {args.tune_threshold_on_val}")
    print("=" * 100)

    all_metric_rows = []
    report_models = []

    thresholds = [round(x, 2) for x in np.arange(0.05, 0.96, 0.01)]

    for model_name in args.models:
        print("=" * 100)
        print(f"Evaluating model: {model_name.upper()}")
        print("=" * 100)

        split_predictions = {}

        for split in SPLITS:
            path = prediction_path(predictions_dir, split, model_name)
            df = load_prediction_csv(path)
            split_predictions[split] = df

            print(
                f"{model_name.upper()} | {split}: "
                f"sequences={len(df)} | videos={df['video_id'].nunique()}"
            )

        # Sequence-level metrics
        for split in SPLITS:
            seq_df = split_predictions[split]
            metrics = sequence_level_metrics(seq_df)

            add_metric_row(
                rows=all_metric_rows,
                model_name=model_name,
                split=split,
                level="sequence",
                aggregation="none",
                threshold=None,
                metrics=metrics,
            )

        # Video-level majority-vote metrics
        for split in SPLITS:
            seq_df = split_predictions[split]
            video_df = aggregate_video_majority_vote(seq_df)

            out_video_path = predictions_dir / f"06_video_predictions_{split}_{model_name}_majority_vote.csv"
            video_df.to_csv(out_video_path, index=False)

            metrics = video_level_metrics(video_df)

            add_metric_row(
                rows=all_metric_rows,
                model_name=model_name,
                split=split,
                level="video",
                aggregation="majority_vote",
                threshold=None,
                metrics=metrics,
            )

        # Video-level mean-prob metrics
        tuned_threshold = float(args.threshold)
        tuned_val_metrics = None

        if args.tune_threshold_on_val:
            tuned_threshold, tuned_val_metrics = find_best_threshold_on_val(
                val_df=split_predictions["val"],
                thresholds=thresholds,
                metric_name="macro_f1",
            )

            print(
                f"{model_name.upper()} best val threshold: "
                f"{tuned_threshold:.2f} | val_macro_f1={tuned_val_metrics['macro_f1']:.4f}"
            )

        for split in SPLITS:
            seq_df = split_predictions[split]
            video_df = aggregate_video_mean_prob(seq_df, threshold=tuned_threshold)

            out_video_path = predictions_dir / f"06_video_predictions_{split}_{model_name}_mean_prob.csv"
            video_df.to_csv(out_video_path, index=False)

            metrics = video_level_metrics(video_df)

            add_metric_row(
                rows=all_metric_rows,
                model_name=model_name,
                split=split,
                level="video",
                aggregation="mean_prob",
                threshold=tuned_threshold,
                metrics=metrics,
            )

        report_models.append({
            "model_name": model_name,
            "tuned_threshold": tuned_threshold,
            "tuned_threshold_source": "val_macro_f1" if args.tune_threshold_on_val else "fixed_default",
        })

    metrics_df = pd.DataFrame(all_metric_rows)

    # Sort for readability
    metrics_df = metrics_df.sort_values(
        ["split", "level", "aggregation", "macro_f1", "accuracy"],
        ascending=[True, True, True, False, False],
    ).reset_index(drop=True)

    metrics_csv = metrics_dir / "06_evaluate_cnn_lstm_convlstm_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # Save confusion matrices for test rows only.
    cm_dir = metrics_dir / "confusion_matrices_06"
    ensure_dir(cm_dir)

    test_rows = metrics_df[metrics_df["split"] == "test"].copy()

    for _, row in test_rows.iterrows():
        cm_path = cm_dir / f"cm_test_{row['model_name']}_{row['level']}_{row['aggregation']}.csv"
        save_confusion_matrix_csv(row.to_dict(), cm_path)

    # Pick best rows for summary.
    test_video_mean = metrics_df[
        (metrics_df["split"] == "test")
        & (metrics_df["level"] == "video")
        & (metrics_df["aggregation"] == "mean_prob")
    ].copy()

    if len(test_video_mean) > 0:
        best_test_video_mean = test_video_mean.sort_values(
            ["macro_f1", "accuracy"],
            ascending=[False, False],
        ).iloc[0].to_dict()
    else:
        best_test_video_mean = {}

    test_sequence = metrics_df[
        (metrics_df["split"] == "test")
        & (metrics_df["level"] == "sequence")
    ].copy()

    if len(test_sequence) > 0:
        best_test_sequence = test_sequence.sort_values(
            ["macro_f1", "accuracy"],
            ascending=[False, False],
        ).iloc[0].to_dict()
    else:
        best_test_sequence = {}

    report = {
        "status": "completed",
        "pipeline_note": "Evaluate CNN, LSTM, and sequential ConvLSTM at both sequence-level and video-level. Video-level mean-probability aggregation is most comparable to other video-level baselines.",
        "predictions_dir": str(predictions_dir),
        "metrics_dir": str(metrics_dir),
        "metrics_csv": str(metrics_csv),
        "confusion_matrix_dir": str(cm_dir),
        "models": report_models,
        "default_threshold": float(args.threshold),
        "tune_threshold_on_val": bool(args.tune_threshold_on_val),
        "best_test_video_mean_prob_row": best_test_video_mean,
        "best_test_sequence_row": best_test_sequence,
        "test_metrics_preview": test_rows.to_dict(orient="records"),
    }

    report_path = reports_dir / "06_evaluate_cnn_lstm_convlstm_report.json"
    save_json(report, report_path)

    print("=" * 100)
    print("FILE 06 completed.")
    print("=" * 100)
    print(f"Metrics CSV: {metrics_csv}")
    print(f"Report:      {report_path}")
    print(f"CM dir:      {cm_dir}")
    print("=" * 100)

    print("TEST RESULTS:")
    display_cols = [
        "model_name",
        "level",
        "aggregation",
        "threshold",
        "accuracy",
        "macro_f1",
        "fall_recall",
        "fall_f1",
        "specificity_not_fall_recall",
        "tn",
        "fp",
        "fn",
        "tp",
        "num_sequences",
        "num_videos",
    ]

    print(test_rows[display_cols].to_string(index=False))
    print("=" * 100)

    if best_test_video_mean:
        print("BEST TEST VIDEO MEAN-PROB:")
        for key in [
            "model_name",
            "accuracy",
            "macro_f1",
            "fall_recall",
            "fall_f1",
            "specificity_not_fall_recall",
            "tn",
            "fp",
            "fn",
            "tp",
            "num_videos",
        ]:
            print(f"{key}: {best_test_video_mean.get(key)}")
        print("=" * 100)


if __name__ == "__main__":
    main()
