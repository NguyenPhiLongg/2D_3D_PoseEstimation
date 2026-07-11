from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


SCRIPT_PATH = Path(__file__).resolve()
BASELINE_ROOT = SCRIPT_PATH.parents[1]


LABEL_NAMES = {
    0: "Not_Fall",
    1: "Fall",
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def read_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_safe(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e

    raise last_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),

        "sensitivity_fall_recall": float(sensitivity),
        "specificity_not_fall_recall": float(specificity),

        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),

        "fall_precision": float(precision_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "fall_recall": float(recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "fall_f1": float(f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),

        "not_fall_precision": float(precision_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),
        "not_fall_recall": float(recall_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),
        "not_fall_f1": float(f1_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def validate_prediction_df(df: pd.DataFrame, path: Path):
    required = [
        "video_id",
        "true_label",
        "pred_label",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Prediction CSV missing columns in {path}: {missing}")


def evaluate_prediction_csv(split_name: str, prediction_csv: Path) -> Dict:
    if not prediction_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {prediction_csv}")

    df = read_csv_safe(prediction_csv)
    validate_prediction_df(df, prediction_csv)

    y_true = pd.to_numeric(df["true_label"], errors="coerce").fillna(-1).astype(int).to_numpy()
    y_pred = pd.to_numeric(df["pred_label"], errors="coerce").fillna(-1).astype(int).to_numpy()

    valid_mask = np.isin(y_true, [0, 1]) & np.isin(y_pred, [0, 1])

    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    metrics = compute_metrics(y_true, y_pred)

    num_videos = int(len(y_true))
    num_true_fall = int((y_true == 1).sum())
    num_true_not_fall = int((y_true == 0).sum())
    num_pred_fall = int((y_pred == 1).sum())
    num_pred_not_fall = int((y_pred == 0).sum())

    out = {
        "split": split_name,
        "prediction_csv": str(prediction_csv),

        "num_videos": num_videos,
        "num_true_fall": num_true_fall,
        "num_true_not_fall": num_true_not_fall,
        "num_pred_fall": num_pred_fall,
        "num_pred_not_fall": num_pred_not_fall,

        **metrics,
    }

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Chen 2020-style three-threshold method on train/val/test predictions."
    )

    parser.add_argument(
        "--train-predictions",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "predictions" / "06_train_video_predictions.csv"),
    )

    parser.add_argument(
        "--val-predictions",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "predictions" / "06_val_video_predictions.csv"),
    )

    parser.add_argument(
        "--test-predictions",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "predictions" / "06_test_video_predictions.csv"),
    )

    parser.add_argument(
        "--threshold-json",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "metrics" / "05_best_thresholds.json"),
    )

    parser.add_argument(
        "--apply-report",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "reports" / "06_apply_three_conditions_report.json"),
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

    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    reports_dir = Path(args.reports_dir)

    ensure_dir(metrics_dir)
    ensure_dir(reports_dir)

    threshold_json = Path(args.threshold_json)
    apply_report_path = Path(args.apply_report)

    thresholds = read_json(threshold_json) if threshold_json.exists() else {}
    apply_report = read_json(apply_report_path) if apply_report_path.exists() else {}

    split_items = [
        ("train", Path(args.train_predictions)),
        ("val", Path(args.val_predictions)),
        ("test", Path(args.test_predictions)),
    ]

    print("=" * 100)
    print("FILE 07 - Evaluate Chen 2020-style threshold method")
    print("=" * 100)
    print(f"Threshold JSON: {threshold_json}")
    print(f"Train pred:     {args.train_predictions}")
    print(f"Val pred:       {args.val_predictions}")
    print(f"Test pred:      {args.test_predictions}")
    print("=" * 100)

    results = []

    for split_name, prediction_csv in split_items:
        result = evaluate_prediction_csv(split_name, prediction_csv)
        results.append(result)

        print(
            f"{split_name.upper()} | "
            f"videos={result['num_videos']} | "
            f"acc={result['accuracy']:.4f} | "
            f"macro_f1={result['macro_f1']:.4f} | "
            f"fall_recall={result['fall_recall']:.4f} | "
            f"specificity={result['specificity_not_fall_recall']:.4f} | "
            f"TN={result['tn']} FP={result['fp']} FN={result['fn']} TP={result['tp']}"
        )

    metrics_df = pd.DataFrame(results)

    metrics_csv = metrics_dir / "07_threshold_method_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    test_row = metrics_df[metrics_df["split"] == "test"].iloc[0].to_dict()

    report = {
        "status": "completed",
        "pipeline_note": "Final evaluation of the Chen et al. 2020-style three-threshold fall detection method. The main test result is computed on the held-out test split.",
        "thresholds": thresholds,
        "apply_report": str(apply_report_path),
        "metrics_csv": str(metrics_csv),
        "results": results,
        "test_result": {
            "num_videos": int(test_row["num_videos"]),
            "accuracy": float(test_row["accuracy"]),
            "sensitivity_fall_recall": float(test_row["sensitivity_fall_recall"]),
            "specificity_not_fall_recall": float(test_row["specificity_not_fall_recall"]),
            "macro_f1": float(test_row["macro_f1"]),
            "fall_precision": float(test_row["fall_precision"]),
            "fall_recall": float(test_row["fall_recall"]),
            "fall_f1": float(test_row["fall_f1"]),
            "not_fall_precision": float(test_row["not_fall_precision"]),
            "not_fall_recall": float(test_row["not_fall_recall"]),
            "not_fall_f1": float(test_row["not_fall_f1"]),
            "tn": int(test_row["tn"]),
            "fp": int(test_row["fp"]),
            "fn": int(test_row["fn"]),
            "tp": int(test_row["tp"]),
        },
        "source_files": {
            "train_predictions": str(args.train_predictions),
            "val_predictions": str(args.val_predictions),
            "test_predictions": str(args.test_predictions),
        },
        "file06_summary": apply_report.get("splits", {}),
    }

    report_path = reports_dir / "07_evaluate_threshold_method_report.json"
    save_json(report, report_path)

    print("=" * 100)
    print("FILE 07 completed.")
    print("=" * 100)
    print("Metrics table:")
    print(metrics_df.to_string(index=False))
    print("-" * 100)
    print("Main TEST result:")
    print(report["test_result"])
    print("-" * 100)
    print(f"Metrics CSV: {metrics_csv}")
    print(f"Report:      {report_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
