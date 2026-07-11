from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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


# ======================================================================================
# FILE 05: TUNE THRESHOLDS ON VALIDATION SET
# ======================================================================================
#
# Chen et al. 2020-style rule:
#
#   Condition 1: hip-center descent velocity is greater than a critical velocity
#   Condition 2: body centerline angle with the ground is less than 45 degrees
#   Condition 3: width / height ratio of external rectangle is greater than 1
#
# In this adapted YOLOv8 version:
#   - angle threshold defaults to 45.0 degrees, following the paper
#   - width/height ratio threshold defaults to 1.0, following the paper
#   - hip velocity threshold is tuned on validation set because YOLOv8 gives pixel
#     coordinates and the correct critical velocity depends on video resolution/FPS
#
# This file does NOT evaluate on test.
# Test evaluation will be done in File 07.
# ======================================================================================


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def read_csv_safe(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e

    raise last_error


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip() != ""]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip() != ""]


def label_to_name(label: int) -> str:
    if int(label) == 1:
        return "Fall"
    if int(label) == 0:
        return "Not_Fall"
    return "Unknown"


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


def load_validation_videos(val_index_csv: Path) -> List[Dict]:
    if not val_index_csv.exists():
        raise FileNotFoundError(f"Validation index CSV not found: {val_index_csv}")

    index_df = read_csv_safe(val_index_csv)

    required = ["video_id", "label", "label_name", "output_csv"]

    missing = [col for col in required if col not in index_df.columns]

    if missing:
        raise ValueError(f"Validation index missing columns: {missing}")

    videos = []

    for idx, row in index_df.iterrows():
        csv_path = Path(str(row["output_csv"]))

        if not csv_path.exists():
            raise FileNotFoundError(f"Rule feature CSV not found: {csv_path}")

        df = read_csv_safe(csv_path)

        videos.append({
            "video_id": str(row["video_id"]),
            "label": int(row["label"]),
            "label_name": str(row["label_name"]),
            "csv_path": str(csv_path),
            "df": df,
        })

    return videos


def build_velocity_candidates(
    videos: List[Dict],
    num_candidates: int,
    include_zero: bool = True,
) -> List[float]:
    max_values = []

    for item in videos:
        df = item["df"]

        if "hip_descent_velocity" not in df.columns:
            raise ValueError(f"hip_descent_velocity column not found in {item['csv_path']}")

        values = pd.to_numeric(df["hip_descent_velocity"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        max_values.append(float(np.max(values)))

    max_values = np.asarray(max_values, dtype=np.float32)

    upper = float(np.percentile(max_values, 99.0))

    if upper <= 0:
        candidates = [0.0]
    else:
        candidates = np.linspace(0.0, upper, num_candidates).tolist()

    # Add meaningful percentiles to make search more stable.
    for p in [50, 60, 70, 75, 80, 85, 90, 95]:
        candidates.append(float(np.percentile(max_values, p)))

    if include_zero:
        candidates.append(0.0)

    candidates = sorted(set([round(float(x), 4) for x in candidates if float(x) >= 0.0]))

    return candidates


def predict_one_video(
    df: pd.DataFrame,
    velocity_threshold: float,
    angle_threshold: float,
    ratio_threshold: float,
    min_event_frames: int,
) -> Dict:
    required = [
        "hip_descent_velocity",
        "body_centerline_angle_degrees",
        "width_height_ratio",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Feature CSV missing required columns: {missing}")

    velocity = pd.to_numeric(df["hip_descent_velocity"], errors="coerce").fillna(0.0)
    angle = pd.to_numeric(df["body_centerline_angle_degrees"], errors="coerce").fillna(999.0)
    ratio = pd.to_numeric(df["width_height_ratio"], errors="coerce").fillna(0.0)

    if "all_rule_features_valid" in df.columns:
        valid = pd.to_numeric(df["all_rule_features_valid"], errors="coerce").fillna(0).astype(int) == 1
    else:
        valid = pd.Series([True] * len(df))

    condition_velocity = velocity >= velocity_threshold
    condition_angle = angle < angle_threshold
    condition_ratio = ratio > ratio_threshold

    event_mask = valid & condition_velocity & condition_angle & condition_ratio

    event_count = int(event_mask.sum())
    pred_label = 1 if event_count >= min_event_frames else 0

    if event_count > 0:
        first_event_idx = int(np.where(event_mask.to_numpy())[0][0])
        first_event_frame = int(df.iloc[first_event_idx]["frame_idx"])
    else:
        first_event_idx = -1
        first_event_frame = -1

    return {
        "pred_label": int(pred_label),
        "event_count": int(event_count),
        "first_event_idx": int(first_event_idx),
        "first_event_frame": int(first_event_frame),
        "max_hip_descent_velocity": float(velocity.max()),
        "min_body_angle": float(angle.min()),
        "max_width_height_ratio": float(ratio.max()),
        "condition_velocity_frames": int(condition_velocity.sum()),
        "condition_angle_frames": int(condition_angle.sum()),
        "condition_ratio_frames": int(condition_ratio.sum()),
    }


def evaluate_thresholds(
    videos: List[Dict],
    velocity_threshold: float,
    angle_threshold: float,
    ratio_threshold: float,
    min_event_frames: int,
) -> Tuple[Dict, pd.DataFrame]:
    rows = []

    for item in videos:
        pred = predict_one_video(
            df=item["df"],
            velocity_threshold=velocity_threshold,
            angle_threshold=angle_threshold,
            ratio_threshold=ratio_threshold,
            min_event_frames=min_event_frames,
        )

        true_label = int(item["label"])
        pred_label = int(pred["pred_label"])

        rows.append({
            "video_id": item["video_id"],
            "true_label": true_label,
            "true_label_name": label_to_name(true_label),
            "pred_label": pred_label,
            "pred_label_name": label_to_name(pred_label),
            "correct": int(true_label == pred_label),
            "csv_path": item["csv_path"],
            **pred,
        })

    pred_df = pd.DataFrame(rows)

    y_true = pred_df["true_label"].astype(int).to_numpy()
    y_pred = pred_df["pred_label"].astype(int).to_numpy()

    metrics = compute_metrics(y_true, y_pred)

    return metrics, pred_df


def main():
    parser = argparse.ArgumentParser(
        description="Tune Chen 2020-style three-condition thresholds on validation set."
    )

    parser.add_argument(
        "--val-index-csv",
        type=str,
        default=str(BASELINE_ROOT / "data" / "splits" / "chen2020_val_rule_features_index.csv"),
        help="Validation index CSV from File 04.",
    )

    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "metrics"),
        help="Directory to save threshold search metrics.",
    )

    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "predictions"),
        help="Directory to save validation predictions.",
    )

    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "reports"),
        help="Directory to save report JSON.",
    )

    parser.add_argument(
        "--num-velocity-candidates",
        type=int,
        default=80,
        help="Number of velocity threshold candidates.",
    )

    parser.add_argument(
        "--angle-candidates",
        type=str,
        default="45",
        help="Comma-separated angle thresholds. Default 45 follows the paper.",
    )

    parser.add_argument(
        "--ratio-candidates",
        type=str,
        default="1.0",
        help="Comma-separated width/height ratio thresholds. Default 1.0 follows the paper.",
    )

    parser.add_argument(
        "--min-event-frames-candidates",
        type=str,
        default="1",
        help="Comma-separated minimum event frames. Default 1 follows direct rule-based detection.",
    )

    args = parser.parse_args()

    val_index_csv = Path(args.val_index_csv)
    metrics_dir = Path(args.metrics_dir)
    predictions_dir = Path(args.predictions_dir)
    reports_dir = Path(args.reports_dir)

    ensure_dir(metrics_dir)
    ensure_dir(predictions_dir)
    ensure_dir(reports_dir)

    print("=" * 100)
    print("FILE 05 - Tune Chen 2020-style thresholds on validation set")
    print("=" * 100)
    print(f"Validation index:              {val_index_csv}")
    print(f"Angle candidates:              {args.angle_candidates}")
    print(f"Ratio candidates:              {args.ratio_candidates}")
    print(f"Min event frames candidates:   {args.min_event_frames_candidates}")
    print("=" * 100)

    videos = load_validation_videos(val_index_csv)

    velocity_candidates = build_velocity_candidates(
        videos=videos,
        num_candidates=args.num_velocity_candidates,
        include_zero=True,
    )

    angle_candidates = parse_float_list(args.angle_candidates)
    ratio_candidates = parse_float_list(args.ratio_candidates)
    min_event_frames_candidates = parse_int_list(args.min_event_frames_candidates)

    print(f"Validation videos:             {len(videos)}")
    print(f"Velocity candidates:           {len(velocity_candidates)}")
    print(f"Velocity range:                {min(velocity_candidates):.4f} -> {max(velocity_candidates):.4f}")
    print(f"Angle candidates:              {angle_candidates}")
    print(f"Ratio candidates:              {ratio_candidates}")
    print(f"Min event frames candidates:   {min_event_frames_candidates}")
    print("=" * 100)

    search_rows = []
    best = None
    best_pred_df = None

    total_combinations = (
        len(velocity_candidates)
        * len(angle_candidates)
        * len(ratio_candidates)
        * len(min_event_frames_candidates)
    )

    combo_idx = 0

    for velocity_threshold in velocity_candidates:
        for angle_threshold in angle_candidates:
            for ratio_threshold in ratio_candidates:
                for min_event_frames in min_event_frames_candidates:
                    combo_idx += 1

                    metrics, pred_df = evaluate_thresholds(
                        videos=videos,
                        velocity_threshold=velocity_threshold,
                        angle_threshold=angle_threshold,
                        ratio_threshold=ratio_threshold,
                        min_event_frames=min_event_frames,
                    )

                    row = {
                        "velocity_threshold": float(velocity_threshold),
                        "angle_threshold": float(angle_threshold),
                        "ratio_threshold": float(ratio_threshold),
                        "min_event_frames": int(min_event_frames),
                        **metrics,
                    }

                    search_rows.append(row)

                    # Main objective:
                    #   macro_f1 first,
                    #   fall recall second,
                    #   accuracy third,
                    #   specificity fourth.
                    score_tuple = (
                        row["macro_f1"],
                        row["fall_recall"],
                        row["accuracy"],
                        row["specificity_not_fall_recall"],
                    )

                    if best is None or score_tuple > best["score_tuple"]:
                        best = {
                            "row": row,
                            "score_tuple": score_tuple,
                        }
                        best_pred_df = pred_df.copy()

                    if combo_idx % 20 == 0 or combo_idx == total_combinations:
                        print(
                            f"[{combo_idx}/{total_combinations}] "
                            f"current best macro_f1={best['row']['macro_f1']:.4f}, "
                            f"fall_recall={best['row']['fall_recall']:.4f}, "
                            f"vel={best['row']['velocity_threshold']:.4f}, "
                            f"angle={best['row']['angle_threshold']:.2f}, "
                            f"ratio={best['row']['ratio_threshold']:.2f}"
                        )

    search_df = pd.DataFrame(search_rows)

    search_df = search_df.sort_values(
        ["macro_f1", "fall_recall", "accuracy", "specificity_not_fall_recall"],
        ascending=False,
    ).reset_index(drop=True)

    best_row = search_df.iloc[0].to_dict()

    search_csv = metrics_dir / "05_threshold_search_results.csv"
    best_thresholds_json = metrics_dir / "05_best_thresholds.json"
    best_val_predictions_csv = predictions_dir / "05_val_predictions_best_thresholds.csv"

    search_df.to_csv(search_csv, index=False)

    if best_pred_df is not None:
        best_pred_df.to_csv(best_val_predictions_csv, index=False)

    best_thresholds = {
        "velocity_threshold": float(best_row["velocity_threshold"]),
        "angle_threshold": float(best_row["angle_threshold"]),
        "ratio_threshold": float(best_row["ratio_threshold"]),
        "min_event_frames": int(best_row["min_event_frames"]),

        "paper_defaults": {
            "angle_threshold": 45.0,
            "ratio_threshold": 1.0,
            "rule": "velocity >= critical_velocity AND angle < 45 AND width_height_ratio > 1",
        },

        "validation_metrics": {
            "accuracy": float(best_row["accuracy"]),
            "sensitivity_fall_recall": float(best_row["sensitivity_fall_recall"]),
            "specificity_not_fall_recall": float(best_row["specificity_not_fall_recall"]),
            "macro_f1": float(best_row["macro_f1"]),
            "fall_precision": float(best_row["fall_precision"]),
            "fall_recall": float(best_row["fall_recall"]),
            "fall_f1": float(best_row["fall_f1"]),
            "not_fall_precision": float(best_row["not_fall_precision"]),
            "not_fall_recall": float(best_row["not_fall_recall"]),
            "not_fall_f1": float(best_row["not_fall_f1"]),
            "tn": int(best_row["tn"]),
            "fp": int(best_row["fp"]),
            "fn": int(best_row["fn"]),
            "tp": int(best_row["tp"]),
        },
    }

    save_json(best_thresholds, best_thresholds_json)

    report = {
        "status": "completed",
        "pipeline_note": "Validation set is used to tune the critical hip descent velocity threshold. Angle threshold and width/height ratio default to the paper values unless candidate lists are changed by arguments.",
        "val_index_csv": str(val_index_csv),
        "num_validation_videos": len(videos),
        "num_combinations": int(total_combinations),
        "search_csv": str(search_csv),
        "best_thresholds_json": str(best_thresholds_json),
        "best_val_predictions_csv": str(best_val_predictions_csv),
        "best_thresholds": best_thresholds,
        "top_10": search_df.head(10).to_dict(orient="records"),
    }

    report_path = reports_dir / "05_tune_thresholds_on_val_report.json"
    save_json(report, report_path)

    print("=" * 100)
    print("FILE 05 completed.")
    print("=" * 100)
    print("Best thresholds:")
    print(best_thresholds)
    print("-" * 100)
    print(f"Search CSV:              {search_csv}")
    print(f"Best thresholds JSON:    {best_thresholds_json}")
    print(f"Best val predictions:    {best_val_predictions_csv}")
    print(f"Report:                  {report_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
