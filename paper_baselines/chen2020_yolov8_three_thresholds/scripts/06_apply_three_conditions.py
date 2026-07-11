from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


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


def label_to_name(label: int) -> str:
    return LABEL_NAMES.get(int(label), "Unknown")


def load_thresholds(threshold_json: Path) -> Dict:
    data = read_json(threshold_json)

    required_keys = [
        "velocity_threshold",
        "angle_threshold",
        "ratio_threshold",
        "min_event_frames",
    ]

    missing = [key for key in required_keys if key not in data]

    if missing:
        raise ValueError(f"Threshold JSON missing keys: {missing}")

    return {
        "velocity_threshold": float(data["velocity_threshold"]),
        "angle_threshold": float(data["angle_threshold"]),
        "ratio_threshold": float(data["ratio_threshold"]),
        "min_event_frames": int(data["min_event_frames"]),
        "source_json": str(threshold_json),
    }


def validate_feature_df(df: pd.DataFrame, csv_path: Path):
    required_cols = [
        "video_id",
        "frame_idx",
        "label",
        "label_name",
        "hip_descent_velocity",
        "body_centerline_angle_degrees",
        "width_height_ratio",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Feature CSV missing columns in {csv_path}: {missing}")


def apply_three_conditions(df: pd.DataFrame, thresholds: Dict) -> pd.DataFrame:
    out = df.copy()

    velocity = pd.to_numeric(out["hip_descent_velocity"], errors="coerce").fillna(0.0)
    angle = pd.to_numeric(out["body_centerline_angle_degrees"], errors="coerce").fillna(999.0)
    ratio = pd.to_numeric(out["width_height_ratio"], errors="coerce").fillna(0.0)

    if "all_rule_features_valid" in out.columns:
        valid = pd.to_numeric(out["all_rule_features_valid"], errors="coerce").fillna(0).astype(int) == 1
    else:
        valid = pd.Series([True] * len(out), index=out.index)

    out["condition_1_velocity"] = (velocity >= thresholds["velocity_threshold"]).astype(int)
    out["condition_2_angle"] = (angle < thresholds["angle_threshold"]).astype(int)
    out["condition_3_ratio"] = (ratio > thresholds["ratio_threshold"]).astype(int)
    out["all_features_valid_for_rule"] = valid.astype(int)

    event_mask = (
        valid
        & (out["condition_1_velocity"] == 1)
        & (out["condition_2_angle"] == 1)
        & (out["condition_3_ratio"] == 1)
    )

    out["fall_event_frame"] = event_mask.astype(int)

    return out


def stand_up_diagnostic(
    frame_df: pd.DataFrame,
    first_event_idx: int,
    check_after_frames: int,
    stand_angle_threshold: float,
    stand_ratio_threshold: float,
) -> Dict:
    if first_event_idx < 0:
        return {
            "stand_up_checked": 0,
            "can_stand_after_event": 0,
            "stand_up_frame_idx": -1,
            "stand_up_frame": -1,
            "stand_up_frames_count": 0,
        }

    start = first_event_idx + 1
    end = min(len(frame_df), first_event_idx + 1 + check_after_frames)

    if start >= end:
        return {
            "stand_up_checked": 1,
            "can_stand_after_event": 0,
            "stand_up_frame_idx": -1,
            "stand_up_frame": -1,
            "stand_up_frames_count": 0,
        }

    sub = frame_df.iloc[start:end].copy()

    angle = pd.to_numeric(sub["body_centerline_angle_degrees"], errors="coerce").fillna(0.0)
    ratio = pd.to_numeric(sub["width_height_ratio"], errors="coerce").fillna(999.0)

    stand_like = (angle >= stand_angle_threshold) & (ratio <= stand_ratio_threshold)
    count = int(stand_like.sum())

    if count > 0:
        rel_idx = int(np.where(stand_like.to_numpy())[0][0])
        abs_idx = int(start + rel_idx)
        frame_number = int(frame_df.iloc[abs_idx]["frame_idx"])
        can_stand = 1
    else:
        abs_idx = -1
        frame_number = -1
        can_stand = 0

    return {
        "stand_up_checked": 1,
        "can_stand_after_event": int(can_stand),
        "stand_up_frame_idx": int(abs_idx),
        "stand_up_frame": int(frame_number),
        "stand_up_frames_count": int(count),
    }


def summarize_video(
    frame_df: pd.DataFrame,
    csv_path: Path,
    thresholds: Dict,
    use_stand_up_filter: bool,
    check_after_frames: int,
    stand_angle_threshold: float,
    stand_ratio_threshold: float,
) -> Dict:
    video_id = str(frame_df["video_id"].iloc[0])
    true_label = int(pd.to_numeric(frame_df["label"].iloc[0], errors="coerce"))

    event_mask = frame_df["fall_event_frame"].astype(int) == 1
    event_count = int(event_mask.sum())

    raw_pred_label = 1 if event_count >= thresholds["min_event_frames"] else 0

    if event_count > 0:
        first_event_idx = int(np.where(event_mask.to_numpy())[0][0])
        first_event_frame = int(frame_df.iloc[first_event_idx]["frame_idx"])
    else:
        first_event_idx = -1
        first_event_frame = -1

    stand_info = stand_up_diagnostic(
        frame_df=frame_df,
        first_event_idx=first_event_idx,
        check_after_frames=check_after_frames,
        stand_angle_threshold=stand_angle_threshold,
        stand_ratio_threshold=stand_ratio_threshold,
    )

    if use_stand_up_filter and raw_pred_label == 1 and stand_info["can_stand_after_event"] == 1:
        pred_label = 0
    else:
        pred_label = raw_pred_label

    velocity = pd.to_numeric(frame_df["hip_descent_velocity"], errors="coerce").fillna(0.0)
    angle = pd.to_numeric(frame_df["body_centerline_angle_degrees"], errors="coerce").fillna(999.0)
    ratio = pd.to_numeric(frame_df["width_height_ratio"], errors="coerce").fillna(0.0)

    return {
        "video_id": video_id,
        "true_label": int(true_label),
        "true_label_name": label_to_name(true_label),

        "raw_pred_label": int(raw_pred_label),
        "raw_pred_label_name": label_to_name(raw_pred_label),

        "pred_label": int(pred_label),
        "pred_label_name": label_to_name(pred_label),
        "correct": int(pred_label == true_label),

        "num_frames": int(len(frame_df)),
        "event_count": int(event_count),
        "min_event_frames": int(thresholds["min_event_frames"]),
        "first_event_idx": int(first_event_idx),
        "first_event_frame": int(first_event_frame),

        "condition_1_velocity_frames": int(frame_df["condition_1_velocity"].sum()),
        "condition_2_angle_frames": int(frame_df["condition_2_angle"].sum()),
        "condition_3_ratio_frames": int(frame_df["condition_3_ratio"].sum()),
        "fall_event_frames": int(frame_df["fall_event_frame"].sum()),

        "max_hip_descent_velocity": float(velocity.max()),
        "min_body_angle": float(angle.min()),
        "max_width_height_ratio": float(ratio.max()),

        "use_stand_up_filter": int(use_stand_up_filter),
        **stand_info,

        "source_csv": str(csv_path),
    }


def process_one_video(
    csv_path: Path,
    thresholds: Dict,
    frame_output_dir: Path,
    save_frame_predictions: bool,
    use_stand_up_filter: bool,
    check_after_frames: int,
    stand_angle_threshold: float,
    stand_ratio_threshold: float,
) -> Dict:
    df = read_csv_safe(csv_path)
    validate_feature_df(df, csv_path)

    if "frame_idx" in df.columns:
        df = df.sort_values("frame_idx").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    frame_pred_df = apply_three_conditions(df, thresholds)

    video_result = summarize_video(
        frame_df=frame_pred_df,
        csv_path=csv_path,
        thresholds=thresholds,
        use_stand_up_filter=use_stand_up_filter,
        check_after_frames=check_after_frames,
        stand_angle_threshold=stand_angle_threshold,
        stand_ratio_threshold=stand_ratio_threshold,
    )

    if save_frame_predictions:
        ensure_dir(frame_output_dir)
        safe_id = video_result["video_id"].replace("\\", "__").replace("/", "__").replace(":", "")
        frame_csv = frame_output_dir / f"{safe_id}_frame_predictions.csv"
        frame_pred_df.to_csv(frame_csv, index=False)
        video_result["frame_prediction_csv"] = str(frame_csv)
    else:
        video_result["frame_prediction_csv"] = ""

    return video_result


def load_split_index(index_csv: Path) -> pd.DataFrame:
    if not index_csv.exists():
        raise FileNotFoundError(f"Split index CSV not found: {index_csv}")

    df = read_csv_safe(index_csv)

    required_cols = ["video_id", "label", "label_name", "output_csv"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Split index missing columns in {index_csv}: {missing}")

    return df


def process_split(
    split_name: str,
    index_csv: Path,
    thresholds: Dict,
    predictions_dir: Path,
    save_frame_predictions: bool,
    use_stand_up_filter: bool,
    check_after_frames: int,
    stand_angle_threshold: float,
    stand_ratio_threshold: float,
) -> Dict:
    split_df = load_split_index(index_csv)

    rows = []
    frame_output_dir = predictions_dir / f"06_{split_name}_frame_predictions"

    print(f"Processing split: {split_name} | videos={len(split_df)}")

    for idx, row in split_df.iterrows():
        csv_path = Path(str(row["output_csv"]))

        if not csv_path.exists():
            raise FileNotFoundError(f"Feature CSV not found: {csv_path}")

        result = process_one_video(
            csv_path=csv_path,
            thresholds=thresholds,
            frame_output_dir=frame_output_dir,
            save_frame_predictions=save_frame_predictions,
            use_stand_up_filter=use_stand_up_filter,
            check_after_frames=check_after_frames,
            stand_angle_threshold=stand_angle_threshold,
            stand_ratio_threshold=stand_ratio_threshold,
        )

        result["split"] = split_name
        rows.append(result)

        if (idx + 1) % 500 == 0 or (idx + 1) == len(split_df):
            print(f"  [{idx + 1}/{len(split_df)}] done")

    pred_df = pd.DataFrame(rows)

    output_csv = predictions_dir / f"06_{split_name}_video_predictions.csv"
    pred_df.to_csv(output_csv, index=False)

    summary = {
        "split": split_name,
        "num_videos": int(len(pred_df)),
        "num_true_fall": int((pred_df["true_label"] == 1).sum()),
        "num_true_not_fall": int((pred_df["true_label"] == 0).sum()),
        "num_pred_fall": int((pred_df["pred_label"] == 1).sum()),
        "num_pred_not_fall": int((pred_df["pred_label"] == 0).sum()),
        "num_raw_pred_fall": int((pred_df["raw_pred_label"] == 1).sum()),
        "num_raw_pred_not_fall": int((pred_df["raw_pred_label"] == 0).sum()),
        "videos_with_event": int((pred_df["event_count"] > 0).sum()),
        "total_event_frames": int(pred_df["event_count"].sum()),
        "use_stand_up_filter": int(use_stand_up_filter),
        "videos_can_stand_after_event": int(pred_df["can_stand_after_event"].sum()),
        "prediction_csv": str(output_csv),
    }

    return {
        "summary": summary,
        "prediction_csv": str(output_csv),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Apply Chen 2020-style three fall conditions to train/val/test splits."
    )

    parser.add_argument(
        "--threshold-json",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "metrics" / "05_best_thresholds.json"),
    )

    parser.add_argument(
        "--train-index-csv",
        type=str,
        default=str(BASELINE_ROOT / "data" / "splits" / "chen2020_train_rule_features_index.csv"),
    )

    parser.add_argument(
        "--val-index-csv",
        type=str,
        default=str(BASELINE_ROOT / "data" / "splits" / "chen2020_val_rule_features_index.csv"),
    )

    parser.add_argument(
        "--test-index-csv",
        type=str,
        default=str(BASELINE_ROOT / "data" / "splits" / "chen2020_test_rule_features_index.csv"),
    )

    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "predictions"),
    )

    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "reports"),
    )

    parser.add_argument(
        "--save-frame-predictions",
        action="store_true",
    )

    parser.add_argument(
        "--use-stand-up-filter",
        action="store_true",
    )

    parser.add_argument("--check-after-frames", type=int, default=60)
    parser.add_argument("--stand-angle-threshold", type=float, default=45.0)
    parser.add_argument("--stand-ratio-threshold", type=float, default=1.0)

    args = parser.parse_args()

    thresholds = load_thresholds(Path(args.threshold_json))

    predictions_dir = Path(args.predictions_dir)
    reports_dir = Path(args.reports_dir)

    ensure_dir(predictions_dir)
    ensure_dir(reports_dir)

    print("=" * 100)
    print("FILE 06 - Apply Chen 2020 three conditions")
    print("=" * 100)
    print(f"Velocity threshold:     {thresholds['velocity_threshold']}")
    print(f"Angle threshold:        {thresholds['angle_threshold']}")
    print(f"Ratio threshold:        {thresholds['ratio_threshold']}")
    print(f"Min event frames:       {thresholds['min_event_frames']}")
    print(f"Use stand-up filter:    {args.use_stand_up_filter}")
    print(f"Save frame predictions: {args.save_frame_predictions}")
    print("=" * 100)

    split_results = {}

    split_results["train"] = process_split(
        split_name="train",
        index_csv=Path(args.train_index_csv),
        thresholds=thresholds,
        predictions_dir=predictions_dir,
        save_frame_predictions=args.save_frame_predictions,
        use_stand_up_filter=args.use_stand_up_filter,
        check_after_frames=args.check_after_frames,
        stand_angle_threshold=args.stand_angle_threshold,
        stand_ratio_threshold=args.stand_ratio_threshold,
    )

    split_results["val"] = process_split(
        split_name="val",
        index_csv=Path(args.val_index_csv),
        thresholds=thresholds,
        predictions_dir=predictions_dir,
        save_frame_predictions=args.save_frame_predictions,
        use_stand_up_filter=args.use_stand_up_filter,
        check_after_frames=args.check_after_frames,
        stand_angle_threshold=args.stand_angle_threshold,
        stand_ratio_threshold=args.stand_ratio_threshold,
    )

    split_results["test"] = process_split(
        split_name="test",
        index_csv=Path(args.test_index_csv),
        thresholds=thresholds,
        predictions_dir=predictions_dir,
        save_frame_predictions=args.save_frame_predictions,
        use_stand_up_filter=args.use_stand_up_filter,
        check_after_frames=args.check_after_frames,
        stand_angle_threshold=args.stand_angle_threshold,
        stand_ratio_threshold=args.stand_ratio_threshold,
    )

    report = {
        "status": "completed",
        "pipeline_note": "Three Chen et al. 2020-style conditions were applied: hip descent velocity, body centerline angle, and width/height ratio. Stand-up checking is diagnostic unless --use-stand-up-filter is enabled.",
        "thresholds": thresholds,
        "use_stand_up_filter": bool(args.use_stand_up_filter),
        "save_frame_predictions": bool(args.save_frame_predictions),
        "check_after_frames": int(args.check_after_frames),
        "stand_angle_threshold": float(args.stand_angle_threshold),
        "stand_ratio_threshold": float(args.stand_ratio_threshold),
        "splits": split_results,
    }

    report_path = reports_dir / "06_apply_three_conditions_report.json"
    save_json(report, report_path)

    print("=" * 100)
    print("FILE 06 completed.")
    print("=" * 100)
    print("Train:")
    print(split_results["train"]["summary"])
    print("Val:")
    print(split_results["val"]["summary"])
    print("Test:")
    print(split_results["test"]["summary"])
    print("-" * 100)
    print(f"Report: {report_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
