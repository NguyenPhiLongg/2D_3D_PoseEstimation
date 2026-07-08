"""
Compare Phase 3 results.

Purpose:
- Read all Phase 3 result JSON files.
- Compare:
    1. 2D Common
    2. 3D Common
    3. Concat Fusion Common
    4. Gated Fusion

- Save:
    outputs/ablation_results/ablation_summary.csv
    outputs/ablation_results/phase3_comparison_summary.json
    outputs/ablation_results/phase3_comparison_report.md

Run:
    python phase3_common_set_gated_fusion/compare_phase3_results.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from phase3_utils import OUTPUT_DIR, PHASE3_DIR


# =========================
# PATHS
# =========================

ABLATION_DIR = OUTPUT_DIR / "ablation_results"
ABLATION_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILES = [
    {
        "experiment": "2D Common",
        "task": "binary",
        "fusion_type": "single_stream",
        "input": "2D",
        "input_dim": "40D",
        "path": OUTPUT_DIR / "training_2d_common" / "results_2d_common_binary.json",
    },
    {
        "experiment": "2D Common",
        "task": "action",
        "fusion_type": "single_stream",
        "input": "2D",
        "input_dim": "40D",
        "path": OUTPUT_DIR / "training_2d_common" / "results_2d_common_action.json",
    },
    {
        "experiment": "3D Common",
        "task": "binary",
        "fusion_type": "single_stream",
        "input": "3D",
        "input_dim": "59D",
        "path": OUTPUT_DIR / "training_3d_common" / "results_3d_common_binary.json",
    },
    {
        "experiment": "3D Common",
        "task": "action",
        "fusion_type": "single_stream",
        "input": "3D",
        "input_dim": "59D",
        "path": OUTPUT_DIR / "training_3d_common" / "results_3d_common_action.json",
    },
    {
        "experiment": "Concat Fusion Common",
        "task": "binary",
        "fusion_type": "early_concat",
        "input": "2D+3D",
        "input_dim": "99D",
        "path": OUTPUT_DIR / "training_concat_common" / "results_concat_common_binary.json",
    },
    {
        "experiment": "Concat Fusion Common",
        "task": "action",
        "fusion_type": "early_concat",
        "input": "2D+3D",
        "input_dim": "99D",
        "path": OUTPUT_DIR / "training_concat_common" / "results_concat_common_action.json",
    },
    {
        "experiment": "Gated Fusion",
        "task": "binary",
        "fusion_type": "adaptive_gated_fusion",
        "input": "2D+3D",
        "input_dim": "40D+59D",
        "path": OUTPUT_DIR / "training_gated_fusion" / "results_gated_fusion_binary.json",
    },
    {
        "experiment": "Gated Fusion",
        "task": "action",
        "fusion_type": "adaptive_gated_fusion",
        "input": "2D+3D",
        "input_dim": "40D+59D",
        "path": OUTPUT_DIR / "training_gated_fusion" / "results_gated_fusion_action.json",
    },
]


# =========================
# BASIC HELPERS
# =========================

def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_round(value, digits: int = 4):
    if value is None:
        return None

    try:
        return round(float(value), digits)
    except Exception:
        return None


def get_report_dict(data: Dict) -> Dict:
    report = data.get("classification_report_dict", {})

    if not isinstance(report, dict):
        return {}

    return report


def get_class_metric(data: Dict, class_name: str, metric_name: str) -> Optional[float]:
    report = get_report_dict(data)

    if class_name not in report:
        return None

    class_data = report[class_name]

    if not isinstance(class_data, dict):
        return None

    value = class_data.get(metric_name)

    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def get_existing_result_records() -> List[Dict]:
    records = []

    for item in RESULT_FILES:
        data = load_json(item["path"])

        if data is None:
            print("Missing result file:", item["path"])
            continue

        record = {
            "experiment": item["experiment"],
            "task": item["task"],
            "fusion_type": item["fusion_type"],
            "input": item["input"],
            "input_dim": item["input_dim"],
            "result_path": str(item["path"]),
            "data": data,
        }

        records.append(record)

    return records


# =========================
# SUMMARY TABLES
# =========================

def build_summary_rows(records: List[Dict]) -> List[Dict]:
    rows = []

    for record in records:
        data = record["data"]
        task = record["task"]

        row = {
            "task": task,
            "experiment": record["experiment"],
            "fusion_type": record["fusion_type"],
            "input": record["input"],
            "input_dim": record["input_dim"],
            "accuracy": safe_round(data.get("final_test_accuracy")),
            "macro_f1": safe_round(data.get("final_test_macro_f1")),
            "best_val_macro_f1": safe_round(data.get("best_val_macro_f1")),
            "best_epoch": data.get("best_epoch"),
            "num_test_samples": data.get("num_test_samples"),
            "num_test_unique_videos": data.get("num_test_unique_videos"),
            "checkpoint_path": data.get("checkpoint_path"),
        }

        if task == "binary":
            row["not_fall_precision"] = safe_round(get_class_metric(data, "Not_Fall", "precision"))
            row["not_fall_recall"] = safe_round(get_class_metric(data, "Not_Fall", "recall"))
            row["not_fall_f1"] = safe_round(get_class_metric(data, "Not_Fall", "f1-score"))

            row["fall_precision"] = safe_round(get_class_metric(data, "Fall", "precision"))
            row["fall_recall"] = safe_round(get_class_metric(data, "Fall", "recall"))
            row["fall_f1"] = safe_round(get_class_metric(data, "Fall", "f1-score"))

        elif task == "action":
            for class_name in ["Sitting", "Sleeping", "Standing", "Walking"]:
                key = class_name.lower()
                row[f"{key}_precision"] = safe_round(get_class_metric(data, class_name, "precision"))
                row[f"{key}_recall"] = safe_round(get_class_metric(data, class_name, "recall"))
                row[f"{key}_f1"] = safe_round(get_class_metric(data, class_name, "f1-score"))

        if record["experiment"] == "Gated Fusion":
            row["mean_gate"] = safe_round(data.get("mean_gate"))
            row["std_gate"] = safe_round(data.get("std_gate"))
            row["min_gate"] = safe_round(data.get("min_gate"))
            row["max_gate"] = safe_round(data.get("max_gate"))
        else:
            row["mean_gate"] = None
            row["std_gate"] = None
            row["min_gate"] = None
            row["max_gate"] = None

        rows.append(row)

    return rows


def create_summary_dataframe(records: List[Dict]) -> pd.DataFrame:
    rows = build_summary_rows(records)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    base_cols = [
        "task",
        "experiment",
        "fusion_type",
        "input",
        "input_dim",
        "accuracy",
        "macro_f1",
        "best_val_macro_f1",
        "best_epoch",
        "num_test_samples",
        "num_test_unique_videos",
    ]

    other_cols = [col for col in df.columns if col not in base_cols + ["checkpoint_path"]]
    ordered_cols = base_cols + other_cols + ["checkpoint_path"]

    df = df[ordered_cols]

    return df


# =========================
# BEST MODEL ANALYSIS
# =========================

def get_best_by_task(df: pd.DataFrame, task: str, metric: str) -> Optional[Dict]:
    task_df = df[df["task"] == task].copy()

    if task_df.empty:
        return None

    if metric not in task_df.columns:
        return None

    task_df = task_df.dropna(subset=[metric])

    if task_df.empty:
        return None

    best_idx = task_df[metric].astype(float).idxmax()
    row = task_df.loc[best_idx].to_dict()

    return row


def build_best_summary(df: pd.DataFrame) -> Dict:
    summary = {}

    for task in ["binary", "action"]:
        summary[task] = {}

        for metric in ["accuracy", "macro_f1"]:
            best = get_best_by_task(df, task, metric)

            if best is not None:
                summary[task][f"best_{metric}"] = {
                    "experiment": best.get("experiment"),
                    "value": best.get(metric),
                    "input": best.get("input"),
                    "input_dim": best.get("input_dim"),
                }

    binary_fall_recall_best = get_best_by_task(df, "binary", "fall_recall")

    if binary_fall_recall_best is not None:
        summary.setdefault("binary", {})
        summary["binary"]["best_fall_recall"] = {
            "experiment": binary_fall_recall_best.get("experiment"),
            "value": binary_fall_recall_best.get("fall_recall"),
            "input": binary_fall_recall_best.get("input"),
            "input_dim": binary_fall_recall_best.get("input_dim"),
        }

    return summary


# =========================
# IMPROVEMENT ANALYSIS
# =========================

def find_metric(df: pd.DataFrame, task: str, experiment: str, metric: str) -> Optional[float]:
    row = df[(df["task"] == task) & (df["experiment"] == experiment)]

    if row.empty:
        return None

    value = row.iloc[0].get(metric)

    if value is None or pd.isna(value):
        return None

    try:
        return float(value)
    except Exception:
        return None


def compute_improvement(df: pd.DataFrame) -> List[Dict]:
    """
    Compute simple improvement values:
    - Concat Fusion vs 2D Common
    - Concat Fusion vs 3D Common
    - Gated Fusion vs Concat Fusion
    - Gated Fusion vs 2D Common
    - Gated Fusion vs 3D Common
    """
    comparisons = [
        ("Concat Fusion Common", "2D Common"),
        ("Concat Fusion Common", "3D Common"),
        ("Gated Fusion", "Concat Fusion Common"),
        ("Gated Fusion", "2D Common"),
        ("Gated Fusion", "3D Common"),
    ]

    metrics = ["accuracy", "macro_f1"]

    rows = []

    for task in ["binary", "action"]:
        for target, baseline in comparisons:
            for metric in metrics:
                target_value = find_metric(df, task, target, metric)
                baseline_value = find_metric(df, task, baseline, metric)

                if target_value is None or baseline_value is None:
                    continue

                diff = target_value - baseline_value

                rows.append(
                    {
                        "task": task,
                        "target_model": target,
                        "baseline_model": baseline,
                        "metric": metric,
                        "target_value": safe_round(target_value),
                        "baseline_value": safe_round(baseline_value),
                        "absolute_improvement": safe_round(diff),
                        "relative_improvement_percent": safe_round(
                            (diff / baseline_value * 100.0) if baseline_value != 0 else None
                        ),
                    }
                )

    return rows


# =========================
# MARKDOWN REPORT
# =========================

def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data available."

    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def build_markdown_report(
    summary_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    best_summary: Dict,
) -> str:
    binary_df = summary_df[summary_df["task"] == "binary"].copy()
    action_df = summary_df[summary_df["task"] == "action"].copy()

    binary_cols = [
        "experiment",
        "input",
        "input_dim",
        "accuracy",
        "macro_f1",
        "fall_precision",
        "fall_recall",
        "fall_f1",
        "num_test_samples",
        "num_test_unique_videos",
        "mean_gate",
    ]

    action_cols = [
        "experiment",
        "input",
        "input_dim",
        "accuracy",
        "macro_f1",
        "sitting_f1",
        "sleeping_f1",
        "standing_f1",
        "walking_f1",
        "num_test_samples",
        "num_test_unique_videos",
        "mean_gate",
    ]

    binary_cols = [col for col in binary_cols if col in binary_df.columns]
    action_cols = [col for col in action_cols if col in action_df.columns]

    lines = []

    lines.append("# Phase 3 Result Comparison")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "Phase 3 evaluates 2D-only, 3D-only, concat fusion, and gated fusion "
        "on the same common video subset. This ensures that the comparison is fair."
    )
    lines.append("")
    lines.append("## Binary Task: Fall / Not_Fall")
    lines.append("")
    lines.append(dataframe_to_markdown(binary_df[binary_cols] if not binary_df.empty else binary_df))
    lines.append("")
    lines.append("## Action Task: Sitting / Sleeping / Standing / Walking")
    lines.append("")
    lines.append(dataframe_to_markdown(action_df[action_cols] if not action_df.empty else action_df))
    lines.append("")
    lines.append("## Improvement Analysis")
    lines.append("")
    lines.append(dataframe_to_markdown(improvement_df))
    lines.append("")
    lines.append("## Best Model Summary")
    lines.append("")

    for task, task_summary in best_summary.items():
        lines.append(f"### {task.capitalize()}")

        for metric_name, item in task_summary.items():
            lines.append(
                f"- **{metric_name}**: {item['experiment']} "
                f"({item['input']}, {item['input_dim']}) = {item['value']}"
            )

        lines.append("")

    lines.append("## Interpretation Template")
    lines.append("")
    lines.append(
        "If Gated Fusion achieves higher Macro F1 than Concat Fusion, it suggests that "
        "adaptive weighting between 2D and estimated 3D skeleton streams is more effective "
        "than fixed early concatenation."
    )
    lines.append("")
    lines.append(
        "If Concat Fusion or Gated Fusion outperforms both 2D Common and 3D Common, it suggests "
        "that 2D and estimated 3D pose representations provide complementary information."
    )
    lines.append("")
    lines.append(
        "If 3D Common does not outperform 2D Common, this can be explained by the fact that the "
        "3D pose is estimated from 2D keypoints rather than obtained from ground-truth 3D annotations. "
        "Errors from 2D detection and 2D-to-3D lifting may affect the quality of the 3D representation."
    )
    lines.append("")

    return "\n".join(lines)


# =========================
# SAVE OUTPUTS
# =========================

def save_outputs(
    summary_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    best_summary: Dict,
    markdown_report: str,
) -> None:
    summary_csv_path = ABLATION_DIR / "ablation_summary.csv"
    improvement_csv_path = ABLATION_DIR / "improvement_summary.csv"
    summary_json_path = ABLATION_DIR / "phase3_comparison_summary.json"
    report_path = ABLATION_DIR / "phase3_comparison_report.md"

    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    improvement_df.to_csv(improvement_csv_path, index=False, encoding="utf-8-sig")

    json_data = {
        "best_summary": best_summary,
        "summary_records": summary_df.to_dict(orient="records"),
        "improvement_records": improvement_df.to_dict(orient="records"),
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)

    print("=" * 80)
    print("Saved Phase 3 comparison outputs")
    print("=" * 80)
    print("Ablation summary CSV:", summary_csv_path)
    print("Improvement summary CSV:", improvement_csv_path)
    print("Comparison summary JSON:", summary_json_path)
    print("Markdown report:", report_path)
    print("=" * 80)


# =========================
# PRINTING
# =========================

def print_summary(summary_df: pd.DataFrame, improvement_df: pd.DataFrame, best_summary: Dict) -> None:
    print("\n" + "=" * 80)
    print("PHASE 3 SUMMARY TABLE")
    print("=" * 80)

    if summary_df.empty:
        print("No result files found.")
        return

    display_cols = [
        "task",
        "experiment",
        "input",
        "input_dim",
        "accuracy",
        "macro_f1",
        "fall_recall",
        "fall_f1",
        "sitting_f1",
        "sleeping_f1",
        "standing_f1",
        "walking_f1",
        "mean_gate",
        "num_test_samples",
        "num_test_unique_videos",
    ]

    display_cols = [col for col in display_cols if col in summary_df.columns]

    print(summary_df[display_cols].to_string(index=False))

    print("\n" + "=" * 80)
    print("IMPROVEMENT SUMMARY")
    print("=" * 80)

    if improvement_df.empty:
        print("No improvement rows available.")
    else:
        print(improvement_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("BEST MODEL SUMMARY")
    print("=" * 80)

    print(json.dumps(best_summary, indent=4, ensure_ascii=False))


# =========================
# MAIN
# =========================

def main() -> None:
    print("=" * 80)
    print("PHASE 3 RESULT COMPARISON")
    print("=" * 80)
    print("Phase 3 directory:", PHASE3_DIR)
    print("Output directory:", OUTPUT_DIR)
    print("Ablation directory:", ABLATION_DIR)
    print("=" * 80)

    records = get_existing_result_records()

    if not records:
        print("No result files found.")
        print("Please train at least one Phase 3 model first.")
        print("")
        print("Expected training commands:")
        print("python phase3_common_set_gated_fusion/train_2d_common.py --task binary")
        print("python phase3_common_set_gated_fusion/train_2d_common.py --task action")
        print("python phase3_common_set_gated_fusion/train_3d_common.py --task binary")
        print("python phase3_common_set_gated_fusion/train_3d_common.py --task action")
        print("python phase3_common_set_gated_fusion/train_concat_common.py --task binary")
        print("python phase3_common_set_gated_fusion/train_concat_common.py --task action")
        print("python phase3_common_set_gated_fusion/train_gated_fusion.py --task binary")
        print("python phase3_common_set_gated_fusion/train_gated_fusion.py --task action")
        return

    summary_df = create_summary_dataframe(records)
    best_summary = build_best_summary(summary_df)

    improvement_rows = compute_improvement(summary_df)
    improvement_df = pd.DataFrame(improvement_rows)

    markdown_report = build_markdown_report(
        summary_df=summary_df,
        improvement_df=improvement_df,
        best_summary=best_summary,
    )

    print_summary(summary_df, improvement_df, best_summary)

    save_outputs(
        summary_df=summary_df,
        improvement_df=improvement_df,
        best_summary=best_summary,
        markdown_report=markdown_report,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()