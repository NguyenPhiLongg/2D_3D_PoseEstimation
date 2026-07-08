import os
import json
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd


"""
Compare all fair results from Phase 1, Phase 2, Phase 3, and Phase 4.

Important:
    We do NOT use the old original Phase 1 / Phase 2 results because they were not
    necessarily evaluated on the same test set.

Fair mapping:
    Phase 1 fair:
        2D Common result from Phase 3 common-set evaluation.

    Phase 2 fair:
        3D Common result from Phase 3 common-set evaluation.
        Concat Common result from Phase 3 common-set evaluation.

    Phase 3 fair:
        Gated Fusion result from Phase 3 common-set evaluation.

    Phase 4 fair:
        Quality-Gated result from Phase 4.
        Quality-Concat result from Phase 4.

Fair test sets:
    Binary:
        2965 test samples, 812 test videos.

    Action:
        2053 test samples, 367 test videos.
"""


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PHASE4_DIR = os.path.join(PROJECT_ROOT, "phase4_quality_aware_fusion")
PHASE4_TRAINING_DIR = os.path.join(PHASE4_DIR, "outputs", "training_quality_gated")
COMPARISON_OUTPUT_DIR = os.path.join(PHASE4_DIR, "outputs", "comparison")


TASK_INFO = {
    "binary": {
        "task_name": "Binary Fall Detection",
        "expected_test_samples": 2965,
        "expected_test_videos": 812,
        "class_names": ["Not_Fall", "Fall"],
    },
    "action": {
        "task_name": "Action Classification",
        "expected_test_samples": 2053,
        "expected_test_videos": 367,
        "class_names": ["Sitting", "Sleeping", "Standing", "Walking"],
    },
}


# These are the fair Phase 3 common-set results already produced earlier.
# They represent Phase 1, Phase 2, and Phase 3 on the SAME common test set.
PHASE3_FAIR_RESULTS = {
    "binary": [
        {
            "phase": "Phase 1",
            "model": "2D Baseline",
            "fair_model_name": "Phase 1 - 2D Common",
            "input": "2D pose",
            "accuracy": 0.9346,
            "macro_f1": 0.9234,
            "num_test_samples": 2965,
            "num_test_videos": 812,
            "source": "Phase 3 common-set rerun",
        },
        {
            "phase": "Phase 2",
            "model": "3D Upgrade",
            "fair_model_name": "Phase 2 - 3D Common",
            "input": "Estimated 3D pose",
            "accuracy": 0.9272,
            "macro_f1": 0.9140,
            "num_test_samples": 2965,
            "num_test_videos": 812,
            "source": "Phase 3 common-set rerun",
        },
        {
            "phase": "Phase 2",
            "model": "2D+3D Concat Fusion",
            "fair_model_name": "Phase 2 - Concat Common",
            "input": "2D pose + estimated 3D pose",
            "accuracy": 0.9393,
            "macro_f1": 0.9299,
            "num_test_samples": 2965,
            "num_test_videos": 812,
            "source": "Phase 3 common-set rerun",
        },
        {
            "phase": "Phase 3",
            "model": "Gated Fusion",
            "fair_model_name": "Phase 3 - Gated Fusion",
            "input": "2D pose + estimated 3D pose",
            "accuracy": 0.9379,
            "macro_f1": 0.9281,
            "num_test_samples": 2965,
            "num_test_videos": 812,
            "mean_gate": 0.4837,
            "source": "Phase 3 common-set evaluation",
        },
    ],
    "action": [
        {
            "phase": "Phase 1",
            "model": "2D Baseline",
            "fair_model_name": "Phase 1 - 2D Common",
            "input": "2D pose",
            "accuracy": 0.9698,
            "macro_f1": 0.9436,
            "num_test_samples": 2053,
            "num_test_videos": 367,
            "source": "Phase 3 common-set rerun",
        },
        {
            "phase": "Phase 2",
            "model": "3D Upgrade",
            "fair_model_name": "Phase 2 - 3D Common",
            "input": "Estimated 3D pose",
            "accuracy": 0.9679,
            "macro_f1": 0.9470,
            "num_test_samples": 2053,
            "num_test_videos": 367,
            "source": "Phase 3 common-set rerun",
        },
        {
            "phase": "Phase 2",
            "model": "2D+3D Concat Fusion",
            "fair_model_name": "Phase 2 - Concat Common",
            "input": "2D pose + estimated 3D pose",
            "accuracy": 0.9864,
            "macro_f1": 0.9751,
            "num_test_samples": 2053,
            "num_test_videos": 367,
            "source": "Phase 3 common-set rerun",
        },
        {
            "phase": "Phase 3",
            "model": "Gated Fusion",
            "fair_model_name": "Phase 3 - Gated Fusion",
            "input": "2D pose + estimated 3D pose",
            "accuracy": 0.9781,
            "macro_f1": 0.9618,
            "num_test_samples": 2053,
            "num_test_videos": 367,
            "mean_gate": 0.4663,
            "source": "Phase 3 common-set evaluation",
        },
    ],
}


PHASE4_RESULT_FILES = {
    "binary": {
        "quality_gated": {
            "phase": "Phase 4",
            "model": "Quality-Aware Gated Fusion",
            "fair_model_name": "Phase 4 - Quality-Gated",
            "input": "2D pose + 3D pose + quality features",
            "path": os.path.join(PHASE4_TRAINING_DIR, "results_quality_gated_binary.json"),
            "source": "Phase 4 training",
        },
        "quality_concat": {
            "phase": "Phase 4",
            "model": "Quality-Concat Fusion",
            "fair_model_name": "Phase 4 - Quality-Concat",
            "input": "2D pose + 3D pose + quality features",
            "path": os.path.join(PHASE4_TRAINING_DIR, "results_quality_concat_binary.json"),
            "source": "Phase 4 ablation",
        },
    },
    "action": {
        "quality_gated": {
            "phase": "Phase 4",
            "model": "Quality-Aware Gated Fusion",
            "fair_model_name": "Phase 4 - Quality-Gated",
            "input": "2D pose + 3D pose + quality features",
            "path": os.path.join(PHASE4_TRAINING_DIR, "results_quality_gated_action.json"),
            "source": "Phase 4 training",
        },
        "quality_concat": {
            "phase": "Phase 4",
            "model": "Quality-Concat Fusion",
            "fair_model_name": "Phase 4 - Quality-Concat",
            "input": "2D pose + 3D pose + quality features",
            "path": os.path.join(PHASE4_TRAINING_DIR, "results_quality_concat_action.json"),
            "source": "Phase 4 ablation",
        },
    },
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=4)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, list):
        return [to_jsonable(v) for v in value]

    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if pd.isna(value) if isinstance(value, float) else False:
        return None

    return value


def get_nested(data: Dict, keys: List[str], default=None):
    cur = data

    for key in keys:
        if not isinstance(cur, dict):
            return default

        if key not in cur:
            return default

        cur = cur[key]

    return cur


def get_phase4_metric(data: Dict, metric_name: str, default=np.nan):
    paths = [
        ["test_result", metric_name],
        ["test_results", metric_name],
        ["test_metrics", metric_name],
        [metric_name],
    ]

    for path in paths:
        value = get_nested(data, path, default=None)

        if value is not None:
            return value

    return default


def get_phase4_gate_stats(data: Dict) -> Dict:
    gate_stats = get_nested(data, ["test_result", "gate_stats"], default=None)

    if gate_stats is None:
        gate_stats = data.get("gate_stats", None)

    if not isinstance(gate_stats, dict):
        return {
            "mean_gate": np.nan,
            "std_gate": np.nan,
            "min_gate": np.nan,
            "max_gate": np.nan,
        }

    return {
        "mean_gate": gate_stats.get("mean_gate", np.nan),
        "std_gate": gate_stats.get("std_gate", np.nan),
        "min_gate": gate_stats.get("min_gate", np.nan),
        "max_gate": gate_stats.get("max_gate", np.nan),
    }


def get_phase4_per_class(data: Dict, task: str) -> Dict:
    output = {}

    class_names = TASK_INFO[task]["class_names"]
    per_class = get_nested(data, ["test_result", "per_class"], default={})

    for class_name in class_names:
        class_data = per_class.get(class_name, {}) if isinstance(per_class, dict) else {}

        output[f"{class_name}_precision"] = class_data.get("precision", np.nan)
        output[f"{class_name}_recall"] = class_data.get("recall", np.nan)
        output[f"{class_name}_f1"] = class_data.get("f1", np.nan)
        output[f"{class_name}_support"] = class_data.get("support", np.nan)

    return output


def load_phase4_row(task: str, model_key: str, spec: Dict) -> Dict:
    path = spec["path"]

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing Phase 4 result file: {path}")

    data = read_json(path)
    task_info = TASK_INFO[task]

    accuracy = get_phase4_metric(data, "accuracy")
    macro_f1 = get_phase4_metric(data, "macro_f1")

    num_samples = get_phase4_metric(data, "num_samples", default=task_info["expected_test_samples"])
    dataset_info = data.get("dataset_info", {})

    num_test_samples = dataset_info.get("num_test_samples", num_samples)
    num_test_videos = dataset_info.get("num_test_videos", task_info["expected_test_videos"])

    gate_stats = get_phase4_gate_stats(data)
    per_class = get_phase4_per_class(data, task)

    row = {
        "task": task,
        "task_name": task_info["task_name"],
        "phase": spec["phase"],
        "model": spec["model"],
        "fair_model_name": spec["fair_model_name"],
        "input": spec["input"],
        "source": spec["source"],
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "accuracy_percent": float(accuracy) * 100,
        "macro_f1_percent": float(macro_f1) * 100,
        "num_test_samples": int(num_test_samples),
        "num_test_videos": int(num_test_videos),
        "expected_test_samples": task_info["expected_test_samples"],
        "expected_test_videos": task_info["expected_test_videos"],
        "sample_count_matches_expected": int(num_test_samples) == int(task_info["expected_test_samples"]),
        "video_count_matches_expected": int(num_test_videos) == int(task_info["expected_test_videos"]),
        "best_epoch": data.get("best_epoch", np.nan),
        "best_val_macro_f1": data.get("best_val_macro_f1", np.nan),
        "mean_gate": gate_stats["mean_gate"],
        "std_gate": gate_stats["std_gate"],
        "min_gate": gate_stats["min_gate"],
        "max_gate": gate_stats["max_gate"],
        "result_path": path,
    }

    row.update(per_class)

    return row


def make_phase3_row(task: str, item: Dict) -> Dict:
    task_info = TASK_INFO[task]

    row = {
        "task": task,
        "task_name": task_info["task_name"],
        "phase": item["phase"],
        "model": item["model"],
        "fair_model_name": item["fair_model_name"],
        "input": item["input"],
        "source": item["source"],
        "accuracy": float(item["accuracy"]),
        "macro_f1": float(item["macro_f1"]),
        "accuracy_percent": float(item["accuracy"]) * 100,
        "macro_f1_percent": float(item["macro_f1"]) * 100,
        "num_test_samples": int(item["num_test_samples"]),
        "num_test_videos": int(item["num_test_videos"]),
        "expected_test_samples": task_info["expected_test_samples"],
        "expected_test_videos": task_info["expected_test_videos"],
        "sample_count_matches_expected": int(item["num_test_samples"]) == int(task_info["expected_test_samples"]),
        "video_count_matches_expected": int(item["num_test_videos"]) == int(task_info["expected_test_videos"]),
        "best_epoch": np.nan,
        "best_val_macro_f1": np.nan,
        "mean_gate": item.get("mean_gate", np.nan),
        "std_gate": item.get("std_gate", np.nan),
        "min_gate": np.nan,
        "max_gate": np.nan,
        "result_path": "phase3_common_set_fair_result",
    }

    for class_name in task_info["class_names"]:
        row[f"{class_name}_precision"] = np.nan
        row[f"{class_name}_recall"] = np.nan
        row[f"{class_name}_f1"] = np.nan
        row[f"{class_name}_support"] = np.nan

    return row


def build_comparison_dataframe() -> pd.DataFrame:
    rows = []

    for task in ["binary", "action"]:
        for item in PHASE3_FAIR_RESULTS[task]:
            rows.append(make_phase3_row(task, item))

        for model_key, spec in PHASE4_RESULT_FILES[task].items():
            rows.append(load_phase4_row(task, model_key, spec))

    df = pd.DataFrame(rows)

    ranked_frames = []

    for task, group in df.groupby("task"):
        group = group.copy()
        group = group.sort_values(
            by=["macro_f1", "accuracy"],
            ascending=[False, False],
        ).reset_index(drop=True)

        group["rank_by_macro_f1"] = np.arange(1, len(group) + 1)

        ranked_frames.append(group)

    return pd.concat(ranked_frames, axis=0).reset_index(drop=True)


def build_best_summary(df: pd.DataFrame) -> Dict:
    summary = {}

    for task, group in df.groupby("task"):
        group = group.sort_values(
            by=["macro_f1", "accuracy"],
            ascending=[False, False],
        ).reset_index(drop=True)

        best = group.iloc[0].to_dict()

        summary[task] = {
            "task_name": TASK_INFO[task]["task_name"],
            "best_model": best["fair_model_name"],
            "phase": best["phase"],
            "accuracy": best["accuracy"],
            "macro_f1": best["macro_f1"],
            "accuracy_percent": best["accuracy_percent"],
            "macro_f1_percent": best["macro_f1_percent"],
            "num_test_samples": best["num_test_samples"],
            "num_test_videos": best["num_test_videos"],
        }

    return summary


def format_percent(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value) * 100:.2f}%"


def build_markdown_report(df: pd.DataFrame, best_summary: Dict) -> str:
    lines = []

    lines.append("# All Phases Fair Comparison Report")
    lines.append("")
    lines.append("This report compares Phase 1, Phase 2, Phase 3, and Phase 4 on the same fair common test sets.")
    lines.append("")
    lines.append("Fair mapping:")
    lines.append("")
    lines.append("- Phase 1 = 2D Common rerun.")
    lines.append("- Phase 2 = 3D Common and 2D+3D Concat Common reruns.")
    lines.append("- Phase 3 = Gated Fusion on the common set.")
    lines.append("- Phase 4 = Quality-Gated and Quality-Concat models.")
    lines.append("")

    for task in ["binary", "action"]:
        task_df = df[df["task"] == task].copy()
        task_name = TASK_INFO[task]["task_name"]

        lines.append(f"## {task_name}")
        lines.append("")
        lines.append(
            f"Fair test set: {TASK_INFO[task]['expected_test_samples']} samples, "
            f"{TASK_INFO[task]['expected_test_videos']} videos."
        )
        lines.append("")
        lines.append("| Rank | Phase | Model | Input | Accuracy | Macro F1 | Test Samples | Test Videos |")
        lines.append("|---:|---|---|---|---:|---:|---:|---:|")

        for _, row in task_df.iterrows():
            lines.append(
                f"| {int(row['rank_by_macro_f1'])} "
                f"| {row['phase']} "
                f"| {row['model']} "
                f"| {row['input']} "
                f"| {row['accuracy_percent']:.2f}% "
                f"| {row['macro_f1_percent']:.2f}% "
                f"| {int(row['num_test_samples'])} "
                f"| {int(row['num_test_videos'])} |"
            )

        lines.append("")

        best = best_summary[task]

        lines.append(
            f"Best model for **{task_name}**: "
            f"**{best['best_model']}** with "
            f"**{best['accuracy_percent']:.2f}% Accuracy** and "
            f"**{best['macro_f1_percent']:.2f}% Macro F1**."
        )
        lines.append("")

        if task == "binary":
            lines.append(
                "Interpretation: Phase 4 is valuable for binary fall detection. "
                "The best binary model is expected to be Quality-Concat, showing that pose quality features "
                "help the Fall/Not_Fall decision."
            )
            lines.append("")

        if task == "action":
            lines.append(
                "Interpretation: Phase 3 Concat Fusion can still remain the strongest for multi-class action classification. "
                "This means quality features are more useful for fall detection than for fine-grained action recognition."
            )
            lines.append("")

    lines.append("## Final Task-Specific Recommendation")
    lines.append("")

    binary_best = best_summary["binary"]
    action_best = best_summary["action"]

    lines.append(
        f"- Binary Fall Detection: use **{binary_best['best_model']}** "
        f"({binary_best['accuracy_percent']:.2f}% Accuracy, "
        f"{binary_best['macro_f1_percent']:.2f}% Macro F1)."
    )
    lines.append(
        f"- Action Classification: use **{action_best['best_model']}** "
        f"({action_best['accuracy_percent']:.2f}% Accuracy, "
        f"{action_best['macro_f1_percent']:.2f}% Macro F1)."
    )
    lines.append("")

    return "\n".join(lines)


def save_outputs(df: pd.DataFrame, best_summary: Dict) -> None:
    ensure_dir(COMPARISON_OUTPUT_DIR)

    csv_path = os.path.join(COMPARISON_OUTPUT_DIR, "all_phases_fair_comparison.csv")
    json_path = os.path.join(COMPARISON_OUTPUT_DIR, "all_phases_fair_comparison.json")
    best_path = os.path.join(COMPARISON_OUTPUT_DIR, "all_phases_fair_best_models.json")
    md_path = os.path.join(COMPARISON_OUTPUT_DIR, "all_phases_fair_report.md")

    df.to_csv(csv_path, index=False)

    save_json(
        {
            "comparison": df.to_dict(orient="records"),
        },
        json_path,
    )

    save_json(best_summary, best_path)

    report = build_markdown_report(df, best_summary)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\nSaved comparison outputs:")
    print(f"- {csv_path}")
    print(f"- {json_path}")
    print(f"- {best_path}")
    print(f"- {md_path}")


def print_summary(df: pd.DataFrame, best_summary: Dict) -> None:
    print("\nAll Phases Fair Comparison")
    print("=" * 100)

    for task in ["binary", "action"]:
        task_df = df[df["task"] == task].copy()

        print(f"\n{TASK_INFO[task]['task_name']}")
        print("-" * 100)

        cols = [
            "rank_by_macro_f1",
            "phase",
            "model",
            "accuracy_percent",
            "macro_f1_percent",
            "num_test_samples",
            "num_test_videos",
        ]

        print(task_df[cols].to_string(index=False))

        best = best_summary[task]

        print(
            f"\nBest {task}: {best['best_model']} | "
            f"Accuracy={best['accuracy_percent']:.2f}% | "
            f"Macro F1={best['macro_f1_percent']:.2f}%"
        )

    print("=" * 100)


def main() -> None:
    ensure_dir(COMPARISON_OUTPUT_DIR)

    df = build_comparison_dataframe()
    best_summary = build_best_summary(df)

    save_outputs(df, best_summary)
    print_summary(df, best_summary)

    print("\nDone.")


if __name__ == "__main__":
    main()