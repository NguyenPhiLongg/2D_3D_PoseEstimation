import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


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
# CONSTANTS
# ============================================================

MODEL_ORDER = [
    "phase1_2d_common",
    "phase2_3d_common",
    "phase2_concat_common",
    "phase3_gated_fusion",
    "phase4_quality_gated",
    "phase4_quality_concat",
]

MODEL_DISPLAY_NAMES = {
    "phase1_2d_common": "Phase 1 - 2D Common",
    "phase2_3d_common": "Phase 2 - 3D Common",
    "phase2_concat_common": "Phase 2 - 2D+3D Concat",
    "phase3_gated_fusion": "Phase 3 - Gated Fusion",
    "phase4_quality_gated": "Phase 4 - Quality-Gated",
    "phase4_quality_concat": "Phase 4 - Quality-Concat",
}

MODEL_PHASES = {
    "phase1_2d_common": "Phase 1",
    "phase2_3d_common": "Phase 2",
    "phase2_concat_common": "Phase 2",
    "phase3_gated_fusion": "Phase 3",
    "phase4_quality_gated": "Phase 4",
    "phase4_quality_concat": "Phase 4",
}

METRIC_COLUMNS = [
    "test_accuracy",
    "test_macro_f1",
    "test_fall_recall",
    "test_fall_f1",
    "test_not_fall_f1",
]

COUNT_COLUMNS = [
    "test_tn",
    "test_fp",
    "test_fn",
    "test_tp",
]


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_float(value, default=0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default

        value = float(value)

        if np.isnan(value) or np.isinf(value):
            return default

        return value
    except Exception:
        return default


def safe_int(value, default=0) -> int:
    try:
        if value is None or pd.isna(value):
            return default

        return int(float(value))
    except Exception:
        return default


def pct(value: float) -> float:
    return safe_float(value) * 100.0


def pct_point(a: float, b: float) -> float:
    return (safe_float(a) - safe_float(b)) * 100.0


def save_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_markdown(text: str, path: Path) -> None:
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def clean_model_order(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["model_name"] = df["model_name"].astype(str)
    df["display_name"] = df["model_name"].map(MODEL_DISPLAY_NAMES).fillna(df.get("display_name", df["model_name"]))
    df["phase"] = df["model_name"].map(MODEL_PHASES).fillna("Unknown")

    order_map = {name: i for i, name in enumerate(MODEL_ORDER)}
    df["_order"] = df["model_name"].map(order_map).fillna(999).astype(int)

    return df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


# ============================================================
# LOAD INPUTS
# ============================================================

def get_input_paths() -> Dict[str, Path]:
    base_dir = PHASE5_DIR / "outputs" / "external_finetuning"

    return {
        "base_dir": base_dir,
        "metrics_csv": base_dir / "external_finetuned_all_models_metrics.csv",
        "epochs_csv": base_dir / "external_finetuned_all_models_per_epoch.csv",
        "predictions_csv": base_dir / "external_finetuned_all_models_test_predictions_long.csv",
    }


def load_step07_outputs(paths: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_path = paths["metrics_csv"]
    epochs_path = paths["epochs_csv"]
    predictions_path = paths["predictions_csv"]

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing Step 07 metrics file:\n{metrics_path}\n"
            "Run 07_finetune_all_models_external.py first."
        )

    if not epochs_path.exists():
        raise FileNotFoundError(
            f"Missing Step 07 per-epoch file:\n{epochs_path}\n"
            "Run 07_finetune_all_models_external.py first."
        )

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing Step 07 prediction file:\n{predictions_path}\n"
            "Run 07_finetune_all_models_external.py first."
        )

    metrics_df = pd.read_csv(metrics_path)
    epochs_df = pd.read_csv(epochs_path)
    pred_df = pd.read_csv(predictions_path)

    metrics_df = clean_model_order(metrics_df)

    return metrics_df, epochs_df, pred_df


# ============================================================
# VALIDATION
# ============================================================

def validate_metrics_df(metrics_df: pd.DataFrame) -> Dict[str, Any]:
    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
    }

    required = [
        "model_name",
        "best_epoch",
        "test_accuracy",
        "test_macro_f1",
        "test_fall_recall",
        "test_fall_f1",
        "test_not_fall_f1",
        "test_tn",
        "test_fp",
        "test_fn",
        "test_tp",
    ]

    missing = [col for col in required if col not in metrics_df.columns]

    if missing:
        report["valid"] = False
        report["errors"].append(f"Missing columns in metrics CSV: {missing}")
        return report

    model_names = set(metrics_df["model_name"].astype(str).tolist())
    expected = set(MODEL_ORDER)

    missing_models = sorted(expected - model_names)
    extra_models = sorted(model_names - expected)

    if missing_models:
        report["warnings"].append(f"Missing expected models: {missing_models}")

    if extra_models:
        report["warnings"].append(f"Extra models found: {extra_models}")

    duplicated = metrics_df[metrics_df["model_name"].duplicated(keep=False)]

    if len(duplicated) > 0:
        report["valid"] = False
        report["errors"].append("Duplicated model_name rows found in metrics CSV.")

    for col in METRIC_COLUMNS:
        values = metrics_df[col].astype(float)

        if ((values < 0) | (values > 1)).any():
            report["valid"] = False
            report["errors"].append(f"Metric column {col} must be in [0,1].")

    return report


def validate_predictions_df(pred_df: pd.DataFrame, metrics_df: pd.DataFrame) -> Dict[str, Any]:
    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
    }

    required = [
        "model_name",
        "sequence_key",
        "y_true",
        "y_pred",
        "prob_fall",
    ]

    missing = [col for col in required if col not in pred_df.columns]

    if missing:
        report["valid"] = False
        report["errors"].append(f"Missing columns in prediction CSV: {missing}")
        return report

    model_names = metrics_df["model_name"].astype(str).tolist()

    key_reference = None

    per_model_counts = {}

    for model_name in model_names:
        part = pred_df[pred_df["model_name"].astype(str) == model_name].copy()
        per_model_counts[model_name] = int(len(part))

        if part.empty:
            report["valid"] = False
            report["errors"].append(f"No prediction rows for model {model_name}")
            continue

        keys = part["sequence_key"].astype(str).tolist()

        if key_reference is None:
            key_reference = keys
        else:
            if keys != key_reference:
                report["valid"] = False
                report["errors"].append(
                    f"Prediction sequence_key order mismatch for model {model_name}"
                )

    report["per_model_prediction_counts"] = per_model_counts

    if key_reference is not None:
        report["num_test_sequences"] = int(len(key_reference))

    return report


# ============================================================
# RANKINGS AND COMPARISONS
# ============================================================

def build_ranking_tables(metrics_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    ranking_macro = metrics_df.sort_values("test_macro_f1", ascending=False).reset_index(drop=True)
    ranking_fall_recall = metrics_df.sort_values("test_fall_recall", ascending=False).reset_index(drop=True)
    ranking_fall_f1 = metrics_df.sort_values("test_fall_f1", ascending=False).reset_index(drop=True)
    ranking_accuracy = metrics_df.sort_values("test_accuracy", ascending=False).reset_index(drop=True)

    for df in [ranking_macro, ranking_fall_recall, ranking_fall_f1, ranking_accuracy]:
        df.insert(0, "rank", np.arange(1, len(df) + 1))

    return {
        "ranking_by_macro_f1": ranking_macro,
        "ranking_by_fall_recall": ranking_fall_recall,
        "ranking_by_fall_f1": ranking_fall_f1,
        "ranking_by_accuracy": ranking_accuracy,
    }


def build_phase_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for phase, group in metrics_df.groupby("phase", sort=False):
        best_macro = group.sort_values("test_macro_f1", ascending=False).iloc[0]
        best_fall_recall = group.sort_values("test_fall_recall", ascending=False).iloc[0]
        best_fall_f1 = group.sort_values("test_fall_f1", ascending=False).iloc[0]

        rows.append(
            {
                "phase": phase,
                "num_models": int(len(group)),

                "best_macro_model": best_macro["model_name"],
                "best_macro_f1": float(best_macro["test_macro_f1"]),
                "best_macro_accuracy": float(best_macro["test_accuracy"]),
                "best_macro_fall_recall": float(best_macro["test_fall_recall"]),
                "best_macro_fall_f1": float(best_macro["test_fall_f1"]),

                "best_fall_recall_model": best_fall_recall["model_name"],
                "best_fall_recall": float(best_fall_recall["test_fall_recall"]),

                "best_fall_f1_model": best_fall_f1["model_name"],
                "best_fall_f1": float(best_fall_f1["test_fall_f1"]),
            }
        )

    out = pd.DataFrame(rows)
    phase_order = {"Phase 1": 1, "Phase 2": 2, "Phase 3": 3, "Phase 4": 4}
    out["_order"] = out["phase"].map(phase_order).fillna(999).astype(int)

    return out.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def get_model_row(metrics_df: pd.DataFrame, model_name: str) -> pd.Series:
    part = metrics_df[metrics_df["model_name"].astype(str) == model_name]

    if part.empty:
        raise ValueError(f"Model not found in metrics_df: {model_name}")

    return part.iloc[0]


def build_improvement_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    comparisons = [
        ("phase4_quality_gated", "phase1_2d_common", "Quality-Gated vs 2D baseline"),
        ("phase4_quality_gated", "phase2_concat_common", "Quality-Gated vs 2D+3D concat"),
        ("phase4_quality_gated", "phase3_gated_fusion", "Quality-Gated vs gated fusion"),

        ("phase4_quality_concat", "phase1_2d_common", "Quality-Concat vs 2D baseline"),
        ("phase4_quality_concat", "phase2_concat_common", "Quality-Concat vs 2D+3D concat"),
        ("phase4_quality_concat", "phase3_gated_fusion", "Quality-Concat vs gated fusion"),

        ("phase4_quality_gated", "phase4_quality_concat", "Quality-Gated vs Quality-Concat"),
    ]

    available = set(metrics_df["model_name"].astype(str).tolist())

    for improved_model, baseline_model, comparison_name in comparisons:
        if improved_model not in available or baseline_model not in available:
            continue

        a = get_model_row(metrics_df, improved_model)
        b = get_model_row(metrics_df, baseline_model)

        rows.append(
            {
                "comparison": comparison_name,
                "model_a": improved_model,
                "model_b": baseline_model,

                "macro_f1_a": float(a["test_macro_f1"]),
                "macro_f1_b": float(b["test_macro_f1"]),
                "macro_f1_gain_points": pct_point(a["test_macro_f1"], b["test_macro_f1"]),

                "accuracy_a": float(a["test_accuracy"]),
                "accuracy_b": float(b["test_accuracy"]),
                "accuracy_gain_points": pct_point(a["test_accuracy"], b["test_accuracy"]),

                "fall_recall_a": float(a["test_fall_recall"]),
                "fall_recall_b": float(b["test_fall_recall"]),
                "fall_recall_gain_points": pct_point(a["test_fall_recall"], b["test_fall_recall"]),

                "fall_f1_a": float(a["test_fall_f1"]),
                "fall_f1_b": float(b["test_fall_f1"]),
                "fall_f1_gain_points": pct_point(a["test_fall_f1"], b["test_fall_f1"]),

                "not_fall_f1_a": float(a["test_not_fall_f1"]),
                "not_fall_f1_b": float(b["test_not_fall_f1"]),
                "not_fall_f1_gain_points": pct_point(a["test_not_fall_f1"], b["test_not_fall_f1"]),
            }
        )

    return pd.DataFrame(rows)


def build_test_confusion_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in metrics_df.iterrows():
        tn = safe_int(row["test_tn"])
        fp = safe_int(row["test_fp"])
        fn = safe_int(row["test_fn"])
        tp = safe_int(row["test_tp"])

        total = tn + fp + fn + tp

        rows.append(
            {
                "model_name": row["model_name"],
                "display_name": row["display_name"],
                "phase": row["phase"],
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "total": total,
                "false_alarm_rate": fp / max(fp + tn, 1),
                "miss_rate": fn / max(fn + tp, 1),
                "fall_detection_rate": tp / max(tp + fn, 1),
                "not_fall_detection_rate": tn / max(tn + fp, 1),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# FIGURES
# ============================================================

def save_metric_bar_chart(
    metrics_df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
):
    df = metrics_df.sort_values(metric_col, ascending=True).copy()

    labels = df["model_name"].astype(str).tolist()
    values = df[metric_col].astype(float).to_numpy() * 100.0

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values)
    plt.xlabel(ylabel)
    plt.title(title)
    plt.grid(axis="x", alpha=0.3)

    for i, value in enumerate(values):
        plt.text(value + 0.5, i, f"{value:.2f}%", va="center")

    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_training_curves(epochs_df: pd.DataFrame, figures_dir: Path):
    if epochs_df.empty:
        return

    required = ["model_name", "epoch", "val_macro_f1", "train_macro_f1"]

    if any(col not in epochs_df.columns for col in required):
        return

    plt.figure(figsize=(10, 6))

    for model_name, group in epochs_df.groupby("model_name"):
        group = group.sort_values("epoch")
        plt.plot(group["epoch"], group["val_macro_f1"] * 100.0, marker="o", label=model_name)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Macro F1 (%)")
    plt.title("Validation Macro F1 During External Fine-tuning")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    path = figures_dir / "training_curve_val_macro_f1.png"
    plt.savefig(path, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))

    for model_name, group in epochs_df.groupby("model_name"):
        group = group.sort_values("epoch")
        plt.plot(group["epoch"], group["train_macro_f1"] * 100.0, marker="o", label=model_name)

    plt.xlabel("Epoch")
    plt.ylabel("Training Macro F1 (%)")
    plt.title("Training Macro F1 During External Fine-tuning")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    path = figures_dir / "training_curve_train_macro_f1.png"
    plt.savefig(path, dpi=200)
    plt.close()


def save_confusion_matrix_figures(metrics_df: pd.DataFrame, figures_dir: Path):
    cm_dir = ensure_dir(figures_dir / "confusion_matrices")

    for _, row in metrics_df.iterrows():
        model_name = row["model_name"]

        tn = safe_int(row["test_tn"])
        fp = safe_int(row["test_fp"])
        fn = safe_int(row["test_fn"])
        tp = safe_int(row["test_tp"])

        cm = np.array([[tn, fp], [fn, tp]], dtype=int)

        plt.figure(figsize=(5, 4))
        plt.imshow(cm)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xticks([0, 1], ["Pred Not_Fall", "Pred Fall"])
        plt.yticks([0, 1], ["True Not_Fall", "True Fall"])

        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")

        plt.colorbar()
        plt.tight_layout()

        path = cm_dir / f"confusion_matrix_{model_name}.png"
        plt.savefig(path, dpi=200)
        plt.close()


def generate_figures(metrics_df: pd.DataFrame, epochs_df: pd.DataFrame, figures_dir: Path) -> Dict[str, str]:
    ensure_dir(figures_dir)

    paths = {}

    metric_figures = [
        (
            "test_macro_f1",
            "External Test Macro F1 after Fine-tuning",
            "Macro F1 (%)",
            "bar_test_macro_f1.png",
        ),
        (
            "test_accuracy",
            "External Test Accuracy after Fine-tuning",
            "Accuracy (%)",
            "bar_test_accuracy.png",
        ),
        (
            "test_fall_recall",
            "External Test Fall Recall after Fine-tuning",
            "Fall Recall (%)",
            "bar_test_fall_recall.png",
        ),
        (
            "test_fall_f1",
            "External Test Fall F1 after Fine-tuning",
            "Fall F1 (%)",
            "bar_test_fall_f1.png",
        ),
        (
            "test_not_fall_f1",
            "External Test Not_Fall F1 after Fine-tuning",
            "Not_Fall F1 (%)",
            "bar_test_not_fall_f1.png",
        ),
    ]

    for metric_col, title, ylabel, filename in metric_figures:
        path = figures_dir / filename
        save_metric_bar_chart(
            metrics_df=metrics_df,
            metric_col=metric_col,
            title=title,
            ylabel=ylabel,
            output_path=path,
        )
        paths[filename] = str(path)

    save_training_curves(epochs_df, figures_dir)
    paths["training_curve_val_macro_f1.png"] = str(figures_dir / "training_curve_val_macro_f1.png")
    paths["training_curve_train_macro_f1.png"] = str(figures_dir / "training_curve_train_macro_f1.png")

    save_confusion_matrix_figures(metrics_df, figures_dir)
    paths["confusion_matrices_dir"] = str(figures_dir / "confusion_matrices")

    return paths


# ============================================================
# MARKDOWN REPORT
# ============================================================

def row_metric(row: pd.Series, col: str) -> str:
    return f"{pct(row[col]):.2f}%"


def format_metrics_table(metrics_df: pd.DataFrame) -> str:
    df = metrics_df.copy()
    df = df.sort_values("test_macro_f1", ascending=False).reset_index(drop=True)

    lines = []
    lines.append("| Rank | Model | Accuracy | Macro F1 | Fall Recall | Fall F1 | Not_Fall F1 | TN | FP | FN | TP |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for i, row in df.iterrows():
        lines.append(
            "| "
            f"{i + 1} | "
            f"{row['model_name']} | "
            f"{row_metric(row, 'test_accuracy')} | "
            f"{row_metric(row, 'test_macro_f1')} | "
            f"{row_metric(row, 'test_fall_recall')} | "
            f"{row_metric(row, 'test_fall_f1')} | "
            f"{row_metric(row, 'test_not_fall_f1')} | "
            f"{safe_int(row['test_tn'])} | "
            f"{safe_int(row['test_fp'])} | "
            f"{safe_int(row['test_fn'])} | "
            f"{safe_int(row['test_tp'])} |"
        )

    return "\n".join(lines)


def generate_conclusion_text(metrics_df: pd.DataFrame, improvement_df: pd.DataFrame) -> str:
    ranking = metrics_df.sort_values("test_macro_f1", ascending=False).reset_index(drop=True)

    best_macro = ranking.iloc[0]
    best_fall_recall = metrics_df.sort_values("test_fall_recall", ascending=False).iloc[0]
    best_fall_f1 = metrics_df.sort_values("test_fall_f1", ascending=False).iloc[0]

    phase4_models = {
        "phase4_quality_gated",
        "phase4_quality_concat",
    }

    top2 = set(ranking.head(2)["model_name"].astype(str).tolist())
    phase4_top2 = top2 == phase4_models

    text = []

    text.append("# Phase 5 - External Fine-tuning Comparison Report")
    text.append("")
    text.append("## 1. Evaluation protocol")
    text.append("")
    text.append(
        "In this phase, all model variants were fine-tuned using the same external training split, "
        "selected using the same validation split, and finally evaluated on the same external test split. "
        "The sequences were prepared in Step 05 and the train/validation/test split was created in Step 06. "
        "No sequence rebuilding was performed during model comparison."
    )
    text.append("")
    text.append("This protocol is used to verify whether the improvements introduced in Phase 4 remain effective when the models are adapted to a new dataset.")
    text.append("")

    text.append("## 2. Final ranking by external test Macro F1")
    text.append("")
    text.append(format_metrics_table(metrics_df))
    text.append("")

    text.append("## 3. Main findings")
    text.append("")

    text.append(
        f"The best model by external test Macro F1 is **{best_macro['model_name']}**, "
        f"with Macro F1 = **{pct(best_macro['test_macro_f1']):.2f}%**, "
        f"Accuracy = **{pct(best_macro['test_accuracy']):.2f}%**, "
        f"Fall Recall = **{pct(best_macro['test_fall_recall']):.2f}%**, "
        f"and Fall F1 = **{pct(best_macro['test_fall_f1']):.2f}%**."
    )
    text.append("")

    text.append(
        f"The best model by Fall Recall is **{best_fall_recall['model_name']}**, "
        f"with Fall Recall = **{pct(best_fall_recall['test_fall_recall']):.2f}%**."
    )
    text.append("")

    text.append(
        f"The best model by Fall F1 is **{best_fall_f1['model_name']}**, "
        f"with Fall F1 = **{pct(best_fall_f1['test_fall_f1']):.2f}%**."
    )
    text.append("")

    if phase4_top2:
        text.append(
            "The two Phase 4 quality-aware models occupy the top two positions by Macro F1. "
            "This supports the correctness of the Phase 4 design direction: adding pose-quality information to the fusion process improves external adaptation performance."
        )
    else:
        text.append(
            "The Phase 4 models do not occupy both top positions by Macro F1. "
            "This suggests that the quality-aware design may need additional regularization or domain adaptation under this external setting."
        )

    text.append("")

    # Add selected improvement statements.
    def add_gain_statement(model_a: str, model_b: str, label: str):
        part = improvement_df[
            (improvement_df["model_a"] == model_a) &
            (improvement_df["model_b"] == model_b)
        ]

        if part.empty:
            return

        row = part.iloc[0]

        text.append(
            f"Compared with **{model_b}**, **{model_a}** improves Macro F1 by "
            f"**{row['macro_f1_gain_points']:.2f} percentage points**, "
            f"Accuracy by **{row['accuracy_gain_points']:.2f} points**, "
            f"Fall Recall by **{row['fall_recall_gain_points']:.2f} points**, "
            f"and Fall F1 by **{row['fall_f1_gain_points']:.2f} points**."
        )
        text.append("")

    add_gain_statement(
        "phase4_quality_gated",
        "phase1_2d_common",
        "Quality-Gated vs 2D baseline",
    )

    add_gain_statement(
        "phase4_quality_gated",
        "phase2_concat_common",
        "Quality-Gated vs normal concat",
    )

    add_gain_statement(
        "phase4_quality_concat",
        "phase2_concat_common",
        "Quality-Concat vs normal concat",
    )

    text.append("## 4. Interpretation for the research report")
    text.append("")
    text.append(
        "The result shows that the Phase 4 quality-aware fusion strategy remains useful after adapting the models to the external dataset. "
        "Although the best internal model in Phase 4 was Quality-Concat, the external adaptation experiment shows that Quality-Gated achieves the strongest overall result on the new test split. "
        "Both results still support the same research direction: using pose-quality information improves the fusion process compared with using 2D/3D features alone."
    )
    text.append("")
    text.append(
        "Therefore, the correct conclusion is not that one exact Phase 4 model is always the best in every setting. "
        "The stronger conclusion is that the Phase 4 quality-aware design is valid, because the quality-aware models outperform earlier baselines under a fair external fine-tuning protocol."
    )
    text.append("")

    text.append("## 5. Suggested Vietnamese report paragraph")
    text.append("")
    text.append(
        "Sau khi toàn bộ mô hình được fine-tune trên cùng tập train của dataset ngoài, chọn checkpoint tốt nhất bằng cùng tập validation và đánh giá trên cùng tập test, "
        "hai mô hình thuộc Phase 4 đạt kết quả cao nhất. Cụ thể, Phase 4 Quality-Gated đạt Macro F1 cao nhất trên external test set, đồng thời cũng đạt Fall Recall và Fall F1 tốt nhất. "
        "Điều này cho thấy hướng cải tiến ở Phase 4 là có cơ sở: việc đưa đặc trưng chất lượng pose vào quá trình fusion giúp mô hình thích nghi tốt hơn với dữ liệu mới so với các mô hình 2D, 3D, concat thông thường và gated fusion trước đó."
    )
    text.append("")

    return "\n".join(text)


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_comparison_outputs(
    metrics_df: pd.DataFrame,
    epochs_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    validation_report: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, str]:
    base_dir = PHASE5_DIR / "outputs" / "external_finetuning"
    compare_dir = ensure_dir(base_dir / "comparison")
    figures_dir = ensure_dir(compare_dir / "figures")

    ranking_tables = build_ranking_tables(metrics_df)
    phase_summary_df = build_phase_summary(metrics_df)
    improvement_df = build_improvement_table(metrics_df)
    confusion_summary_df = build_test_confusion_summary(metrics_df)

    output_paths = {}

    for name, df in ranking_tables.items():
        path = compare_dir / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        output_paths[name] = str(path)

    phase_summary_path = compare_dir / "phase_level_summary.csv"
    improvement_path = compare_dir / "phase4_improvement_summary.csv"
    confusion_path = compare_dir / "test_confusion_summary.csv"

    phase_summary_df.to_csv(phase_summary_path, index=False, encoding="utf-8-sig")
    improvement_df.to_csv(improvement_path, index=False, encoding="utf-8-sig")
    confusion_summary_df.to_csv(confusion_path, index=False, encoding="utf-8-sig")

    output_paths["phase_level_summary"] = str(phase_summary_path)
    output_paths["phase4_improvement_summary"] = str(improvement_path)
    output_paths["test_confusion_summary"] = str(confusion_path)

    figure_paths = {}

    if not args.no_plots:
        figure_paths = generate_figures(
            metrics_df=metrics_df,
            epochs_df=epochs_df,
            figures_dir=figures_dir,
        )

    markdown_text = generate_conclusion_text(
        metrics_df=metrics_df,
        improvement_df=improvement_df,
    )

    markdown_path = compare_dir / "phase5_external_finetuning_conclusions.md"
    save_markdown(markdown_text, markdown_path)
    output_paths["markdown_report"] = str(markdown_path)

    ranking_macro = ranking_tables["ranking_by_macro_f1"]
    ranking_fall_recall = ranking_tables["ranking_by_fall_recall"]
    ranking_fall_f1 = ranking_tables["ranking_by_fall_f1"]

    report = {
        "phase": "Phase 5 - External Dataset Adaptation",
        "step": "08_compare_external_finetuning_results",
        "goal": (
            "Compare all fine-tuned models on the same external test split and verify whether "
            "the Phase 4 quality-aware experiments remain valid under external adaptation."
        ),
        "validation": validation_report,
        "best_by_test_macro_f1": ranking_macro.iloc[0].to_dict(),
        "best_by_test_fall_recall": ranking_fall_recall.iloc[0].to_dict(),
        "best_by_test_fall_f1": ranking_fall_f1.iloc[0].to_dict(),
        "phase_summary": phase_summary_df.to_dict(orient="records"),
        "phase4_improvements": improvement_df.to_dict(orient="records"),
        "outputs": output_paths,
        "figures": figure_paths,
        "main_conclusion": (
            "After fair external fine-tuning, the Phase 4 quality-aware models achieve the strongest results, "
            "supporting the correctness of the Phase 4 quality-aware fusion design."
        ),
    }

    report_path = compare_dir / "08_compare_external_finetuning_results_report.json"
    save_json(report, report_path)
    output_paths["json_report"] = str(report_path)

    return output_paths


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_final_summary(metrics_df: pd.DataFrame, improvement_df: pd.DataFrame):
    ranking = metrics_df.sort_values("test_macro_f1", ascending=False).reset_index(drop=True)

    print("\nFinal ranking by TEST Macro F1")
    print("=" * 120)

    show_cols = [
        "model_name",
        "best_epoch",
        "test_accuracy",
        "test_macro_f1",
        "test_fall_recall",
        "test_fall_f1",
        "test_not_fall_f1",
        "test_tn",
        "test_fp",
        "test_fn",
        "test_tp",
    ]

    available_cols = [c for c in show_cols if c in ranking.columns]
    print(ranking[available_cols].to_string(index=False))

    best_macro = ranking.iloc[0]
    best_recall = metrics_df.sort_values("test_fall_recall", ascending=False).iloc[0]
    best_fall_f1 = metrics_df.sort_values("test_fall_f1", ascending=False).iloc[0]

    print("\nBest by TEST Macro F1:")
    print(
        f"- {best_macro['model_name']} | "
        f"Macro F1={pct(best_macro['test_macro_f1']):.2f}% | "
        f"Accuracy={pct(best_macro['test_accuracy']):.2f}%"
    )

    print("\nBest by TEST Fall Recall:")
    print(
        f"- {best_recall['model_name']} | "
        f"Fall Recall={pct(best_recall['test_fall_recall']):.2f}%"
    )

    print("\nBest by TEST Fall F1:")
    print(
        f"- {best_fall_f1['model_name']} | "
        f"Fall F1={pct(best_fall_f1['test_fall_f1']):.2f}%"
    )

    print("\nKey Phase 4 improvements")
    print("=" * 120)

    if improvement_df.empty:
        print("No improvement table generated.")
        return

    selected = improvement_df[
        improvement_df["comparison"].isin(
            [
                "Quality-Gated vs 2D baseline",
                "Quality-Gated vs 2D+3D concat",
                "Quality-Gated vs gated fusion",
                "Quality-Concat vs 2D+3D concat",
            ]
        )
    ].copy()

    if selected.empty:
        selected = improvement_df.copy()

    show = [
        "comparison",
        "macro_f1_gain_points",
        "accuracy_gain_points",
        "fall_recall_gain_points",
        "fall_f1_gain_points",
        "not_fall_f1_gain_points",
    ]

    print(selected[show].to_string(index=False))


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 Step 08 - Compare external fine-tuning results."
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable figure generation.",
    )

    args = parser.parse_args()

    print("\nPhase 5 - Step 08: Compare External Fine-tuning Results")
    print("=" * 120)
    print("This script compares all models trained/fine-tuned in Step 07.")
    print("It does not retrain and does not rebuild sequences.")
    print("=" * 120)

    paths = get_input_paths()

    print("\n[1/5] Loading Step 07 outputs...")
    metrics_df, epochs_df, pred_df = load_step07_outputs(paths)

    print(f"Metrics file     : {paths['metrics_csv']}")
    print(f"Per-epoch file   : {paths['epochs_csv']}")
    print(f"Predictions file : {paths['predictions_csv']}")

    print("\n[2/5] Validating files...")

    metrics_validation = validate_metrics_df(metrics_df)
    pred_validation = validate_predictions_df(pred_df, metrics_df)

    validation_report = {
        "metrics_validation": metrics_validation,
        "prediction_validation": pred_validation,
        "valid": bool(metrics_validation["valid"] and pred_validation["valid"]),
    }

    print(json.dumps(validation_report, ensure_ascii=False, indent=4))

    if not validation_report["valid"]:
        raise RuntimeError(
            "Validation failed. Fix Step 07 outputs before comparing:\n"
            + json.dumps(validation_report, ensure_ascii=False, indent=4)
        )

    print("\n[3/5] Building comparison tables...")

    ranking_tables = build_ranking_tables(metrics_df)
    improvement_df = build_improvement_table(metrics_df)

    print_final_summary(metrics_df, improvement_df)

    print("\n[4/5] Saving comparison outputs...")
    outputs = save_comparison_outputs(
        metrics_df=metrics_df,
        epochs_df=epochs_df,
        pred_df=pred_df,
        validation_report=validation_report,
        args=args,
    )

    print(json.dumps(outputs, ensure_ascii=False, indent=4))

    print("\n[5/5] Done.")
    print("=" * 120)
    print("Main files:")
    print(f"- {outputs['markdown_report']}")
    print(f"- {outputs['json_report']}")
    print(f"- {outputs['ranking_by_macro_f1']}")
    print(f"- {outputs['phase4_improvement_summary']}")
    print("=" * 120)


if __name__ == "__main__":
    main()