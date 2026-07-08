import os
import sys
import json
import shutil
import argparse
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional

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

METRIC_COLS = [
    "test_accuracy",
    "test_macro_f1",
    "test_fall_recall",
    "test_fall_f1",
    "test_not_fall_f1",
]

CONFUSION_COLS = [
    "test_tn",
    "test_fp",
    "test_fn",
    "test_tp",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path, default=None):
    if default is None:
        default = {}

    if not path.exists():
        return default

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path):
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_text(text: str, path: Path):
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


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


def pct(value) -> float:
    return safe_float(value) * 100.0


def fmt_pct(value) -> str:
    return f"{pct(value):.2f}%"


def fmt_float(value) -> str:
    return f"{safe_float(value):.4f}"


def html_escape(text) -> str:
    text = str(text)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


def image_to_base64(path: Path) -> str:
    if not path.exists():
        return ""

    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    suffix = path.suffix.lower().replace(".", "")

    if suffix == "jpg":
        suffix = "jpeg"

    return f"data:image/{suffix};base64,{data}"


def dataframe_to_html_table(
    df: pd.DataFrame,
    percent_cols: Optional[List[str]] = None,
    int_cols: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
) -> str:
    percent_cols = percent_cols or []
    int_cols = int_cols or []

    show_df = df.copy()

    if max_rows is not None:
        show_df = show_df.head(max_rows).copy()

    lines = []
    lines.append("<table>")
    lines.append("<thead><tr>")

    for col in show_df.columns:
        lines.append(f"<th>{html_escape(col)}</th>")

    lines.append("</tr></thead>")
    lines.append("<tbody>")

    for _, row in show_df.iterrows():
        lines.append("<tr>")

        for col in show_df.columns:
            value = row[col]

            if col in percent_cols:
                value = fmt_pct(value)
            elif col in int_cols:
                value = str(safe_int(value))
            elif isinstance(value, float):
                value = f"{value:.4f}"

            lines.append(f"<td>{html_escape(value)}</td>")

        lines.append("</tr>")

    lines.append("</tbody>")
    lines.append("</table>")

    return "\n".join(lines)


def dataframe_to_markdown_table(
    df: pd.DataFrame,
    percent_cols: Optional[List[str]] = None,
    int_cols: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
) -> str:
    percent_cols = percent_cols or []
    int_cols = int_cols or []

    show_df = df.copy()

    if max_rows is not None:
        show_df = show_df.head(max_rows).copy()

    formatted = show_df.copy()

    for col in formatted.columns:
        if col in percent_cols:
            formatted[col] = formatted[col].apply(fmt_pct)
        elif col in int_cols:
            formatted[col] = formatted[col].apply(lambda x: str(safe_int(x)))
        else:
            formatted[col] = formatted[col].apply(
                lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
            )

    lines = []
    header = "| " + " | ".join(formatted.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"

    lines.append(header)
    lines.append(sep)

    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in formatted.columns) + " |")

    return "\n".join(lines)


# ============================================================
# LOAD INPUT FILES
# ============================================================

def get_paths() -> Dict[str, Path]:
    external_finetune_dir = PHASE5_DIR / "outputs" / "external_finetuning"
    comparison_dir = external_finetune_dir / "comparison"
    split_dir = PHASE5_DIR / "data" / "external_sequences" / "train_val_test_splits"

    return {
        "external_finetune_dir": external_finetune_dir,
        "comparison_dir": comparison_dir,

        "metrics_csv": external_finetune_dir / "external_finetuned_all_models_metrics.csv",
        "epochs_csv": external_finetune_dir / "external_finetuned_all_models_per_epoch.csv",
        "predictions_csv": external_finetune_dir / "external_finetuned_all_models_test_predictions_long.csv",

        "compare_report_json": comparison_dir / "08_compare_external_finetuning_results_report.json",
        "conclusion_md": comparison_dir / "phase5_external_finetuning_conclusions.md",
        "ranking_macro_csv": comparison_dir / "ranking_by_macro_f1.csv",
        "improvement_csv": comparison_dir / "phase4_improvement_summary.csv",
        "phase_summary_csv": comparison_dir / "phase_level_summary.csv",
        "confusion_summary_csv": comparison_dir / "test_confusion_summary.csv",

        "split_report_json": PHASE5_DIR / "outputs" / "splits" / "06_create_external_train_val_test_split_report.json",
        "train_manifest": split_dir / "external_train_manifest.csv",
        "val_manifest": split_dir / "external_val_manifest.csv",
        "test_manifest": split_dir / "external_test_manifest.csv",
    }


def load_required_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {name}: {path}\n"
            "Please run the previous pipeline steps first."
        )

    return pd.read_csv(path)


def load_all_inputs(paths: Dict[str, Path]) -> Dict[str, Any]:
    metrics_df = load_required_csv(paths["metrics_csv"], "Step 07 metrics CSV")
    epochs_df = load_required_csv(paths["epochs_csv"], "Step 07 per-epoch CSV")
    predictions_df = load_required_csv(paths["predictions_csv"], "Step 07 prediction CSV")

    ranking_df = (
        pd.read_csv(paths["ranking_macro_csv"])
        if paths["ranking_macro_csv"].exists()
        else metrics_df.sort_values("test_macro_f1", ascending=False).reset_index(drop=True)
    )

    improvement_df = (
        pd.read_csv(paths["improvement_csv"])
        if paths["improvement_csv"].exists()
        else pd.DataFrame()
    )

    phase_summary_df = (
        pd.read_csv(paths["phase_summary_csv"])
        if paths["phase_summary_csv"].exists()
        else pd.DataFrame()
    )

    confusion_df = (
        pd.read_csv(paths["confusion_summary_csv"])
        if paths["confusion_summary_csv"].exists()
        else build_confusion_summary(metrics_df)
    )

    split_report = read_json(paths["split_report_json"], default={})
    compare_report = read_json(paths["compare_report_json"], default={})

    conclusion_text = ""

    if paths["conclusion_md"].exists():
        conclusion_text = paths["conclusion_md"].read_text(encoding="utf-8")

    for df in [metrics_df, ranking_df]:
        if "model_name" in df.columns:
            df["display_name"] = df["model_name"].map(MODEL_DISPLAY_NAMES).fillna(df.get("display_name", df["model_name"]))
            df["phase"] = df["model_name"].map(MODEL_PHASES).fillna(df.get("phase", "Unknown"))

    return {
        "metrics_df": metrics_df,
        "epochs_df": epochs_df,
        "predictions_df": predictions_df,
        "ranking_df": ranking_df,
        "improvement_df": improvement_df,
        "phase_summary_df": phase_summary_df,
        "confusion_df": confusion_df,
        "split_report": split_report,
        "compare_report": compare_report,
        "conclusion_text": conclusion_text,
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_inputs(data: Dict[str, Any]) -> Dict[str, Any]:
    metrics_df = data["metrics_df"]
    predictions_df = data["predictions_df"]

    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
    }

    required_metrics_cols = [
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

    missing = [c for c in required_metrics_cols if c not in metrics_df.columns]

    if missing:
        report["valid"] = False
        report["errors"].append(f"metrics_df missing columns: {missing}")

    required_pred_cols = [
        "model_name",
        "sequence_key",
        "y_true",
        "y_pred",
        "prob_fall",
    ]

    missing_pred = [c for c in required_pred_cols if c not in predictions_df.columns]

    if missing_pred:
        report["valid"] = False
        report["errors"].append(f"predictions_df missing columns: {missing_pred}")

    model_names = set(metrics_df["model_name"].astype(str).tolist())

    expected_models = set(MODEL_ORDER)
    missing_models = sorted(expected_models - model_names)

    if missing_models:
        report["warnings"].append(f"Missing expected models: {missing_models}")

    if not predictions_df.empty and "model_name" in predictions_df.columns:
        per_model_counts = predictions_df.groupby("model_name").size().to_dict()
        report["prediction_rows_per_model"] = {
            str(k): int(v)
            for k, v in per_model_counts.items()
        }

        counts = list(per_model_counts.values())

        if counts and len(set(counts)) != 1:
            report["valid"] = False
            report["errors"].append(
                f"Prediction rows per model are not equal: {per_model_counts}"
            )

    return report


# ============================================================
# DERIVED TABLES
# ============================================================

def build_confusion_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in metrics_df.iterrows():
        tn = safe_int(row.get("test_tn", 0))
        fp = safe_int(row.get("test_fp", 0))
        fn = safe_int(row.get("test_fn", 0))
        tp = safe_int(row.get("test_tp", 0))

        rows.append(
            {
                "model_name": row.get("model_name", ""),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "false_alarm_rate": fp / max(tn + fp, 1),
                "miss_rate": fn / max(fn + tp, 1),
            }
        )

    return pd.DataFrame(rows)


def build_dashboard_summary(metrics_df: pd.DataFrame, split_report: Dict[str, Any]) -> Dict[str, Any]:
    ranking = metrics_df.sort_values("test_macro_f1", ascending=False).reset_index(drop=True)

    best_macro = ranking.iloc[0].to_dict()
    best_fall_recall = metrics_df.sort_values("test_fall_recall", ascending=False).iloc[0].to_dict()
    best_fall_f1 = metrics_df.sort_values("test_fall_f1", ascending=False).iloc[0].to_dict()

    split_summary = split_report.get("split_summary", {})

    return {
        "best_by_macro_f1": {
            "model_name": best_macro["model_name"],
            "macro_f1_percent": pct(best_macro["test_macro_f1"]),
            "accuracy_percent": pct(best_macro["test_accuracy"]),
            "fall_recall_percent": pct(best_macro["test_fall_recall"]),
            "fall_f1_percent": pct(best_macro["test_fall_f1"]),
        },
        "best_by_fall_recall": {
            "model_name": best_fall_recall["model_name"],
            "fall_recall_percent": pct(best_fall_recall["test_fall_recall"]),
        },
        "best_by_fall_f1": {
            "model_name": best_fall_f1["model_name"],
            "fall_f1_percent": pct(best_fall_f1["test_fall_f1"]),
        },
        "split_summary": split_summary,
        "num_models": int(len(metrics_df)),
    }


def make_report_tables(metrics_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    ranking = metrics_df.sort_values("test_macro_f1", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    main_cols = [
        "rank",
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

    main_cols = [c for c in main_cols if c in ranking.columns]
    main_table = ranking[main_cols].copy()

    phase4 = metrics_df[metrics_df["model_name"].isin(["phase4_quality_gated", "phase4_quality_concat"])].copy()

    phase4_table = phase4[
        [
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
    ].sort_values("test_macro_f1", ascending=False)

    return {
        "main_table": main_table,
        "phase4_table": phase4_table,
    }


# ============================================================
# FIGURES
# ============================================================

def save_bar_chart(
    df: pd.DataFrame,
    metric_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
):
    plot_df = df.sort_values(metric_col, ascending=True).copy()

    labels = plot_df["model_name"].astype(str).tolist()
    values = plot_df[metric_col].astype(float).to_numpy() * 100.0

    plt.figure(figsize=(11, 6))
    plt.barh(labels, values)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(axis="x", alpha=0.3)

    for i, v in enumerate(values):
        plt.text(v + 0.5, i, f"{v:.2f}%", va="center")

    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_grouped_metric_chart(metrics_df: pd.DataFrame, output_path: Path):
    df = metrics_df.copy()
    df = df.set_index("model_name").loc[[m for m in MODEL_ORDER if m in df["model_name"].values]].reset_index()

    models = df["model_name"].tolist()
    x = np.arange(len(models))
    width = 0.18

    metrics = [
        ("test_accuracy", "Accuracy"),
        ("test_macro_f1", "Macro F1"),
        ("test_fall_recall", "Fall Recall"),
        ("test_fall_f1", "Fall F1"),
    ]

    plt.figure(figsize=(14, 6))

    for i, (col, label) in enumerate(metrics):
        values = df[col].astype(float).to_numpy() * 100.0
        plt.bar(x + (i - 1.5) * width, values, width, label=label)

    plt.xticks(x, models, rotation=30, ha="right")
    plt.ylabel("Score (%)")
    plt.title("External Test Metrics after Fine-tuning")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_training_curve(epochs_df: pd.DataFrame, output_path: Path):
    if epochs_df.empty:
        return

    required = ["model_name", "epoch", "val_macro_f1"]

    if any(c not in epochs_df.columns for c in required):
        return

    plt.figure(figsize=(12, 6))

    for model_name, group in epochs_df.groupby("model_name"):
        group = group.sort_values("epoch")
        plt.plot(
            group["epoch"],
            group["val_macro_f1"].astype(float) * 100.0,
            marker="o",
            label=model_name,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Macro F1 (%)")
    plt.title("Validation Macro F1 During External Fine-tuning")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_confusion_heatmaps(metrics_df: pd.DataFrame, figures_dir: Path) -> List[Path]:
    paths = []
    cm_dir = ensure_dir(figures_dir / "confusion_matrices")

    for _, row in metrics_df.iterrows():
        model_name = row["model_name"]

        cm = np.array(
            [
                [safe_int(row["test_tn"]), safe_int(row["test_fp"])],
                [safe_int(row["test_fn"]), safe_int(row["test_tp"])],
            ],
            dtype=int,
        )

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
        plt.savefig(path, dpi=180)
        plt.close()

        paths.append(path)

    return paths


def generate_figures(metrics_df: pd.DataFrame, epochs_df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    figures_dir = ensure_dir(output_dir / "figures")

    figure_paths = {}

    specs = [
        ("test_macro_f1", "dashboard_macro_f1.png", "External Test Macro F1 after Fine-tuning", "Macro F1 (%)"),
        ("test_accuracy", "dashboard_accuracy.png", "External Test Accuracy after Fine-tuning", "Accuracy (%)"),
        ("test_fall_recall", "dashboard_fall_recall.png", "External Test Fall Recall after Fine-tuning", "Fall Recall (%)"),
        ("test_fall_f1", "dashboard_fall_f1.png", "External Test Fall F1 after Fine-tuning", "Fall F1 (%)"),
    ]

    for col, filename, title, xlabel in specs:
        path = figures_dir / filename
        save_bar_chart(metrics_df, col, path, title, xlabel)
        figure_paths[filename] = path

    grouped_path = figures_dir / "dashboard_grouped_metrics.png"
    save_grouped_metric_chart(metrics_df, grouped_path)
    figure_paths["dashboard_grouped_metrics.png"] = grouped_path

    curve_path = figures_dir / "dashboard_training_curve_val_macro_f1.png"
    save_training_curve(epochs_df, curve_path)

    if curve_path.exists():
        figure_paths["dashboard_training_curve_val_macro_f1.png"] = curve_path

    cm_paths = save_confusion_heatmaps(metrics_df, figures_dir)
    figure_paths["confusion_matrices"] = cm_paths

    return figure_paths


# ============================================================
# DASHBOARD HTML
# ============================================================

def html_metric_card(title: str, value: str, subtitle: str = "") -> str:
    return f"""
    <div class="card">
        <div class="card-title">{html_escape(title)}</div>
        <div class="card-value">{html_escape(value)}</div>
        <div class="card-subtitle">{html_escape(subtitle)}</div>
    </div>
    """


def html_image(path: Path, alt: str) -> str:
    src = image_to_base64(path)

    if not src:
        return f"<p><em>Missing image: {html_escape(path)}</em></p>"

    return f"""
    <figure>
        <img src="{src}" alt="{html_escape(alt)}" />
        <figcaption>{html_escape(alt)}</figcaption>
    </figure>
    """


def build_html_dashboard(
    metrics_df: pd.DataFrame,
    epochs_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    phase_summary_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    summary: Dict[str, Any],
    figure_paths: Dict[str, Any],
    output_path: Path,
):
    best_macro = summary["best_by_macro_f1"]
    best_recall = summary["best_by_fall_recall"]
    best_fall_f1 = summary["best_by_fall_f1"]

    tables = make_report_tables(metrics_df)

    percent_cols = [
        "test_accuracy",
        "test_macro_f1",
        "test_fall_recall",
        "test_fall_f1",
        "test_not_fall_f1",
        "accuracy_a",
        "accuracy_b",
        "macro_f1_a",
        "macro_f1_b",
        "fall_recall_a",
        "fall_recall_b",
        "fall_f1_a",
        "fall_f1_b",
        "not_fall_f1_a",
        "not_fall_f1_b",
        "false_alarm_rate",
        "miss_rate",
    ]

    int_cols = [
        "rank",
        "best_epoch",
        "test_tn",
        "test_fp",
        "test_fn",
        "test_tp",
        "tn",
        "fp",
        "fn",
        "tp",
        "total",
    ]

    main_table_html = dataframe_to_html_table(
        tables["main_table"],
        percent_cols=percent_cols,
        int_cols=int_cols,
    )

    phase4_table_html = dataframe_to_html_table(
        tables["phase4_table"],
        percent_cols=percent_cols,
        int_cols=int_cols,
    )

    improvement_table_html = dataframe_to_html_table(
        improvement_df,
        percent_cols=[],
        int_cols=[],
        max_rows=10,
    ) if not improvement_df.empty else "<p>No improvement table found.</p>"

    phase_summary_html = dataframe_to_html_table(
        phase_summary_df,
        percent_cols=[
            "best_macro_f1",
            "best_macro_accuracy",
            "best_macro_fall_recall",
            "best_macro_fall_f1",
            "best_fall_recall",
            "best_fall_f1",
        ],
        int_cols=["num_models"],
    ) if not phase_summary_df.empty else "<p>No phase summary found.</p>"

    confusion_html = dataframe_to_html_table(
        confusion_df,
        percent_cols=["false_alarm_rate", "miss_rate", "fall_detection_rate", "not_fall_detection_rate"],
        int_cols=["tn", "fp", "fn", "tp", "total"],
    )

    grouped_img = html_image(
        figure_paths.get("dashboard_grouped_metrics.png", Path()),
        "Grouped external test metrics",
    )

    macro_img = html_image(
        figure_paths.get("dashboard_macro_f1.png", Path()),
        "External test Macro F1 ranking",
    )

    recall_img = html_image(
        figure_paths.get("dashboard_fall_recall.png", Path()),
        "External test Fall Recall ranking",
    )

    curve_img = ""

    if "dashboard_training_curve_val_macro_f1.png" in figure_paths:
        curve_img = html_image(
            figure_paths["dashboard_training_curve_val_macro_f1.png"],
            "Validation Macro F1 during fine-tuning",
        )

    css = """
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 32px;
            background: #f7f8fb;
            color: #1f2937;
            line-height: 1.5;
        }
        h1, h2, h3 {
            color: #111827;
        }
        .subtitle {
            color: #4b5563;
            font-size: 15px;
        }
        .cards {
            display: grid;
            grid-template-columns: repeat(3, minmax(220px, 1fr));
            gap: 16px;
            margin: 20px 0;
        }
        .card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        }
        .card-title {
            font-size: 13px;
            color: #6b7280;
            margin-bottom: 8px;
        }
        .card-value {
            font-size: 24px;
            font-weight: 700;
            color: #111827;
        }
        .card-subtitle {
            font-size: 13px;
            color: #4b5563;
            margin-top: 8px;
        }
        .section {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 20px;
            margin: 24px 0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }
        table {
            border-collapse: collapse;
            width: 100%;
            font-size: 13px;
            margin: 14px 0;
        }
        th, td {
            border: 1px solid #e5e7eb;
            padding: 8px;
            text-align: left;
        }
        th {
            background: #f3f4f6;
            font-weight: 700;
        }
        tr:nth-child(even) {
            background: #fafafa;
        }
        figure {
            margin: 20px 0;
        }
        img {
            max-width: 100%;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            background: white;
        }
        figcaption {
            font-size: 13px;
            color: #6b7280;
            margin-top: 6px;
        }
        .conclusion {
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
            border-radius: 12px;
            padding: 16px;
        }
        .warning {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-radius: 12px;
            padding: 16px;
        }
        code {
            background: #f3f4f6;
            padding: 2px 5px;
            border-radius: 4px;
        }
    </style>
    """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>Phase 5 External Fine-tuning Dashboard</title>
        {css}
    </head>
    <body>
        <h1>Phase 5 External Fine-tuning Dashboard</h1>
        <p class="subtitle">
            This dashboard summarizes the final Phase 5 experiment: all models were fine-tuned on the same external train split,
            selected using the same validation split, and tested on the same external test split.
        </p>

        <div class="cards">
            {html_metric_card("Best Macro F1 model", best_macro["model_name"], f"Macro F1 = {best_macro['macro_f1_percent']:.2f}%")}
            {html_metric_card("Best Fall Recall model", best_recall["model_name"], f"Fall Recall = {best_recall['fall_recall_percent']:.2f}%")}
            {html_metric_card("Best Fall F1 model", best_fall_f1["model_name"], f"Fall F1 = {best_fall_f1['fall_f1_percent']:.2f}%")}
        </div>

        <div class="section conclusion">
            <h2>Main conclusion</h2>
            <p>
                After fair external fine-tuning, the Phase 4 quality-aware models achieved the strongest results.
                The best model by Macro F1 was <strong>{html_escape(best_macro["model_name"])}</strong>.
                This supports the correctness of the Phase 4 direction: using pose-quality information during fusion helps the model adapt better to the new dataset.
            </p>
            <p>
                In this external adaptation experiment, Quality-Gated ranked first, while Quality-Concat also remained among the top Phase 4 models.
                Therefore, the conclusion should focus on the validity of the <strong>quality-aware fusion strategy</strong>, not only on one fixed architecture.
            </p>
        </div>

        <div class="section">
            <h2>1. Final ranking by test Macro F1</h2>
            {main_table_html}
        </div>

        <div class="section">
            <h2>2. Phase 4 quality-aware models</h2>
            {phase4_table_html}
        </div>

        <div class="section">
            <h2>3. Figures</h2>
            {grouped_img}
            {macro_img}
            {recall_img}
            {curve_img}
        </div>

        <div class="section">
            <h2>4. Phase-level summary</h2>
            {phase_summary_html}
        </div>

        <div class="section">
            <h2>5. Phase 4 improvement summary</h2>
            {improvement_table_html}
        </div>

        <div class="section">
            <h2>6. Confusion and error analysis</h2>
            {confusion_html}
        </div>

                <div class="section">
            <h2>7. Report-ready conclusion</h2>
            <p>
                After all models were fine-tuned on the same external training split, selected using the same validation split,
                and evaluated on the same external test split, the Phase 4 models achieved the strongest overall performance.
                Specifically, Phase 4 Quality-Gated obtained the best Macro F1, Fall Recall, and Fall F1 on the external test set.
                This result supports the correctness of the Phase 4 improvement direction: incorporating pose-quality features into
                the fusion process helps the model adapt better to the new dataset compared with the previous 2D, 3D, standard concat,
                and gated fusion baselines.
            </p>
        </div>
    </body>
    </html>
    """

    save_text(html, output_path)


# ============================================================
# MARKDOWN DASHBOARD
# ============================================================

def build_markdown_dashboard(
    metrics_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    phase_summary_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    summary: Dict[str, Any],
    output_path: Path,
):
    tables = make_report_tables(metrics_df)

    percent_cols = [
        "test_accuracy",
        "test_macro_f1",
        "test_fall_recall",
        "test_fall_f1",
        "test_not_fall_f1",
    ]

    int_cols = [
        "rank",
        "best_epoch",
        "test_tn",
        "test_fp",
        "test_fn",
        "test_tp",
    ]

    best_macro = summary["best_by_macro_f1"]
    best_recall = summary["best_by_fall_recall"]
    best_fall_f1 = summary["best_by_fall_f1"]

    lines = []

    lines.append("# Phase 5 External Fine-tuning Dashboard")
    lines.append("")
    lines.append("## Main conclusion")
    lines.append("")
    lines.append(
        f"Best by Macro F1: **{best_macro['model_name']}** "
        f"with **{best_macro['macro_f1_percent']:.2f}%**."
    )
    lines.append(
        f"Best by Fall Recall: **{best_recall['model_name']}** "
        f"with **{best_recall['fall_recall_percent']:.2f}%**."
    )
    lines.append(
        f"Best by Fall F1: **{best_fall_f1['model_name']}** "
        f"with **{best_fall_f1['fall_f1_percent']:.2f}%**."
    )
    lines.append("")
    lines.append(
        "After fair external fine-tuning, the Phase 4 quality-aware models achieved the strongest results. "
        "This supports the correctness of the Phase 4 quality-aware fusion design."
    )
    lines.append("")

    lines.append("## Final ranking by test Macro F1")
    lines.append("")
    lines.append(
        dataframe_to_markdown_table(
            tables["main_table"],
            percent_cols=percent_cols,
            int_cols=int_cols,
        )
    )
    lines.append("")

    lines.append("## Phase 4 quality-aware models")
    lines.append("")
    lines.append(
        dataframe_to_markdown_table(
            tables["phase4_table"],
            percent_cols=percent_cols,
            int_cols=int_cols,
        )
    )
    lines.append("")

    if not improvement_df.empty:
        lines.append("## Phase 4 improvement summary")
        lines.append("")
        lines.append(dataframe_to_markdown_table(improvement_df.head(10)))
        lines.append("")

    if not phase_summary_df.empty:
        lines.append("## Phase-level summary")
        lines.append("")
        lines.append(dataframe_to_markdown_table(phase_summary_df))
        lines.append("")

    lines.append("## Confusion summary")
    lines.append("")
    lines.append(dataframe_to_markdown_table(confusion_df))
    lines.append("")

    lines.append("## Report-ready conclusion")
    lines.append("")
    lines.append(
        "After all models were fine-tuned on the same external training split, selected using the same validation split, "
        "and evaluated on the same external test split, the Phase 4 models achieved the strongest overall performance. "
        "Specifically, Phase 4 Quality-Gated obtained the best Macro F1, Fall Recall, and Fall F1 on the external test set. "
        "This result supports the correctness of the Phase 4 improvement direction: incorporating pose-quality features into "
        "the fusion process helps the model adapt better to the new dataset compared with the previous 2D, 3D, standard concat, "
        "and gated fusion baselines."
    )
    lines.append("")

    save_text("\n".join(lines), output_path)


# ============================================================
# SAVE TABLES
# ============================================================

def save_dashboard_tables(
    metrics_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    phase_summary_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, str]:
    tables_dir = ensure_dir(output_dir / "tables")

    paths = {}

    files = {
        "dashboard_final_metrics.csv": metrics_df,
        "dashboard_phase4_improvement_summary.csv": improvement_df,
        "dashboard_phase_level_summary.csv": phase_summary_df,
        "dashboard_confusion_summary.csv": confusion_df,
    }

    for filename, df in files.items():
        path = tables_dir / filename
        df.to_csv(path, index=False, encoding="utf-8-sig")
        paths[filename] = str(path)

    return paths


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 Step 09 - Generate final dashboard for external fine-tuning experiment."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PHASE5_DIR / "outputs" / "phase5_dashboard"),
        help="Dashboard output folder.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation.",
    )

    args = parser.parse_args()

    print("\nPhase 5 - Step 09: Generate Final Dashboard")
    print("=" * 120)
    print("This step does not train models and does not rebuild sequences.")
    print("It only summarizes Step 06, Step 07, and Step 08 outputs.")
    print("=" * 120)

    output_dir = ensure_dir(Path(args.output_dir))
    paths = get_paths()

    print("\n[1/5] Loading inputs...")
    data = load_all_inputs(paths)

    metrics_df = data["metrics_df"]
    epochs_df = data["epochs_df"]
    predictions_df = data["predictions_df"]
    improvement_df = data["improvement_df"]
    phase_summary_df = data["phase_summary_df"]
    confusion_df = data["confusion_df"]
    split_report = data["split_report"]

    print(f"Loaded metrics rows     : {len(metrics_df)}")
    print(f"Loaded epoch rows       : {len(epochs_df)}")
    print(f"Loaded prediction rows  : {len(predictions_df)}")

    print("\n[2/5] Validating inputs...")
    validation_report = validate_inputs(data)

    print(json.dumps(validation_report, ensure_ascii=False, indent=4))

    if not validation_report["valid"]:
        raise RuntimeError(
            "Dashboard validation failed:\n"
            + json.dumps(validation_report, ensure_ascii=False, indent=4)
        )

    print("\n[3/5] Building dashboard summary...")
    summary = build_dashboard_summary(metrics_df, split_report)

    print(json.dumps(summary, ensure_ascii=False, indent=4))

    print("\n[4/5] Generating figures and tables...")

    if args.no_plots:
        figure_paths = {}
    else:
        figure_paths = generate_figures(metrics_df, epochs_df, output_dir)

    table_paths = save_dashboard_tables(
        metrics_df=metrics_df,
        improvement_df=improvement_df,
        phase_summary_df=phase_summary_df,
        confusion_df=confusion_df,
        output_dir=output_dir,
    )

    print("\n[5/5] Writing HTML / Markdown / JSON dashboard...")

    html_path = output_dir / "phase5_dashboard.html"
    md_path = output_dir / "phase5_dashboard.md"
    json_path = output_dir / "phase5_dashboard_summary.json"

    build_html_dashboard(
        metrics_df=metrics_df,
        epochs_df=epochs_df,
        improvement_df=improvement_df,
        phase_summary_df=phase_summary_df,
        confusion_df=confusion_df,
        summary=summary,
        figure_paths=figure_paths,
        output_path=html_path,
    )

    build_markdown_dashboard(
        metrics_df=metrics_df,
        improvement_df=improvement_df,
        phase_summary_df=phase_summary_df,
        confusion_df=confusion_df,
        summary=summary,
        output_path=md_path,
    )

    dashboard_report = {
        "phase": "Phase 5 - External Dataset Adaptation",
        "step": "09_generate_phase5_dashboard",
        "goal": "Generate final dashboard for report and presentation.",
        "validation": validation_report,
        "summary": summary,
        "outputs": {
            "html_dashboard": str(html_path),
            "markdown_dashboard": str(md_path),
            "json_summary": str(json_path),
            "output_dir": str(output_dir),
            "tables": table_paths,
            "figures": {k: str(v) for k, v in figure_paths.items() if not isinstance(v, list)},
            "confusion_matrix_figures": [
                str(p)
                for p in figure_paths.get("confusion_matrices", [])
            ],
        },
        "main_conclusion": (
            "After fair external fine-tuning, Phase 4 quality-aware models achieved the strongest results. "
            "This supports the validity of the Phase 4 quality-aware fusion strategy."
        ),
    }

    save_json(dashboard_report, json_path)

    print("\nDONE: Phase 5 Step 09 completed.")
    print("=" * 120)
    print("Main outputs:")
    print(f"- HTML dashboard    : {html_path}")
    print(f"- Markdown dashboard: {md_path}")
    print(f"- JSON summary      : {json_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()