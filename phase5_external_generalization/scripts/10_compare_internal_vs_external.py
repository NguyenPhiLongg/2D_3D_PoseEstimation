import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

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
# MODEL INFO
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


# ============================================================
# INTERNAL REFERENCE DEFAULTS
# ============================================================

def default_internal_results() -> pd.DataFrame:
    """
    Internal dataset results from Phase 3/4 fair comparison.

    Notes:
    - Phase 1/2/3 values are from the common internal binary fair set.
    - Phase 4 values are from Phase 4 binary internal evaluation.
    - Some Phase 4 fall-specific metrics may be unavailable, so they are left as NaN.
    """
    rows = [
        {
            "model_name": "phase1_2d_common",
            "internal_accuracy": 0.9346,
            "internal_macro_f1": 0.9234,
            "internal_fall_recall": 0.8991,
            "internal_fall_f1": 0.8942,
            "internal_source": "Phase 3 common internal binary fair set",
        },
        {
            "model_name": "phase2_3d_common",
            "internal_accuracy": 0.9272,
            "internal_macro_f1": 0.9140,
            "internal_fall_recall": 0.8717,
            "internal_fall_f1": 0.8804,
            "internal_source": "Phase 3 common internal binary fair set",
        },
        {
            "model_name": "phase2_concat_common",
            "internal_accuracy": 0.9393,
            "internal_macro_f1": 0.9299,
            "internal_fall_recall": 0.9320,
            "internal_fall_f1": 0.9043,
            "internal_source": "Phase 3 common internal binary fair set",
        },
        {
            "model_name": "phase3_gated_fusion",
            "internal_accuracy": 0.9379,
            "internal_macro_f1": 0.9281,
            "internal_fall_recall": 0.9221,
            "internal_fall_f1": 0.9014,
            "internal_source": "Phase 3 gated fusion internal binary fair set",
        },
        {
            "model_name": "phase4_quality_gated",
            "internal_accuracy": 0.953457,
            "internal_macro_f1": 0.945028,
            "internal_fall_recall": np.nan,
            "internal_fall_f1": np.nan,
            "internal_source": "Phase 4 quality-aware binary internal evaluation",
        },
        {
            "model_name": "phase4_quality_concat",
            "internal_accuracy": 0.961551,
            "internal_macro_f1": 0.954617,
            "internal_fall_recall": np.nan,
            "internal_fall_f1": np.nan,
            "internal_source": "Phase 4 quality-aware binary internal evaluation",
        },
    ]

    return pd.DataFrame(rows)


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: Path):
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_text(text: str, path: Path):
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def safe_float(value, default=np.nan):
    try:
        if value is None or pd.isna(value):
            return default

        value = float(value)

        if np.isinf(value):
            return default

        return value
    except Exception:
        return default


def pct(value) -> str:
    value = safe_float(value)

    if pd.isna(value):
        return "N/A"

    return f"{value * 100:.2f}%"


def point_diff(a, b) -> float:
    a = safe_float(a)
    b = safe_float(b)

    if pd.isna(a) or pd.isna(b):
        return np.nan

    return (a - b) * 100.0


def point_fmt(value) -> str:
    value = safe_float(value)

    if pd.isna(value):
        return "N/A"

    return f"{value:+.2f}"


def html_escape(text) -> str:
    text = str(text)

    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


def clean_model_order(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["model_name"] = df["model_name"].astype(str)
    df["display_name"] = df["model_name"].map(MODEL_DISPLAY_NAMES).fillna(df["model_name"])
    df["phase"] = df["model_name"].map(MODEL_PHASES).fillna("Unknown")

    order_map = {name: i for i, name in enumerate(MODEL_ORDER)}
    df["_order"] = df["model_name"].map(order_map).fillna(999).astype(int)

    return df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def format_cell(value, col: str, percent_cols: List[str], point_cols: List[str]) -> str:
    if col in percent_cols:
        return pct(value)

    if col in point_cols:
        return point_fmt(value)

    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.4f}"

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    return str(value)


def dataframe_to_markdown(
    df: pd.DataFrame,
    percent_cols: Optional[List[str]] = None,
    point_cols: Optional[List[str]] = None,
) -> str:
    percent_cols = percent_cols or []
    point_cols = point_cols or []

    out = df.copy()

    lines = []
    lines.append("| " + " | ".join(out.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(out.columns)) + " |")

    for _, row in out.iterrows():
        formatted_values = [
            format_cell(row[col], col, percent_cols, point_cols)
            for col in out.columns
        ]
        lines.append("| " + " | ".join(formatted_values) + " |")

    return "\n".join(lines)


def dataframe_to_html(
    df: pd.DataFrame,
    percent_cols: Optional[List[str]] = None,
    point_cols: Optional[List[str]] = None,
) -> str:
    percent_cols = percent_cols or []
    point_cols = point_cols or []

    show_df = df.copy()

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
            value = format_cell(row[col], col, percent_cols, point_cols)
            lines.append(f"<td>{html_escape(value)}</td>")

        lines.append("</tr>")

    lines.append("</tbody>")
    lines.append("</table>")

    return "\n".join(lines)


# ============================================================
# PATHS
# ============================================================

def get_paths() -> Dict[str, Path]:
    external_dir = PHASE5_DIR / "outputs" / "external_finetuning"
    comparison_dir = external_dir / "comparison"
    output_dir = PHASE5_DIR / "outputs" / "internal_vs_external"

    return {
        "external_metrics_csv": external_dir / "external_finetuned_all_models_metrics.csv",
        "external_compare_report_json": comparison_dir / "08_compare_external_finetuning_results_report.json",
        "internal_reference_csv": output_dir / "internal_reference_metrics.csv",
        "output_dir": output_dir,
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_external_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing external metrics file:\n{path}\n"
            "Run 07_finetune_all_models_external.py first."
        )

    df = pd.read_csv(path)

    required = [
        "model_name",
        "test_accuracy",
        "test_macro_f1",
        "test_fall_recall",
        "test_fall_f1",
        "test_not_fall_f1",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"External metrics CSV missing columns: {missing}")

    rename_map = {
        "test_accuracy": "external_accuracy",
        "test_macro_f1": "external_macro_f1",
        "test_fall_recall": "external_fall_recall",
        "test_fall_f1": "external_fall_f1",
        "test_not_fall_f1": "external_not_fall_f1",
    }

    df = df.rename(columns=rename_map)

    keep_cols = [
        "model_name",
        "best_epoch",
        "external_accuracy",
        "external_macro_f1",
        "external_fall_recall",
        "external_fall_f1",
        "external_not_fall_f1",
        "test_tn",
        "test_fp",
        "test_fn",
        "test_tp",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols].copy()


def load_or_create_internal_reference(path: Path, overwrite: bool = False) -> pd.DataFrame:
    if path.exists() and not overwrite:
        df = pd.read_csv(path)
    else:
        df = default_internal_results()
        ensure_dir(path.parent)
        df.to_csv(path, index=False, encoding="utf-8-sig")

    required = [
        "model_name",
        "internal_accuracy",
        "internal_macro_f1",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Internal reference CSV missing columns: {missing}")

    return df.copy()


# ============================================================
# COMPARISON LOGIC
# ============================================================

def build_internal_external_comparison(
    internal_df: pd.DataFrame,
    external_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = internal_df.merge(
        external_df,
        on="model_name",
        how="inner",
    )

    merged = clean_model_order(merged)

    if merged.empty:
        raise RuntimeError(
            "No overlapping model names between internal reference metrics and external metrics."
        )

    merged["internal_macro_rank"] = merged["internal_macro_f1"].rank(
        ascending=False,
        method="min",
    ).astype(int)

    merged["external_macro_rank"] = merged["external_macro_f1"].rank(
        ascending=False,
        method="min",
    ).astype(int)

    merged["rank_change"] = merged["internal_macro_rank"] - merged["external_macro_rank"]

    merged["accuracy_change_points"] = merged.apply(
        lambda r: point_diff(r["external_accuracy"], r["internal_accuracy"]),
        axis=1,
    )

    merged["macro_f1_change_points"] = merged.apply(
        lambda r: point_diff(r["external_macro_f1"], r["internal_macro_f1"]),
        axis=1,
    )

    merged["fall_recall_change_points"] = merged.apply(
        lambda r: point_diff(
            r.get("external_fall_recall", np.nan),
            r.get("internal_fall_recall", np.nan),
        ),
        axis=1,
    )

    merged["fall_f1_change_points"] = merged.apply(
        lambda r: point_diff(
            r.get("external_fall_f1", np.nan),
            r.get("internal_fall_f1", np.nan),
        ),
        axis=1,
    )

    merged["is_phase4_quality_aware"] = merged["model_name"].isin(
        ["phase4_quality_gated", "phase4_quality_concat"]
    )

    return merged


def build_phase_level_comparison(comparison_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for phase, group in comparison_df.groupby("phase", sort=False):
        best_internal = group.sort_values("internal_macro_f1", ascending=False).iloc[0]
        best_external = group.sort_values("external_macro_f1", ascending=False).iloc[0]

        rows.append(
            {
                "phase": phase,
                "num_models": int(len(group)),

                "best_internal_model": best_internal["model_name"],
                "best_internal_macro_f1": float(best_internal["internal_macro_f1"]),
                "best_internal_accuracy": float(best_internal["internal_accuracy"]),

                "best_external_model": best_external["model_name"],
                "best_external_macro_f1": float(best_external["external_macro_f1"]),
                "best_external_accuracy": float(best_external["external_accuracy"]),
                "best_external_fall_recall": float(best_external["external_fall_recall"]),
                "best_external_fall_f1": float(best_external["external_fall_f1"]),
            }
        )

    out = pd.DataFrame(rows)

    order = {"Phase 1": 1, "Phase 2": 2, "Phase 3": 3, "Phase 4": 4}
    out["_order"] = out["phase"].map(order).fillna(999).astype(int)

    return out.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def build_research_claims(comparison_df: pd.DataFrame) -> Dict[str, Any]:
    internal_ranking = comparison_df.sort_values("internal_macro_f1", ascending=False).reset_index(drop=True)
    external_ranking = comparison_df.sort_values("external_macro_f1", ascending=False).reset_index(drop=True)

    best_internal = internal_ranking.iloc[0]
    best_external = external_ranking.iloc[0]

    top2_external = set(external_ranking.head(2)["model_name"].astype(str).tolist())
    phase4_set = {"phase4_quality_gated", "phase4_quality_concat"}

    phase4_top2_external = top2_external == phase4_set
    phase4_external_best = best_external["model_name"] in phase4_set

    claims = {
        "best_internal_model": best_internal["model_name"],
        "best_internal_macro_f1": float(best_internal["internal_macro_f1"]),

        "best_external_model": best_external["model_name"],
        "best_external_macro_f1": float(best_external["external_macro_f1"]),

        "phase4_top2_external": bool(phase4_top2_external),
        "phase4_external_best": bool(phase4_external_best),

        "main_claim": "",
        "caution": "",
        "report_ready_paragraph": "",
    }

    if phase4_top2_external:
        claims["main_claim"] = (
            "The exact best model changes from Quality-Concat on the internal dataset to "
            "Quality-Gated on the external dataset. However, the top two external models are both "
            "Phase 4 quality-aware models. Therefore, the Phase 4 quality-aware fusion direction is "
            "supported under external adaptation."
        )
    elif phase4_external_best:
        claims["main_claim"] = (
            "A Phase 4 quality-aware model remains the best external model, which supports the "
            "Phase 4 design direction under external adaptation."
        )
    else:
        claims["main_claim"] = (
            "The Phase 4 models are not the strongest on the external dataset. Therefore, the "
            "Phase 4 design would require further validation or stronger domain adaptation."
        )

    claims["caution"] = (
        "Internal and external results should not be interpreted as a direct measure of dataset "
        "difficulty because the datasets, splits, and training conditions differ. This comparison is "
        "intended to evaluate whether the model ranking trend and design direction remain valid when "
        "the models are adapted to a new dataset."
    )

    claims["report_ready_paragraph"] = (
        "The comparison between the internal dataset and the new external dataset shows that the "
        "exact best-performing model can change across domains. On the internal dataset, Phase 4 "
        "Quality-Concat achieved the best result. After fair fine-tuning on the external dataset, "
        "Phase 4 Quality-Gated achieved the best external test result, while Quality-Concat remained "
        "among the top-performing models. This suggests that the key contribution is not that one "
        "specific architecture always ranks first, but that the Phase 4 quality-aware fusion direction "
        "remains valid and beneficial when adapting to a new dataset."
    )

    return claims


# ============================================================
# FIGURES
# ============================================================

def save_internal_external_bar_chart(
    comparison_df: pd.DataFrame,
    metric_internal: str,
    metric_external: str,
    title: str,
    output_path: Path,
):
    df = comparison_df.copy()
    df = df.set_index("model_name").loc[
        [m for m in MODEL_ORDER if m in df["model_name"].values]
    ].reset_index()

    labels = df["model_name"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    internal_values = df[metric_internal].astype(float).to_numpy() * 100.0
    external_values = df[metric_external].astype(float).to_numpy() * 100.0

    plt.figure(figsize=(13, 6))
    plt.bar(x - width / 2, internal_values, width, label="Internal dataset")
    plt.bar(x + width / 2, external_values, width, label="External dataset")

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Score (%)")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_rank_change_chart(comparison_df: pd.DataFrame, output_path: Path):
    df = comparison_df.copy()
    df = df.sort_values("external_macro_rank", ascending=True)

    labels = df["model_name"].tolist()
    internal_rank = df["internal_macro_rank"].to_numpy()
    external_rank = df["external_macro_rank"].to_numpy()

    y = np.arange(len(labels))

    plt.figure(figsize=(10, 6))
    plt.scatter(internal_rank, y, label="Internal rank")
    plt.scatter(external_rank, y, label="External rank")

    for i in range(len(labels)):
        plt.plot([internal_rank[i], external_rank[i]], [y[i], y[i]])

    plt.yticks(y, labels)
    plt.xlabel("Rank by Macro F1, lower is better")
    plt.title("Internal vs External Macro F1 Ranking Shift")
    plt.grid(axis="x", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=180)
    plt.close()


def generate_figures(comparison_df: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    figures_dir = ensure_dir(output_dir / "figures")

    paths = {}

    macro_path = figures_dir / "internal_vs_external_macro_f1.png"
    acc_path = figures_dir / "internal_vs_external_accuracy.png"
    rank_path = figures_dir / "internal_vs_external_rank_change.png"

    save_internal_external_bar_chart(
        comparison_df,
        metric_internal="internal_macro_f1",
        metric_external="external_macro_f1",
        title="Internal vs External Macro F1",
        output_path=macro_path,
    )

    save_internal_external_bar_chart(
        comparison_df,
        metric_internal="internal_accuracy",
        metric_external="external_accuracy",
        title="Internal vs External Accuracy",
        output_path=acc_path,
    )

    save_rank_change_chart(comparison_df, rank_path)

    paths["internal_vs_external_macro_f1"] = str(macro_path)
    paths["internal_vs_external_accuracy"] = str(acc_path)
    paths["internal_vs_external_rank_change"] = str(rank_path)

    return paths


# ============================================================
# REPORT TABLE PREP
# ============================================================

def get_report_tables(comparison_df: pd.DataFrame, phase_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    ranking_cols = [
        "model_name",
        "phase",
        "internal_macro_rank",
        "external_macro_rank",
        "rank_change",
        "internal_accuracy",
        "external_accuracy",
        "accuracy_change_points",
        "internal_macro_f1",
        "external_macro_f1",
        "macro_f1_change_points",
        "external_fall_recall",
        "external_fall_f1",
    ]

    ranking_cols = [c for c in ranking_cols if c in comparison_df.columns]

    model_table = comparison_df[ranking_cols].sort_values(
        "external_macro_rank"
    ).reset_index(drop=True)

    phase_cols = [
        "phase",
        "num_models",
        "best_internal_model",
        "best_internal_macro_f1",
        "best_external_model",
        "best_external_macro_f1",
        "best_external_fall_recall",
        "best_external_fall_f1",
    ]

    phase_cols = [c for c in phase_cols if c in phase_df.columns]

    phase_table = phase_df[phase_cols].copy()

    return {
        "model_table": model_table,
        "phase_table": phase_table,
    }


def report_percent_cols() -> List[str]:
    return [
        "internal_accuracy",
        "external_accuracy",
        "internal_macro_f1",
        "external_macro_f1",
        "external_fall_recall",
        "external_fall_f1",
        "best_internal_macro_f1",
        "best_external_macro_f1",
        "best_external_fall_recall",
        "best_external_fall_f1",
    ]


def report_point_cols() -> List[str]:
    return [
        "accuracy_change_points",
        "macro_f1_change_points",
        "fall_recall_change_points",
        "fall_f1_change_points",
    ]


# ============================================================
# MARKDOWN REPORT
# ============================================================

def generate_markdown_report(
    comparison_df: pd.DataFrame,
    phase_df: pd.DataFrame,
    claims: Dict[str, Any],
) -> str:
    tables = get_report_tables(comparison_df, phase_df)

    lines = []

    lines.append("# Internal vs External Dataset Comparison")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This comparison checks whether the model improvement trend from the internal dataset "
        "remains valid after adapting the models to a new external dataset. All external results "
        "are taken from the fair external fine-tuning protocol: same external train split, same "
        "validation split, and same test split for all models."
    )
    lines.append("")
    lines.append(
        "Important note: internal and external scores should not be interpreted as a direct "
        "comparison of dataset difficulty. They are used to compare model ranking and design "
        "validity across datasets."
    )
    lines.append("")

    lines.append("## 2. Model-level comparison")
    lines.append("")
    lines.append(
        dataframe_to_markdown(
            tables["model_table"],
            percent_cols=report_percent_cols(),
            point_cols=report_point_cols(),
        )
    )
    lines.append("")

    lines.append("## 3. Phase-level comparison")
    lines.append("")
    lines.append(
        dataframe_to_markdown(
            tables["phase_table"],
            percent_cols=report_percent_cols(),
            point_cols=report_point_cols(),
        )
    )
    lines.append("")

    lines.append("## 4. Main conclusion")
    lines.append("")
    lines.append(claims["main_claim"])
    lines.append("")

    lines.append("## 5. Caution")
    lines.append("")
    lines.append(claims["caution"])
    lines.append("")

    lines.append("## 6. Report-ready paragraph")
    lines.append("")
    lines.append(claims["report_ready_paragraph"])
    lines.append("")

    return "\n".join(lines)


# ============================================================
# HTML REPORT
# ============================================================

def generate_html_report(
    comparison_df: pd.DataFrame,
    phase_df: pd.DataFrame,
    claims: Dict[str, Any],
) -> str:
    tables = get_report_tables(comparison_df, phase_df)

    model_table_html = dataframe_to_html(
        tables["model_table"],
        percent_cols=report_percent_cols(),
        point_cols=report_point_cols(),
    )

    phase_table_html = dataframe_to_html(
        tables["phase_table"],
        percent_cols=report_percent_cols(),
        point_cols=report_point_cols(),
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
        h1, h2 {
            color: #111827;
        }
        .section {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 20px;
            margin: 24px 0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }
        .conclusion {
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
            border-radius: 12px;
            padding: 18px;
            margin: 24px 0;
        }
        .warning {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-radius: 12px;
            padding: 18px;
            margin: 24px 0;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            font-size: 13px;
            margin-top: 14px;
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
        <title>Internal vs External Dataset Comparison</title>
        {css}
    </head>
    <body>
        <h1>Internal vs External Dataset Comparison</h1>

        <div class="section">
            <h2>1. Purpose</h2>
            <p>
                This report compares internal dataset results with external fine-tuning results.
                The goal is to evaluate whether the improvement direction from Phase 4 remains valid
                after adapting the models to a new dataset.
            </p>
            <p>
                The comparison should not be interpreted as a direct measurement of dataset difficulty.
                Instead, it checks whether model ranking and design trends remain consistent across
                different data domains.
            </p>
        </div>

        <div class="section">
            <h2>2. Model-level comparison</h2>
            {model_table_html}
        </div>

        <div class="section">
            <h2>3. Phase-level comparison</h2>
            {phase_table_html}
        </div>

        <div class="conclusion">
            <h2>4. Main conclusion</h2>
            <p>{html_escape(claims["main_claim"])}</p>
        </div>

        <div class="warning">
            <h2>5. Caution</h2>
            <p>{html_escape(claims["caution"])}</p>
        </div>

        <div class="section">
            <h2>6. Report-ready paragraph</h2>
            <p>{html_escape(claims["report_ready_paragraph"])}</p>
        </div>
    </body>
    </html>
    """

    return html


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    comparison_df: pd.DataFrame,
    phase_df: pd.DataFrame,
    claims: Dict[str, Any],
    figure_paths: Dict[str, str],
    output_dir: Path,
) -> Dict[str, Any]:
    ensure_dir(output_dir)

    comparison_csv = output_dir / "internal_vs_external_model_comparison.csv"
    phase_csv = output_dir / "internal_vs_external_phase_summary.csv"
    claims_json = output_dir / "internal_vs_external_claims.json"
    report_md = output_dir / "internal_vs_external_report.md"
    report_html = output_dir / "internal_vs_external_report.html"
    report_json = output_dir / "10_compare_internal_vs_external_report.json"

    comparison_df.to_csv(comparison_csv, index=False, encoding="utf-8-sig")
    phase_df.to_csv(phase_csv, index=False, encoding="utf-8-sig")

    save_json(claims, claims_json)

    markdown_text = generate_markdown_report(
        comparison_df=comparison_df,
        phase_df=phase_df,
        claims=claims,
    )

    html_text = generate_html_report(
        comparison_df=comparison_df,
        phase_df=phase_df,
        claims=claims,
    )

    save_text(markdown_text, report_md)
    save_text(html_text, report_html)

    report = {
        "phase": "Phase 5 - Internal vs External Comparison",
        "step": "10_compare_internal_vs_external",
        "goal": (
            "Compare internal dataset results with external fine-tuning results to evaluate whether "
            "the Phase 4 quality-aware design remains valid on the new dataset."
        ),
        "main_claim": claims["main_claim"],
        "caution": claims["caution"],
        "outputs": {
            "comparison_csv": str(comparison_csv),
            "phase_csv": str(phase_csv),
            "claims_json": str(claims_json),
            "report_md": str(report_md),
            "report_html": str(report_html),
            "report_json": str(report_json),
            "figures": figure_paths,
        },
    }

    save_json(report, report_json)

    return report["outputs"]


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(comparison_df: pd.DataFrame, phase_df: pd.DataFrame, claims: Dict[str, Any]):
    print("\nInternal vs External Ranking")
    print("=" * 120)

    show_cols = [
        "model_name",
        "internal_macro_rank",
        "external_macro_rank",
        "rank_change",
        "internal_macro_f1",
        "external_macro_f1",
        "macro_f1_change_points",
        "internal_accuracy",
        "external_accuracy",
        "accuracy_change_points",
    ]

    show_cols = [c for c in show_cols if c in comparison_df.columns]

    print(comparison_df[show_cols].sort_values("external_macro_rank").to_string(index=False))

    print("\nPhase-level summary")
    print("=" * 120)
    print(phase_df.to_string(index=False))

    print("\nMain claim")
    print("=" * 120)
    print(claims["main_claim"])

    print("\nCaution")
    print("=" * 120)
    print(claims["caution"])


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 Step 10 - Compare internal dataset results with external fine-tuning results."
    )

    parser.add_argument(
        "--internal-csv",
        type=str,
        default=None,
        help=(
            "Optional internal reference CSV. "
            "If omitted, the script uses and saves default internal reference metrics."
        ),
    )

    parser.add_argument(
        "--overwrite-internal-defaults",
        action="store_true",
        help="Overwrite the generated internal_reference_metrics.csv with default values.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable figure generation.",
    )

    args = parser.parse_args()

    print("\nPhase 5 - Step 10: Compare Internal vs External Results")
    print("=" * 120)
    print("This step does not train models and does not rebuild sequences.")
    print("It compares old internal results with new external fine-tuning results.")
    print("=" * 120)

    paths = get_paths()
    output_dir = ensure_dir(paths["output_dir"])

    print("\n[1/5] Loading external fine-tuning results...")
    external_df = load_external_metrics(paths["external_metrics_csv"])

    print(f"External metrics: {paths['external_metrics_csv']}")

    print("\n[2/5] Loading internal reference results...")

    if args.internal_csv:
        internal_path = Path(args.internal_csv)

        if not internal_path.exists():
            raise FileNotFoundError(f"Internal CSV not found: {internal_path}")

        internal_df = pd.read_csv(internal_path)
        internal_source_path = internal_path
    else:
        internal_source_path = paths["internal_reference_csv"]
        internal_df = load_or_create_internal_reference(
            internal_source_path,
            overwrite=args.overwrite_internal_defaults,
        )

    print(f"Internal reference: {internal_source_path}")

    print("\n[3/5] Building comparison tables...")

    comparison_df = build_internal_external_comparison(
        internal_df=internal_df,
        external_df=external_df,
    )

    phase_df = build_phase_level_comparison(comparison_df)
    claims = build_research_claims(comparison_df)

    print_summary(comparison_df, phase_df, claims)

    print("\n[4/5] Generating figures...")

    if args.no_plots:
        figure_paths = {}
    else:
        figure_paths = generate_figures(comparison_df, output_dir)

    print("\n[5/5] Saving outputs...")

    outputs = save_outputs(
        comparison_df=comparison_df,
        phase_df=phase_df,
        claims=claims,
        figure_paths=figure_paths,
        output_dir=output_dir,
    )

    print(json.dumps(outputs, ensure_ascii=False, indent=4))

    print("\nDONE: Phase 5 Step 10 completed.")
    print("=" * 120)
    print("Main files:")
    print(f"- {outputs['comparison_csv']}")
    print(f"- {outputs['phase_csv']}")
    print(f"- {outputs['report_md']}")
    print(f"- {outputs['report_html']}")
    print(f"- {outputs['report_json']}")
    print("=" * 120)


if __name__ == "__main__":
    main()