import os
import json
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


"""
Generate final Phase 4 HTML dashboard.

This dashboard reads:
    1. Clean fair comparison:
        phase4_quality_aware_fusion/outputs/comparison/all_phases_fair_comparison.csv
        phase4_quality_aware_fusion/outputs/comparison/all_phases_fair_best_models.json

    2. Robustness comparison:
        phase4_quality_aware_fusion/outputs/robustness/all_models_robustness_binary.csv
        phase4_quality_aware_fusion/outputs/robustness/all_models_robustness_action.csv

Output:
    phase4_quality_aware_fusion/outputs/comparison/phase4_dashboard.html

Purpose:
    Create one final dashboard containing:
        - Clean performance comparison
        - Binary robustness comparison
        - Action robustness comparison
        - Final task-specific recommendation
"""


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PHASE4_DIR = os.path.join(PROJECT_ROOT, "phase4_quality_aware_fusion")

COMPARISON_DIR = os.path.join(PHASE4_DIR, "outputs", "comparison")
ROBUSTNESS_DIR = os.path.join(PHASE4_DIR, "outputs", "robustness")

DEFAULT_COMPARISON_CSV = os.path.join(
    COMPARISON_DIR,
    "all_phases_fair_comparison.csv",
)

DEFAULT_BEST_MODELS_JSON = os.path.join(
    COMPARISON_DIR,
    "all_phases_fair_best_models.json",
)

DEFAULT_ROBUSTNESS_BINARY_CSV = os.path.join(
    ROBUSTNESS_DIR,
    "all_models_robustness_binary.csv",
)

DEFAULT_ROBUSTNESS_ACTION_CSV = os.path.join(
    ROBUSTNESS_DIR,
    "all_models_robustness_action.csv",
)

DEFAULT_OUTPUT_HTML = os.path.join(
    COMPARISON_DIR,
    "phase4_dashboard.html",
)


TASK_ORDER = ["binary", "action"]

TASK_DISPLAY_NAMES = {
    "binary": "Binary Fall Detection",
    "action": "Action Classification",
}

TASK_DESCRIPTIONS = {
    "binary": (
        "Binary Fall Detection evaluates whether the model can distinguish "
        "Fall and Not_Fall. This is the main safety-critical task."
    ),
    "action": (
        "Action Classification evaluates whether the model can classify "
        "Sitting, Sleeping, Standing, and Walking."
    ),
}

SELECTED_ROBUSTNESS_SCENARIOS = [
    "clean",
    "combined_light",
    "combined_heavy",
    "missing_joint_0.30",
]

SCENARIO_DISPLAY_NAMES = {
    "clean": "Clean",
    "combined_light": "Combined Light",
    "combined_heavy": "Combined Heavy",
    "missing_joint_0.10": "Missing Joint 10%",
    "missing_joint_0.20": "Missing Joint 20%",
    "missing_joint_0.30": "Missing Joint 30%",
    "frame_drop_0.10": "Frame Drop 10%",
    "frame_drop_0.20": "Frame Drop 20%",
    "frame_drop_0.30": "Frame Drop 30%",
    "gaussian_2d_0.01": "2D Noise 0.01",
    "gaussian_2d_0.03": "2D Noise 0.03",
    "gaussian_2d_0.05": "2D Noise 0.05",
    "gaussian_3d_0.01": "3D Noise 0.01",
    "gaussian_3d_0.03": "3D Noise 0.03",
    "gaussian_3d_0.05": "3D Noise 0.05",
}

PHASE_BADGE_CLASS = {
    "Phase 1": "phase1",
    "Phase 2": "phase2",
    "Phase 3": "phase3",
    "Phase 4": "phase4",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_required(path: str, description: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{description} not found:\n{path}\n\n"
            "Please run the required previous script first."
        )

    return pd.read_csv(path)


def escape_html(text) -> str:
    text = str(text)

    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


def is_nan(value) -> bool:
    try:
        return pd.isna(value)
    except Exception:
        return False


def format_percent(value) -> str:
    if value is None or is_nan(value):
        return "N/A"

    return f"{float(value):.2f}%"


def format_number(value) -> str:
    if value is None or is_nan(value):
        return "N/A"

    return str(int(value))


def format_pp(value) -> str:
    if value is None or is_nan(value):
        return "N/A"

    value = float(value)

    if value > 0:
        return f"+{value:.2f} pp"

    return f"{value:.2f} pp"


def safe_float(value, default=np.nan) -> float:
    try:
        if pd.isna(value):
            return default

        return float(value)
    except Exception:
        return default


def phase_badge(phase: str) -> str:
    css_class = PHASE_BADGE_CLASS.get(str(phase), "phase-default")
    return f'<span class="badge {css_class}">{escape_html(phase)}</span>'


def scenario_label(scenario: str) -> str:
    return SCENARIO_DISPLAY_NAMES.get(str(scenario), str(scenario))


def bar_width(value: float, min_value: float, max_value: float) -> float:
    if is_nan(value):
        return 0.0

    value = float(value)

    if max_value <= min_value:
        return 100.0

    return 8.0 + 92.0 * ((value - min_value) / (max_value - min_value))


def get_best_row(df: pd.DataFrame, task: str, scenario: Optional[str] = None) -> Optional[pd.Series]:
    task_df = df[df["task"] == task].copy()

    if scenario is not None and "scenario" in task_df.columns:
        task_df = task_df[task_df["scenario"] == scenario].copy()

    if task_df.empty:
        return None

    task_df = task_df.sort_values(
        by=["macro_f1_percent", "accuracy_percent"],
        ascending=[False, False],
    )

    return task_df.iloc[0]


def normalize_clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "rank_by_macro_f1" not in df.columns:
        ranked = []

        for task, group in df.groupby("task"):
            group = group.copy()
            group = group.sort_values(
                by=["macro_f1_percent", "accuracy_percent"],
                ascending=[False, False],
            )
            group["rank_by_macro_f1"] = np.arange(1, len(group) + 1)
            ranked.append(group)

        df = pd.concat(ranked, axis=0).reset_index(drop=True)

    return df


def normalize_robustness_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "rank_by_macro_f1_scenario" not in df.columns:
        ranked = []

        for (task, scenario), group in df.groupby(["task", "scenario"]):
            group = group.copy()
            group = group.sort_values(
                by=["macro_f1_percent", "accuracy_percent"],
                ascending=[False, False],
            )
            group["rank_by_macro_f1_scenario"] = np.arange(1, len(group) + 1)
            ranked.append(group)

        df = pd.concat(ranked, axis=0).reset_index(drop=True)

    return df


# ============================================================
# SUMMARY
# ============================================================

def build_metric_card(title: str, value: str, subtitle: str, accent: str = "") -> str:
    return f"""
    <div class="metric-card {accent}">
        <div class="metric-title">{escape_html(title)}</div>
        <div class="metric-value">{escape_html(value)}</div>
        <div class="metric-subtitle">{escape_html(subtitle)}</div>
    </div>
    """


def build_summary_cards(
    clean_df: pd.DataFrame,
    robustness_binary_df: pd.DataFrame,
    robustness_action_df: pd.DataFrame,
    best_models: Dict,
) -> str:
    binary_best = best_models.get("binary", {})
    action_best = best_models.get("action", {})

    binary_model = binary_best.get("best_model", "N/A")
    binary_acc = binary_best.get("accuracy_percent", np.nan)
    binary_f1 = binary_best.get("macro_f1_percent", np.nan)

    action_model = action_best.get("best_model", "N/A")
    action_acc = action_best.get("accuracy_percent", np.nan)
    action_f1 = action_best.get("macro_f1_percent", np.nan)

    binary_heavy = get_best_row(robustness_binary_df, "binary", "combined_heavy")
    action_heavy = get_best_row(robustness_action_df, "action", "combined_heavy")

    binary_heavy_text = "N/A"
    binary_heavy_subtitle = "Run robustness binary first."

    if binary_heavy is not None:
        binary_heavy_text = str(binary_heavy["model_name"])
        binary_heavy_subtitle = (
            f"Heavy corruption Macro F1 {format_percent(binary_heavy['macro_f1_percent'])}"
        )

    action_heavy_text = "N/A"
    action_heavy_subtitle = "Run robustness action first."

    if action_heavy is not None:
        action_heavy_text = str(action_heavy["model_name"])
        action_heavy_subtitle = (
            f"Heavy corruption Macro F1 {format_percent(action_heavy['macro_f1_percent'])}"
        )

    return f"""
    <section class="summary-grid">
        {build_metric_card(
            "Best Clean Binary",
            binary_model,
            f"Accuracy {format_percent(binary_acc)} | Macro F1 {format_percent(binary_f1)}",
            "accent-green",
        )}
        {build_metric_card(
            "Best Clean Action",
            action_model,
            f"Accuracy {format_percent(action_acc)} | Macro F1 {format_percent(action_f1)}",
            "accent-blue",
        )}
        {build_metric_card(
            "Best Heavy Binary",
            binary_heavy_text,
            binary_heavy_subtitle,
            "accent-purple",
        )}
        {build_metric_card(
            "Best Heavy Action",
            action_heavy_text,
            action_heavy_subtitle,
            "accent-orange",
        )}
    </section>
    """


# ============================================================
# CLEAN PERFORMANCE SECTIONS
# ============================================================

def build_clean_table(task_df: pd.DataFrame) -> str:
    task_df = task_df.sort_values("rank_by_macro_f1", ascending=True).copy()

    rows = []

    for _, row in task_df.iterrows():
        rank = int(row["rank_by_macro_f1"])
        rank_class = "rank-first" if rank == 1 else ""

        model_text = row.get("model", row.get("fair_model_name", "N/A"))
        input_text = row.get("input", "N/A")

        sample_match = bool(row.get("sample_count_matches_expected", True))
        video_match = bool(row.get("video_count_matches_expected", True))

        fair_status = (
            '<span class="status-ok">OK</span>'
            if sample_match and video_match
            else '<span class="status-warning">Check</span>'
        )

        rows.append(
            f"""
            <tr class="{rank_class}">
                <td class="rank-cell">{rank}</td>
                <td>{phase_badge(row["phase"])}</td>
                <td class="model-cell">{escape_html(model_text)}</td>
                <td>{escape_html(input_text)}</td>
                <td class="num-cell">{format_percent(row["accuracy_percent"])}</td>
                <td class="num-cell strong">{format_percent(row["macro_f1_percent"])}</td>
                <td class="num-cell">{format_number(row.get("num_test_samples", np.nan))}</td>
                <td class="num-cell">{format_number(row.get("num_test_videos", np.nan))}</td>
                <td>{fair_status}</td>
            </tr>
            """
        )

    return f"""
    <div class="table-wrap">
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Phase</th>
                    <th>Model</th>
                    <th>Input</th>
                    <th>Accuracy</th>
                    <th>Macro F1</th>
                    <th>Test Samples</th>
                    <th>Test Videos</th>
                    <th>Fair Set</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """


def build_clean_bar_chart(task_df: pd.DataFrame, metric_col: str, title: str) -> str:
    task_df = task_df.sort_values("rank_by_macro_f1", ascending=True).copy()

    values = task_df[metric_col].astype(float)

    min_value = float(values.min())
    max_value = float(values.max())

    bars = []

    for _, row in task_df.iterrows():
        value = float(row[metric_col])
        width = bar_width(value, min_value, max_value)
        model_text = row.get("model", row.get("fair_model_name", "N/A"))

        bars.append(
            f"""
            <div class="bar-row">
                <div class="bar-label">
                    <span class="bar-phase">{escape_html(row["phase"])}</span>
                    <span>{escape_html(model_text)}</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="width: {width:.2f}%"></div>
                </div>
                <div class="bar-value">{value:.2f}%</div>
            </div>
            """
        )

    return f"""
    <div class="chart-card">
        <h3>{escape_html(title)}</h3>
        <div class="bar-chart">
            {''.join(bars)}
        </div>
    </div>
    """


def build_clean_insight(task: str, task_df: pd.DataFrame) -> str:
    best = task_df.sort_values(
        by=["macro_f1_percent", "accuracy_percent"],
        ascending=[False, False],
    ).iloc[0]

    if task == "binary":
        phase2_concat = task_df[task_df["fair_model_name"] == "Phase 2 - Concat Common"]
        phase4_concat = task_df[task_df["fair_model_name"] == "Phase 4 - Quality-Concat"]

        delta_acc = np.nan
        delta_f1 = np.nan

        if not phase2_concat.empty and not phase4_concat.empty:
            delta_acc = (
                float(phase4_concat.iloc[0]["accuracy_percent"])
                - float(phase2_concat.iloc[0]["accuracy_percent"])
            )
            delta_f1 = (
                float(phase4_concat.iloc[0]["macro_f1_percent"])
                - float(phase2_concat.iloc[0]["macro_f1_percent"])
            )

        return f"""
        <div class="insight-card positive">
            <h3>Clean Binary Insight</h3>
            <p>
                The best clean binary model is
                <strong>{escape_html(best["fair_model_name"])}</strong>,
                with <strong>{format_percent(best["accuracy_percent"])}</strong> Accuracy
                and <strong>{format_percent(best["macro_f1_percent"])}</strong> Macro F1.
            </p>
            <p>
                Compared with Phase 2 Concat Common, Phase 4 Quality-Concat improves
                Accuracy by <strong>{delta_acc:.2f} percentage points</strong>
                and Macro F1 by <strong>{delta_f1:.2f} percentage points</strong>.
            </p>
            <p>
                This supports the value of quality-aware features for the main Fall/Not_Fall task.
            </p>
        </div>
        """

    if task == "action":
        return f"""
        <div class="insight-card neutral">
            <h3>Clean Action Insight</h3>
            <p>
                The best clean action model is
                <strong>{escape_html(best["fair_model_name"])}</strong>,
                with <strong>{format_percent(best["accuracy_percent"])}</strong> Accuracy
                and <strong>{format_percent(best["macro_f1_percent"])}</strong> Macro F1.
            </p>
            <p>
                Phase 4 does not outperform the direct 2D+3D Concat Common model for
                multi-class action classification.
            </p>
        </div>
        """

    return ""


def build_clean_task_section(task: str, clean_df: pd.DataFrame) -> str:
    task_df = clean_df[clean_df["task"] == task].copy()

    if task_df.empty:
        return ""

    task_df = task_df.sort_values("rank_by_macro_f1", ascending=True)

    task_name = TASK_DISPLAY_NAMES.get(task, task)

    return f"""
    <section class="task-section" id="clean-{task}">
        <div class="section-header">
            <h2>Clean Performance - {escape_html(task_name)}</h2>
            <p>{escape_html(TASK_DESCRIPTIONS.get(task, ""))}</p>
        </div>

        {build_clean_insight(task, task_df)}

        <div class="chart-grid">
            {build_clean_bar_chart(task_df, "accuracy_percent", "Clean Accuracy")}
            {build_clean_bar_chart(task_df, "macro_f1_percent", "Clean Macro F1")}
        </div>

        {build_clean_table(task_df)}
    </section>
    """


# ============================================================
# ROBUSTNESS SECTIONS
# ============================================================

def build_robustness_aggregate(robust_df: pd.DataFrame, task: str) -> pd.DataFrame:
    task_df = robust_df[robust_df["task"] == task].copy()

    if task_df.empty:
        return pd.DataFrame()

    clean_rows = task_df[task_df["scenario"] == "clean"].copy()
    corrupted_rows = task_df[task_df["scenario"] != "clean"].copy()

    rows = []

    for model_id, group in task_df.groupby("model_id"):
        clean = clean_rows[clean_rows["model_id"] == model_id]
        corrupted = corrupted_rows[corrupted_rows["model_id"] == model_id]

        if clean.empty or corrupted.empty:
            continue

        clean_row = clean.iloc[0]

        worst_row = corrupted.sort_values(
            by=["macro_f1_percent", "accuracy_percent"],
            ascending=[True, True],
        ).iloc[0]

        best_corrupted_row = corrupted.sort_values(
            by=["macro_f1_percent", "accuracy_percent"],
            ascending=[False, False],
        ).iloc[0]

        row = {
            "task": task,
            "phase": clean_row["phase"],
            "model_id": model_id,
            "model_name": clean_row["model_name"],
            "clean_accuracy_percent": clean_row["accuracy_percent"],
            "clean_macro_f1_percent": clean_row["macro_f1_percent"],
            "mean_corrupted_accuracy_percent": corrupted["accuracy_percent"].mean(),
            "mean_corrupted_macro_f1_percent": corrupted["macro_f1_percent"].mean(),
            "mean_delta_accuracy_percent": corrupted["delta_accuracy_percent_from_clean"].mean(),
            "mean_delta_macro_f1_percent": corrupted["delta_macro_f1_percent_from_clean"].mean(),
            "worst_corrupted_scenario": worst_row["scenario"],
            "worst_corrupted_accuracy_percent": worst_row["accuracy_percent"],
            "worst_corrupted_macro_f1_percent": worst_row["macro_f1_percent"],
            "worst_delta_macro_f1_percent": worst_row["delta_macro_f1_percent_from_clean"],
            "best_corrupted_scenario": best_corrupted_row["scenario"],
            "best_corrupted_macro_f1_percent": best_corrupted_row["macro_f1_percent"],
        }

        rows.append(row)

    agg = pd.DataFrame(rows)

    if agg.empty:
        return agg

    agg = agg.sort_values(
        by=["mean_corrupted_macro_f1_percent", "mean_corrupted_accuracy_percent"],
        ascending=[False, False],
    ).reset_index(drop=True)

    agg["rank_by_mean_corrupted_macro_f1"] = np.arange(1, len(agg) + 1)

    return agg


def build_robustness_aggregate_table(agg_df: pd.DataFrame) -> str:
    if agg_df.empty:
        return """
        <div class="empty-box">
            Robustness aggregate table is not available.
        </div>
        """

    rows = []

    for _, row in agg_df.iterrows():
        rank = int(row["rank_by_mean_corrupted_macro_f1"])
        rank_class = "rank-first" if rank == 1 else ""

        rows.append(
            f"""
            <tr class="{rank_class}">
                <td class="rank-cell">{rank}</td>
                <td>{phase_badge(row["phase"])}</td>
                <td class="model-cell">{escape_html(row["model_name"])}</td>
                <td class="num-cell">{format_percent(row["clean_macro_f1_percent"])}</td>
                <td class="num-cell strong">{format_percent(row["mean_corrupted_macro_f1_percent"])}</td>
                <td class="num-cell">{format_pp(row["mean_delta_macro_f1_percent"])}</td>
                <td>{escape_html(scenario_label(row["worst_corrupted_scenario"]))}</td>
                <td class="num-cell">{format_percent(row["worst_corrupted_macro_f1_percent"])}</td>
                <td class="num-cell">{format_pp(row["worst_delta_macro_f1_percent"])}</td>
            </tr>
            """
        )

    return f"""
    <div class="table-wrap">
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Phase</th>
                    <th>Model</th>
                    <th>Clean Macro F1</th>
                    <th>Mean Corrupted Macro F1</th>
                    <th>Mean Drop</th>
                    <th>Worst Scenario</th>
                    <th>Worst Macro F1</th>
                    <th>Worst Drop</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """


def build_scenario_table(task_df: pd.DataFrame, scenario: str) -> str:
    scenario_df = task_df[task_df["scenario"] == scenario].copy()

    if scenario_df.empty:
        return f"""
        <div class="empty-box">
            Scenario {escape_html(scenario)} is not available.
        </div>
        """

    scenario_df = scenario_df.sort_values(
        by=["macro_f1_percent", "accuracy_percent"],
        ascending=[False, False],
    ).reset_index(drop=True)

    rows = []

    for idx, row in scenario_df.iterrows():
        rank = idx + 1
        rank_class = "rank-first" if rank == 1 else ""

        rows.append(
            f"""
            <tr class="{rank_class}">
                <td class="rank-cell">{rank}</td>
                <td>{phase_badge(row["phase"])}</td>
                <td class="model-cell">{escape_html(row["model_name"])}</td>
                <td class="num-cell">{format_percent(row["accuracy_percent"])}</td>
                <td class="num-cell strong">{format_percent(row["macro_f1_percent"])}</td>
                <td class="num-cell">{format_pp(row["delta_accuracy_percent_from_clean"])}</td>
                <td class="num-cell">{format_pp(row["delta_macro_f1_percent_from_clean"])}</td>
            </tr>
            """
        )

    return f"""
    <div class="scenario-card">
        <h3>{escape_html(scenario_label(scenario))}</h3>
        <div class="table-wrap compact">
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Phase</th>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>Macro F1</th>
                        <th>Δ Acc</th>
                        <th>Δ F1</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
    </div>
    """


def build_robustness_bar_chart(
    task_df: pd.DataFrame,
    scenario: str,
    metric_col: str,
    title: str,
) -> str:
    scenario_df = task_df[task_df["scenario"] == scenario].copy()

    if scenario_df.empty:
        return ""

    scenario_df = scenario_df.sort_values(
        by=[metric_col, "accuracy_percent"],
        ascending=[False, False],
    )

    values = scenario_df[metric_col].astype(float)

    min_value = float(values.min())
    max_value = float(values.max())

    bars = []

    for _, row in scenario_df.iterrows():
        value = float(row[metric_col])
        width = bar_width(value, min_value, max_value)

        bars.append(
            f"""
            <div class="bar-row">
                <div class="bar-label">
                    <span class="bar-phase">{escape_html(row["phase"])}</span>
                    <span>{escape_html(row["model_name"])}</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill robust-fill" style="width: {width:.2f}%"></div>
                </div>
                <div class="bar-value">{value:.2f}%</div>
            </div>
            """
        )

    return f"""
    <div class="chart-card">
        <h3>{escape_html(title)}</h3>
        <div class="bar-chart">
            {''.join(bars)}
        </div>
    </div>
    """


def build_robustness_insight(task: str, robust_df: pd.DataFrame) -> str:
    task_df = robust_df[robust_df["task"] == task].copy()

    if task_df.empty:
        return ""

    heavy = task_df[task_df["scenario"] == "combined_heavy"].copy()
    missing30 = task_df[task_df["scenario"] == "missing_joint_0.30"].copy()

    heavy_best = None
    missing_best = None

    if not heavy.empty:
        heavy_best = heavy.sort_values(
            by=["macro_f1_percent", "accuracy_percent"],
            ascending=[False, False],
        ).iloc[0]

    if not missing30.empty:
        missing_best = missing30.sort_values(
            by=["macro_f1_percent", "accuracy_percent"],
            ascending=[False, False],
        ).iloc[0]

    if task == "binary":
        quality_concat_heavy = heavy[heavy["model_id"] == "phase4_quality_concat"]
        phase2_concat_heavy = heavy[heavy["model_id"] == "phase2_concat_common"]

        heavy_delta_text = "N/A"

        if not quality_concat_heavy.empty and not phase2_concat_heavy.empty:
            heavy_delta = (
                float(quality_concat_heavy.iloc[0]["macro_f1_percent"])
                - float(phase2_concat_heavy.iloc[0]["macro_f1_percent"])
            )
            heavy_delta_text = f"{heavy_delta:.2f} percentage points"

        return f"""
        <div class="insight-card positive">
            <h3>Binary Robustness Insight</h3>
            <p>
                Under heavy combined corruption, the strongest binary model is
                <strong>{escape_html(heavy_best["model_name"]) if heavy_best is not None else "N/A"}</strong>
                with <strong>{format_percent(heavy_best["macro_f1_percent"]) if heavy_best is not None else "N/A"}</strong>
                Macro F1.
            </p>
            <p>
                In the heavy corruption scenario, Phase 4 Quality-Concat is ahead of
                Phase 2 Concat Common by <strong>{escape_html(heavy_delta_text)}</strong>
                in Macro F1.
            </p>
            <p>
                This is the key robustness evidence for Phase 4 in the main fall detection task.
            </p>
        </div>
        """

    if task == "action":
        return f"""
        <div class="insight-card warning">
            <h3>Action Robustness Insight</h3>
            <p>
                For action classification, the best heavy-corruption model is
                <strong>{escape_html(heavy_best["model_name"]) if heavy_best is not None else "N/A"}</strong>
                with <strong>{format_percent(heavy_best["macro_f1_percent"]) if heavy_best is not None else "N/A"}</strong>
                Macro F1.
            </p>
            <p>
                Under 30% missing joints, the best model is
                <strong>{escape_html(missing_best["model_name"]) if missing_best is not None else "N/A"}</strong>
                with <strong>{format_percent(missing_best["macro_f1_percent"]) if missing_best is not None else "N/A"}</strong>
                Macro F1.
            </p>
            <p>
                This shows that Phase 4 is not universally better for action classification.
                Its strongest contribution is clearer in binary fall detection.
            </p>
        </div>
        """

    return ""


def build_robustness_task_section(task: str, robust_df: pd.DataFrame) -> str:
    task_df = robust_df[robust_df["task"] == task].copy()

    if task_df.empty:
        return ""

    task_name = TASK_DISPLAY_NAMES.get(task, task)
    agg_df = build_robustness_aggregate(robust_df, task)

    scenario_tables = []

    for scenario in SELECTED_ROBUSTNESS_SCENARIOS:
        scenario_tables.append(build_scenario_table(task_df, scenario))

    heavy_chart = build_robustness_bar_chart(
        task_df,
        scenario="combined_heavy",
        metric_col="macro_f1_percent",
        title="Macro F1 under Combined Heavy Corruption",
    )

    missing_chart = build_robustness_bar_chart(
        task_df,
        scenario="missing_joint_0.30",
        metric_col="macro_f1_percent",
        title="Macro F1 under 30% Missing Joints",
    )

    return f"""
    <section class="task-section" id="robustness-{task}">
        <div class="section-header">
            <h2>Robustness Analysis - {escape_html(task_name)}</h2>
            <p>
                Robustness is evaluated by corrupting pose data in memory using Gaussian noise,
                missing joints, frame dropping, and combined perturbations.
            </p>
        </div>

        {build_robustness_insight(task, robust_df)}

        <h3 class="subsection-title">Aggregate Robustness Ranking</h3>
        {build_robustness_aggregate_table(agg_df)}

        <div class="chart-grid">
            {heavy_chart}
            {missing_chart}
        </div>

        <h3 class="subsection-title">Selected Scenario Rankings</h3>
        <div class="scenario-grid">
            {''.join(scenario_tables)}
        </div>
    </section>
    """


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

def build_final_recommendation(
    clean_df: pd.DataFrame,
    robustness_binary_df: pd.DataFrame,
    robustness_action_df: pd.DataFrame,
    best_models: Dict,
) -> str:
    binary = best_models.get("binary", {})
    action = best_models.get("action", {})

    binary_heavy = get_best_row(robustness_binary_df, "binary", "combined_heavy")
    binary_missing30 = get_best_row(robustness_binary_df, "binary", "missing_joint_0.30")

    action_heavy = get_best_row(robustness_action_df, "action", "combined_heavy")
    action_missing30 = get_best_row(robustness_action_df, "action", "missing_joint_0.30")

    return f"""
    <section class="recommendation">
        <h2>Final Task-Specific Recommendation</h2>

        <div class="recommend-grid">
            <div class="recommend-card">
                <h3>Binary Fall Detection</h3>
                <p class="recommend-model">{escape_html(binary.get("best_model", "N/A"))}</p>
                <p>
                    Clean Accuracy:
                    <strong>{format_percent(binary.get("accuracy_percent", np.nan))}</strong><br>
                    Clean Macro F1:
                    <strong>{format_percent(binary.get("macro_f1_percent", np.nan))}</strong>
                </p>
                <p>
                    Best under combined heavy corruption:
                    <strong>{escape_html(binary_heavy["model_name"]) if binary_heavy is not None else "N/A"}</strong>
                    ({format_percent(binary_heavy["macro_f1_percent"]) if binary_heavy is not None else "N/A"} Macro F1).
                </p>
                <p>
                    Best under 30% missing joints:
                    <strong>{escape_html(binary_missing30["model_name"]) if binary_missing30 is not None else "N/A"}</strong>
                    ({format_percent(binary_missing30["macro_f1_percent"]) if binary_missing30 is not None else "N/A"} Macro F1).
                </p>
                <p>
                    Recommendation: use Phase 4 Quality-Concat for the main binary fall detection task.
                </p>
            </div>

            <div class="recommend-card">
                <h3>Action Classification</h3>
                <p class="recommend-model">{escape_html(action.get("best_model", "N/A"))}</p>
                <p>
                    Clean Accuracy:
                    <strong>{format_percent(action.get("accuracy_percent", np.nan))}</strong><br>
                    Clean Macro F1:
                    <strong>{format_percent(action.get("macro_f1_percent", np.nan))}</strong>
                </p>
                <p>
                    Best under combined heavy corruption:
                    <strong>{escape_html(action_heavy["model_name"]) if action_heavy is not None else "N/A"}</strong>
                    ({format_percent(action_heavy["macro_f1_percent"]) if action_heavy is not None else "N/A"} Macro F1).
                </p>
                <p>
                    Best under 30% missing joints:
                    <strong>{escape_html(action_missing30["model_name"]) if action_missing30 is not None else "N/A"}</strong>
                    ({format_percent(action_missing30["macro_f1_percent"]) if action_missing30 is not None else "N/A"} Macro F1).
                </p>
                <p>
                    Recommendation: keep Phase 2 Concat Common as the clean action classifier.
                    Phase 4 should not be claimed as universally better for action classification.
                </p>
            </div>
        </div>

        <div class="final-note">
            <strong>Research conclusion:</strong>
            Phase 4 is a task-specific improvement. It clearly improves binary fall detection
            and shows strong robustness for the main safety-critical task, but it does not
            consistently outperform direct 2D+3D fusion for multi-class action classification.
        </div>
    </section>
    """


# ============================================================
# CSS
# ============================================================

def build_css() -> str:
    return """
    <style>
        :root {
            --bg: #f6f8fb;
            --card: #ffffff;
            --text: #172033;
            --muted: #657083;
            --line: #e4e8f0;
            --blue: #2563eb;
            --green: #16a34a;
            --purple: #7c3aed;
            --orange: #ea580c;
            --red: #dc2626;
            --yellow: #f59e0b;
            --shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Inter, Segoe UI, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
        }

        .page {
            max-width: 1380px;
            margin: 0 auto;
            padding: 28px;
        }

        .hero {
            background: linear-gradient(135deg, #172033, #263b72);
            color: white;
            padding: 34px;
            border-radius: 24px;
            box-shadow: var(--shadow);
            margin-bottom: 24px;
        }

        .hero h1 {
            margin: 0 0 8px 0;
            font-size: 34px;
            letter-spacing: -0.5px;
        }

        .hero p {
            margin: 5px 0;
            color: #dbeafe;
            font-size: 15px;
        }

        .nav-box {
            background: var(--card);
            border: 1px solid var(--line);
            box-shadow: var(--shadow);
            border-radius: 18px;
            padding: 16px 20px;
            margin-bottom: 24px;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .nav-box a {
            text-decoration: none;
            color: var(--blue);
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            padding: 8px 12px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 13px;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }

        .metric-card {
            background: var(--card);
            padding: 20px;
            border-radius: 18px;
            box-shadow: var(--shadow);
            border-top: 5px solid var(--blue);
        }

        .metric-card.accent-green {
            border-top-color: var(--green);
        }

        .metric-card.accent-blue {
            border-top-color: var(--blue);
        }

        .metric-card.accent-purple {
            border-top-color: var(--purple);
        }

        .metric-card.accent-orange {
            border-top-color: var(--orange);
        }

        .metric-title {
            color: var(--muted);
            font-size: 13px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .metric-value {
            font-size: 21px;
            font-weight: 850;
            margin-top: 8px;
            margin-bottom: 6px;
        }

        .metric-subtitle {
            color: var(--muted);
            font-size: 13px;
        }

        .task-section,
        .recommendation {
            background: var(--card);
            border-radius: 24px;
            box-shadow: var(--shadow);
            padding: 26px;
            margin-bottom: 24px;
        }

        .section-header h2,
        .recommendation h2 {
            margin: 0 0 6px 0;
            font-size: 26px;
        }

        .section-header p {
            margin: 0 0 20px 0;
            color: var(--muted);
        }

        .subsection-title {
            margin-top: 26px;
            margin-bottom: 12px;
            font-size: 20px;
        }

        .insight-card {
            border-radius: 18px;
            padding: 18px;
            margin-bottom: 22px;
            border: 1px solid var(--line);
        }

        .insight-card h3 {
            margin: 0 0 10px 0;
        }

        .insight-card p {
            margin: 8px 0;
        }

        .insight-card.positive {
            background: #f0fdf4;
            border-color: #bbf7d0;
        }

        .insight-card.neutral {
            background: #eff6ff;
            border-color: #bfdbfe;
        }

        .insight-card.warning {
            background: #fffbeb;
            border-color: #fde68a;
        }

        .chart-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 18px;
            margin-bottom: 24px;
        }

        .chart-card {
            background: #fbfcff;
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 18px;
        }

        .chart-card h3 {
            margin: 0 0 14px 0;
            font-size: 17px;
        }

        .bar-row {
            display: grid;
            grid-template-columns: 270px 1fr 78px;
            gap: 12px;
            align-items: center;
            margin-bottom: 12px;
        }

        .bar-label {
            font-size: 13px;
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        .bar-phase {
            color: var(--muted);
            font-size: 12px;
            font-weight: 800;
        }

        .bar-track {
            height: 12px;
            background: #e5e7eb;
            border-radius: 999px;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--blue), var(--purple));
            border-radius: 999px;
        }

        .robust-fill {
            background: linear-gradient(90deg, var(--green), var(--blue));
        }

        .bar-value {
            text-align: right;
            font-weight: 800;
            font-size: 13px;
        }

        .scenario-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 18px;
            margin-top: 12px;
        }

        .scenario-card {
            background: #fbfcff;
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 18px;
        }

        .scenario-card h3 {
            margin-top: 0;
            margin-bottom: 12px;
        }

        .table-wrap {
            overflow-x: auto;
            border: 1px solid var(--line);
            border-radius: 16px;
        }

        .table-wrap.compact table {
            min-width: 760px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            min-width: 1060px;
            background: white;
        }

        th {
            text-align: left;
            background: #f8fafc;
            color: #475569;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            padding: 13px;
            border-bottom: 1px solid var(--line);
        }

        td {
            padding: 13px;
            border-bottom: 1px solid var(--line);
            font-size: 14px;
            vertical-align: middle;
        }

        tr:last-child td {
            border-bottom: none;
        }

        .rank-first {
            background: #fff7ed;
        }

        .rank-cell {
            font-weight: 900;
            font-size: 16px;
        }

        .model-cell {
            font-weight: 800;
        }

        .num-cell {
            text-align: right;
            font-variant-numeric: tabular-nums;
        }

        .strong {
            font-weight: 900;
        }

        .badge {
            display: inline-block;
            color: white;
            font-weight: 800;
            font-size: 12px;
            padding: 5px 9px;
            border-radius: 999px;
            white-space: nowrap;
        }

        .phase1 {
            background: #64748b;
        }

        .phase2 {
            background: #2563eb;
        }

        .phase3 {
            background: #7c3aed;
        }

        .phase4 {
            background: #16a34a;
        }

        .phase-default {
            background: #334155;
        }

        .status-ok {
            color: var(--green);
            font-weight: 900;
        }

        .status-warning {
            color: var(--red);
            font-weight: 900;
        }

        .recommend-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 18px;
            margin-top: 18px;
        }

        .recommend-card {
            background: #fbfcff;
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 20px;
        }

        .recommend-card h3 {
            margin-top: 0;
        }

        .recommend-model {
            font-size: 22px;
            font-weight: 900;
            color: var(--blue);
            margin: 8px 0;
        }

        .final-note {
            margin-top: 18px;
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-radius: 16px;
            padding: 16px;
        }

        .empty-box {
            border: 1px dashed var(--line);
            border-radius: 16px;
            padding: 18px;
            color: var(--muted);
            background: #fbfcff;
        }

        .footer {
            color: var(--muted);
            font-size: 13px;
            text-align: center;
            padding: 16px;
        }

        @media (max-width: 1100px) {
            .summary-grid,
            .chart-grid,
            .scenario-grid,
            .recommend-grid {
                grid-template-columns: 1fr;
            }

            .bar-row {
                grid-template-columns: 1fr;
                gap: 6px;
            }

            .bar-value {
                text-align: left;
            }
        }
    </style>
    """


# ============================================================
# HTML GENERATION
# ============================================================

def build_nav() -> str:
    return """
    <nav class="nav-box">
        <a href="#clean-binary">Clean Binary</a>
        <a href="#clean-action">Clean Action</a>
        <a href="#robustness-binary">Robustness Binary</a>
        <a href="#robustness-action">Robustness Action</a>
        <a href="#final">Final Recommendation</a>
    </nav>
    """


def generate_dashboard(
    comparison_csv: str = DEFAULT_COMPARISON_CSV,
    best_models_json: str = DEFAULT_BEST_MODELS_JSON,
    robustness_binary_csv: str = DEFAULT_ROBUSTNESS_BINARY_CSV,
    robustness_action_csv: str = DEFAULT_ROBUSTNESS_ACTION_CSV,
    output_html: str = DEFAULT_OUTPUT_HTML,
) -> None:
    clean_df = load_csv_required(
        comparison_csv,
        "Clean comparison CSV",
    )

    best_models = load_json(best_models_json)

    robustness_binary_df = load_csv_required(
        robustness_binary_csv,
        "Binary robustness CSV",
    )

    robustness_action_df = load_csv_required(
        robustness_action_csv,
        "Action robustness CSV",
    )

    clean_df = normalize_clean_df(clean_df)
    robustness_binary_df = normalize_robustness_df(robustness_binary_df)
    robustness_action_df = normalize_robustness_df(robustness_action_df)

    ensure_dir(os.path.dirname(output_html))

    clean_sections = []

    for task in TASK_ORDER:
        clean_sections.append(build_clean_task_section(task, clean_df))

    robustness_sections = [
        build_robustness_task_section("binary", robustness_binary_df),
        build_robustness_task_section("action", robustness_action_df),
    ]

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Final Phase 4 Quality-Aware Fusion Dashboard</title>
        {build_css()}
    </head>
    <body>
        <main class="page">
            <section class="hero">
                <h1>Final Phase 4 Quality-Aware Fusion Dashboard</h1>
                <p>
                    Clean performance and robustness comparison across Phase 1, Phase 2, Phase 3, and Phase 4.
                </p>
                <p>
                    Binary test set: 2965 samples. Action test set: 2053 samples.
                </p>
                <p>Generated at: {escape_html(generated_at)}</p>
            </section>

            {build_nav()}

            {build_summary_cards(clean_df, robustness_binary_df, robustness_action_df, best_models)}

            {''.join(clean_sections)}

            {''.join(robustness_sections)}

            <section id="final">
                {build_final_recommendation(
                    clean_df,
                    robustness_binary_df,
                    robustness_action_df,
                    best_models,
                )}
            </section>

            <div class="footer">
                Generated by generate_phase4_dashboard.py
            </div>
        </main>
    </body>
    </html>
    """

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)

    print("\nFinal dashboard generated successfully.")
    print("=" * 100)
    print(f"Clean comparison CSV     : {comparison_csv}")
    print(f"Best models JSON         : {best_models_json}")
    print(f"Binary robustness CSV    : {robustness_binary_csv}")
    print(f"Action robustness CSV    : {robustness_action_csv}")
    print(f"Output HTML              : {output_html}")
    print("=" * 100)


def main() -> None:
    generate_dashboard()


if __name__ == "__main__":
    main()