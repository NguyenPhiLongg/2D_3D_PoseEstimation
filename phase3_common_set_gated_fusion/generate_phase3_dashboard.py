"""
Generate Phase 3 HTML dashboard.

Purpose:
- Read Phase 3 result JSON files.
- Generate a visual HTML dashboard for:
    1. 2D Common
    2. 3D Common
    3. Concat Fusion Common
    4. Gated Fusion

Output:
    outputs/ablation_results/phase3_dashboard.html

Run:
    python phase3_common_set_gated_fusion/generate_phase3_dashboard.py

Recommended order:
    python phase3_common_set_gated_fusion/build_common_split.py

    python phase3_common_set_gated_fusion/train_2d_common.py --task binary
    python phase3_common_set_gated_fusion/train_2d_common.py --task action

    python phase3_common_set_gated_fusion/train_3d_common.py --task binary
    python phase3_common_set_gated_fusion/train_3d_common.py --task action

    python phase3_common_set_gated_fusion/train_concat_common.py --task binary
    python phase3_common_set_gated_fusion/train_concat_common.py --task action

    python phase3_common_set_gated_fusion/train_gated_fusion.py --task binary
    python phase3_common_set_gated_fusion/train_gated_fusion.py --task action

    python phase3_common_set_gated_fusion/compare_phase3_results.py
    python phase3_common_set_gated_fusion/generate_phase3_dashboard.py
"""

from __future__ import annotations

import json
import math
import html
from pathlib import Path
from typing import Dict, List, Optional

from phase3_utils import OUTPUT_DIR, PHASE3_DIR


# =========================
# PATHS
# =========================

ABLATION_DIR = OUTPUT_DIR / "ablation_results"
ABLATION_DIR.mkdir(parents=True, exist_ok=True)

DASHBOARD_PATH = ABLATION_DIR / "phase3_dashboard.html"

RESULT_FILES = [
    {
        "experiment": "2D Common",
        "short_name": "2D",
        "task": "binary",
        "input": "2D",
        "input_dim": "40D",
        "path": OUTPUT_DIR / "training_2d_common" / "results_2d_common_binary.json",
    },
    {
        "experiment": "2D Common",
        "short_name": "2D",
        "task": "action",
        "input": "2D",
        "input_dim": "40D",
        "path": OUTPUT_DIR / "training_2d_common" / "results_2d_common_action.json",
    },
    {
        "experiment": "3D Common",
        "short_name": "3D",
        "task": "binary",
        "input": "3D",
        "input_dim": "59D",
        "path": OUTPUT_DIR / "training_3d_common" / "results_3d_common_binary.json",
    },
    {
        "experiment": "3D Common",
        "short_name": "3D",
        "task": "action",
        "input": "3D",
        "input_dim": "59D",
        "path": OUTPUT_DIR / "training_3d_common" / "results_3d_common_action.json",
    },
    {
        "experiment": "Concat Fusion Common",
        "short_name": "Concat",
        "task": "binary",
        "input": "2D + 3D",
        "input_dim": "99D",
        "path": OUTPUT_DIR / "training_concat_common" / "results_concat_common_binary.json",
    },
    {
        "experiment": "Concat Fusion Common",
        "short_name": "Concat",
        "task": "action",
        "input": "2D + 3D",
        "input_dim": "99D",
        "path": OUTPUT_DIR / "training_concat_common" / "results_concat_common_action.json",
    },
    {
        "experiment": "Gated Fusion",
        "short_name": "Gated",
        "task": "binary",
        "input": "2D + 3D",
        "input_dim": "40D + 59D",
        "path": OUTPUT_DIR / "training_gated_fusion" / "results_gated_fusion_binary.json",
    },
    {
        "experiment": "Gated Fusion",
        "short_name": "Gated",
        "task": "action",
        "input": "2D + 3D",
        "input_dim": "40D + 59D",
        "path": OUTPUT_DIR / "training_gated_fusion" / "results_gated_fusion_action.json",
    },
]


# =========================
# BASIC HELPERS
# =========================

def escape_text(value) -> str:
    if value is None:
        return ""

    return html.escape(str(value))


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value) -> Optional[float]:
    if value is None:
        return None

    try:
        value = float(value)

        if math.isnan(value):
            return None

        return value

    except Exception:
        return None


def fmt_score(value) -> str:
    value = safe_float(value)

    if value is None:
        return "-"

    return f"{value:.4f}"


def fmt_percent(value) -> str:
    value = safe_float(value)

    if value is None:
        return "-"

    return f"{value * 100.0:.2f}%"


def fmt_int(value) -> str:
    if value is None:
        return "-"

    try:
        return f"{int(value):,}"
    except Exception:
        return "-"


def get_report_dict(data: Dict) -> Dict:
    report = data.get("classification_report_dict", {})

    if isinstance(report, dict):
        return report

    return {}


def get_class_metric(data: Dict, class_name: str, metric_name: str) -> Optional[float]:
    report = get_report_dict(data)

    if class_name not in report:
        return None

    class_item = report[class_name]

    if not isinstance(class_item, dict):
        return None

    return safe_float(class_item.get(metric_name))


def get_records() -> List[Dict]:
    records = []

    for item in RESULT_FILES:
        data = load_json(item["path"])

        records.append(
            {
                "experiment": item["experiment"],
                "short_name": item["short_name"],
                "task": item["task"],
                "input": item["input"],
                "input_dim": item["input_dim"],
                "path": item["path"],
                "exists": data is not None,
                "data": data,
            }
        )

    return records


def get_existing_records(records: List[Dict]) -> List[Dict]:
    return [record for record in records if record["exists"]]


def find_record(records: List[Dict], task: str, experiment: str) -> Optional[Dict]:
    for record in records:
        if record["task"] == task and record["experiment"] == experiment and record["exists"]:
            return record

    return None


# =========================
# HTML COMPONENTS
# =========================

def make_table(headers: List[str], rows: List[List[str]], css_class: str = "data-table") -> str:
    lines = []

    lines.append(f'<table class="{css_class}">')
    lines.append("<thead>")
    lines.append("<tr>")

    for header in headers:
        lines.append(f"<th>{escape_text(header)}</th>")

    lines.append("</tr>")
    lines.append("</thead>")
    lines.append("<tbody>")

    for row in rows:
        lines.append("<tr>")

        for cell in row:
            lines.append(f"<td>{cell}</td>")

        lines.append("</tr>")

    lines.append("</tbody>")
    lines.append("</table>")

    return "\n".join(lines)


def metric_card(title: str, value: str, subtitle: str, tag: str = "") -> str:
    tag_html = f'<div class="card-tag">{escape_text(tag)}</div>' if tag else ""

    return f"""
    <div class="metric-card">
        {tag_html}
        <div class="metric-title">{escape_text(title)}</div>
        <div class="metric-value">{escape_text(value)}</div>
        <div class="metric-subtitle">{escape_text(subtitle)}</div>
    </div>
    """


def make_missing_files_section(records: List[Dict]) -> str:
    missing = [record for record in records if not record["exists"]]

    if not missing:
        return ""

    rows = []

    for record in missing:
        rows.append(
            [
                escape_text(record["task"]),
                escape_text(record["experiment"]),
                escape_text(str(record["path"])),
            ]
        )

    return f"""
    <section class="panel warning-panel">
        <h2>Missing Result Files</h2>
        <p>These files have not been generated yet. Train the corresponding models to complete the dashboard.</p>
        {make_table(["Task", "Experiment", "Expected JSON Path"], rows)}
    </section>
    """


def make_overview_cards(records: List[Dict]) -> str:
    existing = get_existing_records(records)

    if not existing:
        return """
        <section class="panel">
            <h2>Overview</h2>
            <p>No Phase 3 result files found yet.</p>
        </section>
        """

    cards = []

    for task in ["binary", "action"]:
        task_records = [r for r in existing if r["task"] == task]

        if not task_records:
            continue

        best_acc = max(
            task_records,
            key=lambda r: safe_float(r["data"].get("final_test_accuracy")) or -1,
        )

        best_f1 = max(
            task_records,
            key=lambda r: safe_float(r["data"].get("final_test_macro_f1")) or -1,
        )

        cards.append(
            metric_card(
                title=f"Best {task.capitalize()} Accuracy",
                value=fmt_percent(best_acc["data"].get("final_test_accuracy")),
                subtitle=best_acc["experiment"],
                tag=best_acc["input_dim"],
            )
        )

        cards.append(
            metric_card(
                title=f"Best {task.capitalize()} Macro F1",
                value=fmt_percent(best_f1["data"].get("final_test_macro_f1")),
                subtitle=best_f1["experiment"],
                tag=best_f1["input_dim"],
            )
        )

    return f"""
    <section class="panel">
        <h2>Overview</h2>
        <div class="metric-grid">
            {''.join(cards)}
        </div>
    </section>
    """


def make_binary_table(records: List[Dict]) -> str:
    rows = []

    for experiment in ["2D Common", "3D Common", "Concat Fusion Common", "Gated Fusion"]:
        record = find_record(records, "binary", experiment)

        if record is None:
            continue

        data = record["data"]

        rows.append(
            [
                escape_text(record["experiment"]),
                escape_text(record["input"]),
                escape_text(record["input_dim"]),
                fmt_percent(data.get("final_test_accuracy")),
                fmt_percent(data.get("final_test_macro_f1")),
                fmt_percent(get_class_metric(data, "Fall", "precision")),
                fmt_percent(get_class_metric(data, "Fall", "recall")),
                fmt_percent(get_class_metric(data, "Fall", "f1-score")),
                fmt_int(data.get("num_test_samples")),
                fmt_int(data.get("num_test_unique_videos")),
                fmt_score(data.get("mean_gate")) if record["experiment"] == "Gated Fusion" else "-",
            ]
        )

    if not rows:
        return """
        <section class="panel">
            <h2>Binary Task: Fall / Not_Fall</h2>
            <p>No binary result files found.</p>
        </section>
        """

    return f"""
    <section class="panel">
        <h2>Binary Task: Fall / Not_Fall</h2>
        <p>This table compares all binary models on the same common set.</p>
        {make_table(
            [
                "Model",
                "Input",
                "Input Dim",
                "Accuracy",
                "Macro F1",
                "Fall Precision",
                "Fall Recall",
                "Fall F1",
                "Test Samples",
                "Test Videos",
                "Mean Gate",
            ],
            rows,
            css_class="data-table compact-table",
        )}
    </section>
    """


def make_action_table(records: List[Dict]) -> str:
    rows = []

    for experiment in ["2D Common", "3D Common", "Concat Fusion Common", "Gated Fusion"]:
        record = find_record(records, "action", experiment)

        if record is None:
            continue

        data = record["data"]

        rows.append(
            [
                escape_text(record["experiment"]),
                escape_text(record["input"]),
                escape_text(record["input_dim"]),
                fmt_percent(data.get("final_test_accuracy")),
                fmt_percent(data.get("final_test_macro_f1")),
                fmt_percent(get_class_metric(data, "Sitting", "f1-score")),
                fmt_percent(get_class_metric(data, "Sleeping", "f1-score")),
                fmt_percent(get_class_metric(data, "Standing", "f1-score")),
                fmt_percent(get_class_metric(data, "Walking", "f1-score")),
                fmt_int(data.get("num_test_samples")),
                fmt_int(data.get("num_test_unique_videos")),
                fmt_score(data.get("mean_gate")) if record["experiment"] == "Gated Fusion" else "-",
            ]
        )

    if not rows:
        return """
        <section class="panel">
            <h2>Action Task: Sitting / Sleeping / Standing / Walking</h2>
            <p>No action result files found.</p>
        </section>
        """

    return f"""
    <section class="panel">
        <h2>Action Task: Sitting / Sleeping / Standing / Walking</h2>
        <p>This table compares action classification models on the same common set.</p>
        {make_table(
            [
                "Model",
                "Input",
                "Input Dim",
                "Accuracy",
                "Macro F1",
                "Sitting F1",
                "Sleeping F1",
                "Standing F1",
                "Walking F1",
                "Test Samples",
                "Test Videos",
                "Mean Gate",
            ],
            rows,
            css_class="data-table compact-table",
        )}
    </section>
    """


def make_bar(value: Optional[float], label: str) -> str:
    value = safe_float(value)

    if value is None:
        width = 0.0
        text = "-"
    else:
        width = max(0.0, min(100.0, value * 100.0))
        text = f"{width:.2f}%"

    return f"""
    <div class="bar-row">
        <div class="bar-label">{escape_text(label)}</div>
        <div class="bar-track">
            <div class="bar-fill" style="width: {width:.2f}%"></div>
        </div>
        <div class="bar-value">{escape_text(text)}</div>
    </div>
    """


def make_metric_bars(records: List[Dict], task: str, metric: str, title: str) -> str:
    rows = []

    for experiment in ["2D Common", "3D Common", "Concat Fusion Common", "Gated Fusion"]:
        record = find_record(records, task, experiment)

        if record is None:
            continue

        value = record["data"].get(metric)
        rows.append(make_bar(value, experiment))

    if not rows:
        return ""

    return f"""
    <div class="bar-card">
        <h3>{escape_text(title)}</h3>
        {''.join(rows)}
    </div>
    """


def make_bar_section(records: List[Dict]) -> str:
    content = []

    content.append(
        make_metric_bars(
            records,
            task="binary",
            metric="final_test_accuracy",
            title="Binary Accuracy",
        )
    )

    content.append(
        make_metric_bars(
            records,
            task="binary",
            metric="final_test_macro_f1",
            title="Binary Macro F1",
        )
    )

    content.append(
        make_metric_bars(
            records,
            task="action",
            metric="final_test_accuracy",
            title="Action Accuracy",
        )
    )

    content.append(
        make_metric_bars(
            records,
            task="action",
            metric="final_test_macro_f1",
            title="Action Macro F1",
        )
    )

    content = [item for item in content if item.strip()]

    if not content:
        return ""

    return f"""
    <section class="panel">
        <h2>Visual Metric Comparison</h2>
        <div class="bar-grid">
            {''.join(content)}
        </div>
    </section>
    """


def make_confusion_matrix_html(cm: List[List[int]], class_names: List[str]) -> str:
    if not cm:
        return "<p>No confusion matrix available.</p>"

    max_value = max(max(row) for row in cm) if cm else 1

    if max_value <= 0:
        max_value = 1

    lines = []

    lines.append('<table class="confusion-matrix">')
    lines.append("<thead>")
    lines.append("<tr>")
    lines.append("<th>True \\ Pred</th>")

    for class_name in class_names:
        lines.append(f"<th>{escape_text(class_name)}</th>")

    lines.append("</tr>")
    lines.append("</thead>")
    lines.append("<tbody>")

    for i, row in enumerate(cm):
        row_sum = sum(row)

        lines.append("<tr>")
        lines.append(f"<th>{escape_text(class_names[i] if i < len(class_names) else str(i))}</th>")

        for j, value in enumerate(row):
            alpha = value / max_value

            if i == j:
                background = f"rgba(34, 197, 94, {0.20 + 0.65 * alpha:.3f})"
                label = "Correct"
            else:
                background = f"rgba(239, 68, 68, {0.15 + 0.65 * alpha:.3f})"
                label = "Wrong"

            percent = (value / row_sum * 100.0) if row_sum > 0 else 0.0

            lines.append(
                f"""
                <td style="background: {background};">
                    <div class="cm-value">{value}</div>
                    <div class="cm-percent">{percent:.1f}%</div>
                    <div class="cm-label">{label}</div>
                </td>
                """
            )

        lines.append("</tr>")

    lines.append("</tbody>")
    lines.append("</table>")

    return "\n".join(lines)


def make_confusion_section(records: List[Dict]) -> str:
    existing = get_existing_records(records)

    if not existing:
        return ""

    cards = []

    for record in existing:
        data = record["data"]
        cm = data.get("confusion_matrix", [])
        class_names = data.get("class_names", [])

        cards.append(
            f"""
            <div class="cm-card">
                <h3>{escape_text(record["task"].capitalize())} - {escape_text(record["experiment"])}</h3>
                <div class="cm-subtitle">
                    Input: {escape_text(record["input"])} | Dim: {escape_text(record["input_dim"])}
                </div>
                {make_confusion_matrix_html(cm, class_names)}
            </div>
            """
        )

    return f"""
    <section class="panel">
        <h2>Confusion Matrices</h2>
        <p>Green diagonal cells are correct predictions. Red cells are wrong predictions.</p>
        <div class="cm-grid">
            {''.join(cards)}
        </div>
    </section>
    """


def make_gate_section(records: List[Dict]) -> str:
    gated_records = [
        record for record in records
        if record["exists"] and record["experiment"] == "Gated Fusion"
    ]

    if not gated_records:
        return ""

    rows = []

    for record in gated_records:
        data = record["data"]

        rows.append(
            [
                escape_text(record["task"]),
                fmt_score(data.get("mean_gate")),
                fmt_score(data.get("std_gate")),
                fmt_score(data.get("min_gate")),
                fmt_score(data.get("max_gate")),
            ]
        )

    explanation = """
    <p>
        Gate interpretation: a higher gate value means the model gives more weight to the 2D stream,
        while a lower gate value means the model gives more weight to the estimated 3D stream.
        This interpretation is approximate because vector-gate mode learns one gate value per hidden dimension.
    </p>
    """

    return f"""
    <section class="panel">
        <h2>Gated Fusion Analysis</h2>
        {explanation}
        {make_table(
            ["Task", "Mean Gate", "Std Gate", "Min Gate", "Max Gate"],
            rows,
            css_class="data-table compact-table",
        )}
    </section>
    """


def make_history_table(records: List[Dict]) -> str:
    rows = []

    for record in get_existing_records(records):
        data = record["data"]
        history = data.get("history", [])

        if not history:
            continue

        best_epoch = data.get("best_epoch")
        last_epoch = history[-1] if history else {}

        rows.append(
            [
                escape_text(record["task"]),
                escape_text(record["experiment"]),
                fmt_int(best_epoch),
                fmt_percent(data.get("best_val_macro_f1")),
                fmt_percent(data.get("final_test_accuracy")),
                fmt_percent(data.get("final_test_macro_f1")),
                fmt_score(last_epoch.get("train_loss")),
            ]
        )

    if not rows:
        return ""

    return f"""
    <section class="panel">
        <h2>Training Summary</h2>
        {make_table(
            [
                "Task",
                "Model",
                "Best Epoch",
                "Best Val Macro F1",
                "Test Accuracy",
                "Test Macro F1",
                "Last Train Loss",
            ],
            rows,
            css_class="data-table compact-table",
        )}
    </section>
    """


def make_interpretation_section() -> str:
    return """
    <section class="panel">
        <h2>How to Interpret This Dashboard</h2>

        <div class="note-box">
            <h3>1. Fair comparison</h3>
            <p>
                Phase 3 is designed to compare all models on the same common video subset.
                This avoids unfair comparison between the old 2D baseline and 3D/Fusion models.
            </p>
        </div>

        <div class="note-box">
            <h3>2. Concat Fusion</h3>
            <p>
                Concat Fusion directly joins 2D features and 3D features:
                40D + 59D = 99D. This is a fixed early-fusion strategy.
            </p>
        </div>

        <div class="note-box">
            <h3>3. Gated Fusion</h3>
            <p>
                Gated Fusion learns how much the model should rely on 2D or estimated 3D features.
                It is useful because estimated 3D pose may contain errors from YOLOv8-Pose and PoseFormerV2.
            </p>
        </div>

        <div class="note-box">
            <h3>4. Important metric for fall detection</h3>
            <p>
                Fall Recall is especially important because missing a real fall can be more serious
                than raising a false alarm.
            </p>
        </div>
    </section>
    """


# =========================
# HTML PAGE
# =========================

def get_css() -> str:
    return """
    <style>
        :root {
            --bg: #0f172a;
            --panel: #111827;
            --panel-2: #1f2937;
            --text: #e5e7eb;
            --muted: #9ca3af;
            --border: #374151;
            --blue: #38bdf8;
            --green: #22c55e;
            --red: #ef4444;
            --yellow: #facc15;
            --purple: #a78bfa;
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            padding: 0;
            background: var(--bg);
            color: var(--text);
            font-family: Arial, Helvetica, sans-serif;
            line-height: 1.5;
        }

        .page {
            max-width: 1500px;
            margin: 0 auto;
            padding: 28px;
        }

        .header {
            padding: 28px;
            border: 1px solid var(--border);
            border-radius: 18px;
            background: linear-gradient(135deg, #111827, #172554);
            margin-bottom: 24px;
        }

        .header h1 {
            margin: 0 0 12px 0;
            font-size: 34px;
        }

        .header p {
            margin: 6px 0;
            color: var(--muted);
        }

        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 16px;
        }

        .badge {
            padding: 6px 12px;
            background: rgba(56, 189, 248, 0.15);
            color: #bae6fd;
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 999px;
            font-size: 13px;
        }

        .panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 22px;
            margin-bottom: 24px;
            overflow-x: auto;
        }

        .panel h2 {
            margin: 0 0 14px 0;
            font-size: 24px;
        }

        .panel h3 {
            margin: 0 0 10px 0;
        }

        .panel p {
            color: var(--muted);
        }

        .warning-panel {
            border-color: rgba(250, 204, 21, 0.4);
            background: rgba(250, 204, 21, 0.08);
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 16px;
        }

        .metric-card {
            position: relative;
            background: var(--panel-2);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px;
            min-height: 130px;
        }

        .card-tag {
            position: absolute;
            top: 12px;
            right: 12px;
            padding: 4px 9px;
            background: rgba(167, 139, 250, 0.15);
            color: #ddd6fe;
            border-radius: 999px;
            font-size: 12px;
            border: 1px solid rgba(167, 139, 250, 0.35);
        }

        .metric-title {
            color: var(--muted);
            font-size: 14px;
            margin-bottom: 10px;
        }

        .metric-value {
            font-size: 34px;
            font-weight: bold;
            color: var(--blue);
            margin-bottom: 8px;
        }

        .metric-subtitle {
            color: var(--text);
            font-size: 15px;
        }

        .data-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            margin-top: 14px;
        }

        .data-table th {
            background: #020617;
            color: #f9fafb;
            text-align: left;
            padding: 10px;
            border: 1px solid var(--border);
            white-space: nowrap;
        }

        .data-table td {
            padding: 10px;
            border: 1px solid var(--border);
            color: var(--text);
            white-space: nowrap;
        }

        .data-table tr:nth-child(even) td {
            background: rgba(255, 255, 255, 0.03);
        }

        .compact-table td,
        .compact-table th {
            text-align: center;
        }

        .compact-table td:first-child,
        .compact-table th:first-child {
            text-align: left;
        }

        .bar-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
            gap: 18px;
        }

        .bar-card {
            background: var(--panel-2);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px;
        }

        .bar-card h3 {
            margin-bottom: 18px;
        }

        .bar-row {
            display: grid;
            grid-template-columns: 160px 1fr 80px;
            gap: 12px;
            align-items: center;
            margin: 12px 0;
        }

        .bar-label {
            color: var(--text);
            font-size: 14px;
        }

        .bar-track {
            height: 14px;
            background: #020617;
            border: 1px solid var(--border);
            border-radius: 999px;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--blue), var(--green));
            border-radius: 999px;
        }

        .bar-value {
            color: var(--muted);
            font-size: 13px;
            text-align: right;
        }

        .cm-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
            gap: 18px;
        }

        .cm-card {
            background: var(--panel-2);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px;
            overflow-x: auto;
        }

        .cm-card h3 {
            margin-bottom: 4px;
        }

        .cm-subtitle {
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 14px;
        }

        .confusion-matrix {
            width: 100%;
            border-collapse: collapse;
            text-align: center;
            font-size: 13px;
        }

        .confusion-matrix th {
            background: #020617;
            color: #f9fafb;
            padding: 8px;
            border: 1px solid var(--border);
        }

        .confusion-matrix td {
            min-width: 90px;
            padding: 10px;
            border: 1px solid var(--border);
            color: #f9fafb;
        }

        .cm-value {
            font-size: 20px;
            font-weight: bold;
        }

        .cm-percent {
            font-size: 12px;
            color: #e5e7eb;
        }

        .cm-label {
            margin-top: 4px;
            font-size: 11px;
            opacity: 0.85;
        }

        .note-box {
            background: var(--panel-2);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px;
            margin: 14px 0;
        }

        .note-box h3 {
            color: var(--blue);
        }

        .footer {
            color: var(--muted);
            text-align: center;
            padding: 20px;
            font-size: 13px;
        }
    </style>
    """


def build_html(records: List[Dict]) -> str:
    existing_count = len(get_existing_records(records))
    total_count = len(records)

    body = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Phase 3 Dashboard</title>
        {get_css()}
    </head>
    <body>
        <div class="page">
            <div class="header">
                <h1>Phase 3 Dashboard</h1>
                <p>Common-set Fair Comparison and Gated Fusion for Fall Detection and Action Recognition.</p>
                <p>Generated from Phase 3 result JSON files.</p>

                <div class="badge-row">
                    <div class="badge">2D Common</div>
                    <div class="badge">3D Common</div>
                    <div class="badge">Concat Fusion</div>
                    <div class="badge">Gated Fusion</div>
                    <div class="badge">{existing_count}/{total_count} result files found</div>
                </div>
            </div>

            {make_missing_files_section(records)}
            {make_overview_cards(records)}
            {make_binary_table(records)}
            {make_action_table(records)}
            {make_bar_section(records)}
            {make_gate_section(records)}
            {make_history_table(records)}
            {make_confusion_section(records)}
            {make_interpretation_section()}

            <div class="footer">
                Phase 3 directory: {escape_text(PHASE3_DIR)}
                <br />
                Dashboard path: {escape_text(DASHBOARD_PATH)}
            </div>
        </div>
    </body>
    </html>
    """

    return body


def main() -> None:
    print("=" * 80)
    print("GENERATE PHASE 3 HTML DASHBOARD")
    print("=" * 80)
    print("Phase 3 directory:", PHASE3_DIR)
    print("Output directory:", OUTPUT_DIR)
    print("Dashboard path:", DASHBOARD_PATH)
    print("=" * 80)

    records = get_records()

    existing = get_existing_records(records)

    print("Result files found:", len(existing), "/", len(records))

    for record in records:
        status = "FOUND" if record["exists"] else "MISSING"
        print(f"[{status}] {record['task']} - {record['experiment']} -> {record['path']}")

    html_text = build_html(records)

    with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
        f.write(html_text)

    print("=" * 80)
    print("Dashboard generated successfully.")
    print("Saved to:", DASHBOARD_PATH)
    print("=" * 80)
    print("Open on Windows:")
    print(f"start {DASHBOARD_PATH}")


if __name__ == "__main__":
    main()