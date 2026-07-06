import os
import json
import html
from datetime import datetime


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

OUTPUT_HTML = os.path.join(
    PROJECT_ROOT,
    "phase2_3d_upgrade",
    "outputs",
    "model_comparison_dashboard.html"
)


RESULT_FILES = [
    {
        "task": "Binary",
        "name": "2D",
        "full_name": "CNN1D-BiLSTM 2D",
        "path": "phase1_2d_baseline/outputs/training_2d/results_2d_binary.json",
    },
    {
        "task": "Binary",
        "name": "3D",
        "full_name": "CNN1D-BiLSTM 3D",
        "path": "phase2_3d_upgrade/outputs/training_3d/results_3d_binary_cnn_lstm.json",
    },
    {
        "task": "Binary",
        "name": "Fusion 2D+3D",
        "full_name": "CNN1D-BiLSTM Fusion 2D+3D",
        "path": "phase2_3d_upgrade/outputs/training_fusion_2d3d/results_fusion_2d3d_binary.json",
    },
    {
        "task": "Action",
        "name": "2D",
        "full_name": "CNN1D-BiLSTM 2D",
        "path": "phase1_2d_baseline/outputs/training_2d/results_2d_action.json",
    },
    {
        "task": "Action",
        "name": "3D",
        "full_name": "CNN1D-BiLSTM 3D",
        "path": "phase2_3d_upgrade/outputs/training_3d/results_3d_action_cnn_lstm.json",
    },
    {
        "task": "Action",
        "name": "Fusion 2D+3D",
        "full_name": "CNN1D-BiLSTM Fusion 2D+3D",
        "path": "phase2_3d_upgrade/outputs/training_fusion_2d3d/results_fusion_2d3d_action.json",
    },
]


def safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def fmt_float(value, digits=4):
    value = safe_float(value)

    if value is None:
        return "N/A"

    return f"{value:.{digits}f}"


def fmt_percent(value):
    value = safe_float(value)

    if value is None:
        return "N/A"

    return f"{value * 100:.2f}%"


def escape(value):
    return html.escape(str(value))


def load_json(path):
    full_path = os.path.join(PROJECT_ROOT, path)

    if not os.path.exists(full_path):
        return None

    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_accuracy(data):
    return safe_float(
        data.get("final_test_accuracy", data.get("best_accuracy"))
    )


def get_macro_f1(data):
    return safe_float(
        data.get("final_test_macro_f1", data.get("best_f1"))
    )


def get_report_text(data):
    report_text = data.get("classification_report_text")

    if isinstance(report_text, str):
        return report_text

    report = data.get("classification_report")

    if isinstance(report, str):
        return report

    return ""


def parse_report_text(report_text, class_names):
    result = {}

    if not report_text:
        return result

    lines = report_text.splitlines()

    for class_name in class_names:
        for line in lines:
            stripped = line.strip()

            if not stripped.startswith(class_name):
                continue

            parts = stripped.split()

            if len(parts) >= 5:
                try:
                    result[class_name] = {
                        "precision": float(parts[-4]),
                        "recall": float(parts[-3]),
                        "f1-score": float(parts[-2]),
                        "support": int(float(parts[-1])),
                    }
                except Exception:
                    pass

    return result


def get_report_dict(data):
    report = data.get("classification_report")

    if isinstance(report, dict):
        return report

    class_names = data.get("class_names", [])
    report_text = get_report_text(data)

    return parse_report_text(report_text, class_names)


def get_class_metric(data, class_name, metric_name):
    report = get_report_dict(data)

    if not isinstance(report, dict):
        return None

    class_report = report.get(class_name)

    if not isinstance(class_report, dict):
        return None

    return safe_float(class_report.get(metric_name))


def make_bar(value, max_value=1.0):
    value = safe_float(value, 0.0)

    if value is None:
        value = 0.0

    width = max(0.0, min(100.0, (value / max_value) * 100.0))

    return f"""
    <div class="bar-wrap">
        <div class="bar-fill" style="width: {width:.2f}%"></div>
    </div>
    """


def make_metric_card(title, value, subtitle=""):
    return f"""
    <div class="metric-card">
        <div class="metric-title">{escape(title)}</div>
        <div class="metric-value">{escape(value)}</div>
        <div class="metric-subtitle">{escape(subtitle)}</div>
    </div>
    """


def find_record(records, task, name):
    for r in records:
        if r["task"] == task and r["name"] == name:
            return r

    return None


def make_best_cards(records):
    binary_records = [r for r in records if r["task"] == "Binary"]
    action_records = [r for r in records if r["task"] == "Action"]

    cards = ""

    if binary_records:
        best_binary_acc = max(binary_records, key=lambda r: get_accuracy(r["data"]) or 0)
        best_binary_f1 = max(binary_records, key=lambda r: get_macro_f1(r["data"]) or 0)

        cards += make_metric_card(
            "Best Binary Accuracy",
            f'{best_binary_acc["name"]} - {fmt_percent(get_accuracy(best_binary_acc["data"]))}',
            "Fall / Not_Fall"
        )

        cards += make_metric_card(
            "Best Binary Macro F1",
            f'{best_binary_f1["name"]} - {fmt_percent(get_macro_f1(best_binary_f1["data"]))}',
            "Fall / Not_Fall"
        )

    if action_records:
        best_action_acc = max(action_records, key=lambda r: get_accuracy(r["data"]) or 0)
        best_action_f1 = max(action_records, key=lambda r: get_macro_f1(r["data"]) or 0)

        cards += make_metric_card(
            "Best Action Accuracy",
            f'{best_action_acc["name"]} - {fmt_percent(get_accuracy(best_action_acc["data"]))}',
            "Sitting / Sleeping / Standing / Walking"
        )

        cards += make_metric_card(
            "Best Action Macro F1",
            f'{best_action_f1["name"]} - {fmt_percent(get_macro_f1(best_action_f1["data"]))}',
            "Sitting / Sleeping / Standing / Walking"
        )

    return f"""
    <div class="metric-grid">
        {cards}
    </div>
    """


def make_short_metric_tables(records):
    binary_order = [
        ("2D Binary", "2D"),
        ("3D Binary", "3D"),
        ("Fusion 2D+3D Binary", "Fusion 2D+3D"),
    ]

    action_order = [
        ("2D Action", "2D"),
        ("3D Action", "3D"),
        ("Fusion 2D+3D Action", "Fusion 2D+3D"),
    ]

    binary_rows = ""

    for label, name in binary_order:
        r = find_record(records, "Binary", name)

        if r is None:
            continue

        data = r["data"]

        acc = get_accuracy(data)
        macro_f1 = get_macro_f1(data)
        fall_precision = get_class_metric(data, "Fall", "precision")
        fall_recall = get_class_metric(data, "Fall", "recall")
        fall_f1 = get_class_metric(data, "Fall", "f1-score")

        row_class = "best-row" if "Fusion" in label else ""

        binary_rows += f"""
        <tr class="{row_class}">
            <td>{escape(label)}</td>
            <td>{fmt_float(acc)}</td>
            <td>{fmt_float(macro_f1)}</td>
            <td>{fmt_float(fall_precision, 2)}</td>
            <td>{fmt_float(fall_recall, 2)}</td>
            <td>{fmt_float(fall_f1, 2)}</td>
        </tr>
        """

    action_rows = ""

    for label, name in action_order:
        r = find_record(records, "Action", name)

        if r is None:
            continue

        data = r["data"]

        acc = get_accuracy(data)
        macro_f1 = get_macro_f1(data)

        sitting_f1 = get_class_metric(data, "Sitting", "f1-score")
        sleeping_f1 = get_class_metric(data, "Sleeping", "f1-score")
        standing_f1 = get_class_metric(data, "Standing", "f1-score")
        walking_f1 = get_class_metric(data, "Walking", "f1-score")

        row_class = "best-row" if "Fusion" in label else ""

        action_rows += f"""
        <tr class="{row_class}">
            <td>{escape(label)}</td>
            <td>{fmt_float(acc)}</td>
            <td>{fmt_float(macro_f1)}</td>
            <td>{fmt_float(sitting_f1, 2)}</td>
            <td>{fmt_float(sleeping_f1, 2)}</td>
            <td>{fmt_float(standing_f1, 2)}</td>
            <td>{fmt_float(walking_f1, 2)}</td>
        </tr>
        """

    return f"""
    <div class="section compact-section">
        <h2>Compact Metric Tables</h2>

        <h3>Binary Task: Fall / Not_Fall</h3>
        <table class="compact-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Macro F1</th>
                    <th>Fall Precision</th>
                    <th>Fall Recall</th>
                    <th>Fall F1</th>
                </tr>
            </thead>
            <tbody>
                {binary_rows}
            </tbody>
        </table>

        <h3 style="margin-top: 30px;">Action Task: Sitting / Sleeping / Standing / Walking</h3>
        <table class="compact-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Macro F1</th>
                    <th>Sitting F1</th>
                    <th>Sleeping F1</th>
                    <th>Standing F1</th>
                    <th>Walking F1</th>
                </tr>
            </thead>
            <tbody>
                {action_rows}
            </tbody>
        </table>
    </div>
    """


def make_summary_table(records, task):
    task_records = [r for r in records if r["task"] == task]

    rows = ""

    for r in task_records:
        data = r["data"]

        accuracy = get_accuracy(data)
        macro_f1 = get_macro_f1(data)

        rows += f"""
        <tr>
            <td><b>{escape(r["name"])}</b></td>
            <td>{escape(data.get("model_type", r["full_name"]))}</td>
            <td>{escape(data.get("normalization", "N/A"))}</td>
            <td>{escape(data.get("input_dim", "N/A"))}</td>
            <td>{escape(data.get("sequence_length", "N/A"))}</td>
            <td>{escape(data.get("stride", "N/A"))}</td>
            <td>{escape(data.get("best_epoch", "N/A"))}</td>
            <td>
                <b>{fmt_percent(accuracy)}</b>
                {make_bar(accuracy)}
            </td>
            <td>
                <b>{fmt_percent(macro_f1)}</b>
                {make_bar(macro_f1)}
            </td>
        </tr>
        """

    return f"""
    <div class="section">
        <h2>{escape(task)} Result Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Input</th>
                    <th>Model</th>
                    <th>Normalization</th>
                    <th>Input dim</th>
                    <th>Seq length</th>
                    <th>Stride</th>
                    <th>Best epoch</th>
                    <th>Accuracy</th>
                    <th>Macro F1</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """


def make_comparison_chart(records, task):
    task_records = [r for r in records if r["task"] == task]

    rows = ""

    for r in task_records:
        acc = get_accuracy(r["data"])
        f1 = get_macro_f1(r["data"])

        acc_width = acc * 100 if acc is not None else 0
        f1_width = f1 * 100 if f1 is not None else 0

        rows += f"""
        <div class="chart-row">
            <div class="chart-label">{escape(r["name"])} Accuracy</div>
            <div class="chart-bar-bg">
                <div class="chart-bar" style="width: {acc_width:.2f}%"></div>
            </div>
            <div class="chart-value">{fmt_percent(acc)}</div>
        </div>

        <div class="chart-row">
            <div class="chart-label">{escape(r["name"])} Macro F1</div>
            <div class="chart-bar-bg">
                <div class="chart-bar alt" style="width: {f1_width:.2f}%"></div>
            </div>
            <div class="chart-value">{fmt_percent(f1)}</div>
        </div>
        """

    return f"""
    <div class="section">
        <h2>{escape(task)} Visual Comparison</h2>
        <div class="chart-box">
            {rows}
        </div>
    </div>
    """


def make_confusion_matrix(cm, class_names):
    if not cm:
        return "<p>N/A</p>"

    if not class_names:
        class_names = [f"Class {i}" for i in range(len(cm))]

    max_value = max(max(row) for row in cm) if cm else 1

    if max_value <= 0:
        max_value = 1

    header = "<th class='cm-corner'>True \\ Pred</th>"

    for name in class_names:
        header += f"<th class='cm-header'>{escape(name)}</th>"

    rows = ""

    for i, row in enumerate(cm):
        true_label = class_names[i] if i < len(class_names) else f"Class {i}"
        row_sum = sum(row) if sum(row) > 0 else 1

        cells = f"<th class='cm-row-label'>{escape(true_label)}</th>"

        for j, value in enumerate(row):
            intensity = value / max_value
            row_percent = value / row_sum * 100

            if i == j:
                bg = f"rgba(34, 197, 94, {0.18 + 0.70 * intensity:.3f})"
                border = "rgba(34, 197, 94, 0.85)"
                tag = "Correct"
                tag_class = "correct"
            else:
                bg = f"rgba(239, 68, 68, {0.12 + 0.65 * intensity:.3f})"
                border = "rgba(239, 68, 68, 0.75)"
                tag = "Wrong"
                tag_class = "wrong"

            cells += f"""
            <td class="cm-heat-cell" style="background: {bg}; border-color: {border};">
                <div class="cm-number">{escape(value)}</div>
                <div class="cm-percent">{row_percent:.1f}% of true {escape(true_label)}</div>
                <div class="cm-tag {tag_class}">{tag}</div>
            </td>
            """

        rows += f"<tr>{cells}</tr>"

    return f"""
    <div class="cm-wrapper">
        <table class="cm-heatmap">
            <thead>
                <tr>{header}</tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>

    <div class="cm-note">
        Green diagonal cells are correct predictions. Red/orange cells are misclassifications.
    </div>
    """


def make_class_report_table(data):
    report = get_report_dict(data)
    class_names = data.get("class_names", [])

    if not report:
        text = get_report_text(data)

        if text:
            return f"<pre>{escape(text)}</pre>"

        return "<p>N/A</p>"

    rows = ""

    for class_name in class_names:
        item = report.get(class_name)

        if not item:
            continue

        precision = safe_float(item.get("precision"))
        recall = safe_float(item.get("recall"))
        f1 = safe_float(item.get("f1-score"))
        support = item.get("support", "N/A")

        rows += f"""
        <tr>
            <td><b>{escape(class_name)}</b></td>
            <td>{fmt_float(precision)}</td>
            <td>{fmt_float(recall)}</td>
            <td>{fmt_float(f1)}</td>
            <td>{escape(support)}</td>
        </tr>
        """

    return f"""
    <table>
        <thead>
            <tr>
                <th>Class</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-score</th>
                <th>Support</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """


def make_detail_section(records, task):
    task_records = [r for r in records if r["task"] == task]

    blocks = ""

    for r in task_records:
        data = r["data"]
        class_names = data.get("class_names", [])
        cm = data.get("confusion_matrix")

        blocks += f"""
        <div class="model-detail">
            <h3>{escape(task)} - {escape(r["name"])}</h3>

            <div class="metric-grid small">
                {make_metric_card("Accuracy", fmt_percent(get_accuracy(data)))}
                {make_metric_card("Macro F1", fmt_percent(get_macro_f1(data)))}
                {make_metric_card("Best epoch", data.get("best_epoch", "N/A"))}
                {make_metric_card("Input dim", data.get("input_dim", "N/A"))}
                {make_metric_card("Train sequences", data.get("train_sequences", "N/A"))}
                {make_metric_card("Test sequences", data.get("test_sequences", "N/A"))}
            </div>

            <h4>Classification Report</h4>
            {make_class_report_table(data)}

            <h4>Confusion Matrix Heatmap</h4>
            {make_confusion_matrix(cm, class_names)}
        </div>
        """

    return f"""
    <div class="section">
        <h2>{escape(task)} Detailed Results</h2>
        {blocks}
    </div>
    """


def build_html(records):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>2D vs 3D vs Fusion Result Dashboard</title>

    <style>
        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            background: #0f172a;
            color: #e5e7eb;
        }}

        .container {{
            max-width: 1550px;
            margin: 0 auto;
            padding: 32px;
        }}

        h1 {{
            margin: 0 0 8px 0;
            font-size: 34px;
        }}

        h2 {{
            margin-top: 0;
            color: #f8fafc;
            border-left: 5px solid #38bdf8;
            padding-left: 12px;
        }}

        h3 {{
            margin-top: 0;
            color: #bae6fd;
        }}

        h4 {{
            margin-bottom: 10px;
            color: #f8fafc;
        }}

        .subtitle {{
            color: #94a3b8;
            margin-bottom: 28px;
        }}

        .section {{
            background: #111827;
            border: 1px solid #263244;
            border-radius: 16px;
            padding: 22px;
            margin-bottom: 24px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        }}

        .compact-section {{
            border: 1px solid rgba(56, 189, 248, 0.35);
        }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}

        .metric-grid.small {{
            grid-template-columns: repeat(6, minmax(140px, 1fr));
            margin-bottom: 18px;
        }}

        .metric-card {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 16px;
        }}

        .metric-title {{
            font-size: 13px;
            color: #94a3b8;
            margin-bottom: 8px;
        }}

        .metric-value {{
            font-size: 22px;
            font-weight: bold;
            color: #f8fafc;
        }}

        .metric-subtitle {{
            font-size: 12px;
            color: #94a3b8;
            margin-top: 6px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            overflow: hidden;
            border-radius: 12px;
            margin-bottom: 12px;
        }}

        th, td {{
            padding: 12px 10px;
            border-bottom: 1px solid #263244;
            text-align: left;
            vertical-align: middle;
        }}

        th {{
            background: #1e293b;
            color: #cbd5e1;
            font-size: 13px;
        }}

        td {{
            color: #e5e7eb;
            font-size: 14px;
        }}

        tr:hover td {{
            background: #162033;
        }}

        .compact-table {{
            margin-top: 14px;
            margin-bottom: 10px;
        }}

        .compact-table th {{
            font-size: 15px;
            color: #f8fafc;
            border-bottom: 1px solid rgba(148, 163, 184, 0.25);
        }}

        .compact-table td {{
            font-size: 16px;
            font-weight: 600;
            padding-top: 18px;
            padding-bottom: 18px;
        }}

        .compact-table td:first-child {{
            font-weight: 800;
            color: #f8fafc;
        }}

        .compact-table .best-row td {{
            font-weight: 900;
            color: #ffffff;
            background: rgba(34, 197, 94, 0.08);
        }}

        .compact-table .best-row:hover td {{
            background: rgba(34, 197, 94, 0.14);
        }}

        .bar-wrap {{
            height: 8px;
            background: #334155;
            border-radius: 999px;
            margin-top: 6px;
            overflow: hidden;
        }}

        .bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, #38bdf8, #22c55e);
            border-radius: 999px;
        }}

        .chart-box {{
            display: flex;
            flex-direction: column;
            gap: 14px;
        }}

        .chart-row {{
            display: grid;
            grid-template-columns: 180px 1fr 90px;
            gap: 14px;
            align-items: center;
        }}

        .chart-label {{
            color: #cbd5e1;
            font-size: 14px;
        }}

        .chart-value {{
            text-align: right;
            font-weight: bold;
        }}

        .chart-bar-bg {{
            height: 20px;
            background: #334155;
            border-radius: 999px;
            overflow: hidden;
        }}

        .chart-bar {{
            height: 100%;
            background: linear-gradient(90deg, #38bdf8, #2563eb);
            border-radius: 999px;
        }}

        .chart-bar.alt {{
            background: linear-gradient(90deg, #22c55e, #84cc16);
        }}

        .model-detail {{
            background: #0f172a;
            border: 1px solid #263244;
            border-radius: 14px;
            padding: 18px;
            margin-bottom: 18px;
        }}

        .cm-wrapper {{
            width: 100%;
            overflow-x: auto;
            margin-top: 14px;
            padding-bottom: 6px;
        }}

        .cm-heatmap {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 10px;
            margin-bottom: 0;
        }}

        .cm-heatmap th,
        .cm-heatmap td {{
            border-bottom: none;
        }}

        .cm-corner,
        .cm-header,
        .cm-row-label {{
            background: #1e293b;
            color: #e5e7eb;
            text-align: center;
            font-weight: bold;
            border-radius: 12px;
            padding: 14px;
            white-space: nowrap;
        }}

        .cm-row-label {{
            min-width: 130px;
        }}

        .cm-heat-cell {{
            min-width: 165px;
            height: 110px;
            text-align: center;
            vertical-align: middle;
            border: 1px solid;
            border-radius: 16px;
            box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.18);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}

        .cm-heat-cell:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.25);
        }}

        .cm-number {{
            font-size: 28px;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 6px;
        }}

        .cm-percent {{
            font-size: 12px;
            color: #e5e7eb;
            opacity: 0.9;
        }}

        .cm-tag {{
            display: inline-block;
            margin-top: 7px;
            padding: 3px 8px;
            border-radius: 999px;
            font-size: 11px;
            background: rgba(15, 23, 42, 0.55);
            color: #f8fafc;
        }}

        .cm-tag.correct {{
            border: 1px solid rgba(34, 197, 94, 0.75);
        }}

        .cm-tag.wrong {{
            border: 1px solid rgba(239, 68, 68, 0.75);
        }}

        .cm-note {{
            color: #94a3b8;
            font-size: 13px;
            margin-top: 8px;
        }}

        pre {{
            white-space: pre-wrap;
            background: #020617;
            border: 1px solid #263244;
            border-radius: 12px;
            padding: 14px;
            color: #e5e7eb;
            overflow-x: auto;
        }}

        .warning {{
            background: #422006;
            border: 1px solid #854d0e;
            color: #fde68a;
            padding: 12px 16px;
            border-radius: 12px;
            margin-bottom: 20px;
        }}

        .footer {{
            color: #64748b;
            text-align: center;
            margin: 32px 0 12px;
            font-size: 13px;
        }}

        @media (max-width: 1100px) {{
            .metric-grid,
            .metric-grid.small {{
                grid-template-columns: repeat(2, 1fr);
            }}

            .chart-row {{
                grid-template-columns: 1fr;
                gap: 6px;
            }}

            .chart-value {{
                text-align: left;
            }}

            .cm-heat-cell {{
                min-width: 140px;
            }}
        }}
    </style>
</head>

<body>
    <div class="container">
        <h1>2D vs 3D vs Fusion Result Dashboard</h1>

        <div class="subtitle">
            Generated at: {escape(generated_at)}
        </div>

        {make_best_cards(records)}

        {make_short_metric_tables(records)}

        {make_summary_table(records, "Binary")}
        {make_comparison_chart(records, "Binary")}
        {make_detail_section(records, "Binary")}

        {make_summary_table(records, "Action")}
        {make_comparison_chart(records, "Action")}
        {make_detail_section(records, "Action")}

        <div class="footer">
            Dashboard generated from training result JSON files.
        </div>
    </div>
</body>
</html>
"""

    return html_content


def main():
    records = []
    missing_files = []

    for item in RESULT_FILES:
        data = load_json(item["path"])

        if data is None:
            missing_files.append(item["path"])
            continue

        records.append({
            "task": item["task"],
            "name": item["name"],
            "full_name": item["full_name"],
            "path": item["path"],
            "data": data,
        })

    if len(records) == 0:
        raise RuntimeError("No result JSON files found. Please train models first.")

    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)

    html_content = build_html(records)

    if missing_files:
        missing_html = "<div class='warning'><b>Missing files:</b><br>"
        missing_html += "<br>".join(escape(p) for p in missing_files)
        missing_html += "</div>"

        html_content = html_content.replace(
            "<div class=\"subtitle\">",
            missing_html + "\n        <div class=\"subtitle\">"
        )

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("Dashboard saved to:")
    print(OUTPUT_HTML)


if __name__ == "__main__":
    main()