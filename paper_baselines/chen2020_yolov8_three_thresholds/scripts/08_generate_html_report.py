from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
BASELINE_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = BASELINE_ROOT.parents[1]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def safe_float(value, default=None):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def pct(value) -> str:
    value = safe_float(value)
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def int_or_na(value) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return str(int(round(float(value))))
    except Exception:
        return "N/A"


def derive_confusion_from_support_recall(
    fall_support,
    fall_recall,
    not_fall_support,
    not_fall_recall,
) -> Dict:
    fall_support = safe_float(fall_support)
    fall_recall = safe_float(fall_recall)
    not_fall_support = safe_float(not_fall_support)
    not_fall_recall = safe_float(not_fall_recall)

    if fall_support is None or fall_recall is None or not_fall_support is None or not_fall_recall is None:
        return {
            "tn": None,
            "fp": None,
            "fn": None,
            "tp": None,
        }

    tp = int(round(fall_support * fall_recall))
    fn = int(round(fall_support - tp))

    tn = int(round(not_fall_support * not_fall_recall))
    fn = int(round(fall_support - tp))

    tn = int(round(not_fall_support * not_fall_recall))
    fp = int(round(not_fall_support - tn))

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def load_chen_result(report_path: Path) -> Dict:
    report = read_json(report_path)
    test = report["test_result"]

    return {
        "family": "Chen2020-style baseline",
        "model": "Chen2020-style YOLOv8 Three-Thresholds",
        "method_type": "Rule-based geometric thresholds",
        "aggregation": "Video-level",
        "accuracy": safe_float(test.get("accuracy")),
        "macro_f1": safe_float(test.get("macro_f1")),
        "fall_recall": safe_float(test.get("fall_recall")),
        "specificity": safe_float(test.get("specificity_not_fall_recall")),
        "fall_f1": safe_float(test.get("fall_f1")),
        "not_fall_f1": safe_float(test.get("not_fall_f1")),
        "tn": test.get("tn"),
        "fp": test.get("fp"),
        "fn": test.get("fn"),
        "tp": test.get("tp"),
        "test_samples": test.get("num_videos"),
        "test_videos": test.get("num_videos"),
        "source": str(report_path),
        "note": "Adapted Chen et al. 2020 rule-based method using YOLOv8-Pose instead of OpenPose.",
    }


def load_lin_result(report_path: Path) -> Dict:
    report = read_json(report_path)
    results = report.get("results", [])

    if len(results) == 0:
        raise ValueError(f"No results found in Lin report: {report_path}")

    best = sorted(
        results,
        key=lambda r: (
            safe_float(r.get("video_mean_prob_macro_f1"), 0.0),
            safe_float(r.get("video_mean_prob_accuracy"), 0.0),
        ),
        reverse=True,
    )[0]

    tn = best.get("video_mean_prob_tn")
    fp = best.get("video_mean_prob_fp")
    fn = best.get("video_mean_prob_fn")
    tp = best.get("video_mean_prob_tp")

    tn_f = safe_float(tn)
    fp_f = safe_float(fp)

    if tn_f is not None and fp_f is not None and (tn_f + fp_f) > 0:
        specificity = tn_f / (tn_f + fp_f)
    else:
        specificity = None

    model_type = str(best.get("model_type", "unknown")).upper()

    return {
        "family": "Lin2021-style baseline",
        "model": f"Lin-style YOLOv8 + {model_type}",
        "method_type": "Recurrent skeleton sequence model",
        "aggregation": "Video-level mean probability",
        "accuracy": safe_float(best.get("video_mean_prob_accuracy")),
        "macro_f1": safe_float(best.get("video_mean_prob_macro_f1")),
        "fall_recall": safe_float(best.get("video_mean_prob_fall_recall")),
        "specificity": specificity,
        "fall_f1": safe_float(best.get("video_mean_prob_fall_f1")),
        "not_fall_f1": safe_float(best.get("video_mean_prob_not_fall_f1")),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "test_samples": best.get("num_test_videos"),
        "test_videos": best.get("num_test_videos"),
        "source": str(report_path),
        "note": "Best video-level recurrent baseline from the adapted Lin-style YOLOv8 pipeline.",
    }


def load_phase4_binary_rows(comparison_json: Path) -> List[Dict]:
    data = read_json(comparison_json)
    rows = data.get("comparison", [])

    wanted = {
        "Phase 4 - Quality-Concat",
        "Phase 4 - Quality-Gated",
    }

    output = []

    for row in rows:
        if row.get("task") != "binary":
            continue

        fair_name = row.get("fair_model_name")

        if fair_name not in wanted:
            continue

        cm = derive_confusion_from_support_recall(
            fall_support=row.get("Fall_support"),
            fall_recall=row.get("Fall_recall"),
            not_fall_support=row.get("Not_Fall_support"),
            not_fall_recall=row.get("Not_Fall_recall"),
        )

        if fair_name == "Phase 4 - Quality-Concat":
            method_type = "2D + 3D + quality feature fusion"
        else:
            method_type = "2D + 3D + quality-gated fusion"

        output.append({
            "family": "Phase 4",
            "model": fair_name,
            "method_type": method_type,
            "aggregation": "Sample-level",
            "accuracy": safe_float(row.get("accuracy")),
            "macro_f1": safe_float(row.get("macro_f1")),
            "fall_recall": safe_float(row.get("Fall_recall")),
            "specificity": safe_float(row.get("Not_Fall_recall")),
            "fall_f1": safe_float(row.get("Fall_f1")),
            "not_fall_f1": safe_float(row.get("Not_Fall_f1")),
            "tn": cm["tn"],
            "fp": cm["fp"],
            "fn": cm["fn"],
            "tp": cm["tp"],
            "test_samples": row.get("num_test_samples"),
            "test_videos": row.get("num_test_videos"),
            "source": str(row.get("result_path", comparison_json)),
            "note": "Phase 4 binary result loaded from all_phases_fair_comparison.json.",
        })

    if len(output) != 2:
        raise ValueError(
            f"Expected 2 Phase 4 binary rows, found {len(output)}. Check file: {comparison_json}"
        )

    return output


def build_html(df: pd.DataFrame, output_html: Path) -> str:
    display = df.copy()

    metric_cols = [
        "accuracy",
        "macro_f1",
        "fall_recall",
        "specificity",
        "fall_f1",
        "not_fall_f1",
    ]

    for col in metric_cols:
        display[col] = display[col].apply(pct)

    for col in ["tn", "fp", "fn", "tp", "test_samples", "test_videos"]:
        display[col] = display[col].apply(int_or_na)

    display = display[
        [
            "rank",
            "family",
            "model",
            "method_type",
            "aggregation",
            "accuracy",
            "macro_f1",
            "fall_recall",
            "specificity",
            "fall_f1",
            "not_fall_f1",
            "tn",
            "fp",
            "fn",
            "tp",
            "test_samples",
            "test_videos",
        ]
    ]

    display.columns = [
        "Rank",
        "Family",
        "Model",
        "Method Type",
        "Aggregation",
        "Accuracy",
        "Macro F1",
        "Fall Recall",
        "Specificity",
        "Fall F1",
        "Not-Fall F1",
        "TN",
        "FP",
        "FN",
        "TP",
        "Test Samples",
        "Test Videos",
    ]

    table_html = display.to_html(index=False, escape=False, classes="result-table")

    best = df.iloc[0]

    style = """
<style>
body {
    margin: 0;
    font-family: Segoe UI, Arial, sans-serif;
    background: #f5f7fb;
    color: #172033;
}
.page {
    max-width: 1500px;
    margin: 0 auto;
    padding: 28px;
}
.hero {
    background: linear-gradient(135deg, #111827, #1e3a8a);
    color: white;
    padding: 30px;
    border-radius: 22px;
    margin-bottom: 22px;
}
.section {
    background: white;
    border-radius: 22px;
    padding: 24px;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
    margin-bottom: 22px;
}
.note {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 14px;
    padding: 14px;
    margin: 14px 0;
}
.good {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 14px;
    padding: 14px;
    margin: 14px 0;
}
.table-wrap {
    overflow-x: auto;
}
.result-table {
    border-collapse: collapse;
    width: 100%;
    min-width: 1250px;
    font-size: 14px;
}
.result-table th {
    background: #f8fafc;
    color: #475569;
    text-transform: uppercase;
    font-size: 12px;
    text-align: left;
    padding: 12px;
    border-bottom: 1px solid #e2e8f0;
}
.result-table td {
    padding: 12px;
    border-bottom: 1px solid #e2e8f0;
}
.result-table tr:nth-child(even) td {
    background: #fafafa;
}
.code {
    background: #0f172a;
    color: #e2e8f0;
    border-radius: 12px;
    padding: 14px;
    white-space: pre-wrap;
    font-family: Consolas, monospace;
}
</style>
"""

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Final Comparison Report</title>
{style}
</head>
<body>
<main class="page">
    <section class="hero">
        <h1>Final Ranking - Binary Fall Detection</h1>
        <p>Comparison of Phase 4 fusion models, Lin-style recurrent baseline, and Chen2020-style threshold baseline.</p>
    </section>

    <section class="section">
        <h2>Main Conclusion</h2>
        <div class="good">
            Best model: <strong>{best["model"]}</strong><br>
            Accuracy: <strong>{pct(best["accuracy"])}</strong><br>
            Macro F1: <strong>{pct(best["macro_f1"])}</strong>
        </div>
        <div class="note">
            Phase 4 metrics are sample-level. Chen2020-style and Lin-style metrics are video-level.
            Therefore, both Test Samples and Test Videos are reported to make the comparison transparent.
        </div>
    </section>

    <section class="section">
        <h2>Final Ranking</h2>
        <div class="table-wrap">
            {table_html}
        </div>
    </section>

    <section class="section">
        <h2>Interpretation</h2>
        <div class="code">Chen2020-style is the most interpretable but weakest baseline because it relies on fixed geometric rules.
Lin-style YOLOv8 + GRU is stronger because it learns temporal skeleton patterns.
Phase 4 is strongest because it uses 2D pose, 3D pose, and pose quality features.</div>
    </section>

    <section class="section">
        <h2>Output</h2>
        <div class="code">{output_html}</div>
    </section>
</main>
</body>
</html>
"""

    return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate final HTML comparison report with complete Phase 4 metrics."
    )

    parser.add_argument(
        "--chen-report",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "reports" / "07_evaluate_threshold_method_report.json"),
    )

    parser.add_argument(
        "--lin-report",
        type=str,
        default=str(PROJECT_ROOT / "paper_baselines" / "lin2021_yolov8_lstm_gru" / "outputs" / "reports" / "07_evaluate_rnn_lstm_gru_report.json"),
    )

    parser.add_argument(
        "--phase4-comparison-json",
        type=str,
        default=str(PROJECT_ROOT / "phase4_quality_aware_fusion" / "outputs" / "comparison" / "all_phases_fair_comparison.json"),
    )

    parser.add_argument(
        "--output-html",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "reports" / "08_chen2020_yolov8_final_report.html"),
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "reports" / "08_chen2020_yolov8_final_report.json"),
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "metrics" / "08_final_comparison_table.csv"),
    )

    args = parser.parse_args()

    chen_row = load_chen_result(Path(args.chen_report))
    lin_row = load_lin_result(Path(args.lin_report))
    phase4_rows = load_phase4_binary_rows(Path(args.phase4_comparison_json))

    rows = [
        *phase4_rows,
        lin_row,
        chen_row,
    ]

    df = pd.DataFrame(rows)
    df = df.sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    output_html = Path(args.output_html)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)

    ensure_dir(output_html.parent)
    ensure_dir(output_json.parent)
    ensure_dir(output_csv.parent)

    df.to_csv(output_csv, index=False)

    report = {
        "status": "completed",
        "task": "Binary Fall / Not_Fall",
        "chen_report": str(args.chen_report),
        "lin_report": str(args.lin_report),
        "phase4_comparison_json": str(args.phase4_comparison_json),
        "output_html": str(output_html),
        "output_json": str(output_json),
        "output_csv": str(output_csv),
        "best_overall": {
            "model": str(df.iloc[0]["model"]),
            "accuracy": float(df.iloc[0]["accuracy"]),
            "macro_f1": float(df.iloc[0]["macro_f1"]),
            "aggregation": str(df.iloc[0]["aggregation"]),
        },
        "comparison_table": df.to_dict(orient="records"),
        "note": "Phase 4 metrics are now loaded from all_phases_fair_comparison.json, including Fall Recall, Specificity, Fall F1, Not-Fall F1, and derived TN/FP/FN/TP.",
    }

    save_json(report, output_json)

    html = build_html(df=df, output_html=output_html)

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)

    print("=" * 100)
    print("FILE 08 completed with full Phase 4 metrics.")
    print("=" * 100)
    print(df[[
        "rank",
        "family",
        "model",
        "aggregation",
        "accuracy",
        "macro_f1",
        "fall_recall",
        "specificity",
        "fall_f1",
        "not_fall_f1",
        "tn",
        "fp",
        "fn",
        "tp",
        "test_samples",
        "test_videos",
    ]].to_string(index=False))
    print("-" * 100)
    print(f"CSV:  {output_csv}")
    print(f"JSON: {output_json}")
    print(f"HTML: {output_html}")
    print("=" * 100)


if __name__ == "__main__":
    main()

