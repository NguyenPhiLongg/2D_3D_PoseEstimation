from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
BASELINE_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = BASELINE_ROOT.parents[1]


# ======================================================================================
# FILE 08: COMPARE BEST PAPER-STYLE PIPELINE WITH PHASE 4 BINARY MODELS
# ======================================================================================
#
# This file compares:
#   1. Best Lin-style YOLOv8 + RNN/LSTM/GRU model from File 07
#   2. Phase 4 - Quality-Concat binary model
#   3. Phase 4 - Quality-Gated binary model
#
# Task:
#   Binary Fall / Not_Fall classification only.
#
# Important:
#   The Lin-style pipeline is adapted from the paper-style skeleton RNN/LSTM/GRU pipeline,
#   but uses YOLOv8-Pose instead of OpenPose because OpenPose failed to detect skeletons
#   on this dataset.
#
# Output:
#   outputs/reports/08_compare_paper_pipeline_with_phase4_binary.html
#   outputs/reports/08_compare_paper_pipeline_with_phase4_binary.json
#   outputs/metrics/08_compare_paper_pipeline_with_phase4_binary.csv
# ======================================================================================


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


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def pp(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x * 100:.2f} pp"


def safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ======================================================================================
# LOAD LIN-STYLE PAPER PIPELINE RESULTS
# ======================================================================================

def load_lin_style_results(report_path: Path) -> Dict:
    report = read_json(report_path)

    results = report.get("results", [])

    if len(results) == 0:
        raise ValueError(f"No model results found in {report_path}")

    # Primary ranking from File 07:
    # best by sequence Macro F1.
    best_info = report.get("best_model_by_sequence_macro_f1", {})
    best_model_type = best_info.get("model_type", None)

    if best_model_type is None:
        results_sorted = sorted(
            results,
            key=lambda r: (
                safe_float(r.get("sequence_macro_f1")),
                safe_float(r.get("sequence_fall_recall")),
            ),
            reverse=True,
        )
        best = results_sorted[0]
    else:
        matched = [
            r for r in results
            if str(r.get("model_type")).lower() == str(best_model_type).lower()
        ]

        if len(matched) == 0:
            raise ValueError(f"Best model {best_model_type} not found in results.")
        best = matched[0]

    row = {
        "family": "Paper-style baseline",
        "model": f"Lin-style YOLOv8 + {str(best['model_type']).upper()}",
        "input": "YOLOv8 2D skeleton sequence, 15 joints x 2 coordinates",
        "aggregation": "Sequence-level",
        "accuracy": safe_float(best.get("sequence_accuracy")),
        "macro_f1": safe_float(best.get("sequence_macro_f1")),
        "fall_recall": safe_float(best.get("sequence_fall_recall")),
        "fall_f1": safe_float(best.get("sequence_fall_f1")),
        "not_fall_f1": safe_float(best.get("sequence_not_fall_f1")),
        "tn": int(best.get("sequence_tn", 0)),
        "fp": int(best.get("sequence_fp", 0)),
        "fn": int(best.get("sequence_fn", 0)),
        "tp": int(best.get("sequence_tp", 0)),
        "test_samples": int(best.get("num_test_sequences", report.get("test_shape", [0])[0])),
        "test_videos": int(best.get("num_test_videos", 0)),
        "source": str(report_path),
        "note": "Best RNN/LSTM/GRU model from adapted Lin-style pipeline.",
    }

    video_row = {
        "family": "Paper-style baseline",
        "model": f"Lin-style YOLOv8 + {str(best['model_type']).upper()}",
        "input": "YOLOv8 2D skeleton sequence, 15 joints x 2 coordinates",
        "aggregation": "Video-level mean probability",
        "accuracy": safe_float(best.get("video_mean_prob_accuracy")),
        "macro_f1": safe_float(best.get("video_mean_prob_macro_f1")),
        "fall_recall": safe_float(best.get("video_mean_prob_fall_recall")),
        "fall_f1": safe_float(best.get("video_mean_prob_fall_f1")),
        "not_fall_f1": safe_float(best.get("video_mean_prob_not_fall_f1")),
        "tn": int(best.get("video_mean_prob_tn", 0)),
        "fp": int(best.get("video_mean_prob_fp", 0)),
        "fn": int(best.get("video_mean_prob_fn", 0)),
        "tp": int(best.get("video_mean_prob_tp", 0)),
        "test_samples": int(best.get("num_test_videos", 0)),
        "test_videos": int(best.get("num_test_videos", 0)),
        "source": str(report_path),
        "note": "Same Lin-style model aggregated from sequences to video prediction.",
    }

    all_lin_rows = []

    for r in results:
        all_lin_rows.append({
            "model_type": str(r.get("model_type")).upper(),
            "sequence_accuracy": safe_float(r.get("sequence_accuracy")),
            "sequence_macro_f1": safe_float(r.get("sequence_macro_f1")),
            "sequence_fall_recall": safe_float(r.get("sequence_fall_recall")),
            "sequence_fall_f1": safe_float(r.get("sequence_fall_f1")),
            "video_mean_prob_accuracy": safe_float(r.get("video_mean_prob_accuracy")),
            "video_mean_prob_macro_f1": safe_float(r.get("video_mean_prob_macro_f1")),
            "video_mean_prob_fall_recall": safe_float(r.get("video_mean_prob_fall_recall")),
            "video_mean_prob_fall_f1": safe_float(r.get("video_mean_prob_fall_f1")),
        })

    return {
        "sequence_row": row,
        "video_row": video_row,
        "all_lin_rows": all_lin_rows,
    }


# ======================================================================================
# PHASE 4 RESULTS
# ======================================================================================

def phase4_default_rows() -> List[Dict]:
    """
    Clean binary Phase 4 results taken from the existing Phase 4 dashboard.

    These are binary Fall / Not_Fall results.
    """
    return [
        {
            "family": "Phase 4",
            "model": "Phase 4 - Quality-Concat",
            "input": "2D pose + estimated 3D pose + quality features",
            "aggregation": "Sample-level",
            "accuracy": 0.9616,
            "macro_f1": 0.9546,
            "fall_recall": None,
            "fall_f1": None,
            "not_fall_f1": None,
            "tn": None,
            "fp": None,
            "fn": None,
            "tp": None,
            "test_samples": 2965,
            "test_videos": 812,
            "source": "phase4_dashboard.html",
            "note": "Best clean binary model in Phase 4 dashboard.",
        },
        {
            "family": "Phase 4",
            "model": "Phase 4 - Quality-Gated",
            "input": "2D pose + estimated 3D pose + quality features",
            "aggregation": "Sample-level",
            "accuracy": 0.9535,
            "macro_f1": 0.9450,
            "fall_recall": None,
            "fall_f1": None,
            "not_fall_f1": None,
            "tn": None,
            "fp": None,
            "fn": None,
            "tp": None,
            "test_samples": 2965,
            "test_videos": 812,
            "source": "phase4_dashboard.html",
            "note": "Second-best clean binary model in Phase 4 dashboard.",
        },
    ]


# ======================================================================================
# HTML REPORT
# ======================================================================================

def build_bar(width: float, color: str = "#2563eb") -> str:
    width_pct = max(0.0, min(100.0, width * 100.0))
    return f"""
    <div class="bar-track">
        <div class="bar-fill" style="width:{width_pct:.2f}%; background:{color};"></div>
    </div>
    """


def make_comparison_table(df: pd.DataFrame) -> str:
    rows_html = []

    best_macro = df["macro_f1"].max()

    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        is_best = abs(float(row["macro_f1"]) - float(best_macro)) < 1e-12
        tr_class = "best-row" if is_best else ""

        fall_recall_text = "N/A" if pd.isna(row["fall_recall"]) else pct(float(row["fall_recall"]))
        fall_f1_text = "N/A" if pd.isna(row["fall_f1"]) else pct(float(row["fall_f1"]))

        rows_html.append(f"""
        <tr class="{tr_class}">
            <td class="rank">{rank}</td>
            <td><span class="badge">{row['family']}</span></td>
            <td class="model-cell">{row['model']}</td>
            <td>{row['aggregation']}</td>
            <td>{row['input']}</td>
            <td class="num">{pct(float(row['accuracy']))}</td>
            <td class="num strong">{pct(float(row['macro_f1']))}</td>
            <td class="num">{fall_recall_text}</td>
            <td class="num">{fall_f1_text}</td>
            <td class="num">{int(row['test_samples'])}</td>
            <td class="num">{int(row['test_videos'])}</td>
        </tr>
        """)

    return f"""
    <div class="table-wrap">
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Family</th>
                    <th>Model</th>
                    <th>Aggregation</th>
                    <th>Input</th>
                    <th>Accuracy</th>
                    <th>Macro F1</th>
                    <th>Fall Recall</th>
                    <th>Fall F1</th>
                    <th>Test Samples</th>
                    <th>Test Videos</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
    """


def make_delta_table(df: pd.DataFrame, baseline_model_name: str) -> str:
    baseline_row = df[df["model"] == baseline_model_name]

    if len(baseline_row) == 0:
        return "<p>No baseline row found for delta table.</p>"

    baseline = baseline_row.iloc[0]

    rows_html = []

    for _, row in df.iterrows():
        delta_acc = float(row["accuracy"]) - float(baseline["accuracy"])
        delta_f1 = float(row["macro_f1"]) - float(baseline["macro_f1"])

        rows_html.append(f"""
        <tr>
            <td class="model-cell">{row['model']}</td>
            <td class="num">{pct(float(row['accuracy']))}</td>
            <td class="num strong">{pct(float(row['macro_f1']))}</td>
            <td class="num">{pp(delta_acc)}</td>
            <td class="num">{pp(delta_f1)}</td>
        </tr>
        """)

    return f"""
    <div class="table-wrap compact">
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Macro F1</th>
                    <th>Δ Accuracy vs Paper Pipeline</th>
                    <th>Δ Macro F1 vs Paper Pipeline</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
    """


def make_lin_model_table(all_lin_rows: List[Dict]) -> str:
    df = pd.DataFrame(all_lin_rows)
    df = df.sort_values("sequence_macro_f1", ascending=False).reset_index(drop=True)

    rows = []

    for rank, (_, r) in enumerate(df.iterrows(), start=1):
        rows.append(f"""
        <tr>
            <td class="rank">{rank}</td>
            <td class="model-cell">{r['model_type']}</td>
            <td class="num">{pct(float(r['sequence_accuracy']))}</td>
            <td class="num strong">{pct(float(r['sequence_macro_f1']))}</td>
            <td class="num">{pct(float(r['sequence_fall_recall']))}</td>
            <td class="num">{pct(float(r['sequence_fall_f1']))}</td>
            <td class="num">{pct(float(r['video_mean_prob_accuracy']))}</td>
            <td class="num">{pct(float(r['video_mean_prob_macro_f1']))}</td>
        </tr>
        """)

    return f"""
    <div class="table-wrap compact">
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Paper-style Model</th>
                    <th>Seq Acc</th>
                    <th>Seq Macro F1</th>
                    <th>Seq Fall Recall</th>
                    <th>Seq Fall F1</th>
                    <th>Video Acc</th>
                    <th>Video Macro F1</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """


def build_html_report(
    comparison_df: pd.DataFrame,
    all_lin_rows: List[Dict],
    html_path: Path,
) -> str:
    best_row = comparison_df.sort_values("macro_f1", ascending=False).iloc[0]
    paper_row = comparison_df[comparison_df["family"] == "Paper-style baseline"].iloc[0]

    phase4_concat = comparison_df[comparison_df["model"] == "Phase 4 - Quality-Concat"].iloc[0]
    phase4_gated = comparison_df[comparison_df["model"] == "Phase 4 - Quality-Gated"].iloc[0]

    max_macro = comparison_df["macro_f1"].max()

    cards = f"""
    <section class="cards">
        <div class="card green">
            <div class="card-title">Best Overall Binary Model</div>
            <div class="card-value">{best_row['model']}</div>
            <div class="card-sub">Macro F1 {pct(float(best_row['macro_f1']))} | Accuracy {pct(float(best_row['accuracy']))}</div>
        </div>
        <div class="card blue">
            <div class="card-title">Best Paper-style Pipeline</div>
            <div class="card-value">{paper_row['model']}</div>
            <div class="card-sub">Macro F1 {pct(float(paper_row['macro_f1']))} | Fall Recall {pct(float(paper_row['fall_recall']))}</div>
        </div>
        <div class="card purple">
            <div class="card-title">Phase 4 Quality-Concat</div>
            <div class="card-value">{pct(float(phase4_concat['macro_f1']))}</div>
            <div class="card-sub">Accuracy {pct(float(phase4_concat['accuracy']))}</div>
        </div>
        <div class="card orange">
            <div class="card-title">Phase 4 Quality-Gated</div>
            <div class="card-value">{pct(float(phase4_gated['macro_f1']))}</div>
            <div class="card-sub">Accuracy {pct(float(phase4_gated['accuracy']))}</div>
        </div>
    </section>
    """

    bars = []

    for _, row in comparison_df.sort_values("macro_f1", ascending=False).iterrows():
        ratio = float(row["macro_f1"]) / max_macro if max_macro > 0 else 0.0
        color = "#16a34a" if "Quality-Concat" in row["model"] else "#7c3aed" if "Quality-Gated" in row["model"] else "#2563eb"

        bars.append(f"""
        <div class="bar-row">
            <div class="bar-label">
                <strong>{row['model']}</strong>
                <span>{row['aggregation']}</span>
            </div>
            {build_bar(ratio, color)}
            <div class="bar-value">{pct(float(row['macro_f1']))}</div>
        </div>
        """)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Paper Pipeline vs Phase 4 Binary Comparison</title>
<style>
:root {{
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
    --shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
}}
* {{ box-sizing: border-box; }}
body {{
    margin: 0;
    font-family: Inter, Segoe UI, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
}}
.page {{
    max-width: 1320px;
    margin: 0 auto;
    padding: 28px;
}}
.hero {{
    background: linear-gradient(135deg, #172033, #263b72);
    color: white;
    padding: 32px;
    border-radius: 24px;
    box-shadow: var(--shadow);
    margin-bottom: 24px;
}}
.hero h1 {{ margin: 0 0 8px 0; font-size: 32px; }}
.hero p {{ color: #dbeafe; margin: 6px 0; }}
.cards {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}}
.card {{
    background: var(--card);
    border-radius: 18px;
    padding: 20px;
    box-shadow: var(--shadow);
    border-top: 5px solid var(--blue);
}}
.card.green {{ border-top-color: var(--green); }}
.card.blue {{ border-top-color: var(--blue); }}
.card.purple {{ border-top-color: var(--purple); }}
.card.orange {{ border-top-color: var(--orange); }}
.card-title {{
    color: var(--muted);
    font-size: 13px;
    text-transform: uppercase;
    font-weight: 800;
}}
.card-value {{
    margin-top: 8px;
    font-size: 21px;
    font-weight: 900;
}}
.card-sub {{
    margin-top: 6px;
    color: var(--muted);
    font-size: 13px;
}}
.section {{
    background: var(--card);
    border-radius: 24px;
    padding: 26px;
    box-shadow: var(--shadow);
    margin-bottom: 24px;
}}
.section h2 {{ margin-top: 0; }}
.note {{
    border: 1px solid #fde68a;
    background: #fffbeb;
    padding: 16px;
    border-radius: 16px;
    margin: 16px 0;
}}
.good {{
    border: 1px solid #bbf7d0;
    background: #f0fdf4;
    padding: 16px;
    border-radius: 16px;
    margin: 16px 0;
}}
.table-wrap {{
    overflow-x: auto;
    border: 1px solid var(--line);
    border-radius: 16px;
    margin-top: 14px;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    background: white;
    min-width: 1080px;
}}
.compact table {{ min-width: 850px; }}
th {{
    text-align: left;
    background: #f8fafc;
    color: #475569;
    font-size: 12px;
    text-transform: uppercase;
    padding: 12px;
    border-bottom: 1px solid var(--line);
}}
td {{
    padding: 12px;
    border-bottom: 1px solid var(--line);
    font-size: 14px;
}}
tr:last-child td {{ border-bottom: none; }}
.best-row {{ background: #fff7ed; }}
.rank {{ font-weight: 900; }}
.model-cell {{ font-weight: 800; }}
.num {{
    text-align: right;
    font-variant-numeric: tabular-nums;
}}
.strong {{ font-weight: 900; }}
.badge {{
    display: inline-block;
    background: #334155;
    color: white;
    padding: 5px 9px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 800;
}}
.bar-row {{
    display: grid;
    grid-template-columns: 320px 1fr 90px;
    gap: 14px;
    align-items: center;
    margin-bottom: 14px;
}}
.bar-label {{
    display: flex;
    flex-direction: column;
    gap: 2px;
    font-size: 14px;
}}
.bar-label span {{
    color: var(--muted);
    font-size: 12px;
}}
.bar-track {{
    height: 13px;
    background: #e5e7eb;
    border-radius: 999px;
    overflow: hidden;
}}
.bar-fill {{
    height: 100%;
    border-radius: 999px;
}}
.bar-value {{
    text-align: right;
    font-weight: 900;
}}
.footer {{
    text-align: center;
    color: var(--muted);
    font-size: 13px;
    padding: 18px;
}}
@media (max-width: 1000px) {{
    .cards {{ grid-template-columns: 1fr; }}
    .bar-row {{ grid-template-columns: 1fr; }}
    .bar-value {{ text-align: left; }}
}}
</style>
</head>
<body>
<main class="page">
    <section class="hero">
        <h1>Paper-style Pipeline vs Phase 4 Binary Fall Detection</h1>
        <p>Task: binary Fall / Not_Fall comparison using existing results only.</p>
        <p>Compared models: Lin-style YOLOv8 recurrent baseline, Phase 4 Quality-Concat, and Phase 4 Quality-Gated.</p>
    </section>

    {cards}

    <section class="section">
        <h2>Main Result</h2>
        <div class="good">
            <strong>Conclusion:</strong>
            Phase 4 Quality-Concat remains the best clean binary model by Macro F1.
            The best paper-style recurrent baseline is {paper_row['model']}, but it is below Phase 4 Quality-Concat on Macro F1.
        </div>
        <div class="note">
            <strong>Fairness note:</strong>
            These are existing results from two pipelines. The task is the same binary Fall / Not_Fall task,
            but sample counts are not exactly identical. Phase 4 clean binary uses 2965 test samples and 812 videos,
            while the Lin-style YOLOv8 test split uses 2032 sequences and 823 videos.
        </div>
        {make_comparison_table(comparison_df)}
    </section>

    <section class="section">
        <h2>Macro F1 Ranking</h2>
        {''.join(bars)}
    </section>

    <section class="section">
        <h2>Delta Compared with Paper-style Pipeline</h2>
        <p>This table uses the best sequence-level paper-style model as the comparison baseline.</p>
        {make_delta_table(comparison_df, baseline_model_name=str(paper_row['model']))}
    </section>

    <section class="section">
        <h2>Internal Paper-style Model Ranking</h2>
        <p>Among RNN, LSTM and GRU, GRU is the best in the adapted Lin-style YOLOv8 pipeline.</p>
        {make_lin_model_table(all_lin_rows)}
    </section>

    <section class="section">
        <h2>Recommended Wording for Report</h2>
        <div class="note">
            The original Lin et al. pipeline is adapted in this project because OpenPose failed to extract valid
            skeletons on the dataset. Therefore, YOLOv8-Pose keypoints are used while keeping the skeleton-sequence
            preprocessing and RNN/LSTM/GRU temporal classification procedure. Under this adapted setting, GRU is the
            best paper-style recurrent baseline. However, the strongest binary Fall/Not_Fall model among the available
            results remains Phase 4 Quality-Concat.
        </div>
    </section>

    <div class="footer">
        Output file: {html_path}
    </div>
</main>
</body>
</html>
"""
    return html


def main():
    parser = argparse.ArgumentParser(
        description="Compare best Lin-style YOLOv8 recurrent baseline with Phase 4 Quality-Concat and Quality-Gated binary results."
    )

    parser.add_argument(
        "--lin-report",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "reports" / "07_evaluate_rnn_lstm_gru_report.json"),
        help="File 07 evaluation report.",
    )

    parser.add_argument(
        "--output-html",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "reports" / "08_compare_paper_pipeline_with_phase4_binary.html"),
        help="Output HTML comparison report.",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "reports" / "08_compare_paper_pipeline_with_phase4_binary.json"),
        help="Output JSON comparison report.",
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "metrics" / "08_compare_paper_pipeline_with_phase4_binary.csv"),
        help="Output CSV comparison table.",
    )

    args = parser.parse_args()

    lin_report_path = Path(args.lin_report)
    output_html = Path(args.output_html)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)

    ensure_dir(output_html.parent)
    ensure_dir(output_json.parent)
    ensure_dir(output_csv.parent)

    lin_results = load_lin_style_results(lin_report_path)

    rows = [
        lin_results["sequence_row"],
        *phase4_default_rows(),
    ]

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values(
        ["macro_f1", "accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    comparison_df.to_csv(output_csv, index=False)

    best = comparison_df.iloc[0].to_dict()
    paper = comparison_df[comparison_df["family"] == "Paper-style baseline"].iloc[0].to_dict()

    report = {
        "status": "completed",
        "task": "Binary Fall / Not_Fall",
        "comparison_scope": "Existing clean binary results only",
        "lin_report": str(lin_report_path),
        "output_html": str(output_html),
        "output_json": str(output_json),
        "output_csv": str(output_csv),
        "best_overall": {
            "model": str(best["model"]),
            "accuracy": float(best["accuracy"]),
            "macro_f1": float(best["macro_f1"]),
            "test_samples": int(best["test_samples"]),
            "test_videos": int(best["test_videos"]),
        },
        "best_paper_style_pipeline": {
            "model": str(paper["model"]),
            "accuracy": float(paper["accuracy"]),
            "macro_f1": float(paper["macro_f1"]),
            "fall_recall": float(paper["fall_recall"]),
            "fall_f1": float(paper["fall_f1"]),
            "test_samples": int(paper["test_samples"]),
            "test_videos": int(paper["test_videos"]),
        },
        "fairness_note": "The task is the same binary Fall/Not_Fall task, but the existing test sample counts differ between Phase 4 and the Lin-style YOLOv8 baseline.",
        "table": comparison_df.to_dict(orient="records"),
        "all_lin_style_models": lin_results["all_lin_rows"],
    }

    save_json(report, output_json)

    html = build_html_report(
        comparison_df=comparison_df,
        all_lin_rows=lin_results["all_lin_rows"],
        html_path=output_html,
    )

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)

    print("=" * 100)
    print("FILE 08 completed.")
    print("=" * 100)
    print("Binary Fall / Not_Fall comparison:")
    print(comparison_df[[
        "family",
        "model",
        "aggregation",
        "accuracy",
        "macro_f1",
        "fall_recall",
        "fall_f1",
        "test_samples",
        "test_videos",
    ]].to_string(index=False))
    print("-" * 100)
    print("Best overall:")
    print(report["best_overall"])
    print("-" * 100)
    print("Best paper-style pipeline:")
    print(report["best_paper_style_pipeline"])
    print("-" * 100)
    print(f"CSV:  {output_csv}")
    print(f"JSON: {output_json}")
    print(f"HTML: {output_html}")
    print("=" * 100)


if __name__ == "__main__":
    main()
