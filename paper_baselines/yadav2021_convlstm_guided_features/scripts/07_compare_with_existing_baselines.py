from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
BASELINE_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = BASELINE_ROOT.parents[1]

sys.path.insert(0, str(BASELINE_ROOT))

from utils.io_utils import ensure_dir, save_json


# ======================================================================================
# FILE 07: FINAL BASELINE COMPARISON REPORT
# ======================================================================================
#
# Required report structure:
#
#   MAIN TABLE:
#       Compare only binary Fall/Not_Fall results:
#       - Phase 4 Quality-Concat
#       - Phase 4 Quality-Gated
#       - Best Chen2020-style paper baseline
#       - Best Lin2021-style paper baseline
#       - Best Yadav2021-style paper baseline
#
#   INTERNAL TABLES:
#       1. Chen2020-style model table
#       2. Lin2021-style model table
#       3. Yadav2021-style model table
#
# Important:
#   Phase 4 JSON may contain extra non-binary/action rows.
#   This script keeps only rows with Fall/Not_Fall metrics.
# ======================================================================================


def to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None

        if isinstance(v, str):
            v = v.strip()
            if v == "" or v.lower() in ["nan", "none", "null", "n/a"]:
                return None
            v = v.replace("%", "")

        if pd.isna(v):
            return None

        x = float(v)

        if not np.isfinite(x):
            return None

        if x > 1.0 and x <= 100.0:
            x = x / 100.0

        return x

    except Exception:
        return None


def to_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None

        if isinstance(v, str):
            v = v.strip()
            if v == "" or v.lower() in ["nan", "none", "null", "n/a"]:
                return None

        if pd.isna(v):
            return None

        return int(round(float(v)))

    except Exception:
        return None


def fmt_pct(v: Any) -> str:
    x = to_float(v)
    return "N/A" if x is None else f"{x * 100:.2f}%"


def fmt_int(v: Any) -> str:
    x = to_int(v)
    return "N/A" if x is None else str(x)


def fmt_threshold(v: Any) -> str:
    x = to_float(v)
    return "N/A" if x is None else f"{x:.2f}"


def get_any(d: Dict[str, Any], keys: List[str]) -> Any:
    if d is None:
        return None

    for k in keys:
        if k in d:
            return d[k]

    lower = {str(k).lower(): k for k in d.keys()}

    for k in keys:
        real = lower.get(k.lower())
        if real is not None:
            return d[real]

    return None


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_dicts(obj: Any):
    if isinstance(obj, dict):
        yield obj

        for v in obj.values():
            yield from iter_dicts(v)

    elif isinstance(obj, list):
        for item in obj:
            yield from iter_dicts(item)


def flatten_dict(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}.{k}" if prefix else str(k)

            if isinstance(v, dict):
                out.update(flatten_dict(v, new_key))
            else:
                out[new_key] = v

    return out


def metric_row(
    family: str,
    model_name: str,
    method_type: str,
    split: str,
    level: str,
    aggregation: str,
    threshold: Any,
    accuracy: Any,
    macro_f1: Any,
    fall_recall: Any,
    fall_f1: Any,
    specificity: Any,
    tn: Any,
    fp: Any,
    fn: Any,
    tp: Any,
    num_samples: Any,
    num_videos: Any,
    source_path: Any,
    notes: str,
) -> Dict[str, Any]:
    return {
        "family": family,
        "model_name": model_name,
        "method_type": method_type,
        "split": split,
        "level": level,
        "aggregation": aggregation,
        "threshold": to_float(threshold),

        "accuracy": to_float(accuracy),
        "macro_f1": to_float(macro_f1),
        "fall_recall": to_float(fall_recall),
        "fall_f1": to_float(fall_f1),
        "specificity": to_float(specificity),

        "tn": to_int(tn),
        "fp": to_int(fp),
        "fn": to_int(fn),
        "tp": to_int(tp),

        "num_samples": to_int(num_samples),
        "num_videos": to_int(num_videos),

        "source_path": str(source_path),
        "notes": notes,
    }



def add_sample_counts_from_cm(row: Dict[str, Any]) -> Dict[str, Any]:
    tn = row.get("tn")
    fp = row.get("fp")
    fn = row.get("fn")
    tp = row.get("tp")

    vals = [tn, fp, fn, tp]

    if all(v is not None for v in vals):
        total = int(tn + fp + fn + tp)

        if row.get("num_samples") is None:
            row["num_samples"] = total

        if row.get("level") == "video" and row.get("num_videos") is None:
            row["num_videos"] = total

        if row.get("accuracy") is None:
            row["accuracy"] = (tn + tp) / max(total, 1)

        fall_precision = tp / max(tp + fp, 1)
        fall_recall = tp / max(tp + fn, 1)

        not_fall_precision = tn / max(tn + fn, 1)
        not_fall_recall = tn / max(tn + fp, 1)

        fall_f1 = 2 * fall_precision * fall_recall / max(fall_precision + fall_recall, 1e-12)
        not_fall_f1 = 2 * not_fall_precision * not_fall_recall / max(not_fall_precision + not_fall_recall, 1e-12)

        if row.get("fall_recall") is None:
            row["fall_recall"] = fall_recall

        if row.get("specificity") is None:
            row["specificity"] = not_fall_recall

        if row.get("fall_f1") is None:
            row["fall_f1"] = fall_f1

        if row.get("macro_f1") is None:
            row["macro_f1"] = (fall_f1 + not_fall_f1) / 2.0

    return row



def best_row(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    return df.sort_values(
        ["macro_f1", "accuracy", "fall_recall"],
        ascending=[False, False, False],
    ).head(1).copy()


# ======================================================================================
# CHEN
# ======================================================================================

def load_chen_all(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    rows = []

    for _, r in df.iterrows():
        split = str(get_any(r.to_dict(), ["split", "dataset", "subset"]) or "test")

        row = metric_row(
            family="Chen2020-style",
            model_name="YOLOv8 three-threshold rule",
            method_type="rule_based_geometric_threshold",
            split=split,
            level="video",
            aggregation="threshold_rule",
            threshold=get_any(r.to_dict(), ["velocity_threshold", "threshold"]),
            accuracy=get_any(r.to_dict(), ["accuracy", "acc"]),
            macro_f1=get_any(r.to_dict(), ["macro_f1", "f1_macro"]),
            fall_recall=get_any(r.to_dict(), ["fall_recall", "Fall_recall", "sensitivity"]),
            fall_f1=get_any(r.to_dict(), ["fall_f1", "Fall_f1"]),
            specificity=get_any(r.to_dict(), ["specificity", "not_fall_recall", "Not_Fall_recall"]),
            tn=get_any(r.to_dict(), ["tn", "TN"]),
            fp=get_any(r.to_dict(), ["fp", "FP"]),
            fn=get_any(r.to_dict(), ["fn", "FN"]),
            tp=get_any(r.to_dict(), ["tp", "TP"]),
            num_samples=get_any(r.to_dict(), ["num_samples", "num_videos"]),
            num_videos=get_any(r.to_dict(), ["num_videos", "test_videos"]),
            source_path=path,
            notes="Adapted Chen2020 rule-based baseline using YOLOv8-Pose skeleton instead of OpenPose.",
        )

        rows.append(add_sample_counts_from_cm(row))

    return pd.DataFrame(rows)


# ======================================================================================
# YADAV
# ======================================================================================

def load_yadav_all(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == "test"].copy()

    rows = []

    for _, r in df.iterrows():
        level = str(r.get("level", "sequence"))
        aggregation = str(r.get("aggregation", "none"))

        row = metric_row(
            family="Yadav2021-style",
            model_name=str(r.get("model_name", "unknown")).upper(),
            method_type="guided_features_deep_learning",
            split="test",
            level=level,
            aggregation=aggregation,
            threshold=r.get("threshold"),
            accuracy=r.get("accuracy"),
            macro_f1=r.get("macro_f1"),
            fall_recall=r.get("fall_recall"),
            fall_f1=r.get("fall_f1"),
            specificity=r.get("specificity_not_fall_recall"),
            tn=r.get("tn"),
            fp=r.get("fp"),
            fn=r.get("fn"),
            tp=r.get("tp"),
            num_samples=r.get("num_sequences"),
            num_videos=r.get("num_videos"),
            source_path=path,
            notes="Adapted Yadav2021 baseline using 3D skeleton coordinates plus 14 guided features.",
        )

        rows.append(add_sample_counts_from_cm(row))

    return pd.DataFrame(rows)


def best_yadav_row(yadav_df: pd.DataFrame) -> pd.DataFrame:
    if len(yadav_df) == 0:
        return pd.DataFrame()

    video_df = yadav_df[yadav_df["level"].astype(str).str.lower() == "video"].copy()

    if len(video_df) == 0:
        return best_row(yadav_df)

    return best_row(video_df)


# ======================================================================================
# LIN
# ======================================================================================

def extract_metric(flat: Dict[str, Any], metric: str, prefer_video: bool = True) -> Any:
    keys = list(flat.keys())
    low_map = {k.lower(): k for k in keys}

    metric_low = metric.lower()

    candidates = []

    if prefer_video:
        for k in keys:
            kl = k.lower()
            if metric_low in kl and ("video_mean_prob" in kl or "mean_prob" in kl):
                candidates.append(k)

        for k in keys:
            kl = k.lower()
            if metric_low in kl and "video" in kl:
                candidates.append(k)

    for k in keys:
        kl = k.lower()
        if metric_low in kl:
            candidates.append(k)

    seen = set()

    for k in candidates:
        if k in seen:
            continue

        seen.add(k)
        v = flat.get(k)

        if to_float(v) is not None or to_int(v) is not None:
            return v

    return None


def find_model_entries(obj: Any, path: str = "") -> List[Tuple[str, Dict[str, Any]]]:
    out = []

    if isinstance(obj, dict):
        # Case 1: key itself is model name
        for k, v in obj.items():
            kl = str(k).lower()

            if kl in ["rnn", "lstm", "gru"] and isinstance(v, dict):
                out.append((kl, v))

            out.extend(find_model_entries(v, f"{path}.{k}" if path else str(k)))

        # Case 2: dict contains model name
        model = get_any(obj, ["model_name", "model", "algorithm", "name"])

        if model is not None:
            ml = str(model).lower()

            if ml in ["rnn", "lstm", "gru"]:
                out.append((ml, obj))

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            out.extend(find_model_entries(item, f"{path}[{i}]"))

    return out


def load_lin_from_json(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    data = read_json(path)
    entries = find_model_entries(data)

    rows = []
    seen = set()

    for model_name, d in entries:
        flat = flatten_dict(d)

        accuracy = extract_metric(flat, "accuracy")
        macro_f1 = extract_metric(flat, "macro_f1")

        if accuracy is None and macro_f1 is None:
            continue

        row = metric_row(
            family="Lin2021-style",
            model_name=model_name.upper(),
            method_type="rnn_lstm_gru_temporal_skeleton",
            split="test",
            level="video",
            aggregation="mean_prob",
            threshold=extract_metric(flat, "threshold"),
            accuracy=accuracy,
            macro_f1=macro_f1,
            fall_recall=extract_metric(flat, "fall_recall"),
            fall_f1=extract_metric(flat, "fall_f1"),
            specificity=extract_metric(flat, "specificity") or extract_metric(flat, "not_fall_recall"),
            tn=extract_metric(flat, "tn"),
            fp=extract_metric(flat, "fp"),
            fn=extract_metric(flat, "fn"),
            tp=extract_metric(flat, "tp"),
            num_samples=extract_metric(flat, "num_test_videos") or extract_metric(flat, "num_videos"),
            num_videos=extract_metric(flat, "num_test_videos") or extract_metric(flat, "num_videos"),
            source_path=path,
            notes="Adapted Lin2021 temporal skeleton baseline using YOLOv8-Pose instead of OpenPose.",
        )

        row = add_sample_counts_from_cm(row)

        key = (
            row["model_name"],
            row["accuracy"],
            row["macro_f1"],
            row["tn"],
            row["fp"],
            row["fn"],
            row["tp"],
        )

        if key not in seen:
            seen.add(key)
            rows.append(row)

    return pd.DataFrame(rows)



def load_lin_from_csvs(lin_root: Path) -> pd.DataFrame:
    rows = []

    if not lin_root.exists():
        return pd.DataFrame()

    preferred_csv = lin_root / "outputs" / "metrics" / "lin2021_yolov8_test_evaluation_summary.csv"

    csv_paths = []

    if preferred_csv.exists():
        csv_paths.append(preferred_csv)
    else:
        csv_paths = sorted((lin_root / "outputs").rglob("*.csv")) if (lin_root / "outputs").exists() else []

    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        cols_lower = {c.lower(): c for c in df.columns}

        model_col = (
            cols_lower.get("model_type")
            or cols_lower.get("model_name")
            or cols_lower.get("model")
            or cols_lower.get("algorithm")
        )

        if model_col is None:
            continue

        if "split" in cols_lower:
            split_col = cols_lower["split"]
            df = df[df[split_col].astype(str).str.lower() == "test"].copy()

        for _, r in df.iterrows():
            model_name = str(r.get(model_col, "")).lower()

            if model_name not in ["rnn", "lstm", "gru"]:
                continue

            d = r.to_dict()

            aggregation_specs = [
                {
                    "level": "video",
                    "aggregation": "majority_vote",
                    "prefix": "video_majority",
                },
                {
                    "level": "video",
                    "aggregation": "mean_prob",
                    "prefix": "video_mean_prob",
                },
                {
                    "level": "sequence",
                    "aggregation": "none",
                    "prefix": "sequence",
                },
            ]

            for spec in aggregation_specs:
                prefix = spec["prefix"]

                accuracy = get_any(d, [f"{prefix}_accuracy", f"{prefix}_acc"])
                macro_f1 = get_any(d, [f"{prefix}_macro_f1"])

                if accuracy is None and macro_f1 is None:
                    continue

                row = metric_row(
                    family="Lin2021-style",
                    model_name=model_name.upper(),
                    method_type="rnn_lstm_gru_temporal_skeleton",
                    split="test",
                    level=spec["level"],
                    aggregation=spec["aggregation"],
                    threshold=None,
                    accuracy=accuracy,
                    macro_f1=macro_f1,
                    fall_recall=get_any(d, [f"{prefix}_fall_recall"]),
                    fall_f1=get_any(d, [f"{prefix}_fall_f1"]),
                    specificity=None,
                    tn=get_any(d, [f"{prefix}_tn"]),
                    fp=get_any(d, [f"{prefix}_fp"]),
                    fn=get_any(d, [f"{prefix}_fn"]),
                    tp=get_any(d, [f"{prefix}_tp"]),
                    num_samples=get_any(d, ["num_test_sequences"]) if spec["level"] == "sequence" else get_any(d, ["num_test_videos"]),
                    num_videos=get_any(d, ["num_test_videos"]),
                    source_path=path,
                    notes="Adapted Lin2021 temporal skeleton baseline parsed from lin2021_yolov8_test_evaluation_summary.csv.",
                )

                rows.append(add_sample_counts_from_cm(row))

    if len(rows) == 0:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(
        subset=[
            "model_name",
            "level",
            "aggregation",
            "accuracy",
            "macro_f1",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
    )

    return out




def load_lin_all(lin_root: Path) -> pd.DataFrame:
    # Prefer the evaluation summary CSV because it has clean RNN/LSTM/GRU rows.
    csv_df = load_lin_from_csvs(lin_root)

    if len(csv_df) > 0:
        out = csv_df.copy()
    else:
        candidates = [
            lin_root / "outputs" / "reports" / "07_evaluate_rnn_lstm_gru_report.json",
            lin_root / "outputs" / "reports" / "08_compare_paper_pipeline_with_phase4_binary.json",
        ]

        dfs = []

        for path in candidates:
            df = load_lin_from_json(path)
            if len(df) > 0:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        out = pd.concat(dfs, ignore_index=True)

    # For internal Lin table, keep the best video-level row per model.
    video_out = out[out["level"].astype(str).str.lower() == "video"].copy()

    if len(video_out) == 0:
        video_out = out.copy()

    bests = []

    for model_name, g in video_out.groupby("model_name"):
        bests.append(best_row(g))

    return pd.concat(bests, ignore_index=True) if bests else pd.DataFrame()



# ======================================================================================
# PHASE 4
# ======================================================================================

def phase4_binary_cm(d: Dict[str, Any]) -> Dict[str, Optional[int]]:
    fall_support = to_float(get_any(d, ["Fall_support", "fall_support"]))
    not_fall_support = to_float(get_any(d, ["Not_Fall_support", "not_fall_support"]))

    fall_recall = to_float(get_any(d, ["Fall_recall", "fall_recall"]))
    not_fall_recall = to_float(get_any(d, ["Not_Fall_recall", "not_fall_recall"]))

    if None in [fall_support, not_fall_support, fall_recall, not_fall_recall]:
        return {"tn": None, "fp": None, "fn": None, "tp": None}

    tp = int(round(fall_support * fall_recall))
    fn = int(round(fall_support - tp))

    tn = int(round(not_fall_support * not_fall_recall))
    fp = int(round(not_fall_support - tn))

    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def load_phase4_binary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    data = read_json(path)
    rows = []
    seen = set()

    for d in iter_dicts(data):
        name = get_any(d, ["fair_model_name", "model_name", "name"])

        if name is None:
            continue

        name = str(name)

        if "phase 4" not in name.lower() and "quality" not in name.lower():
            continue

        # Hard filter: keep only binary Fall/Not_Fall rows.
        has_binary_metrics = (
            get_any(d, ["Fall_recall", "fall_recall"]) is not None
            and get_any(d, ["Not_Fall_recall", "not_fall_recall"]) is not None
            and get_any(d, ["Fall_support", "fall_support"]) is not None
            and get_any(d, ["Not_Fall_support", "not_fall_support"]) is not None
        )

        result_path = str(get_any(d, ["result_path", "path"]) or "")

        if not has_binary_metrics and "binary" not in result_path.lower():
            continue

        accuracy = get_any(d, ["accuracy"])
        macro_f1 = get_any(d, ["macro_f1"])

        if accuracy is None or macro_f1 is None:
            continue

        cm = phase4_binary_cm(d)

        clean_name = name.replace("Phase 4 - ", "")

        key = (clean_name, accuracy, macro_f1, cm["tn"], cm["fp"], cm["fn"], cm["tp"])

        if key in seen:
            continue

        seen.add(key)

        row = metric_row(
            family="Phase 4",
            model_name=clean_name,
            method_type="quality_aware_fusion",
            split="test",
            level="sample",
            aggregation="none",
            threshold=None,
            accuracy=accuracy,
            macro_f1=macro_f1,
            fall_recall=get_any(d, ["Fall_recall", "fall_recall"]),
            fall_f1=get_any(d, ["Fall_f1", "fall_f1"]),
            specificity=get_any(d, ["Not_Fall_recall", "not_fall_recall"]),
            tn=cm["tn"],
            fp=cm["fp"],
            fn=cm["fn"],
            tp=cm["tp"],
            num_samples=get_any(d, ["num_test_samples", "num_samples"]),
            num_videos=get_any(d, ["num_test_videos", "num_videos"]),
            source_path=path,
            notes="Phase 4 binary Fall/Not_Fall quality-aware model.",
        )

        rows.append(add_sample_counts_from_cm(row))

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    # Keep only the best row per Phase 4 model name.
    bests = []

    for model_name, g in out.groupby("model_name"):
        bests.append(best_row(g))

    return pd.concat(bests, ignore_index=True) if bests else pd.DataFrame()


# ======================================================================================
# TABLE BUILDING
# ======================================================================================

def build_main_table(
    phase4_df: pd.DataFrame,
    chen_df: pd.DataFrame,
    lin_df: pd.DataFrame,
    yadav_df: pd.DataFrame,
) -> pd.DataFrame:
    parts = []

    if len(phase4_df) > 0:
        parts.append(phase4_df)

    chen_test = chen_df[chen_df["split"].astype(str).str.lower() == "test"].copy() if len(chen_df) else pd.DataFrame()

    if len(chen_test) > 0:
        parts.append(best_row(chen_test))

    if len(lin_df) > 0:
        parts.append(best_row(lin_df))

    if len(yadav_df) > 0:
        parts.append(best_yadav_row(yadav_df))

    if not parts:
        return pd.DataFrame()

    main = pd.concat(parts, ignore_index=True)
    main = main.sort_values(["macro_f1", "accuracy", "fall_recall"], ascending=[False, False, False]).reset_index(drop=True)
    main.insert(0, "rank", np.arange(1, len(main) + 1))

    return main


def build_yadav_internal(yadav_df: pd.DataFrame) -> pd.DataFrame:
    if len(yadav_df) == 0:
        return pd.DataFrame()

    # Keep best video-level row per model.
    video_df = yadav_df[yadav_df["level"].astype(str).str.lower() == "video"].copy()

    if len(video_df) == 0:
        video_df = yadav_df.copy()

    rows = []

    for model_name, g in video_df.groupby("model_name"):
        rows.append(best_row(g))

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out = out.sort_values(["macro_f1", "accuracy"], ascending=[False, False]).reset_index(drop=True)

    if len(out) > 0:
        out.insert(0, "rank", np.arange(1, len(out) + 1))

    return out


def display_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    out = df.copy()

    for c in ["accuracy", "macro_f1", "fall_recall", "fall_f1", "specificity"]:
        if c in out.columns:
            out[c] = out[c].apply(fmt_pct)

    for c in ["tn", "fp", "fn", "tp", "num_samples", "num_videos"]:
        if c in out.columns:
            out[c] = out[c].apply(fmt_int)

    if "threshold" in out.columns:
        out["threshold"] = out["threshold"].apply(fmt_threshold)

    return out


def table_html(df: pd.DataFrame, cols: List[str]) -> str:
    if df is None or len(df) == 0:
        return "<p><em>No data available.</em></p>"

    cols = [c for c in cols if c in df.columns]
    d = display_df(df[cols])

    return d.to_html(index=False, escape=True, classes="metric-table")


def make_html(
    main_df: pd.DataFrame,
    chen_df: pd.DataFrame,
    lin_df: pd.DataFrame,
    yadav_internal_df: pd.DataFrame,
    warnings: List[str],
    out_path: Path,
) -> None:
    main_cols = [
        "rank",
        "family",
        "model_name",
        "method_type",
        "level",
        "aggregation",
        "accuracy",
        "macro_f1",
        "fall_recall",
        "fall_f1",
        "specificity",
        "tn",
        "fp",
        "fn",
        "tp",
        "num_samples",
        "num_videos",
    ]

    internal_cols = [
        "rank",
        "family",
        "model_name",
        "split",
        "level",
        "aggregation",
        "accuracy",
        "macro_f1",
        "fall_recall",
        "fall_f1",
        "specificity",
        "tn",
        "fp",
        "fn",
        "tp",
        "num_samples",
        "num_videos",
    ]

    chen_cols = [
        "family",
        "model_name",
        "split",
        "level",
        "aggregation",
        "accuracy",
        "macro_f1",
        "fall_recall",
        "fall_f1",
        "specificity",
        "tn",
        "fp",
        "fn",
        "tp",
        "num_samples",
        "num_videos",
    ]

    warn_html = "<p>No warnings.</p>"

    if warnings:
        warn_html = "<ul>" + "".join(f"<li>{html.escape(w)}</li>" for w in warnings) + "</ul>"

    html_text = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Final Baseline Comparison</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 28px;
    color: #222;
}}
h1, h2 {{
    color: #111;
}}
.note {{
    background: #fff7d6;
    border-left: 4px solid #d69e2e;
    padding: 12px 16px;
    margin: 14px 0 22px 0;
}}
.metric-table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 13px;
    margin-bottom: 30px;
}}
.metric-table th {{
    background: #222;
    color: white;
    padding: 8px;
    border: 1px solid #555;
    text-align: left;
}}
.metric-table td {{
    padding: 7px;
    border: 1px solid #ccc;
}}
.metric-table tr:nth-child(even) {{
    background: #f7f7f7;
}}
</style>
</head>
<body>
<h1>Final Baseline Comparison Report</h1>

<h2>1. Main comparison: Phase 4 vs best model from each paper</h2>
{table_html(main_df, main_cols)}

<h2>2. Chen2020-style internal table</h2>
{table_html(chen_df, chen_cols)}

<h2>3. Lin2021-style internal table</h2>
{table_html(lin_df, internal_cols)}

<h2>4. Yadav2021-style internal table</h2>
{table_html(yadav_internal_df, internal_cols)}

{("<h2>Warnings</h2>" + warn_html) if warnings else ""}

</body>
</html>
"""

    ensure_dir(out_path.parent)
    out_path.write_text(html_text, encoding="utf-8")


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [json_safe(v) for v in obj]

    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if not np.isfinite(x) else x

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chen-metrics-csv",
        type=str,
        default=str(PROJECT_ROOT / "paper_baselines" / "chen2020_yolov8_three_thresholds" / "outputs" / "metrics" / "07_threshold_method_metrics.csv"),
    )

    parser.add_argument(
        "--lin-root",
        type=str,
        default=str(PROJECT_ROOT / "paper_baselines" / "lin2021_yolov8_lstm_gru"),
    )

    parser.add_argument(
        "--yadav-metrics-csv",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "metrics" / "06_evaluate_cnn_lstm_convlstm_metrics_val_tuned.csv"),
    )

    parser.add_argument(
        "--phase4-json",
        type=str,
        default=str(PROJECT_ROOT / "phase4_quality_aware_fusion" / "outputs" / "comparison" / "all_phases_fair_comparison.json"),
    )

    args = parser.parse_args()

    chen_path = Path(args.chen_metrics_csv)
    lin_root = Path(args.lin_root)
    yadav_path = Path(args.yadav_metrics_csv)
    phase4_path = Path(args.phase4_json)

    metrics_dir = BASELINE_ROOT / "outputs" / "metrics"
    reports_dir = BASELINE_ROOT / "outputs" / "reports"

    ensure_dir(metrics_dir)
    ensure_dir(reports_dir)

    print("=" * 100)
    print("FILE 07 - Final baseline comparison")
    print("=" * 100)
    print(f"Chen CSV:    {chen_path} | exists={chen_path.exists()}")
    print(f"Lin root:    {lin_root} | exists={lin_root.exists()}")
    print(f"Yadav CSV:   {yadav_path} | exists={yadav_path.exists()}")
    print(f"Phase4 JSON: {phase4_path} | exists={phase4_path.exists()}")
    print("=" * 100)

    warnings = []

    if not chen_path.exists():
        warnings.append(f"Chen metrics not found: {chen_path}")

    if not lin_root.exists():
        warnings.append(f"Lin root not found: {lin_root}")

    if not yadav_path.exists():
        fallback = BASELINE_ROOT / "outputs" / "metrics" / "06_evaluate_cnn_lstm_convlstm_metrics.csv"

        if fallback.exists():
            warnings.append("Yadav val_tuned metrics not found. Using non-tuned metrics.")
            yadav_path = fallback
        else:
            warnings.append(f"Yadav metrics not found: {yadav_path}")

    if not phase4_path.exists():
        warnings.append(f"Phase4 comparison JSON not found: {phase4_path}")

    chen_df = load_chen_all(chen_path)
    lin_df = load_lin_all(lin_root)
    yadav_df = load_yadav_all(yadav_path)
    phase4_df = load_phase4_binary(phase4_path)

    # Final report should compare test results only.
    # Chen2020-style has only one threshold-rule model, so keep only its test row.
    if len(chen_df) > 0 and "split" in chen_df.columns:
        chen_df = chen_df[chen_df["split"].astype(str).str.lower() == "test"].copy()

    if len(lin_df) == 0:
        warnings.append("Could not parse Lin2021-style results. Check Lin output report/CSV structure.")

    if len(phase4_df) != 2:
        warnings.append(f"Expected 2 binary Phase 4 rows, got {len(phase4_df)}.")

    yadav_internal_df = build_yadav_internal(yadav_df)
    main_df = build_main_table(phase4_df, chen_df, lin_df, yadav_df)

    # The final selected rows do not use a comparable threshold column:
    # Phase 4 has no threshold, Lin/Yadav selected rows use majority_vote,
    # and Chen has only one final test rule row. Remove threshold for cleaner tables.
    for _df in [main_df, chen_df, lin_df, yadav_internal_df]:
        if len(_df) > 0 and "threshold" in _df.columns:
            _df.drop(columns=["threshold"], inplace=True)

    # Sort internal tables.
    if len(chen_df) > 0:
        chen_df = chen_df.sort_values(["split", "macro_f1"], ascending=[True, False]).reset_index(drop=True)

    if len(lin_df) > 0:
        lin_df = lin_df.sort_values(["macro_f1", "accuracy"], ascending=[False, False]).reset_index(drop=True)
        lin_df.insert(0, "rank", np.arange(1, len(lin_df) + 1))

    outputs = {
        "main_comparison_csv": metrics_dir / "07_main_comparison_phase4_vs_best_papers.csv",
        "chen_internal_csv": metrics_dir / "07_chen2020_internal_table.csv",
        "lin_internal_csv": metrics_dir / "07_lin2021_internal_table.csv",
        "yadav_internal_csv": metrics_dir / "07_yadav2021_internal_table.csv",
        "report_json": reports_dir / "07_final_baseline_comparison_report.json",
        "report_html": reports_dir / "07_final_baseline_comparison_report.html",
    }

    main_df.to_csv(outputs["main_comparison_csv"], index=False)
    chen_df.to_csv(outputs["chen_internal_csv"], index=False)
    lin_df.to_csv(outputs["lin_internal_csv"], index=False)
    yadav_internal_df.to_csv(outputs["yadav_internal_csv"], index=False)

    report = {
        "status": "completed",
        "logic": "Main table contains only 2 binary Phase 4 models plus the best model from Chen2020-style, Lin2021-style, and Yadav2021-style baselines.",
        "input_paths": {
            "chen_csv": str(chen_path),
            "lin_root": str(lin_root),
            "yadav_csv": str(yadav_path),
            "phase4_json": str(phase4_path),
        },
        "outputs": {k: str(v) for k, v in outputs.items()},
        "warnings": warnings,
        "main_comparison": main_df,
        "chen_internal": chen_df,
        "lin_internal": lin_df,
        "yadav_internal": yadav_internal_df,
    }

    save_json(json_safe(report), outputs["report_json"])
    make_html(main_df, chen_df, lin_df, yadav_internal_df, warnings, outputs["report_html"])

    print("=" * 100)
    print("FILE 07 completed.")
    print("=" * 100)
    print(f"Main comparison CSV: {outputs['main_comparison_csv']}")
    print(f"Chen internal CSV:   {outputs['chen_internal_csv']}")
    print(f"Lin internal CSV:    {outputs['lin_internal_csv']}")
    print(f"Yadav internal CSV:  {outputs['yadav_internal_csv']}")
    print(f"Report HTML:         {outputs['report_html']}")
    print("=" * 100)

    main_cols = [
        "rank",
        "family",
        "model_name",
        "level",
        "aggregation",
        "accuracy",
        "macro_f1",
        "fall_recall",
        "fall_f1",
        "specificity",
        "tn",
        "fp",
        "fn",
        "tp",
        "num_samples",
        "num_videos",
    ]

    print("MAIN COMPARISON:")
    print(main_df[main_cols].to_string(index=False))
    print("=" * 100)

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print("-", w)
        print("=" * 100)


if __name__ == "__main__":
    main()
