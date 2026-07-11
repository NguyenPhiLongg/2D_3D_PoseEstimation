from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
BASELINE_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = BASELINE_ROOT.parents[1]


# ======================================================================================
# FILE 01: COLLECT EXISTING YOLOV8-POSE KEYPOINTS
# ======================================================================================
#
# Chen et al. 2020 original method:
#   Surveillance video
#   -> OpenPose obtains skeleton data
#   -> apply three rule-based fall conditions
#
# Adapted method in this project:
#   Surveillance video
#   -> YOLOv8-Pose keypoints
#   -> convert to Chen-style skeleton nodes
#   -> apply three rule-based fall conditions
#
# This file only handles the skeleton/keypoint collection stage.
# It does not compute rules yet.
# It does not tune thresholds yet.
# It does not evaluate yet.
#
# Output per video:
#   video_id, frame_idx, label, label_name, has_person, valid_joint_count,
#   joint_0_x, joint_0_y, joint_0_c, ..., joint_16_x, joint_16_y, joint_16_c
#
# YOLOv8 COCO17 joint order:
#   0 nose
#   1 left_eye
#   2 right_eye
#   3 left_ear
#   4 right_ear
#   5 left_shoulder
#   6 right_shoulder
#   7 left_elbow
#   8 right_elbow
#   9 left_wrist
#   10 right_wrist
#   11 left_hip
#   12 right_hip
#   13 left_knee
#   14 right_knee
#   15 left_ankle
#   16 right_ankle
# ======================================================================================


COCO17_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


LABEL_TEXT_TO_ID = {
    "not_fall": 0,
    "not-fall": 0,
    "not fall": 0,
    "non_fall": 0,
    "non-fall": 0,
    "no_fall": 0,
    "normal": 0,
    "0": 0,
    "fall": 1,
    "falling": 1,
    "1": 1,
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def make_safe_name(text: str) -> str:
    safe = str(text)
    safe = safe.replace("\\", "__")
    safe = safe.replace("/", "__")
    safe = safe.replace(":", "")
    safe = safe.replace(" ", "_")
    safe = safe.replace("-", "_")
    safe = safe.replace(".", "_")
    safe = safe.replace("(", "")
    safe = safe.replace(")", "")
    safe = safe.replace("[", "")
    safe = safe.replace("]", "")
    safe = re.sub(r"_+", "_", safe)
    safe = safe.strip("_")
    safe = re.sub(r"_+", "_", safe)
    safe = safe.strip("_")
    return safe if safe else "unknown"


def label_to_name(label: Optional[int]) -> str:
    if label == 1:
        return "Fall"
    if label == 0:
        return "Not_Fall"
    return "Unknown"


def infer_label_from_text(text: str) -> Optional[int]:
    lower = str(text).lower()

    if (
        "not_fall" in lower
        or "not-fall" in lower
        or "not fall" in lower
        or "non_fall" in lower
        or "non-fall" in lower
        or "no_fall" in lower
    ):
        return 0

    if "fall" in lower:
        return 1

    return None


def parse_label_value(value, fallback_text: str = "") -> Optional[int]:
    if value is None:
        return infer_label_from_text(fallback_text)

    try:
        if pd.isna(value):
            return infer_label_from_text(fallback_text)
    except Exception:
        pass

    if isinstance(value, (int, np.integer)):
        if int(value) in [0, 1]:
            return int(value)

    if isinstance(value, (float, np.floating)):
        if int(value) in [0, 1]:
            return int(value)

    text = str(value).strip().lower()

    if text in LABEL_TEXT_TO_ID:
        return LABEL_TEXT_TO_ID[text]

    return infer_label_from_text(text + " " + fallback_text)


def read_csv_safe(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, nrows=nrows, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e

    raise last_error


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    return None


def infer_video_column(df: pd.DataFrame) -> Optional[str]:
    return find_column(df, [
        "video_id",
        "video",
        "video_name",
        "video_file",
        "video_filename",
        "source_video",
        "filename",
        "file_name",
        "file",
        "path",
        "file_path",
        "video_path",
        "clip",
        "clip_id",
        "sample",
        "sample_id",
    ])


def infer_frame_column(df: pd.DataFrame) -> Optional[str]:
    return find_column(df, [
        "frame_idx",
        "frame_index",
        "frame_id",
        "frame",
        "frame_number",
        "image_id",
        "time_idx",
        "timestep",
        "t",
    ])


def infer_label_column(df: pd.DataFrame) -> Optional[str]:
    return find_column(df, [
        "label",
        "label_id",
        "binary_label",
        "fall_label",
        "is_fall",
        "target",
        "class",
        "class_id",
        "category",
        "action_label",
    ])


def keypoint_column_candidates(index: int, axis: str, name: str) -> List[str]:
    axis = axis.lower()

    if axis == "x":
        aliases = ["x"]
    elif axis == "y":
        aliases = ["y"]
    else:
        aliases = ["c", "conf", "confidence", "score", "visibility", "v"]

    cols = []

    for a in aliases:
        cols.extend([
            f"kp_{index}_{a}",
            f"kp{index}_{a}",
            f"kpt_{index}_{a}",
            f"kpt{index}_{a}",
            f"keypoint_{index}_{a}",
            f"keypoint{index}_{a}",
            f"joint_{index}_{a}",
            f"joint{index}_{a}",
            f"coco_{index}_{a}",
            f"coco{index}_{a}",
            f"yolo_{index}_{a}",
            f"yolo{index}_{a}",
            f"pose_{index}_{a}",
            f"pose{index}_{a}",
            f"{index}_{a}",
            f"{a}_{index}",
            f"{a}{index}",
            f"{name}_{a}",
            f"{name}{a}",
        ])

    return cols


def numeric_ratio(series: pd.Series) -> float:
    converted = pd.to_numeric(series, errors="coerce")
    if len(converted) == 0:
        return 0.0
    return float(converted.notna().mean())


def get_numeric_columns(df: pd.DataFrame, exclude_cols: List[Optional[str]]) -> List[str]:
    exclude = set([c for c in exclude_cols if c is not None])

    numeric_cols = []

    for col in df.columns:
        if col in exclude:
            continue

        if numeric_ratio(df[col]) >= 0.80:
            numeric_cols.append(col)

    return numeric_cols


def detect_named_keypoint_map(df: pd.DataFrame) -> Dict[int, Dict[str, Optional[str]]]:
    mapping: Dict[int, Dict[str, Optional[str]]] = {}

    for i, name in enumerate(COCO17_NAMES):
        x_col = find_column(df, keypoint_column_candidates(i, "x", name))
        y_col = find_column(df, keypoint_column_candidates(i, "y", name))
        c_col = find_column(df, keypoint_column_candidates(i, "c", name))

        mapping[i] = {
            "x": x_col,
            "y": y_col,
            "c": c_col,
        }

    return mapping


def count_detected_points(mapping: Dict[int, Dict[str, Optional[str]]]) -> int:
    count = 0

    for i in range(17):
        if mapping[i]["x"] is not None and mapping[i]["y"] is not None:
            count += 1

    return count


def build_flat_keypoint_map(
    df: pd.DataFrame,
    video_col: Optional[str],
    frame_col: Optional[str],
    label_col: Optional[str],
) -> Tuple[Optional[Dict[int, Dict[str, Optional[str]]]], str]:
    numeric_cols = get_numeric_columns(df, exclude_cols=[video_col, frame_col, label_col])

    if len(numeric_cols) >= 51:
        selected = numeric_cols[:51]
        mapping: Dict[int, Dict[str, Optional[str]]] = {}

        for i in range(17):
            mapping[i] = {
                "x": selected[i * 3 + 0],
                "y": selected[i * 3 + 1],
                "c": selected[i * 3 + 2],
            }

        return mapping, "flat_51_xyc"

    if len(numeric_cols) >= 34:
        selected = numeric_cols[:34]
        mapping = {}

        for i in range(17):
            mapping[i] = {
                "x": selected[i * 2 + 0],
                "y": selected[i * 2 + 1],
                "c": None,
            }

        return mapping, "flat_34_xy"

    return None, "not_found"


def detect_keypoint_format(df: pd.DataFrame) -> Tuple[Optional[Dict[int, Dict[str, Optional[str]]]], Dict]:
    video_col = infer_video_column(df)
    frame_col = infer_frame_column(df)
    label_col = infer_label_column(df)

    named_mapping = detect_named_keypoint_map(df)
    named_count = count_detected_points(named_mapping)

    if named_count >= 10:
        has_conf = any(named_mapping[i]["c"] is not None for i in range(17))

        return named_mapping, {
            "format": "named_columns",
            "num_detected_keypoints": int(named_count),
            "has_confidence": bool(has_conf),
            "video_col": video_col,
            "frame_col": frame_col,
            "label_col": label_col,
        }

    flat_mapping, flat_format = build_flat_keypoint_map(
        df=df,
        video_col=video_col,
        frame_col=frame_col,
        label_col=label_col,
    )

    if flat_mapping is not None:
        has_conf = any(flat_mapping[i]["c"] is not None for i in range(17))

        return flat_mapping, {
            "format": flat_format,
            "num_detected_keypoints": 17,
            "has_confidence": bool(has_conf),
            "video_col": video_col,
            "frame_col": frame_col,
            "label_col": label_col,
        }

    return None, {
        "format": "unusable",
        "num_detected_keypoints": int(named_count),
        "has_confidence": False,
        "video_col": video_col,
        "frame_col": frame_col,
        "label_col": label_col,
    }


def inspect_csv(csv_path: Path) -> Dict:
    try:
        df = read_csv_safe(csv_path, nrows=100)
    except Exception as e:
        return {
            "csv_path": str(csv_path),
            "usable": False,
            "error": repr(e),
        }

    mapping, fmt = detect_keypoint_format(df)

    usable = mapping is not None

    found = []

    if mapping is not None:
        for i in range(17):
            found.append({
                "index": i,
                "name": COCO17_NAMES[i],
                "x": mapping[i]["x"],
                "y": mapping[i]["y"],
                "c": mapping[i]["c"],
            })

    return {
        "csv_path": str(csv_path),
        "usable": bool(usable),
        "format": fmt,
        "num_columns": int(len(df.columns)),
        "columns_preview": [str(c) for c in list(df.columns)[:80]],
        "found_keypoints": found,
    }


def list_csv_files_from_path(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() == ".csv":
        return [path]

    if path.is_dir():
        return sorted(path.rglob("*.csv"))

    return []


def default_candidate_roots(project_root: Path) -> List[Path]:
    return [
        project_root / "data" / "5_extracted_2d_confidence",
        project_root / "data" / "2_extracted_2d",
        project_root / "data" / "master_dataset.csv",
    ]


def collect_candidate_csv_files(project_root: Path, input_path: Optional[str]) -> List[Path]:
    if input_path:
        p = Path(input_path)

        if not p.is_absolute():
            p = project_root / p

        files = list_csv_files_from_path(p)

        if len(files) == 0:
            raise FileNotFoundError(f"No CSV files found from input path: {p}")

        return files

    files = []

    for root in default_candidate_roots(project_root):
        files.extend(list_csv_files_from_path(root))

    seen = set()
    unique = []

    for f in files:
        key = str(f.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(f)

    filtered = []

    for f in unique:
        lower = str(f).lower()
        if "paper_baselines" in lower:
            continue
        filtered.append(f)

    return filtered


def source_priority(csv_path: Path) -> int:
    lower = str(csv_path).lower()

    if "5_extracted_2d_confidence" in lower:
        return 1

    if "2_extracted_2d" in lower:
        return 2

    if "master_dataset.csv" in lower:
        return 3

    return 10


def select_usable_sources(scan_results: List[Dict], use_all_usable: bool) -> List[Path]:
    usable = [
        Path(item["csv_path"])
        for item in scan_results
        if item.get("usable", False)
    ]

    if len(usable) == 0:
        return []

    if use_all_usable:
        return usable

    best_priority = min(source_priority(p) for p in usable)

    selected = [
        p for p in usable
        if source_priority(p) == best_priority
    ]

    return selected


def infer_selected_root(selected_files: List[Path]) -> Path:
    if len(selected_files) == 0:
        return PROJECT_ROOT

    best = min(source_priority(p) for p in selected_files)

    if best == 1:
        return PROJECT_ROOT / "data" / "5_extracted_2d_confidence"

    if best == 2:
        return PROJECT_ROOT / "data" / "2_extracted_2d"

    if best == 3:
        return PROJECT_ROOT / "data"

    return selected_files[0].parent


def clean_video_text(value) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()

    if text == "":
        return ""

    text = text.replace("\\", "/")

    marker = "data/1_raw_videos/"
    lower = text.lower()

    if marker in lower:
        idx = lower.find(marker)
        text = text[idx + len(marker):]

    for suf in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        if text.lower().endswith(suf):
            text = text[: -len(suf)]
            break

    return text


def normalize_video_id(raw_value, csv_path: Path, selected_root: Path, video_col_exists: bool) -> str:
    raw_text = clean_video_text(raw_value)

    if raw_text == "":
        try:
            rel = csv_path.relative_to(selected_root)
            raw_text = str(rel.with_suffix(""))
        except Exception:
            raw_text = csv_path.stem

    if not video_col_exists:
        try:
            rel = csv_path.relative_to(selected_root)
            raw_text = str(rel.with_suffix(""))
        except Exception:
            raw_text = csv_path.stem

    return make_safe_name(raw_text)


def series_to_numeric(df: pd.DataFrame, col: Optional[str], default: float = 0.0) -> pd.Series:
    if col is None:
        return pd.Series([default] * len(df), index=df.index, dtype=np.float32)

    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(np.float32)


def build_raw_yolov8_dataframe(
    df: pd.DataFrame,
    csv_path: Path,
    selected_root: Path,
    mapping: Dict[int, Dict[str, Optional[str]]],
) -> pd.DataFrame:
    video_col = infer_video_column(df)
    frame_col = infer_frame_column(df)
    label_col = infer_label_column(df)

    video_col_exists = video_col is not None

    out = pd.DataFrame(index=df.index)

    if video_col is not None:
        out["video_id_core"] = df[video_col].apply(
            lambda v: normalize_video_id(
                raw_value=v,
                csv_path=csv_path,
                selected_root=selected_root,
                video_col_exists=True,
            )
        )
    else:
        core = normalize_video_id(
            raw_value=None,
            csv_path=csv_path,
            selected_root=selected_root,
            video_col_exists=False,
        )
        out["video_id_core"] = core

    if frame_col is not None:
        out["frame_idx"] = pd.to_numeric(df[frame_col], errors="coerce").fillna(0).astype(int)
    else:
        out["frame_idx"] = out.groupby("video_id_core").cumcount().astype(int)

    source_text = str(csv_path)

    if label_col is not None:
        labels = df[label_col].apply(lambda v: parse_label_value(v, fallback_text=source_text))
    else:
        inferred = infer_label_from_text(source_text)
        labels = pd.Series([inferred] * len(df), index=df.index)

    labels = labels.apply(lambda x: -1 if x is None else int(x))

    out["label"] = labels.astype(int)
    out["label_name"] = out["label"].apply(label_to_name)

    out["video_id"] = out["label_name"].astype(str) + "__" + out["video_id_core"].astype(str)
    out["source_file"] = str(csv_path)

    valid_joint_count = pd.Series([0] * len(df), index=df.index, dtype=np.int32)

    for i in range(17):
        x = series_to_numeric(df, mapping[i]["x"], default=0.0)
        y = series_to_numeric(df, mapping[i]["y"], default=0.0)

        if mapping[i]["c"] is not None:
            c = series_to_numeric(df, mapping[i]["c"], default=0.0)
        else:
            c = ((x != 0.0) | (y != 0.0)).astype(np.float32)

        out[f"joint_{i}_x"] = x
        out[f"joint_{i}_y"] = y
        out[f"joint_{i}_c"] = c

        valid = ((x != 0.0) | (y != 0.0)) & (c > 0.0)
        valid_joint_count += valid.astype(np.int32)

    out["valid_joint_count"] = valid_joint_count.astype(int)
    out["has_person"] = (out["valid_joint_count"] >= 3).astype(int)

    final_cols = [
        "video_id",
        "frame_idx",
        "source_file",
        "label",
        "label_name",
        "has_person",
        "valid_joint_count",
    ]

    for i in range(17):
        final_cols.extend([
            f"joint_{i}_x",
            f"joint_{i}_y",
            f"joint_{i}_c",
        ])

    return out[final_cols]


def export_per_video_csvs(raw_df: pd.DataFrame, output_dir: Path, source_csv: Path) -> List[Dict]:
    results = []

    for video_id, group in raw_df.groupby("video_id"):
        group = group.sort_values("frame_idx").reset_index(drop=True)

        label_values = sorted(group["label"].astype(int).unique().tolist())

        if len(label_values) == 1:
            label = int(label_values[0])
        else:
            label = int(group["label"].value_counts().idxmax())

        label_name = label_to_name(label)
        label_folder = label_name if label_name in ["Fall", "Not_Fall"] else "Unknown"

        video_output_dir = output_dir / label_folder
        ensure_dir(video_output_dir)

        safe_video_id = make_safe_name(video_id)
        output_csv = video_output_dir / f"{safe_video_id}.csv"

        group.to_csv(output_csv, index=False)

        results.append({
            "video_id": safe_video_id,
            "label": int(label),
            "label_name": label_name,
            "num_rows": int(len(group)),
            "num_has_person": int(group["has_person"].sum()),
            "num_no_person": int((group["has_person"] == 0).sum()),
            "mean_valid_joint_count": float(group["valid_joint_count"].mean()),
            "source_csv": str(source_csv),
            "output_csv": str(output_csv),
        })

    return results


def process_one_csv(csv_path: Path, selected_root: Path, output_dir: Path) -> List[Dict]:
    df_sample = read_csv_safe(csv_path, nrows=100)
    mapping, fmt = detect_keypoint_format(df_sample)

    if mapping is None:
        raise ValueError(f"Cannot detect YOLOv8 keypoint columns in {csv_path}")

    df = read_csv_safe(csv_path)

    raw_df = build_raw_yolov8_dataframe(
        df=df,
        csv_path=csv_path,
        selected_root=selected_root,
        mapping=mapping,
    )

    return export_per_video_csvs(raw_df=raw_df, output_dir=output_dir, source_csv=csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="Collect existing YOLOv8-Pose keypoints for Chen et al. 2020-style three-threshold baseline."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional CSV file or folder. If omitted, the script searches data/5_extracted_2d_confidence, data/2_extracted_2d, and data/master_dataset.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BASELINE_ROOT / "data" / "raw_yolov8_keypoints"),
        help="Output folder for per-video raw YOLOv8 COCO17 keypoint CSVs.",
    )

    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only scan available CSV files and report which ones are usable.",
    )

    parser.add_argument(
        "--use-all-usable",
        action="store_true",
        help="Use all usable CSV files. Default chooses the best source priority to avoid duplicates.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    reports_dir = BASELINE_ROOT / "outputs" / "reports"

    ensure_dir(output_dir)
    ensure_dir(reports_dir)

    candidate_csvs = collect_candidate_csv_files(
        project_root=PROJECT_ROOT,
        input_path=args.input,
    )

    print("=" * 100)
    print("FILE 01 - Collect existing YOLOv8-Pose keypoints")
    print("=" * 100)
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Baseline root:    {BASELINE_ROOT}")
    print(f"Output dir:       {output_dir}")
    print(f"Candidate CSVs:   {len(candidate_csvs)}")
    print("=" * 100)

    scan_results = []

    for csv_path in candidate_csvs:
        info = inspect_csv(csv_path)
        scan_results.append(info)

        if info.get("usable", False):
            fmt = info.get("format", {})
            print(f"USABLE: {csv_path}")
            print(f"  format: {fmt.get('format')}")
            print(f"  detected joints: {fmt.get('num_detected_keypoints')}")
            print(f"  has confidence: {fmt.get('has_confidence')}")
        else:
            print(f"SKIP:   {csv_path}")

    selected_files = select_usable_sources(
        scan_results=scan_results,
        use_all_usable=args.use_all_usable,
    )

    selected_root = infer_selected_root(selected_files)

    scan_report = {
        "status": "scan_completed",
        "candidate_csv_count": len(candidate_csvs),
        "usable_csv_count": int(sum(1 for x in scan_results if x.get("usable", False))),
        "selected_csv_count": len(selected_files),
        "selected_root": str(selected_root),
        "selected_files": [str(p) for p in selected_files],
        "scan_results": scan_results,
    }

    scan_report_path = reports_dir / "01_collect_yolov8_keypoints_scan_report.json"
    save_json(scan_report, scan_report_path)

    print("=" * 100)
    print(f"Usable CSVs:      {scan_report['usable_csv_count']}")
    print(f"Selected CSVs:    {len(selected_files)}")
    print(f"Selected root:    {selected_root}")
    print(f"Scan report:      {scan_report_path}")
    print("=" * 100)

    if args.scan_only:
        print("Scan-only mode. No output CSVs were created.")
        return

    if len(selected_files) == 0:
        print("No usable YOLOv8 keypoint CSV files found.")
        print("Open the scan report and check actual column names.")
        return

    all_results = []
    failed = []

    for idx, csv_path in enumerate(selected_files, start=1):
        print(f"[{idx}/{len(selected_files)}] Processing: {csv_path}")

        try:
            results = process_one_csv(
                csv_path=csv_path,
                selected_root=selected_root,
                output_dir=output_dir,
            )

            all_results.extend(results)
            print(f"  OK -> exported videos: {len(results)}")

        except Exception as e:
            failed.append({
                "input_csv": str(csv_path),
                "error": repr(e),
            })

            print(f"  FAILED: {repr(e)}")

    index_df = pd.DataFrame(all_results)
    index_path = output_dir / "all_raw_yolov8_keypoints_index.csv"

    if len(index_df) > 0:
        index_df.to_csv(index_path, index=False)

    total_rows = int(index_df["num_rows"].sum()) if len(index_df) > 0 else 0
    total_has_person = int(index_df["num_has_person"].sum()) if len(index_df) > 0 else 0
    total_no_person = int(index_df["num_no_person"].sum()) if len(index_df) > 0 else 0

    report = {
        "status": "completed",
        "pipeline_note": "Adapted Chen et al. 2020-style pipeline. YOLOv8-Pose replaces OpenPose only at the skeleton extraction stage. Later steps compute the three geometric decision conditions from the paper.",
        "output_dir": str(output_dir),
        "selected_root": str(selected_root),
        "num_selected_input_csv": len(selected_files),
        "num_exported_videos": int(len(index_df)),
        "total_rows": total_rows,
        "total_has_person_rows": total_has_person,
        "total_no_person_rows": total_no_person,
        "index_csv": str(index_path),
        "failed_count": len(failed),
        "failed": failed,
        "results": all_results,
    }

    report_path = reports_dir / "01_collect_yolov8_keypoints_report.json"
    save_json(report, report_path)

    print("=" * 100)
    print("FILE 01 completed.")
    print("=" * 100)
    print(f"Exported videos:          {len(index_df)}")
    print(f"Total rows:               {total_rows}")
    print(f"Total has_person rows:    {total_has_person}")
    print(f"Total no_person rows:     {total_no_person}")
    print(f"Failed input CSVs:        {len(failed)}")
    print(f"Index CSV:                {index_path}")
    print(f"Report:                   {report_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
