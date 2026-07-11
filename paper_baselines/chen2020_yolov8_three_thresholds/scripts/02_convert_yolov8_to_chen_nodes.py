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
# FILE 02: CONVERT YOLOV8 COCO17 KEYPOINTS TO CHEN 2020 NODE MODEL
# ======================================================================================
#
# Chen et al. 2020 original method:
#   OpenPose obtains skeleton data
#   -> human node model s0 ... s13
#   -> compute three geometric decision conditions
#
# Adapted method in this project:
#   YOLOv8-Pose COCO17 keypoints
#   -> convert to Chen-style node model s0 ... s13
#   -> compute three geometric decision conditions
#
# This file only converts skeleton format.
# It does NOT compute velocity, angle, or width-height ratio yet.
# It does NOT tune thresholds.
# It does NOT evaluate.
#
# Input from File 01:
#   joint_0_x, joint_0_y, joint_0_c, ..., joint_16_x, joint_16_y, joint_16_c
#
# Output:
#   s0_x, s0_y, s0_c, ..., s13_x, s13_y, s13_c
#
# Chen-style node mapping:
#   s0  = head / nose
#   s1  = shoulder center
#   s2  = right shoulder
#   s3  = right elbow
#   s4  = right hand / wrist
#   s5  = left shoulder
#   s6  = left elbow
#   s7  = left hand / wrist
#   s8  = right hip
#   s9  = right knee
#   s10 = right ankle
#   s11 = left hip
#   s12 = left knee
#   s13 = left ankle
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


CHEN14_NAMES = [
    "head_or_nose",       # s0
    "shoulder_center",   # s1
    "right_shoulder",    # s2
    "right_elbow",       # s3
    "right_hand",        # s4
    "left_shoulder",     # s5
    "left_elbow",        # s6
    "left_hand",         # s7
    "right_hip",         # s8
    "right_knee",        # s9
    "right_ankle",       # s10
    "left_hip",          # s11
    "left_knee",         # s12
    "left_ankle",        # s13
]


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
    return safe if safe else "unknown"


def label_to_name(label: Optional[int]) -> str:
    if label == 1:
        return "Fall"
    if label == 0:
        return "Not_Fall"
    return "Unknown"


def read_csv_safe(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e

    raise last_error


def get_required_raw_columns() -> List[str]:
    cols = [
        "video_id",
        "frame_idx",
        "label",
        "label_name",
    ]

    for i in range(17):
        cols.extend([
            f"joint_{i}_x",
            f"joint_{i}_y",
            f"joint_{i}_c",
        ])

    return cols


def validate_raw_yolov8_df(df: pd.DataFrame, csv_path: Path):
    required = get_required_raw_columns()

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required YOLOv8 raw columns in {csv_path}. "
            f"Missing examples: {missing[:20]}"
        )


def to_float(value, default: float = 0.0) -> float:
    try:
        out = pd.to_numeric(value, errors="coerce")
        if pd.isna(out):
            return default
        return float(out)
    except Exception:
        return default


def get_coco_point(row: pd.Series, idx: int) -> Tuple[float, float, float]:
    x = to_float(row.get(f"joint_{idx}_x", 0.0))
    y = to_float(row.get(f"joint_{idx}_y", 0.0))
    c = to_float(row.get(f"joint_{idx}_c", 0.0))
    return x, y, c


def is_valid_point(point: Tuple[float, float, float]) -> bool:
    x, y, c = point
    return bool(c > 0.0 and (x != 0.0 or y != 0.0))


def average_points(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    x1, y1, c1 = p1
    x2, y2, c2 = p2

    valid1 = is_valid_point(p1)
    valid2 = is_valid_point(p2)

    if valid1 and valid2:
        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            (c1 + c2) / 2.0,
        )

    if valid1:
        return x1, y1, c1

    if valid2:
        return x2, y2, c2

    return 0.0, 0.0, 0.0


def coco17_to_chen14(row: pd.Series) -> np.ndarray:
    """
    Convert one frame from YOLOv8 COCO17 to Chen et al. 2020 node model.

    YOLOv8 COCO17:
        0  nose
        5  left_shoulder
        6  right_shoulder
        7  left_elbow
        8  right_elbow
        9  left_wrist
        10 right_wrist
        11 left_hip
        12 right_hip
        13 left_knee
        14 right_knee
        15 left_ankle
        16 right_ankle

    Chen-style nodes:
        s0  head / nose
        s1  shoulder center
        s2  right shoulder
        s3  right elbow
        s4  right hand
        s5  left shoulder
        s6  left elbow
        s7  left hand
        s8  right hip
        s9  right knee
        s10 right ankle
        s11 left hip
        s12 left knee
        s13 left ankle
    """
    nose = get_coco_point(row, 0)

    left_shoulder = get_coco_point(row, 5)
    right_shoulder = get_coco_point(row, 6)

    left_elbow = get_coco_point(row, 7)
    right_elbow = get_coco_point(row, 8)

    left_wrist = get_coco_point(row, 9)
    right_wrist = get_coco_point(row, 10)

    left_hip = get_coco_point(row, 11)
    right_hip = get_coco_point(row, 12)

    left_knee = get_coco_point(row, 13)
    right_knee = get_coco_point(row, 14)

    left_ankle = get_coco_point(row, 15)
    right_ankle = get_coco_point(row, 16)

    shoulder_center = average_points(left_shoulder, right_shoulder)

    chen14 = np.zeros((14, 3), dtype=np.float32)

    chen14[0] = nose
    chen14[1] = shoulder_center

    chen14[2] = right_shoulder
    chen14[3] = right_elbow
    chen14[4] = right_wrist

    chen14[5] = left_shoulder
    chen14[6] = left_elbow
    chen14[7] = left_wrist

    chen14[8] = right_hip
    chen14[9] = right_knee
    chen14[10] = right_ankle

    chen14[11] = left_hip
    chen14[12] = left_knee
    chen14[13] = left_ankle

    return chen14


def convert_one_dataframe(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    validate_raw_yolov8_df(df, csv_path)

    if "frame_idx" in df.columns:
        df = df.sort_values("frame_idx").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    rows = []

    for _, row in df.iterrows():
        video_id = str(row["video_id"])
        frame_idx = int(to_float(row["frame_idx"], default=0.0))

        label = int(to_float(row["label"], default=-1.0))
        label_name = str(row["label_name"]) if "label_name" in row else label_to_name(label)

        chen14 = coco17_to_chen14(row)

        valid_node_count = 0

        out_row = {
            "video_id": video_id,
            "frame_idx": frame_idx,
            "source_file": str(csv_path),
            "label": label,
            "label_name": label_name,
        }

        for j in range(14):
            x = float(chen14[j, 0])
            y = float(chen14[j, 1])
            c = float(chen14[j, 2])

            out_row[f"s{j}_name"] = CHEN14_NAMES[j]
            out_row[f"s{j}_x"] = x
            out_row[f"s{j}_y"] = y
            out_row[f"s{j}_c"] = c

            if c > 0.0 and (x != 0.0 or y != 0.0):
                valid_node_count += 1

        out_row["valid_node_count"] = int(valid_node_count)
        out_row["has_person"] = int(valid_node_count >= 3)

        rows.append(out_row)

    out_df = pd.DataFrame(rows)

    meta_cols = [
        "video_id",
        "frame_idx",
        "source_file",
        "label",
        "label_name",
        "has_person",
        "valid_node_count",
    ]

    node_cols = []

    for j in range(14):
        node_cols.extend([
            f"s{j}_name",
            f"s{j}_x",
            f"s{j}_y",
            f"s{j}_c",
        ])

    out_df = out_df[meta_cols + node_cols]

    return out_df


def find_input_csv_files(input_dir: Path) -> List[Path]:
    csv_files = sorted([
        p for p in input_dir.rglob("*.csv")
        if p.name != "all_raw_yolov8_keypoints_index.csv"
    ])

    return csv_files


def process_one_csv(csv_path: Path, output_dir: Path) -> Dict:
    df = read_csv_safe(csv_path)
    out_df = convert_one_dataframe(df, csv_path)

    if len(out_df) == 0:
        raise ValueError(f"No rows after conversion: {csv_path}")

    video_id = str(out_df["video_id"].iloc[0])

    labels = sorted(out_df["label"].astype(int).unique().tolist())

    if len(labels) == 1:
        label = int(labels[0])
    else:
        label = int(out_df["label"].value_counts().idxmax())

    label_name = label_to_name(label)
    label_folder = label_name if label_name in ["Fall", "Not_Fall"] else "Unknown"

    video_output_dir = output_dir / label_folder
    ensure_dir(video_output_dir)

    safe_video_id = make_safe_name(video_id)
    output_csv = video_output_dir / f"{safe_video_id}.csv"

    out_df.to_csv(output_csv, index=False)

    result = {
        "video_id": safe_video_id,
        "label": int(label),
        "label_name": label_name,
        "num_rows": int(len(out_df)),
        "num_has_person": int(out_df["has_person"].sum()),
        "num_no_person": int((out_df["has_person"] == 0).sum()),
        "mean_valid_node_count": float(out_df["valid_node_count"].mean()),
        "input_csv": str(csv_path),
        "output_csv": str(output_csv),
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw YOLOv8 COCO17 keypoints to Chen et al. 2020 node model."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(BASELINE_ROOT / "data" / "raw_yolov8_keypoints"),
        help="Input folder from File 01.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BASELINE_ROOT / "data" / "chen_nodes"),
        help="Output folder for Chen-style node CSV files.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    reports_dir = BASELINE_ROOT / "outputs" / "reports"

    ensure_dir(input_dir)
    ensure_dir(output_dir)
    ensure_dir(reports_dir)

    csv_files = find_input_csv_files(input_dir)

    print("=" * 100)
    print("FILE 02 - Convert YOLOv8 COCO17 to Chen 2020 node model")
    print("=" * 100)
    print(f"Input dir:       {input_dir}")
    print(f"Output dir:      {output_dir}")
    print(f"Input CSV files: {len(csv_files)}")
    print("=" * 100)

    if len(csv_files) == 0:
        report = {
            "status": "no_input_csv",
            "input_dir": str(input_dir),
            "message": "No raw YOLOv8 keypoint CSV files found. Run file 01 first.",
        }

        report_path = reports_dir / "02_convert_yolov8_to_chen_nodes_report.json"
        save_json(report, report_path)

        print("No input CSV files found.")
        print(f"Report: {report_path}")
        return

    results = []
    failed = []

    for idx, csv_path in enumerate(csv_files, start=1):
        print(f"[{idx}/{len(csv_files)}] Processing: {csv_path.name}")

        try:
            result = process_one_csv(
                csv_path=csv_path,
                output_dir=output_dir,
            )

            results.append(result)

            print(
                f"  OK -> rows={result['num_rows']}, "
                f"has_person={result['num_has_person']}, "
                f"mean_valid_nodes={result['mean_valid_node_count']:.2f}"
            )

        except Exception as e:
            failed.append({
                "input_csv": str(csv_path),
                "error": repr(e),
            })

            print(f"  FAILED: {repr(e)}")

    index_df = pd.DataFrame(results)
    index_path = output_dir / "all_chen_nodes_index.csv"

    if len(index_df) > 0:
        index_df.to_csv(index_path, index=False)

    total_rows = int(index_df["num_rows"].sum()) if len(index_df) > 0 else 0
    total_has_person = int(index_df["num_has_person"].sum()) if len(index_df) > 0 else 0
    total_no_person = int(index_df["num_no_person"].sum()) if len(index_df) > 0 else 0

    report = {
        "status": "completed",
        "pipeline_note": "YOLOv8 COCO17 keypoints were converted to the Chen et al. 2020 node model s0...s13. This follows the paper's skeleton node representation, while replacing OpenPose with YOLOv8-Pose at the extraction stage.",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_input_csv": len(csv_files),
        "num_success": len(results),
        "num_failed": len(failed),
        "total_rows": total_rows,
        "total_has_person_rows": total_has_person,
        "total_no_person_rows": total_no_person,
        "chen14_node_names": CHEN14_NAMES,
        "index_csv": str(index_path),
        "failed": failed,
        "results": results,
    }

    report_path = reports_dir / "02_convert_yolov8_to_chen_nodes_report.json"
    save_json(report, report_path)

    print("=" * 100)
    print("FILE 02 completed.")
    print("=" * 100)
    print(f"Success videos:          {len(results)}")
    print(f"Failed videos:           {len(failed)}")
    print(f"Total rows:              {total_rows}")
    print(f"Total has_person rows:   {total_has_person}")
    print(f"Total no_person rows:    {total_no_person}")
    print(f"Index CSV:               {index_path}")
    print(f"Report:                  {report_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
