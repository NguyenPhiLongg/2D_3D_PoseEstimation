import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd


"""
External label mapping for Phase 5.

Purpose:
    Convert labels from external datasets into one common binary format:

        0 = Not_Fall
        1 = Fall

Supported datasets:
    - Le2i
    - MulCamFall

Important:
    This file only handles label mapping and annotation parsing.

Fair evaluation is handled later by:
    phase5_utils.build_sequence_manifest()
    phase5_utils.build_fair_common_manifest()
    phase5_utils.assert_all_models_same_sequence_set()

This prevents the old mistake where different models were evaluated
on different sample sets.
"""


# ============================================================
# IMPORT PHASE 5 UTILS
# ============================================================

PHASE5_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE5_DIR.parent

import sys

if str(PHASE5_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE5_DIR))

from phase5_utils import (
    load_config,
    cfg_path,
    probe_video,
    list_video_files,
    make_video_id,
    normalize_text,
    binary_label_name,
    standardize_binary_label,
    save_csv,
    save_json,
    print_dict,
)


# ============================================================
# COMMON LABEL CONSTANTS
# ============================================================

LABEL_NOT_FALL = 0
LABEL_FALL = 1

LABEL_NAMES = {
    LABEL_NOT_FALL: "Not_Fall",
    LABEL_FALL: "Fall",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or pd.isna(value):
            return default

        return int(float(value))
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default

        return float(value)
    except Exception:
        return default


def normalize_interval(start: int, end: int) -> Tuple[int, int]:
    start = int(start)
    end = int(end)

    if end < start:
        start, end = end, start

    return start, end


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []

    intervals = [
        normalize_interval(start, end)
        for start, end in intervals
        if start is not None and end is not None
    ]

    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda item: (item[0], item[1]))

    merged = [intervals[0]]

    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]

        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def clamp_intervals(
    intervals: List[Tuple[int, int]],
    frame_count: int,
) -> List[Tuple[int, int]]:
    output = []

    max_frame = max(int(frame_count) - 1, 0)

    for start, end in intervals:
        start = max(0, min(int(start), max_frame))
        end = max(0, min(int(end), max_frame))

        if end >= start:
            output.append((start, end))

    return merge_intervals(output)


def complement_intervals(
    fall_intervals: List[Tuple[int, int]],
    frame_count: int,
    min_length: int = 15,
) -> List[Tuple[int, int]]:
    """
    Build Not_Fall intervals outside fall intervals.
    """
    frame_count = int(frame_count)

    if frame_count <= 0:
        return []

    fall_intervals = clamp_intervals(fall_intervals, frame_count)

    not_fall_intervals = []

    cursor = 0

    for start, end in fall_intervals:
        if start > cursor:
            nf_start = cursor
            nf_end = start - 1

            if nf_end - nf_start + 1 >= min_length:
                not_fall_intervals.append((nf_start, nf_end))

        cursor = max(cursor, end + 1)

    if cursor <= frame_count - 1:
        nf_start = cursor
        nf_end = frame_count - 1

        if nf_end - nf_start + 1 >= min_length:
            not_fall_intervals.append((nf_start, nf_end))

    return not_fall_intervals


def build_segment_record(
    dataset: str,
    video_path: Path,
    video_id: str,
    label: int,
    segment_id: str,
    start_frame: int,
    end_frame: int,
    label_source: str,
    scene: str = "",
    scenario: str = "",
    camera: str = "",
    subject: str = "",
    activity: str = "",
    trial: str = "",
    notes: str = "",
) -> Dict:
    label = standardize_binary_label(label)

    probe = probe_video(video_path)

    return {
        "dataset": dataset,
        "scene": scene,
        "scenario": scenario,
        "camera": camera,
        "subject": subject,
        "activity": activity,
        "trial": trial,
        "video_id": video_id,
        "video_path": str(video_path),
        "label": int(label),
        "label_name": binary_label_name(label),
        "has_annotation": True,
        "include_eval": True,
        "label_source": label_source,
        "segment_id": segment_id,
        "segment_start_frame": int(start_frame),
        "segment_end_frame": int(end_frame),
        "frame_count": int(probe.get("frame_count", 0)),
        "fps": float(probe.get("fps", 0.0)),
        "width": int(probe.get("width", 0)),
        "height": int(probe.get("height", 0)),
        "duration_sec": float(probe.get("duration_sec", 0.0)),
        "notes": notes,
    }


# ============================================================
# LE2I HELPERS
# ============================================================

def infer_le2i_scene(video_path: Path, le2i_root: Path) -> str:
    try:
        rel = video_path.relative_to(le2i_root)
        parts = rel.parts

        if len(parts) >= 1:
            return str(parts[0])
    except Exception:
        pass

    return ""


def infer_le2i_video_number(video_path: Path) -> Optional[int]:
    """
    Extract video number from names like:
        video (1).avi
        video_1.avi
        Video1.avi
    """
    stem = video_path.stem

    numbers = re.findall(r"\d+", stem)

    if not numbers:
        return None

    return int(numbers[-1])


def find_annotation_dirs(scene_dir: Path) -> List[Path]:
    candidates = []

    if not scene_dir.exists():
        return candidates

    for path in scene_dir.rglob("*"):
        if path.is_dir():
            name = normalize_text(path.name)

            if any(token in name for token in ["annotation", "annot", "groundtruth", "ground_truth", "truth", "label"]):
                candidates.append(path)

    return sorted(candidates)


def list_annotation_files(annotation_dirs: List[Path]) -> List[Path]:
    files = []

    valid_extensions = {
        ".txt",
        ".csv",
        ".xml",
        ".json",
        ".ann",
        ".data",
    }

    for directory in annotation_dirs:
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in valid_extensions:
                files.append(path)

    return sorted(files)


def annotation_file_score(video_path: Path, annotation_path: Path) -> int:
    """
    Score how likely an annotation file belongs to the video.
    """
    score = 0

    video_stem = normalize_text(video_path.stem)
    ann_stem = normalize_text(annotation_path.stem)

    video_number = infer_le2i_video_number(video_path)

    if video_stem == ann_stem:
        score += 100

    if video_stem in ann_stem or ann_stem in video_stem:
        score += 50

    if video_number is not None:
        ann_numbers = [int(x) for x in re.findall(r"\d+", ann_stem)]

        if video_number in ann_numbers:
            score += 30

        if ann_numbers and video_number == ann_numbers[-1]:
            score += 20

    return score


def find_le2i_annotation_file(video_path: Path, le2i_root: Path) -> Optional[Path]:
    scene = infer_le2i_scene(video_path, le2i_root)

    if not scene:
        return None

    scene_dir = le2i_root / scene

    annotation_dirs = find_annotation_dirs(scene_dir)

    if not annotation_dirs:
        # Some packages put Annotation_files next to the scene folders.
        annotation_dirs = find_annotation_dirs(le2i_root)

    annotation_files = list_annotation_files(annotation_dirs)

    if not annotation_files:
        return None

    scored = []

    for ann_path in annotation_files:
        score = annotation_file_score(video_path, ann_path)

        if score > 0:
            scored.append((score, ann_path))

    if not scored:
        return None

    scored = sorted(scored, key=lambda item: item[0], reverse=True)

    return scored[0][1]


def read_text_file_flexible(path: Path) -> str:
    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin-1",
        "cp1252",
    ]

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue

    with open(path, "rb") as f:
        data = f.read()

    return data.decode("utf-8", errors="ignore")


def parse_le2i_fall_intervals(annotation_path: Path) -> List[Tuple[int, int]]:
    """
    Parse fall intervals from Le2i annotation file.

    This parser is intentionally flexible because Le2i packages can store
    annotation files in slightly different text formats.

    Strategy:
        - Read each line.
        - Extract integer numbers.
        - If a line has at least two integers, use the first two as start/end.
        - Merge all intervals.
    """
    text = read_text_file_flexible(annotation_path)

    intervals = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        nums = re.findall(r"-?\d+", line)

        if len(nums) >= 2:
            start = int(nums[0])
            end = int(nums[1])

            if start >= 0 and end >= 0:
                intervals.append(normalize_interval(start, end))

    if not intervals:
        # Last fallback: use the first two integers in the whole file.
        nums = re.findall(r"-?\d+", text)

        if len(nums) >= 2:
            start = int(nums[0])
            end = int(nums[1])

            if start >= 0 and end >= 0:
                intervals.append(normalize_interval(start, end))

    return merge_intervals(intervals)


def build_le2i_records_for_video(
    video_path: Path,
    le2i_root: Path,
    dataset_name: str = "Le2i",
    min_not_fall_length: int = 15,
) -> List[Dict]:
    scene = infer_le2i_scene(video_path, le2i_root)

    video_id = make_video_id(
        dataset_name=dataset_name,
        video_path=video_path,
        base_dir=le2i_root,
    )

    probe = probe_video(video_path)
    frame_count = int(probe.get("frame_count", 0))

    if frame_count <= 0:
        return [
            {
                "dataset": dataset_name,
                "scene": scene,
                "video_id": video_id,
                "video_path": str(video_path),
                "label": -1,
                "label_name": "Unknown",
                "has_annotation": False,
                "include_eval": False,
                "label_source": "video_unreadable",
                "segment_id": "unreadable",
                "segment_start_frame": np.nan,
                "segment_end_frame": np.nan,
                "frame_count": 0,
                "fps": 0.0,
                "width": 0,
                "height": 0,
                "duration_sec": 0.0,
                "notes": "Cannot read video.",
            }
        ]

    annotation_path = find_le2i_annotation_file(video_path, le2i_root)

    if annotation_path is None:
        return [
            {
                "dataset": dataset_name,
                "scene": scene,
                "video_id": video_id,
                "video_path": str(video_path),
                "label": -1,
                "label_name": "Unknown",
                "has_annotation": False,
                "include_eval": False,
                "label_source": "missing_annotation",
                "segment_id": "missing_annotation",
                "segment_start_frame": np.nan,
                "segment_end_frame": np.nan,
                "frame_count": int(probe.get("frame_count", 0)),
                "fps": float(probe.get("fps", 0.0)),
                "width": int(probe.get("width", 0)),
                "height": int(probe.get("height", 0)),
                "duration_sec": float(probe.get("duration_sec", 0.0)),
                "notes": "No matching Le2i annotation file found.",
            }
        ]

    fall_intervals = parse_le2i_fall_intervals(annotation_path)
    fall_intervals = clamp_intervals(fall_intervals, frame_count)

    if not fall_intervals:
        return [
            {
                "dataset": dataset_name,
                "scene": scene,
                "video_id": video_id,
                "video_path": str(video_path),
                "label": -1,
                "label_name": "Unknown",
                "has_annotation": False,
                "include_eval": False,
                "label_source": str(annotation_path),
                "segment_id": "annotation_parse_failed",
                "segment_start_frame": np.nan,
                "segment_end_frame": np.nan,
                "frame_count": int(probe.get("frame_count", 0)),
                "fps": float(probe.get("fps", 0.0)),
                "width": int(probe.get("width", 0)),
                "height": int(probe.get("height", 0)),
                "duration_sec": float(probe.get("duration_sec", 0.0)),
                "notes": "Annotation file found but no valid interval parsed.",
            }
        ]

    records = []

    not_fall_intervals = complement_intervals(
        fall_intervals=fall_intervals,
        frame_count=frame_count,
        min_length=min_not_fall_length,
    )

    for idx, (start, end) in enumerate(not_fall_intervals):
        records.append(
            build_segment_record(
                dataset=dataset_name,
                video_path=video_path,
                video_id=video_id,
                label=LABEL_NOT_FALL,
                segment_id=f"not_fall_{idx}",
                start_frame=start,
                end_frame=end,
                label_source=str(annotation_path),
                scene=scene,
                notes="Le2i Not_Fall interval generated as complement of fall interval.",
            )
        )

    for idx, (start, end) in enumerate(fall_intervals):
        records.append(
            build_segment_record(
                dataset=dataset_name,
                video_path=video_path,
                video_id=video_id,
                label=LABEL_FALL,
                segment_id=f"fall_{idx}",
                start_frame=start,
                end_frame=end,
                label_source=str(annotation_path),
                scene=scene,
                notes="Le2i Fall interval parsed from annotation file.",
            )
        )

    return records


def build_le2i_metadata(config: Dict) -> pd.DataFrame:
    dataset_cfg = config["datasets"]["Le2i"]

    dataset_name = dataset_cfg.get("dataset_name", "Le2i")
    raw_dir = cfg_path(config, dataset_cfg["raw_dir"])

    include_scenes = set(dataset_cfg.get("include_scenes", []))
    exclude_scenes = set(dataset_cfg.get("exclude_scenes", []))

    video_extensions = dataset_cfg.get("video_extensions", [".avi", ".mp4", ".mov", ".mkv"])

    videos = list_video_files(raw_dir, video_extensions)

    records = []

    for video_path in videos:
        scene = infer_le2i_scene(video_path, raw_dir)

        if include_scenes and scene not in include_scenes:
            continue

        if scene in exclude_scenes:
            continue

        records.extend(
            build_le2i_records_for_video(
                video_path=video_path,
                le2i_root=raw_dir,
                dataset_name=dataset_name,
            )
        )

    df = pd.DataFrame(records)

    return df


# ============================================================
# MULCAMFALL HELPERS
# ============================================================

def infer_mulcam_chute(video_path: Path) -> Optional[int]:
    parts = [normalize_text(part) for part in video_path.parts]

    for part in reversed(parts):
        match = re.search(r"chute[_]?(\d+)", part)

        if match:
            return int(match.group(1))

    return None


def infer_mulcam_camera(video_path: Path) -> Optional[int]:
    stem = normalize_text(video_path.stem)

    match = re.search(r"cam[_]?(\d+)", stem)

    if match:
        return int(match.group(1))

    return None


def build_mulcam_video_index(raw_dir: Path, video_extensions: List[str]) -> Dict[Tuple[int, int], Path]:
    videos = list_video_files(raw_dir, video_extensions)

    index = {}

    for video_path in videos:
        chute = infer_mulcam_chute(video_path)
        cam = infer_mulcam_camera(video_path)

        if chute is None or cam is None:
            continue

        index[(int(chute), int(cam))] = video_path

    return index


def standardize_mulcam_label(raw_label: Any, dataset_cfg: Dict) -> int:
    """
    MulCamFall data_tuple3.csv in this package usually already has:
        label = 0 or 1

    If the label column is not binary, we fall back to position-code mapping
    from the config.
    """
    label_int = safe_int(raw_label, None)

    if label_int in [0, 1]:
        return label_int

    fall_codes = set(int(x) for x in dataset_cfg.get("fall_position_codes", [2, 3]))
    not_fall_codes = set(int(x) for x in dataset_cfg.get("not_fall_position_codes", [1, 4, 5, 6, 7, 8, 9]))

    if label_int in fall_codes:
        return LABEL_FALL

    if label_int in not_fall_codes:
        return LABEL_NOT_FALL

    raise ValueError(f"Unknown MulCamFall label/code: {raw_label}")


def build_mulcamfall_metadata(config: Dict) -> pd.DataFrame:
    dataset_cfg = config["datasets"]["MulCamFall"]

    dataset_name = dataset_cfg.get("dataset_name", "MulCamFall")
    raw_dir = cfg_path(config, dataset_cfg["raw_dir"])
    annotation_csv = cfg_path(config, dataset_cfg["annotation_csv"])
    video_extensions = dataset_cfg.get("video_extensions", [".avi", ".mp4", ".mov", ".mkv"])

    if not annotation_csv.exists():
        raise FileNotFoundError(f"MulCamFall annotation CSV not found: {annotation_csv}")

    ann_df = pd.read_csv(annotation_csv)

    required_cols = ["chute", "cam", "start", "end", "label"]
    missing = [col for col in required_cols if col not in ann_df.columns]

    if missing:
        raise ValueError(f"MulCamFall annotation CSV missing columns: {missing}")

    video_index = build_mulcam_video_index(raw_dir, video_extensions)

    records = []

    for row_idx, row in ann_df.iterrows():
        chute = safe_int(row["chute"])
        cam = safe_int(row["cam"])
        start = safe_int(row["start"])
        end = safe_int(row["end"])
        label = standardize_mulcam_label(row["label"], dataset_cfg)

        if chute is None or cam is None or start is None or end is None:
            continue

        video_path = video_index.get((chute, cam), None)

        if video_path is None:
            records.append(
                {
                    "dataset": dataset_name,
                    "scene": f"chute{chute:02d}",
                    "scenario": f"chute{chute:02d}",
                    "camera": f"cam{cam}",
                    "video_id": f"{normalize_text(dataset_name)}__chute{chute:02d}__cam{cam}",
                    "video_path": "",
                    "label": int(label),
                    "label_name": binary_label_name(label),
                    "has_annotation": True,
                    "include_eval": False,
                    "label_source": str(annotation_csv),
                    "segment_id": f"row_{row_idx}",
                    "segment_start_frame": int(start),
                    "segment_end_frame": int(end),
                    "frame_count": 0,
                    "fps": 0.0,
                    "width": 0,
                    "height": 0,
                    "duration_sec": 0.0,
                    "notes": "Video file not found for this chute/camera.",
                }
            )
            continue

        video_id = make_video_id(
            dataset_name=dataset_name,
            video_path=video_path,
            base_dir=raw_dir,
        )

        records.append(
            build_segment_record(
                dataset=dataset_name,
                video_path=video_path,
                video_id=video_id,
                label=label,
                segment_id=f"tuple_row_{row_idx}",
                start_frame=start,
                end_frame=end,
                label_source=str(annotation_csv),
                scene=f"chute{chute:02d}",
                scenario=f"chute{chute:02d}",
                camera=f"cam{cam}",
                notes="MulCamFall segment parsed from data_tuple3.csv.",
            )
        )

    df = pd.DataFrame(records)

    return df


# ============================================================
# COMBINED BUILDERS
# ============================================================

def build_all_external_metadata(config: Dict) -> pd.DataFrame:
    dfs = []

    datasets_cfg = config.get("datasets", {})

    if datasets_cfg.get("Le2i", {}).get("enabled", False):
        le2i_df = build_le2i_metadata(config)
        dfs.append(le2i_df)

    if datasets_cfg.get("MulCamFall", {}).get("enabled", False):
        mulcam_df = build_mulcamfall_metadata(config)
        dfs.append(mulcam_df)

    if not dfs:
        return pd.DataFrame()

    all_df = pd.concat(dfs, axis=0, ignore_index=True)

    if not all_df.empty:
        sort_cols = [
            col for col in [
                "dataset",
                "scene",
                "video_id",
                "segment_start_frame",
                "segment_end_frame",
            ]
            if col in all_df.columns
        ]

        all_df = all_df.sort_values(sort_cols).reset_index(drop=True)

    return all_df


def metadata_quality_report(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {
            "num_rows": 0,
            "error": "empty_metadata",
        }

    report = {
        "num_rows": int(len(df)),
        "num_eval_rows": int(df["include_eval"].astype(bool).sum()) if "include_eval" in df.columns else 0,
        "datasets": df["dataset"].value_counts(dropna=False).to_dict() if "dataset" in df.columns else {},
        "labels": df["label_name"].value_counts(dropna=False).to_dict() if "label_name" in df.columns else {},
        "has_annotation": df["has_annotation"].value_counts(dropna=False).to_dict() if "has_annotation" in df.columns else {},
        "include_eval": df["include_eval"].value_counts(dropna=False).to_dict() if "include_eval" in df.columns else {},
    }

    if "scene" in df.columns:
        report["scenes"] = df["scene"].value_counts(dropna=False).to_dict()

    if "notes" in df.columns:
        report["notes"] = df["notes"].value_counts(dropna=False).head(20).to_dict()

    return report


def save_metadata_outputs(config: Dict, all_df: pd.DataFrame) -> Dict:
    outputs = {}

    datasets_cfg = config.get("datasets", {})

    if "dataset" in all_df.columns:
        if datasets_cfg.get("Le2i", {}).get("enabled", False):
            le2i_df = all_df[all_df["dataset"] == datasets_cfg["Le2i"].get("dataset_name", "Le2i")].copy()
            le2i_output = cfg_path(config, datasets_cfg["Le2i"]["metadata_output"])
            save_csv(le2i_df, le2i_output)
            outputs["le2i_metadata"] = str(le2i_output)

        if datasets_cfg.get("MulCamFall", {}).get("enabled", False):
            mulcam_df = all_df[all_df["dataset"] == datasets_cfg["MulCamFall"].get("dataset_name", "MulCamFall")].copy()
            mulcam_output = cfg_path(config, datasets_cfg["MulCamFall"]["metadata_output"])
            save_csv(mulcam_df, mulcam_output)
            outputs["mulcamfall_metadata"] = str(mulcam_output)

    all_output = cfg_path(config, config["outputs"]["all_metadata_csv"])
    save_csv(all_df, all_output)
    outputs["all_external_metadata"] = str(all_output)

    report_path = cfg_path(config, config["outputs"]["preparation_dir"]) / "external_metadata_report.json"
    save_json(metadata_quality_report(all_df), report_path)
    outputs["metadata_report"] = str(report_path)

    return outputs


# ============================================================
# QUICK CHECK
# ============================================================

def quick_check_external_label_mapping(config_path: Optional[str] = None) -> None:
    if config_path is None:
        config = load_config()
    else:
        config = load_config(config_path)

    print("\nExternal Label Mapping Quick Check")
    print("=" * 100)

    print("Building Le2i metadata preview...")
    le2i_df = build_le2i_metadata(config)
    print(f"Le2i rows: {len(le2i_df)}")

    if not le2i_df.empty:
        print(le2i_df.head(10).to_string(index=False))
        print("\nLe2i label counts:")
        print(le2i_df["label_name"].value_counts(dropna=False).to_string())

    print("\nBuilding MulCamFall metadata preview...")
    mulcam_df = build_mulcamfall_metadata(config)
    print(f"MulCamFall rows: {len(mulcam_df)}")

    if not mulcam_df.empty:
        print(mulcam_df.head(10).to_string(index=False))
        print("\nMulCamFall label counts:")
        print(mulcam_df["label_name"].value_counts(dropna=False).to_string())

    all_df = pd.concat([le2i_df, mulcam_df], axis=0, ignore_index=True)

    print("\nCombined metadata report")
    print("=" * 100)
    print(json.dumps(metadata_quality_report(all_df), ensure_ascii=False, indent=4))
    print("=" * 100)


if __name__ == "__main__":
    quick_check_external_label_mapping()