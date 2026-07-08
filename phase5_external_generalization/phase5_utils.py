import os
import re
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch


try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "PyYAML is required for Phase 5 config loading. "
        "Install it with: pip install pyyaml"
    ) from exc


try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "OpenCV is required for video probing and frame reading. "
        "Install it with: pip install opencv-python"
    ) from exc


"""
Phase 5 Utilities - External Dataset Generalization.

Main principle:
    All models must be evaluated on the SAME external sequence manifest.

Why:
    Phase 1/2/3/4 models require different inputs:
        - Phase 1 2D        : 2D features only
        - Phase 2 3D        : 3D features only
        - Phase 2 Concat    : 2D + 3D
        - Phase 3 Gated     : 2D + 3D
        - Phase 4 Quality   : 2D + 3D + quality features

    If each model silently drops different failed videos/sequences,
    the comparison becomes unfair.

Therefore:
    1. Prepare external metadata once.
    2. Build sequence manifest once.
    3. Check feature availability.
    4. Build one common fair manifest.
    5. Every model must evaluate on that same common fair manifest.
"""


# ============================================================
# CONSTANTS
# ============================================================

PHASE5_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE5_DIR.parent
DEFAULT_CONFIG_PATH = PHASE5_DIR / "phase5_config.yaml"

SUPPORTED_VIDEO_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mov",
    ".mkv",
    ".mpeg",
    ".mpg",
}

BINARY_LABEL_TO_NAME = {
    0: "Not_Fall",
    1: "Fall",
}

BINARY_NAME_TO_LABEL = {
    "not_fall": 0,
    "not fall": 0,
    "normal": 0,
    "nonfall": 0,
    "non_fall": 0,
    "adl": 0,
    "fall": 1,
    "falling": 1,
}

REQUIRED_METADATA_COLUMNS = [
    "dataset",
    "video_id",
    "video_path",
    "label",
    "label_name",
    "has_annotation",
    "include_eval",
    "label_source",
]

OPTIONAL_METADATA_COLUMNS = [
    "scene",
    "scenario",
    "camera",
    "subject",
    "activity",
    "trial",
    "segment_id",
    "segment_start_frame",
    "segment_end_frame",
    "frame_count",
    "fps",
    "width",
    "height",
    "duration_sec",
    "notes",
]

REQUIRED_SEQUENCE_COLUMNS = [
    "dataset",
    "video_id",
    "sequence_key",
    "video_path",
    "label",
    "label_name",
    "start_frame",
    "end_frame",
    "sequence_length",
    "valid_frame_count",
    "include_eval",
]

FEATURE_AVAILABILITY_COLUMNS = [
    "sequence_key",
    "has_2d",
    "has_3d",
    "has_quality",
]


# ============================================================
# BASIC PATH / CONFIG HELPERS
# ============================================================

def resolve_path(path_value: Any, project_root: Optional[Path] = None) -> Path:
    """
    Resolve a path from config.

    Absolute paths are kept.
    Relative paths are resolved from project root.
    """
    if path_value is None:
        raise ValueError("Path value is None.")

    path_text = str(path_value).replace("\\", "/")
    path = Path(path_text)

    if path.is_absolute():
        return path

    root = project_root if project_root is not None else PROJECT_ROOT
    return root / path


def ensure_dir(path: Any) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(path: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_config(config_path: Any = DEFAULT_CONFIG_PATH) -> Dict:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Phase 5 config not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config is empty: {config_path}")

    return config


def get_project_root(config: Optional[Dict] = None) -> Path:
    if config is None:
        return PROJECT_ROOT

    project_cfg = config.get("project", {})
    root_value = project_cfg.get("root", None)

    if root_value:
        return Path(str(root_value))

    return PROJECT_ROOT


def cfg_path(config: Dict, path_value: Any) -> Path:
    return resolve_path(path_value, get_project_root(config))


def save_json(data: Dict, path: Any) -> None:
    path = ensure_parent_dir(path)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=4)


def load_json(path: Any) -> Dict:
    path = Path(path)

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: Any) -> None:
    path = ensure_parent_dir(path)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read_csv(path: Any) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    return pd.read_csv(path)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, list):
        return [to_jsonable(v) for v in value]

    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    return value


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config: Optional[Dict] = None) -> torch.device:
    if config is not None:
        device_name = str(config.get("runtime", {}).get("device", "cuda"))

        if device_name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# TEXT / ID HELPERS
# ============================================================

def normalize_text(text: Any) -> str:
    text = str(text).strip().lower()
    text = text.replace("-", "_")
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def make_safe_filename(text: Any, max_len: int = 160) -> str:
    text = str(text)
    text = text.replace("\\", "/")
    text = re.sub(r"[^a-zA-Z0-9._/-]+", "_", text)
    text = text.replace("/", "__")
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")

    if len(text) <= max_len:
        return text

    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    return text[: max_len - 11] + "_" + digest


def make_video_id(dataset_name: str, video_path: Any, base_dir: Any) -> str:
    video_path = Path(video_path)
    base_dir = Path(base_dir)

    try:
        rel = video_path.relative_to(base_dir)
    except ValueError:
        rel = video_path.name

    rel_no_ext = str(rel).replace("\\", "/")
    rel_no_ext = str(Path(rel_no_ext).with_suffix(""))
    safe_rel = make_safe_filename(rel_no_ext)

    return f"{normalize_text(dataset_name)}__{safe_rel}"


def make_sequence_key(
    dataset: str,
    video_id: str,
    start_frame: int,
    end_frame: int,
    segment_id: Optional[Any] = None,
) -> str:
    if segment_id is None or pd.isna(segment_id):
        segment_text = "seg0"
    else:
        segment_text = f"seg{segment_id}"

    return (
        f"{normalize_text(dataset)}__"
        f"{make_safe_filename(video_id)}__"
        f"{segment_text}__"
        f"s{int(start_frame):06d}_e{int(end_frame):06d}"
    )


# ============================================================
# LABEL HELPERS
# ============================================================

def standardize_binary_label(value: Any) -> int:
    """
    Convert label into:
        0 = Not_Fall
        1 = Fall
    """
    if value is None or pd.isna(value):
        raise ValueError("Label is missing.")

    if isinstance(value, (int, np.integer)):
        value_int = int(value)

        if value_int in [0, 1]:
            return value_int

    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        value_int = int(value)

        if value_int in [0, 1]:
            return value_int

    text = normalize_text(value)

    if text in BINARY_NAME_TO_LABEL:
        return BINARY_NAME_TO_LABEL[text]

    raise ValueError(f"Cannot standardize binary label: {value}")


def binary_label_name(label: Any) -> str:
    label = standardize_binary_label(label)
    return BINARY_LABEL_TO_NAME[label]


def bool_from_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if value is None or pd.isna(value):
        return False

    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value)

    text = str(value).strip().lower()

    return text in ["1", "true", "yes", "y", "ok", "include"]


# ============================================================
# VIDEO HELPERS
# ============================================================

def is_video_file(path: Any, extensions: Optional[List[str]] = None) -> bool:
    path = Path(path)

    if extensions is None:
        extensions_set = SUPPORTED_VIDEO_EXTENSIONS
    else:
        extensions_set = {ext.lower() for ext in extensions}

    return path.is_file() and path.suffix.lower() in extensions_set


def list_video_files(
    root_dir: Any,
    extensions: Optional[List[str]] = None,
) -> List[Path]:
    root_dir = Path(root_dir)

    if not root_dir.exists():
        raise FileNotFoundError(f"Video root directory not found: {root_dir}")

    videos = []

    for path in root_dir.rglob("*"):
        if is_video_file(path, extensions):
            videos.append(path)

    return sorted(videos)


def probe_video(video_path: Any) -> Dict:
    """
    Read basic metadata from a video.

    Returns:
        frame_count, fps, width, height, duration_sec, readable
    """
    video_path = Path(video_path)

    if not video_path.exists():
        return {
            "video_path": str(video_path),
            "readable": False,
            "error": "file_not_found",
            "frame_count": 0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "duration_sec": 0.0,
        }

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "video_path": str(video_path),
            "readable": False,
            "error": "cannot_open_video",
            "frame_count": 0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "duration_sec": 0.0,
        }

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps <= 1e-6:
        duration_sec = 0.0
    else:
        duration_sec = float(frame_count) / float(fps)

    return {
        "video_path": str(video_path),
        "readable": True,
        "error": "",
        "frame_count": int(frame_count),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "duration_sec": float(duration_sec),
    }


def read_video_frame(video_path: Any, frame_index: int):
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(
            f"Cannot read frame {frame_index} from video: {video_path}"
        )

    return frame


# ============================================================
# METADATA HELPERS
# ============================================================

def standardize_metadata_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize external metadata into a common format.

    Required final columns:
        dataset, video_id, video_path, label, label_name,
        has_annotation, include_eval, label_source
    """
    df = df.copy()

    for col in REQUIRED_METADATA_COLUMNS:
        if col not in df.columns:
            if col in ["has_annotation", "include_eval"]:
                df[col] = False
            elif col == "label":
                df[col] = -1
            elif col == "label_name":
                df[col] = "Unknown"
            else:
                df[col] = ""

    for col in OPTIONAL_METADATA_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["label"] = df["label"].apply(
        lambda x: standardize_binary_label(x) if not pd.isna(x) and str(x) != "-1" else -1
    )

    df["label_name"] = df["label"].apply(
        lambda x: BINARY_LABEL_TO_NAME[int(x)] if int(x) in BINARY_LABEL_TO_NAME else "Unknown"
    )

    df["has_annotation"] = df["has_annotation"].apply(bool_from_value)
    df["include_eval"] = df["include_eval"].apply(bool_from_value)

    df["video_path"] = df["video_path"].astype(str)

    return df


def validate_metadata_df(df: pd.DataFrame, strict: bool = True) -> Dict:
    df = df.copy()

    missing_cols = [
        col for col in REQUIRED_METADATA_COLUMNS
        if col not in df.columns
    ]

    errors = []
    warnings = []

    if missing_cols:
        errors.append(f"Missing metadata columns: {missing_cols}")

    if not missing_cols:
        invalid_labels = df[~df["label"].isin([0, 1, -1])]

        if len(invalid_labels) > 0:
            errors.append(
                f"Metadata contains invalid labels. Count={len(invalid_labels)}"
            )

        eval_df = df[df["include_eval"].apply(bool_from_value)]

        if eval_df.empty:
            warnings.append("No rows are marked include_eval=True.")

        eval_without_label = eval_df[~eval_df["label"].isin([0, 1])]

        if len(eval_without_label) > 0:
            errors.append(
                "Some include_eval=True rows have no valid binary label. "
                f"Count={len(eval_without_label)}"
            )

        duplicated_video_ids = df[df["video_id"].duplicated(keep=False)]

        if len(duplicated_video_ids) > 0:
            warnings.append(
                "Metadata has duplicated video_id values. "
                "This is allowed only for segment-level metadata."
            )

    result = {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "num_rows": int(len(df)),
    }

    if strict and errors:
        raise ValueError(json.dumps(result, ensure_ascii=False, indent=4))

    return result


def filter_eval_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_metadata_df(df)

    eval_df = df[
        (df["include_eval"] == True)
        & (df["has_annotation"] == True)
        & (df["label"].isin([0, 1]))
    ].copy()

    eval_df = eval_df.reset_index(drop=True)

    return eval_df


def add_video_probe_columns(df: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    """
    Add frame_count, fps, width, height, duration_sec, video_readable.
    """
    df = df.copy()

    probe_cols = [
        "frame_count",
        "fps",
        "width",
        "height",
        "duration_sec",
        "video_readable",
        "video_error",
    ]

    for col in probe_cols:
        if col not in df.columns:
            df[col] = np.nan

    for idx, row in df.iterrows():
        already_has_probe = (
            not overwrite
            and not pd.isna(row.get("frame_count", np.nan))
            and int(row.get("frame_count", 0)) > 0
        )

        if already_has_probe:
            continue

        probe = probe_video(row["video_path"])

        df.at[idx, "frame_count"] = int(probe["frame_count"])
        df.at[idx, "fps"] = float(probe["fps"])
        df.at[idx, "width"] = int(probe["width"])
        df.at[idx, "height"] = int(probe["height"])
        df.at[idx, "duration_sec"] = float(probe["duration_sec"])
        df.at[idx, "video_readable"] = bool(probe["readable"])
        df.at[idx, "video_error"] = str(probe.get("error", ""))

    return df


def metadata_summary(df: pd.DataFrame) -> Dict:
    df = standardize_metadata_df(df)

    summary = {
        "num_rows": int(len(df)),
        "num_eval_rows": int(df["include_eval"].sum()),
        "num_annotated_rows": int(df["has_annotation"].sum()),
        "datasets": df["dataset"].value_counts(dropna=False).to_dict(),
        "labels": df["label_name"].value_counts(dropna=False).to_dict(),
    }

    if "scene" in df.columns:
        summary["scenes"] = df["scene"].value_counts(dropna=False).to_dict()

    if "video_readable" in df.columns:
        summary["readable"] = df["video_readable"].value_counts(dropna=False).to_dict()

    return summary


# ============================================================
# SEQUENCE MANIFEST HELPERS
# ============================================================

def to_int_or_none(value: Any) -> Optional[int]:
    if value is None or pd.isna(value):
        return None

    try:
        return int(float(value))
    except Exception:
        return None


def build_sequence_windows(
    total_frames: int,
    sequence_length: int,
    stride: int,
    include_short: bool = True,
) -> List[Tuple[int, int, int]]:
    """
    Build windows in local coordinates.

    Returns:
        [(local_start, local_end, valid_frame_count), ...]

    Fairness note:
        The same function must be used for every model.
    """
    total_frames = int(total_frames)
    sequence_length = int(sequence_length)
    stride = int(stride)

    if total_frames <= 0:
        return []

    if total_frames < sequence_length:
        if not include_short:
            return []

        return [(0, total_frames - 1, total_frames)]

    windows = []

    max_start = total_frames - sequence_length

    for start in range(0, max_start + 1, stride):
        end = start + sequence_length - 1
        windows.append((start, end, sequence_length))

    if not windows and include_short:
        windows.append((0, total_frames - 1, total_frames))

    return windows


def build_sequence_manifest(
    metadata_df: pd.DataFrame,
    config: Dict,
    dataset_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build the official Phase 5 sequence manifest.

    This is the most important fairness file.

    All models must evaluate on rows from this manifest or from
    a common filtered version of this manifest.
    """
    metadata_df = standardize_metadata_df(metadata_df)
    eval_df = filter_eval_metadata(metadata_df)

    if dataset_filter is not None:
        eval_df = eval_df[eval_df["dataset"].isin(dataset_filter)].copy()

    sequence_cfg = config.get("sequence", {})

    sequence_length = int(sequence_cfg.get("sequence_length", 60))
    stride = int(sequence_cfg.get("stride", 15))
    include_short = bool(sequence_cfg.get("include_short", True))

    rows = []

    for _, row in eval_df.iterrows():
        video_path = str(row["video_path"])
        video_id = str(row["video_id"])
        dataset = str(row["dataset"])

        frame_count = to_int_or_none(row.get("frame_count", None))

        if frame_count is None or frame_count <= 0:
            probe = probe_video(video_path)
            frame_count = int(probe["frame_count"])

        if frame_count <= 0:
            continue

        segment_start = to_int_or_none(row.get("segment_start_frame", None))
        segment_end = to_int_or_none(row.get("segment_end_frame", None))

        if segment_start is None:
            segment_start = 0

        if segment_end is None:
            segment_end = frame_count - 1

        segment_start = max(0, int(segment_start))
        segment_end = min(frame_count - 1, int(segment_end))

        if segment_end < segment_start:
            continue

        segment_frame_count = segment_end - segment_start + 1

        windows = build_sequence_windows(
            total_frames=segment_frame_count,
            sequence_length=sequence_length,
            stride=stride,
            include_short=include_short,
        )

        segment_id = row.get("segment_id", 0)

        for local_start, local_end, valid_count in windows:
            start_frame = segment_start + local_start
            end_frame = segment_start + local_end

            sequence_key = make_sequence_key(
                dataset=dataset,
                video_id=video_id,
                start_frame=start_frame,
                end_frame=end_frame,
                segment_id=segment_id,
            )

            rows.append(
                {
                    "dataset": dataset,
                    "scene": row.get("scene", ""),
                    "scenario": row.get("scenario", ""),
                    "camera": row.get("camera", ""),
                    "subject": row.get("subject", ""),
                    "activity": row.get("activity", ""),
                    "trial": row.get("trial", ""),
                    "video_id": video_id,
                    "sequence_key": sequence_key,
                    "video_path": video_path,
                    "label": int(row["label"]),
                    "label_name": binary_label_name(row["label"]),
                    "label_source": row.get("label_source", ""),
                    "segment_id": segment_id,
                    "segment_start_frame": int(segment_start),
                    "segment_end_frame": int(segment_end),
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "sequence_length": int(sequence_length),
                    "valid_frame_count": int(valid_count),
                    "frame_count": int(frame_count),
                    "include_eval": True,
                }
            )

    manifest_df = pd.DataFrame(rows)

    if not manifest_df.empty:
        manifest_df = manifest_df.sort_values(
            by=["dataset", "video_id", "start_frame", "end_frame"]
        ).reset_index(drop=True)

    return manifest_df


def validate_sequence_manifest(manifest_df: pd.DataFrame, strict: bool = True) -> Dict:
    missing_cols = [
        col for col in REQUIRED_SEQUENCE_COLUMNS
        if col not in manifest_df.columns
    ]

    errors = []
    warnings = []

    if missing_cols:
        errors.append(f"Missing sequence manifest columns: {missing_cols}")

    if not missing_cols:
        duplicated = manifest_df[
            manifest_df["sequence_key"].duplicated(keep=False)
        ]

        if len(duplicated) > 0:
            errors.append(
                f"Duplicated sequence_key found. Count={len(duplicated)}"
            )

        invalid_labels = manifest_df[~manifest_df["label"].isin([0, 1])]

        if len(invalid_labels) > 0:
            errors.append(
                f"Invalid sequence labels found. Count={len(invalid_labels)}"
            )

        invalid_lengths = manifest_df[
            manifest_df["valid_frame_count"].astype(int) <= 0
        ]

        if len(invalid_lengths) > 0:
            errors.append(
                f"Invalid valid_frame_count rows. Count={len(invalid_lengths)}"
            )

        if manifest_df.empty:
            errors.append("Sequence manifest is empty.")

        label_counts = manifest_df["label_name"].value_counts().to_dict()

        if len(label_counts) < 2:
            warnings.append(
                f"Only one class found in manifest: {label_counts}. "
                "Metrics like Macro F1 may be misleading."
            )

    result = {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "num_sequences": int(len(manifest_df)),
    }

    if not manifest_df.empty and "dataset" in manifest_df.columns:
        result["datasets"] = manifest_df["dataset"].value_counts().to_dict()
        result["labels"] = manifest_df["label_name"].value_counts().to_dict()

    if strict and errors:
        raise ValueError(json.dumps(result, ensure_ascii=False, indent=4))

    return result


def sequence_manifest_summary(manifest_df: pd.DataFrame) -> Dict:
    summary = {
        "num_sequences": int(len(manifest_df)),
        "num_videos": int(manifest_df["video_id"].nunique()) if "video_id" in manifest_df.columns else 0,
        "datasets": manifest_df["dataset"].value_counts().to_dict() if "dataset" in manifest_df.columns else {},
        "labels": manifest_df["label_name"].value_counts().to_dict() if "label_name" in manifest_df.columns else {},
    }

    if "valid_frame_count" in manifest_df.columns:
        summary["valid_frame_count_min"] = int(manifest_df["valid_frame_count"].min())
        summary["valid_frame_count_max"] = int(manifest_df["valid_frame_count"].max())
        summary["valid_frame_count_mean"] = float(manifest_df["valid_frame_count"].mean())

    return summary


def save_sequence_manifest(config: Dict, manifest_df: pd.DataFrame, filename: str = "all_external_sequences.csv") -> Path:
    output_dir = cfg_path(config, config["outputs"]["external_sequences_dir"])
    ensure_dir(output_dir)

    output_path = output_dir / filename
    save_csv(manifest_df, output_path)

    return output_path


def load_sequence_manifest(config: Dict, filename: str = "all_external_sequences.csv") -> pd.DataFrame:
    input_dir = cfg_path(config, config["outputs"]["external_sequences_dir"])
    input_path = input_dir / filename

    return read_csv(input_path)


# ============================================================
# FAIR FEATURE AVAILABILITY HELPERS
# ============================================================

def validate_feature_availability_df(df: pd.DataFrame, strict: bool = True) -> Dict:
    missing_cols = [
        col for col in FEATURE_AVAILABILITY_COLUMNS
        if col not in df.columns
    ]

    errors = []

    if missing_cols:
        errors.append(f"Missing feature availability columns: {missing_cols}")

    if not missing_cols:
        duplicated = df[df["sequence_key"].duplicated(keep=False)]

        if len(duplicated) > 0:
            errors.append(
                f"Duplicated sequence_key in feature availability. Count={len(duplicated)}"
            )

    result = {
        "valid": len(errors) == 0,
        "errors": errors,
        "num_rows": int(len(df)),
    }

    if strict and errors:
        raise ValueError(json.dumps(result, ensure_ascii=False, indent=4))

    return result


def build_fair_common_manifest(
    sequence_manifest_df: pd.DataFrame,
    feature_availability_df: pd.DataFrame,
    require_2d: bool = True,
    require_3d: bool = True,
    require_quality: bool = True,
) -> pd.DataFrame:
    """
    Build the final fair manifest.

    This prevents the old mistake:
        Model A tests on N samples.
        Model B tests on M samples.
        Then the comparison is unfair.

    For all 6 Phase 1-4 models, the safest setting is:
        require_2d=True
        require_3d=True
        require_quality=True

    That means every kept sequence can be used by every model.
    """
    validate_sequence_manifest(sequence_manifest_df, strict=True)
    validate_feature_availability_df(feature_availability_df, strict=True)

    manifest = sequence_manifest_df.copy()
    availability = feature_availability_df.copy()

    availability["has_2d"] = availability["has_2d"].apply(bool_from_value)
    availability["has_3d"] = availability["has_3d"].apply(bool_from_value)
    availability["has_quality"] = availability["has_quality"].apply(bool_from_value)

    merged = manifest.merge(
        availability,
        on="sequence_key",
        how="left",
        validate="one_to_one",
    )

    for col in ["has_2d", "has_3d", "has_quality"]:
        if col not in merged.columns:
            merged[col] = False

        merged[col] = merged[col].fillna(False).apply(bool_from_value)

    keep_mask = pd.Series(True, index=merged.index)

    if require_2d:
        keep_mask &= merged["has_2d"]

    if require_3d:
        keep_mask &= merged["has_3d"]

    if require_quality:
        keep_mask &= merged["has_quality"]

    fair_df = merged[keep_mask].copy()
    fair_df = fair_df.reset_index(drop=True)

    validate_sequence_manifest(fair_df, strict=True)

    return fair_df


def save_feature_availability(config: Dict, df: pd.DataFrame, filename: str = "feature_availability.csv") -> Path:
    output_dir = cfg_path(config, config["outputs"]["external_sequences_dir"])
    ensure_dir(output_dir)

    output_path = output_dir / filename
    save_csv(df, output_path)

    return output_path


def load_feature_availability(config: Dict, filename: str = "feature_availability.csv") -> pd.DataFrame:
    input_dir = cfg_path(config, config["outputs"]["external_sequences_dir"])
    input_path = input_dir / filename

    return read_csv(input_path)


# ============================================================
# FAIR EVALUATION / PREDICTION HELPERS
# ============================================================

def validate_prediction_df(
    prediction_df: pd.DataFrame,
    fair_manifest_df: pd.DataFrame,
    model_name: str = "model",
    strict: bool = True,
) -> Dict:
    """
    Validate that a model prediction file covers exactly the fair manifest.

    Required prediction columns:
        sequence_key, y_true, y_pred
    """
    required_cols = ["sequence_key", "y_true", "y_pred"]

    errors = []

    for col in required_cols:
        if col not in prediction_df.columns:
            errors.append(f"{model_name}: missing prediction column {col}")

    if not errors:
        pred_keys = set(prediction_df["sequence_key"].astype(str).tolist())
        fair_keys = set(fair_manifest_df["sequence_key"].astype(str).tolist())

        missing_keys = fair_keys - pred_keys
        extra_keys = pred_keys - fair_keys

        if missing_keys:
            errors.append(
                f"{model_name}: missing predictions for {len(missing_keys)} fair sequences."
            )

        if extra_keys:
            errors.append(
                f"{model_name}: prediction contains {len(extra_keys)} extra sequences not in fair manifest."
            )

        duplicated = prediction_df[
            prediction_df["sequence_key"].duplicated(keep=False)
        ]

        if len(duplicated) > 0:
            errors.append(
                f"{model_name}: duplicated sequence_key in predictions. Count={len(duplicated)}"
            )

    result = {
        "valid": len(errors) == 0,
        "errors": errors,
        "num_predictions": int(len(prediction_df)),
        "num_fair_sequences": int(len(fair_manifest_df)),
    }

    if strict and errors:
        raise ValueError(json.dumps(result, ensure_ascii=False, indent=4))

    return result


def assert_all_models_same_sequence_set(
    prediction_dfs: Dict[str, pd.DataFrame],
    fair_manifest_df: pd.DataFrame,
) -> Dict:
    """
    Check that every model prediction uses exactly the same sequence set.
    """
    results = {}

    for model_name, pred_df in prediction_dfs.items():
        results[model_name] = validate_prediction_df(
            prediction_df=pred_df,
            fair_manifest_df=fair_manifest_df,
            model_name=model_name,
            strict=False,
        )

    invalid = {
        model: info
        for model, info in results.items()
        if not info["valid"]
    }

    if invalid:
        raise ValueError(
            "Unfair evaluation detected. Some models do not cover the same sequence set:\n"
            + json.dumps(invalid, ensure_ascii=False, indent=4)
        )

    return results


def compute_binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report,
    )

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    labels = [0, 1]
    class_names = ["Not_Fall", "Fall"]

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    result = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "accuracy_percent": float(accuracy) * 100.0,
        "macro_f1_percent": float(macro_f1) * 100.0,
        "not_fall_precision": float(precision[0]),
        "not_fall_recall": float(recall[0]),
        "not_fall_f1": float(f1[0]),
        "not_fall_support": int(support[0]),
        "fall_precision": float(precision[1]),
        "fall_recall": float(recall[1]),
        "fall_f1": float(f1[1]),
        "fall_support": int(support[1]),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    return result


# ============================================================
# FEATURE PATH HELPERS
# ============================================================

def get_2d_feature_path(config: Dict, dataset: str, video_id: str) -> Path:
    base_dir = cfg_path(config, config["outputs"]["extracted_2d_dir"])
    return base_dir / dataset / f"{make_safe_filename(video_id)}_2d_confidence.csv"


def get_3d_feature_path(config: Dict, dataset: str, video_id: str) -> Path:
    base_dir = cfg_path(config, config["outputs"]["normalized_3d_dir"])
    return base_dir / dataset / f"{make_safe_filename(video_id)}_normalized_3d.csv"


def get_quality_feature_path(config: Dict, dataset: str, video_id: str) -> Path:
    base_dir = cfg_path(config, config["outputs"]["quality_features_dir"])
    return base_dir / dataset / f"{make_safe_filename(video_id)}_quality.csv"


def check_video_level_feature_availability(
    config: Dict,
    sequence_manifest_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build feature availability by checking whether feature files exist per video_id.

    Later scripts may replace this with stricter frame-level checks.
    """
    rows = []

    for _, row in sequence_manifest_df.iterrows():
        dataset = str(row["dataset"])
        video_id = str(row["video_id"])

        path_2d = get_2d_feature_path(config, dataset, video_id)
        path_3d = get_3d_feature_path(config, dataset, video_id)
        path_quality = get_quality_feature_path(config, dataset, video_id)

        rows.append(
            {
                "sequence_key": row["sequence_key"],
                "dataset": dataset,
                "video_id": video_id,
                "has_2d": path_2d.exists(),
                "has_3d": path_3d.exists(),
                "has_quality": path_quality.exists(),
                "path_2d": str(path_2d),
                "path_3d": str(path_3d),
                "path_quality": str(path_quality),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PRINTING HELPERS
# ============================================================

def print_dict(title: str, data: Dict) -> None:
    print("\n" + title)
    print("=" * 100)
    print(json.dumps(to_jsonable(data), ensure_ascii=False, indent=4))
    print("=" * 100)


def print_dataframe_summary(title: str, df: pd.DataFrame, max_rows: int = 10) -> None:
    print("\n" + title)
    print("=" * 100)
    print(f"Shape: {df.shape}")

    if len(df) > 0:
        print(df.head(max_rows).to_string(index=False))

    print("=" * 100)


# ============================================================
# QUICK SELF CHECK
# ============================================================

def quick_check_phase5_utils(config_path: Any = DEFAULT_CONFIG_PATH) -> None:
    config = load_config(config_path)

    print("\nPhase 5 Utils Quick Check")
    print("=" * 100)
    print(f"Project root : {get_project_root(config)}")
    print(f"Phase 5 dir  : {PHASE5_DIR}")
    print(f"Config path  : {Path(config_path).resolve()}")
    print(f"Device       : {get_device(config)}")
    print("=" * 100)

    datasets = config.get("datasets", {})

    for dataset_name, dataset_cfg in datasets.items():
        if not dataset_cfg.get("enabled", False):
            continue

        raw_dir = cfg_path(config, dataset_cfg["raw_dir"])
        exists = raw_dir.exists()

        print(f"\nDataset: {dataset_name}")
        print(f"Raw dir : {raw_dir}")
        print(f"Exists  : {exists}")

        if exists:
            extensions = dataset_cfg.get("video_extensions", list(SUPPORTED_VIDEO_EXTENSIONS))
            videos = list_video_files(raw_dir, extensions)
            print(f"Videos  : {len(videos)}")

            for video in videos[:5]:
                print(f"  - {video}")

    print("\nQuick check done.")


if __name__ == "__main__":
    quick_check_phase5_utils()