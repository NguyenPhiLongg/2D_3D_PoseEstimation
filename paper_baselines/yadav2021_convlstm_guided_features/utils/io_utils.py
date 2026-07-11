from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict, path: Path) -> None:
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def read_csv_safe(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e

    raise last_error


def list_csv_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []

    return sorted([
        p for p in folder.rglob("*.csv")
        if p.is_file()
    ])


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


def label_to_name(label: int) -> str:
    label = int(label)

    if label == 1:
        return "Fall"

    if label == 0:
        return "Not_Fall"

    return "Unknown"


def infer_label_from_path(path: Path) -> Optional[int]:
    parts = [p.lower() for p in path.parts]

    if any(p in ["fall", "falls"] for p in parts):
        return 1

    if any(p in ["not_fall", "notfall", "non_fall", "nonfall", "normal"] for p in parts):
        return 0

    return None


def infer_label_from_df_or_path(df: pd.DataFrame, path: Path) -> int:
    if "label" in df.columns:
        value = pd.to_numeric(df["label"].iloc[0], errors="coerce")

        if not pd.isna(value):
            value = int(value)

            if value in [0, 1]:
                return value

    inferred = infer_label_from_path(path)

    if inferred is not None:
        return inferred

    raise ValueError(f"Cannot infer binary label from CSV or path: {path}")


def get_video_id_from_df_or_path(df: pd.DataFrame, path: Path) -> str:
    for col in ["video_id", "video", "file", "filename", "source_file"]:
        if col in df.columns:
            value = str(df[col].iloc[0])

            if value and value.lower() != "nan":
                return make_safe_name(Path(value).stem)

    return make_safe_name(path.stem)


def resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return project_root / path

