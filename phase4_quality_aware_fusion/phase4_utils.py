import os
import json
import random
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight


"""
Phase 4 shared utilities.

This file provides reusable functions/classes for:
    - loading Phase 4 quality feature sequences
    - loading 2D confidence CSV files
    - loading normalized 3D pose CSV files
    - creating 2D, 3D, and quality tensors
    - filtering by Phase 3 train/val/test split
    - padding short sequences to keep the same test videos as Phase 3
    - creating PyTorch datasets and dataloaders
    - training/evaluation helper functions

Main idea:
    Each sample contains:
        x2d:     [T, 40]
        x3d:     [T, 59]
        quality: [Q]
        label:   int

    where:
        T = fixed sequence length, usually 60
        Q = number of quality features, usually 33

Important:
    Phase 4 must not drop short videos if we want a fair comparison with Phase 3.
    Short sequences are padded by repeating the last frame until they reach 60 frames.
"""


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PHASE3_DIR = os.path.join(PROJECT_ROOT, "phase3_common_set_gated_fusion")
PHASE4_DIR = os.path.join(PROJECT_ROOT, "phase4_quality_aware_fusion")

DEFAULT_QUALITY_CSV = os.path.join(
    PROJECT_ROOT,
    "data",
    "6_pose_quality_features",
    "quality_sequences.csv",
)

DEFAULT_COMMON_SPLIT_DIR = os.path.join(
    PHASE3_DIR,
    "outputs",
    "common_split",
)

DEFAULT_TRAIN_SPLIT = os.path.join(DEFAULT_COMMON_SPLIT_DIR, "train_videos.txt")
DEFAULT_VAL_SPLIT = os.path.join(DEFAULT_COMMON_SPLIT_DIR, "val_videos.txt")
DEFAULT_TEST_SPLIT = os.path.join(DEFAULT_COMMON_SPLIT_DIR, "test_videos.txt")

DEFAULT_CHECKPOINT_DIR = os.path.join(PHASE4_DIR, "checkpoints")
DEFAULT_OUTPUT_DIR = os.path.join(PHASE4_DIR, "outputs", "training_quality_gated")

DEFAULT_SEQUENCE_LENGTH = 60

BINARY_CLASS_NAMES = ["Not_Fall", "Fall"]
ACTION_CLASS_NAMES = ["Sitting", "Sleeping", "Standing", "Walking"]

BINARY_LABEL_TO_ID = {
    "not_fall": 0,
    "not fall": 0,
    "notfall": 0,
    "normal": 0,
    "non_fall": 0,
    "non-fall": 0,
    "0": 0,
    0: 0,
    "fall": 1,
    "1": 1,
    1: 1,
}

ACTION_LABEL_TO_ID = {
    "sitting": 0,
    "sit": 0,
    "sleeping": 1,
    "sleep": 1,
    "standing": 2,
    "stand": 2,
    "walking": 3,
    "walk": 3,
}

COCO_BONES = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (5, 11),
    (6, 12),
]

QUALITY_FEATURE_COLUMNS = [
    "mean_confidence",
    "min_confidence",
    "std_confidence",
    "missing_joint_ratio",
    "low_conf_01_ratio",
    "low_conf_02_ratio",
    "low_conf_03_ratio",
    "low_conf_05_ratio",
    "mean_bbox_conf",
    "min_bbox_conf",
    "multi_person_ratio",
    "bbox_aspect_mean",
    "bbox_aspect_change",
    "bbox_area_mean",
    "bbox_area_change",
    "bbox_center_velocity_mean",
    "bbox_center_velocity_std",
    "temporal_jitter",
    "velocity_mean",
    "velocity_std",
    "bone_length_mean",
    "bone_length_std",
    "bone_length_cv",
    "3d_velocity_mean",
    "3d_velocity_std",
    "3d_bone_length_mean",
    "3d_bone_length_std",
    "3d_bone_length_cv",
    "3d_z_mean",
    "3d_z_std",
    "3d_z_velocity_mean",
    "3d_z_velocity_std",
    "3d_z_instability",
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(data: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=4)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, list):
        return [to_jsonable(v) for v in value]

    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()

    return value


def normalize_key(value: Any) -> str:
    text = str(value).strip().replace("\\", "/")
    base = os.path.basename(text)

    if base.lower().endswith(".csv"):
        base = os.path.splitext(base)[0]

    return base.lower()


def read_split_keys(path: str) -> set:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}")

    keys = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()

            if not value:
                continue

            keys.add(normalize_key(value))

    return keys


def load_quality_dataframe(path: str = DEFAULT_QUALITY_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Quality sequence CSV not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Quality sequence CSV is empty: {path}")

    required = [
        "video_key",
        "csv_file",
        "sequence_index",
        "start_index",
        "end_index",
        "path_2d_conf",
        "path_3d",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"quality_sequences.csv missing columns: {missing}")

    return df


def filter_by_split(df: pd.DataFrame, split_keys: set) -> pd.DataFrame:
    video_keys = df["video_key"].apply(normalize_key)

    if "csv_file" in df.columns:
        csv_keys = df["csv_file"].apply(normalize_key)
        mask = video_keys.isin(split_keys) | csv_keys.isin(split_keys)
    else:
        mask = video_keys.isin(split_keys)

    return df[mask].reset_index(drop=True)


def get_task_class_names(task: str) -> List[str]:
    if task == "binary":
        return BINARY_CLASS_NAMES

    if task == "action":
        return ACTION_CLASS_NAMES

    raise ValueError(f"Invalid task: {task}")


def parse_binary_label_from_text(text: str) -> Optional[int]:
    text = str(text).strip().lower()

    if text in BINARY_LABEL_TO_ID:
        return BINARY_LABEL_TO_ID[text]

    if "not_fall" in text or "not fall" in text or "not-fall" in text:
        return 0

    if text == "fall" or text.startswith("fall_") or "_fall" in text:
        return 1

    return None


def parse_action_label_from_text(text: str) -> Optional[int]:
    text = str(text).strip().lower()

    if text in ACTION_LABEL_TO_ID:
        return ACTION_LABEL_TO_ID[text]

    for key, value in ACTION_LABEL_TO_ID.items():
        if key in text:
            return value

    return None


def parse_action_label_from_value(value: Any, column_name: str = "") -> Optional[int]:
    """
    Robust action label parser.

    Supported final action IDs:
        0 = Sitting
        1 = Sleeping
        2 = Standing
        3 = Walking

    Some generated CSV files may contain:
        action_label:
            0 = Fall
            1 = Sitting
            2 = Sleeping
            3 = Standing
            4 = Walking

    In that case, Fall is ignored for action task.
    """
    if pd.isna(value):
        return None

    column_name = str(column_name).lower()

    if column_name == "action_label":
        try:
            numeric_value = int(value)

            if numeric_value == 0:
                return None

            if numeric_value == 1:
                return 0

            if numeric_value == 2:
                return 1

            if numeric_value == 3:
                return 2

            if numeric_value == 4:
                return 3

        except Exception:
            pass

    return parse_action_label_from_text(str(value))


def label_for_task(row: pd.Series, task: str) -> Optional[int]:
    if task == "binary":
        for col in [
            "binary_label",
            "fall_label",
            "label",
            "class_name",
            "category",
            "video_key",
            "csv_file",
        ]:
            if col not in row.index:
                continue

            value = row[col]

            if pd.isna(value):
                continue

            label = parse_binary_label_from_text(str(value))

            if label is not None:
                return label

        return None

    if task == "action":
        for col in [
            "action_label",
            "action_name",
            "label",
            "class_name",
            "category",
            "video_key",
            "csv_file",
        ]:
            if col not in row.index:
                continue

            value = row[col]

            label = parse_action_label_from_value(value, column_name=col)

            if label is not None:
                return label

        return None

    raise ValueError(f"Invalid task: {task}")


def xy_columns() -> List[str]:
    cols = []

    for i in range(17):
        cols.extend([f"x{i}", f"y{i}"])

    return cols


def conf_columns() -> List[str]:
    return [f"c{i}" for i in range(17)]


def required_2d_conf_columns() -> List[str]:
    cols = [
        "frame",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_conf",
        "person_index",
        "num_persons",
    ]

    for i in range(17):
        cols.extend([f"x{i}", f"y{i}", f"c{i}"])

    return cols


def read_csv_safely(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    return pd.read_csv(path)


def validate_2d_conf_df(df: pd.DataFrame, path: str) -> None:
    missing = [col for col in required_2d_conf_columns() if col not in df.columns]

    if missing:
        raise ValueError(f"2D confidence CSV missing columns {missing[:20]} in {path}")


def extract_2d_raw_arrays(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    validate_2d_conf_df(df, "2D confidence dataframe")

    xy = df[xy_columns()].to_numpy(dtype=np.float32).reshape(-1, 17, 2)
    conf = df[conf_columns()].to_numpy(dtype=np.float32)
    bbox = df[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(dtype=np.float32)
    bbox_conf = df["bbox_conf"].to_numpy(dtype=np.float32)
    num_persons = df["num_persons"].to_numpy(dtype=np.float32)

    return xy, conf, bbox, bbox_conf, num_persons


def normalize_2d_by_bbox(xy: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    eps = 1e-6

    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    width = np.maximum(x2 - x1, eps)
    height = np.maximum(y2 - y1, eps)
    diag = np.sqrt(width ** 2 + height ** 2)

    center = np.stack([cx, cy], axis=1)[:, None, :]
    scale = diag[:, None, None]

    return (xy - center) / (scale + eps)


def extract_2d_features_from_conf_df(df: pd.DataFrame) -> np.ndarray:
    """
    Create 40D frame-level 2D features:
        - 34D normalized keypoint coordinates
        - 6D handcrafted features:
            aspect_ratio
            norm_width
            norm_height
            center_x_norm
            center_y_norm
            velocity
    """
    xy, _, bbox, _, _ = extract_2d_raw_arrays(df)

    eps = 1e-6

    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]

    width = np.maximum(x2 - x1, eps)
    height = np.maximum(y2 - y1, eps)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    xy_norm = normalize_2d_by_bbox(xy, bbox)
    flat_xy = xy_norm.reshape(xy_norm.shape[0], -1)

    aspect_ratio = width / (height + eps)
    norm_width = width / (width + height + eps)
    norm_height = height / (width + height + eps)

    center_x_norm = cx / (np.nanmax(cx) + eps)
    center_y_norm = cy / (np.nanmax(cy) + eps)

    if xy_norm.shape[0] >= 2:
        velocity = xy_norm[1:] - xy_norm[:-1]
        velocity_norm = np.linalg.norm(velocity, axis=-1).mean(axis=1)
        velocity_feature = np.concatenate([[0.0], velocity_norm], axis=0)
    else:
        velocity_feature = np.zeros((xy_norm.shape[0],), dtype=np.float32)

    handcrafted = np.stack(
        [
            aspect_ratio,
            norm_width,
            norm_height,
            center_x_norm,
            center_y_norm,
            velocity_feature,
        ],
        axis=1,
    ).astype(np.float32)

    features = np.concatenate([flat_xy, handcrafted], axis=1).astype(np.float32)

    if features.shape[1] != 40:
        raise ValueError(f"2D feature dimension must be 40, got {features.shape[1]}")

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def find_3d_columns(df: pd.DataFrame) -> Optional[List[str]]:
    patterns = []

    pattern_1 = []
    for i in range(17):
        pattern_1.extend([f"x{i}", f"y{i}", f"z{i}"])
    patterns.append(pattern_1)

    pattern_2 = []
    for i in range(17):
        pattern_2.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
    patterns.append(pattern_2)

    pattern_3 = []
    for i in range(17):
        pattern_3.extend([f"joint{i}_x", f"joint{i}_y", f"joint{i}_z"])
    patterns.append(pattern_3)

    for pattern in patterns:
        if all(col in df.columns for col in pattern):
            return pattern

    numeric_cols = []

    for col in df.columns:
        if col == "frame":
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    if len(numeric_cols) >= 51:
        return numeric_cols[:51]

    return None


def extract_3d_xyz_from_df(df: pd.DataFrame, path: str = "") -> np.ndarray:
    cols = find_3d_columns(df)

    if cols is None:
        raise ValueError(f"Cannot find 3D coordinate columns in {path}")

    values = df[cols].to_numpy(dtype=np.float32)
    xyz = values.reshape(-1, 17, 3)

    return np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)


def compute_3d_frame_features(xyz: np.ndarray) -> np.ndarray:
    """
    Create 59D frame-level 3D features:
        - 51D normalized xyz coordinates
        - 8D handcrafted features:
            width_x
            depth_y
            height_z
            height_width_ratio
            depth_width_ratio
            head_height
            torso_tilt
            velocity
    """
    eps = 1e-6

    flat_xyz = xyz.reshape(xyz.shape[0], -1).astype(np.float32)

    x = xyz[:, :, 0]
    y = xyz[:, :, 1]
    z = xyz[:, :, 2]

    width_x = np.nanmax(x, axis=1) - np.nanmin(x, axis=1)
    depth_y = np.nanmax(y, axis=1) - np.nanmin(y, axis=1)
    height_z = np.nanmax(z, axis=1) - np.nanmin(z, axis=1)

    height_width_ratio = height_z / (width_x + eps)
    depth_width_ratio = depth_y / (width_x + eps)

    head_height = z[:, 0]

    left_shoulder = xyz[:, 5, :]
    right_shoulder = xyz[:, 6, :]
    left_hip = xyz[:, 11, :]
    right_hip = xyz[:, 12, :]

    shoulder_mid = (left_shoulder + right_shoulder) / 2.0
    hip_mid = (left_hip + right_hip) / 2.0
    torso_vec = shoulder_mid - hip_mid

    torso_horizontal = np.linalg.norm(torso_vec[:, :2], axis=1)
    torso_vertical = np.abs(torso_vec[:, 2])
    torso_tilt = torso_horizontal / (torso_vertical + eps)

    if xyz.shape[0] >= 2:
        velocity = xyz[1:] - xyz[:-1]
        velocity_norm = np.linalg.norm(velocity, axis=-1).mean(axis=1)
        velocity_feature = np.concatenate([[0.0], velocity_norm], axis=0)
    else:
        velocity_feature = np.zeros((xyz.shape[0],), dtype=np.float32)

    handcrafted = np.stack(
        [
            width_x,
            depth_y,
            height_z,
            height_width_ratio,
            depth_width_ratio,
            head_height,
            torso_tilt,
            velocity_feature,
        ],
        axis=1,
    ).astype(np.float32)

    features = np.concatenate([flat_xyz, handcrafted], axis=1).astype(np.float32)

    if features.shape[1] != 59:
        raise ValueError(f"3D feature dimension must be 59, got {features.shape[1]}")

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def extract_3d_features_from_df(df: pd.DataFrame, path: str = "") -> np.ndarray:
    xyz = extract_3d_xyz_from_df(df, path=path)
    return compute_3d_frame_features(xyz)


def pad_or_truncate_sequence(features: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad or truncate a sequence to a fixed target length.

    If the sequence is shorter than target_length, repeat the last frame.
    This keeps short videos in the same train/val/test split instead of dropping them.

    Input:
        features: [T, D]

    Output:
        fixed_features: [target_length, D]
    """
    features = np.asarray(features, dtype=np.float32)

    if features.ndim != 2:
        raise ValueError(f"Expected 2D array [T, D], got shape {features.shape}")

    current_length, feature_dim = features.shape

    if current_length == target_length:
        return features.astype(np.float32)

    if current_length > target_length:
        return features[:target_length].astype(np.float32)

    if current_length == 0:
        return np.zeros((target_length, feature_dim), dtype=np.float32)

    pad_length = target_length - current_length
    last_frame = features[-1:, :]
    padding = np.repeat(last_frame, pad_length, axis=0)

    return np.concatenate([features, padding], axis=0).astype(np.float32)


def get_quality_vector(row: pd.Series, quality_columns: List[str]) -> np.ndarray:
    values = []

    for col in quality_columns:
        if col not in row.index:
            values.append(0.0)
            continue

        value = row[col]

        if pd.isna(value):
            values.append(0.0)
        else:
            values.append(float(value))

    return np.asarray(values, dtype=np.float32)


class QualityStandardizer:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, values: np.ndarray) -> "QualityStandardizer":
        values = np.asarray(values, dtype=np.float32)

        self.mean_ = np.nanmean(values, axis=0).astype(np.float32)
        self.std_ = np.nanstd(values, axis=0).astype(np.float32)

        self.std_[self.std_ < 1e-6] = 1.0

        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("QualityStandardizer has not been fitted.")

        values = np.asarray(values, dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        return (values - self.mean_) / self.std_

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        self.fit(values)
        return self.transform(values)

    def to_dict(self) -> Dict:
        return {
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "QualityStandardizer":
        scaler = cls()

        scaler.mean_ = np.asarray(data["mean"], dtype=np.float32)
        scaler.std_ = np.asarray(data["std"], dtype=np.float32)

        return scaler


class Phase4QualityDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        task: str,
        quality_columns: Optional[List[str]] = None,
        quality_standardizer: Optional[QualityStandardizer] = None,
        cache_videos: bool = True,
        target_sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    ):
        self.task = task
        self.quality_columns = quality_columns or QUALITY_FEATURE_COLUMNS
        self.quality_standardizer = quality_standardizer
        self.cache_videos = cache_videos
        self.target_sequence_length = int(target_sequence_length)

        self.rows = []
        self.labels = []

        for _, row in dataframe.iterrows():
            label = label_for_task(row, task)

            if label is None:
                continue

            self.rows.append(row)
            self.labels.append(label)

        self.labels = np.asarray(self.labels, dtype=np.int64)

        if len(self.rows) == 0:
            raise ValueError(f"No valid samples for task={task}")

        self.cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _load_video_features(
        self,
        path_2d_conf: str,
        path_3d: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cache_key = path_2d_conf + "||" + path_3d

        if self.cache_videos and cache_key in self.cache:
            return self.cache[cache_key]

        df2d = read_csv_safely(path_2d_conf)
        df3d = read_csv_safely(path_3d)

        feat2d = extract_2d_features_from_conf_df(df2d)
        feat3d = extract_3d_features_from_df(df3d, path=path_3d)

        min_len = min(feat2d.shape[0], feat3d.shape[0])
        feat2d = feat2d[:min_len]
        feat3d = feat3d[:min_len]

        if self.cache_videos:
            self.cache[cache_key] = (feat2d, feat3d)

        return feat2d, feat3d

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.rows[index]
        label = self.labels[index]

        path_2d_conf = str(row["path_2d_conf"])
        path_3d = str(row["path_3d"])

        start = int(row["start_index"])
        end = int(row["end_index"])

        feat2d, feat3d = self._load_video_features(path_2d_conf, path_3d)

        max_len = min(feat2d.shape[0], feat3d.shape[0])

        if max_len <= 0:
            x2d = np.zeros((self.target_sequence_length, 40), dtype=np.float32)
            x3d = np.zeros((self.target_sequence_length, 59), dtype=np.float32)
        else:
            start = max(0, start)
            end = max(start + 1, end)

            if start >= max_len:
                start = max(0, max_len - 1)

            if end > max_len:
                end = max_len

            x2d = feat2d[start:end]
            x3d = feat3d[start:end]

            if x2d.shape[0] != x3d.shape[0]:
                min_len = min(x2d.shape[0], x3d.shape[0])
                x2d = x2d[:min_len]
                x3d = x3d[:min_len]

            x2d = pad_or_truncate_sequence(x2d, self.target_sequence_length)
            x3d = pad_or_truncate_sequence(x3d, self.target_sequence_length)

        quality = get_quality_vector(row, self.quality_columns)

        if self.quality_standardizer is not None:
            quality = self.quality_standardizer.transform(quality)

        return {
            "x2d": torch.tensor(x2d, dtype=torch.float32),
            "x3d": torch.tensor(x3d, dtype=torch.float32),
            "quality": torch.tensor(quality, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "video_key": str(row["video_key"]),
            "sequence_index": torch.tensor(int(row["sequence_index"]), dtype=torch.long),
        }


def phase4_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    x2d = torch.stack([item["x2d"] for item in batch], dim=0)
    x3d = torch.stack([item["x3d"] for item in batch], dim=0)
    quality = torch.stack([item["quality"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)

    video_keys = [item["video_key"] for item in batch]
    sequence_indices = torch.stack([item["sequence_index"] for item in batch], dim=0)

    return {
        "x2d": x2d,
        "x3d": x3d,
        "quality": quality,
        "label": labels,
        "video_key": video_keys,
        "sequence_index": sequence_indices,
    }


def build_phase4_dataframes(
    quality_csv: str = DEFAULT_QUALITY_CSV,
    train_split: str = DEFAULT_TRAIN_SPLIT,
    val_split: str = DEFAULT_VAL_SPLIT,
    test_split: str = DEFAULT_TEST_SPLIT,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_quality_dataframe(quality_csv)

    train_keys = read_split_keys(train_split)
    val_keys = read_split_keys(val_split)
    test_keys = read_split_keys(test_split)

    train_df = filter_by_split(df, train_keys)
    val_df = filter_by_split(df, val_keys)
    test_df = filter_by_split(df, test_keys)

    return train_df, val_df, test_df


def fit_quality_standardizer(
    train_df: pd.DataFrame,
    quality_columns: Optional[List[str]] = None,
) -> QualityStandardizer:
    quality_columns = quality_columns or QUALITY_FEATURE_COLUMNS

    values = train_df[quality_columns].fillna(0.0).to_numpy(dtype=np.float32)

    scaler = QualityStandardizer()
    scaler.fit(values)

    return scaler


def create_phase4_dataloaders(
    task: str,
    quality_csv: str = DEFAULT_QUALITY_CSV,
    batch_size: int = 32,
    num_workers: int = 0,
    quality_columns: Optional[List[str]] = None,
    cache_videos: bool = True,
    target_sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> Tuple[DataLoader, DataLoader, DataLoader, QualityStandardizer, Dict]:
    train_df, val_df, test_df = build_phase4_dataframes(quality_csv=quality_csv)

    quality_columns = quality_columns or QUALITY_FEATURE_COLUMNS
    scaler = fit_quality_standardizer(train_df, quality_columns=quality_columns)

    train_dataset = Phase4QualityDataset(
        dataframe=train_df,
        task=task,
        quality_columns=quality_columns,
        quality_standardizer=scaler,
        cache_videos=cache_videos,
        target_sequence_length=target_sequence_length,
    )

    val_dataset = Phase4QualityDataset(
        dataframe=val_df,
        task=task,
        quality_columns=quality_columns,
        quality_standardizer=scaler,
        cache_videos=cache_videos,
        target_sequence_length=target_sequence_length,
    )

    test_dataset = Phase4QualityDataset(
        dataframe=test_df,
        task=task,
        quality_columns=quality_columns,
        quality_standardizer=scaler,
        cache_videos=cache_videos,
        target_sequence_length=target_sequence_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=phase4_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=phase4_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=phase4_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    def count_dataset_videos(dataset: Phase4QualityDataset) -> int:
        video_keys = []

        for row in dataset.rows:
            if "video_key" in row.index:
                video_keys.append(normalize_key(row["video_key"]))
            elif "csv_file" in row.index:
                video_keys.append(normalize_key(row["csv_file"]))

        return int(len(set(video_keys)))

    info = {
        "task": task,
        "quality_csv": quality_csv,
        "quality_dim": len(quality_columns),
        "quality_columns": quality_columns,
        "target_sequence_length": int(target_sequence_length),
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "num_test_samples": len(test_dataset),
        "num_train_videos": count_dataset_videos(train_dataset),
        "num_val_videos": count_dataset_videos(val_dataset),
        "num_test_videos": count_dataset_videos(test_dataset),
        "class_names": get_task_class_names(task),
    }

    return train_loader, val_loader, test_loader, scaler, info


def compute_loss_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    labels = np.asarray(labels, dtype=np.int64)
    classes_present = np.unique(labels)

    weights = np.ones((num_classes,), dtype=np.float32)

    if len(classes_present) > 1:
        computed = compute_class_weight(
            class_weight="balanced",
            classes=classes_present,
            y=labels,
        )

        for cls, weight in zip(classes_present, computed):
            weights[int(cls)] = float(weight)

    return torch.tensor(weights, dtype=torch.float32)


def get_dataset_labels(loader: DataLoader) -> np.ndarray:
    if hasattr(loader.dataset, "labels"):
        return np.asarray(loader.dataset.labels, dtype=np.int64)

    labels = []

    for batch in loader:
        labels.extend(batch["label"].cpu().numpy().tolist())

    return np.asarray(labels, dtype=np.int64)


def unpack_model_output(output):
    if isinstance(output, tuple):
        logits = output[0]
        aux = output[1] if len(output) > 1 else {}
        return logits, aux

    if isinstance(output, dict):
        logits = output["logits"]
        aux = {k: v for k, v in output.items() if k != "logits"}
        return logits, aux

    return output, {}


def forward_model(model, batch: Dict[str, torch.Tensor], device: torch.device):
    x2d = batch["x2d"].to(device)
    x3d = batch["x3d"].to(device)
    quality = batch["quality"].to(device)

    output = model(x2d, x3d, quality)
    logits, aux = unpack_model_output(output)

    return logits, aux


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    criterion,
    device: torch.device,
    max_grad_norm: Optional[float] = 1.0,
) -> float:
    model.train()

    losses = []

    for batch in loader:
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits, _ = forward_model(model, batch, device)
        loss = criterion(logits, labels)

        loss.backward()

        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate_model(
    model,
    loader: DataLoader,
    device: torch.device,
    task: str,
) -> Dict:
    model.eval()

    y_true = []
    y_pred = []
    probabilities = []
    gate_values = []

    for batch in loader:
        labels = batch["label"].to(device)

        logits, aux = forward_model(model, batch, device)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        probabilities.extend(probs.detach().cpu().numpy().tolist())

        if "gate" in aux:
            gate_tensor = aux["gate"]

            if isinstance(gate_tensor, torch.Tensor):
                gate_values.extend(gate_tensor.detach().cpu().reshape(-1).numpy().tolist())

        elif "mean_gate" in aux:
            gate_tensor = aux["mean_gate"]

            if isinstance(gate_tensor, torch.Tensor):
                gate_values.extend(gate_tensor.detach().cpu().reshape(-1).numpy().tolist())

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    class_names = get_task_class_names(task)
    num_classes = len(class_names)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}

    for idx, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    result = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "num_samples": int(len(y_true)),
        "num_classes": int(num_classes),
        "class_names": class_names,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }

    if probabilities:
        result["probabilities"] = probabilities

    if gate_values:
        gate_values_np = np.asarray(gate_values, dtype=np.float32)

        result["gate_stats"] = {
            "mean_gate": float(np.mean(gate_values_np)),
            "std_gate": float(np.std(gate_values_np)),
            "min_gate": float(np.min(gate_values_np)),
            "max_gate": float(np.max(gate_values_np)),
        }

    return result


def is_better_score(current_score: float, best_score: float) -> bool:
    return current_score > best_score


def summarize_training_history(history: List[Dict]) -> Dict:
    if not history:
        return {}

    best_item = max(history, key=lambda item: item.get("val_macro_f1", -1.0))

    return {
        "best_epoch": int(best_item["epoch"]),
        "best_val_macro_f1": float(best_item["val_macro_f1"]),
        "last_train_loss": float(history[-1]["train_loss"]),
        "num_epochs": int(len(history)),
    }


def save_checkpoint(
    model,
    path: str,
    extra: Optional[Dict] = None,
) -> None:
    ensure_dir(os.path.dirname(path))

    payload = {
        "model_state_dict": model.state_dict(),
    }

    if extra:
        payload.update(extra)

    torch.save(payload, path)


def load_checkpoint(
    model,
    path: str,
    device: torch.device,
) -> Dict:
    checkpoint = torch.load(path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return checkpoint


def print_dataset_info(info: Dict) -> None:
    print("\nPhase 4 dataset info")
    print("=" * 80)
    print(f"Task              : {info['task']}")
    print(f"Quality dim       : {info['quality_dim']}")
    print(f"Target seq length : {info.get('target_sequence_length', DEFAULT_SEQUENCE_LENGTH)}")
    print(f"Train samples     : {info['num_train_samples']}")
    print(f"Val samples       : {info['num_val_samples']}")
    print(f"Test samples      : {info['num_test_samples']}")
    print(f"Train videos      : {info['num_train_videos']}")
    print(f"Val videos        : {info['num_val_videos']}")
    print(f"Test videos       : {info['num_test_videos']}")
    print(f"Class names       : {info['class_names']}")
    print("=" * 80)


def quick_check_phase4_utils() -> None:
    print("Running quick check for phase4_utils.py")

    quality_csv = DEFAULT_QUALITY_CSV

    if not os.path.exists(quality_csv):
        print(f"Quality CSV not found yet: {quality_csv}")
        return

    df = load_quality_dataframe(quality_csv)
    print(f"Loaded quality dataframe: {df.shape}")

    print("\nAvailable columns:")
    print(df.columns.tolist())

    print("\nRaw label distribution:")

    if "label" in df.columns:
        print("\nlabel:")
        print(df["label"].value_counts(dropna=False).to_dict())

    if "action_label" in df.columns:
        print("\naction_label:")
        print(df["action_label"].value_counts(dropna=False).to_dict())

    if "action_name" in df.columns:
        print("\naction_name:")
        print(df["action_name"].value_counts(dropna=False).to_dict())

    if "sequence_length" in df.columns:
        print("\nsequence_length:")
        print(df["sequence_length"].value_counts(dropna=False).sort_index().to_dict())

    print("\nParsed label distribution:")

    for task in ["binary", "action"]:
        labels = []

        for _, row in df.iterrows():
            label = label_for_task(row, task)

            if label is not None:
                labels.append(label)

        labels = np.asarray(labels, dtype=np.int64)

        if len(labels) == 0:
            print(f"Task={task}: no labels found")
            continue

        unique, counts = np.unique(labels, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(unique, counts)}

        print(f"Task={task}: total valid samples = {len(labels)}")
        print(f"Task={task}: parsed distribution = {dist}")

    print("\nChecking dataloaders and fixed tensor shapes:")

    for task in ["binary", "action"]:
        try:
            train_loader, val_loader, test_loader, _, info = create_phase4_dataloaders(
                task=task,
                quality_csv=quality_csv,
                batch_size=4,
                num_workers=0,
                cache_videos=False,
                target_sequence_length=DEFAULT_SEQUENCE_LENGTH,
            )

            print_dataset_info(info)

            batch = next(iter(train_loader))

            print(f"Task={task}: batch x2d shape     = {tuple(batch['x2d'].shape)}")
            print(f"Task={task}: batch x3d shape     = {tuple(batch['x3d'].shape)}")
            print(f"Task={task}: batch quality shape = {tuple(batch['quality'].shape)}")
            print(f"Task={task}: batch label shape   = {tuple(batch['label'].shape)}")

        except Exception as exc:
            print(f"Task={task}: dataloader check failed: {exc}")

    print("\nQuick check finished.")


if __name__ == "__main__":
    quick_check_phase4_utils()