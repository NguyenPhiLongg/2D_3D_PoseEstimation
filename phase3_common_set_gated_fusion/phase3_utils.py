"""
Shared utilities for Phase 3: Common-set Fair Comparison and Gated Fusion.

This file centralizes:
- common metadata loading
- 2D / 3D / concat feature extraction
- sequence dataset creation
- training and evaluation loops

The goal of Phase 3 is to evaluate 2D-only, 3D-only, concat fusion,
and gated fusion on the exact same common video subset.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset


# =========================
# PATHS
# =========================

PHASE3_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE3_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_2D_DIR = DATA_DIR / "2_extracted_2d"
DATA_3D_DIR = DATA_DIR / "4_normalized_3d"
DATA_3D_RAW_DIR = DATA_DIR / "3_extracted_3d"

OUTPUT_DIR = PHASE3_DIR / "outputs"
COMMON_SPLIT_DIR = OUTPUT_DIR / "common_split"
CHECKPOINT_DIR = PHASE3_DIR / "checkpoints"

REFERENCE_WIDTH = 1920.0
REFERENCE_HEIGHT = 1080.0

BINARY_CLASS_NAMES = ["Not_Fall", "Fall"]
ACTION_CLASS_NAMES = ["Sitting", "Sleeping", "Standing", "Walking"]

ACTION_NAME_TO_LABEL = {
    "sitting": 1,
    "sit": 1,
    "sleeping": 2,
    "sleep": 2,
    "standing": 3,
    "stand": 3,
    "walking": 4,
    "walk": 4,
}

ACTION_LABEL_TO_INDEX = {
    1: 0,  # Sitting
    2: 1,  # Sleeping
    3: 2,  # Standing
    4: 3,  # Walking
}


# =========================
# REPRODUCIBILITY
# =========================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Keep benchmark=True for faster CNN/LSTM training on GPU.
    # deterministic=False because exact reproducibility on GPU is not always guaranteed,
    # but seed still makes the experiment reasonably stable.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =========================
# FILE AND METADATA HELPERS
# =========================

def normalize_video_key(path_or_name: Union[str, Path]) -> str:
    """
    Normalize file stem so 2D and 3D CSV files can be matched robustly.

    Example:
        abc_2d_keypoints.csv
        abc_3d_keypoints.csv
        abc_normalized_3d.csv

    All become:
        abc
    """
    stem = Path(path_or_name).stem.lower().strip()

    removable_suffixes = [
        "_2d_keypoints",
        "_3d_keypoints",
        "_normalized_3d",
        "_normalized",
        "_keypoints",
        "_pose",
        "_2d",
        "_3d",
    ]

    changed = True
    while changed:
        changed = False
        for suffix in removable_suffixes:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True

    stem = stem.replace(" ", "_")
    return stem


def collect_csv_files(root_dir: Path) -> Dict[str, Path]:
    """Collect CSV files recursively and map normalized video key -> path."""
    files: Dict[str, Path] = {}

    if not root_dir.exists():
        return files

    for path in root_dir.rglob("*.csv"):
        key = normalize_video_key(path)
        if key not in files:
            files[key] = path

    return files


def read_video_list(split_name: str) -> List[str]:
    split_path = COMMON_SPLIT_DIR / f"{split_name}_videos.txt"

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}. Run build_common_split.py first."
        )

    with open(split_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_common_metadata() -> pd.DataFrame:
    metadata_path = COMMON_SPLIT_DIR / "common_metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Common metadata not found: {metadata_path}. Run build_common_split.py first."
        )

    df = pd.read_csv(metadata_path)

    required = ["video_key", "path_2d", "path_3d", "label", "action_label", "action_name", "split"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"common_metadata.csv missing columns: {missing}")

    return df


def read_csv_safely(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    if "frame" in df.columns:
        df = df.sort_values("frame").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


# =========================
# LABEL HELPERS
# =========================

def infer_label_from_path(path: Union[str, Path]) -> int:
    """
    Infer binary label from file path when CSV label column is missing.

    Fall    -> 1
    NotFall -> 0
    """
    text = str(path).replace("\\", "/").lower()

    if "/fall/" in text or "fall" in Path(path).stem.lower():
        if "not_fall" not in text and "non_fall" not in text:
            return 1

    return 0


def infer_action_label_from_path(path: Union[str, Path], binary_label: int) -> Tuple[int, str]:
    """
    Infer action label from path when CSV action label is missing.

    action_label:
        0 = Fall
        1 = Sitting
        2 = Sleeping
        3 = Standing
        4 = Walking
    """
    if binary_label == 1:
        return 0, "Fall"

    text = str(path).replace("\\", "/").lower()

    for key, action_label in ACTION_NAME_TO_LABEL.items():
        if key in text:
            return action_label, ACTION_CLASS_NAMES[ACTION_LABEL_TO_INDEX[action_label]]

    return -1, "Unknown"


def extract_label_info(df: pd.DataFrame, path: Union[str, Path]) -> Tuple[int, int, str]:
    """
    Extract label, action_label, action_name from CSV.
    If missing, infer from file path.
    """
    if "label" in df.columns:
        try:
            label = int(df["label"].dropna().iloc[0])
        except Exception:
            label = infer_label_from_path(path)
    else:
        label = infer_label_from_path(path)

    if "action_label" in df.columns:
        try:
            action_label = int(df["action_label"].dropna().iloc[0])
        except Exception:
            action_label = -1
    else:
        action_label = -1

    if "action_name" in df.columns:
        try:
            action_name = str(df["action_name"].dropna().iloc[0])
        except Exception:
            action_name = "Unknown"
    else:
        action_name = "Unknown"

    if action_label < 0 or action_name == "Unknown":
        inferred_action_label, inferred_action_name = infer_action_label_from_path(path, label)

        if action_label < 0:
            action_label = inferred_action_label

        if action_name == "Unknown":
            action_name = inferred_action_name

    return label, action_label, action_name


def label_for_task(record: pd.Series, task: str) -> Optional[int]:
    """
    Return label index for selected task.

    binary:
        Not_Fall = 0
        Fall = 1

    action:
        Sitting = 0
        Sleeping = 1
        Standing = 2
        Walking = 3
    """
    task = task.lower().strip()

    if task == "binary":
        return int(record["label"])

    if task == "action":
        action_label = int(record["action_label"])

        if action_label not in ACTION_LABEL_TO_INDEX:
            return None

        return ACTION_LABEL_TO_INDEX[action_label]

    raise ValueError(f"Invalid task: {task}")


def class_names_for_task(task: str) -> List[str]:
    task = task.lower().strip()

    if task == "binary":
        return BINARY_CLASS_NAMES

    if task == "action":
        return ACTION_CLASS_NAMES

    raise ValueError(f"Invalid task: {task}")


# =========================
# FEATURE EXTRACTION - 2D
# =========================

def get_2d_columns(df: pd.DataFrame) -> List[str]:
    cols = []

    for i in range(17):
        cols.extend([f"x{i}", f"y{i}"])

    missing = [col for col in cols if col not in df.columns]

    if missing:
        raise ValueError(f"2D CSV missing columns: {missing[:10]}")

    return cols


def extract_one_frame_2d_features(
    keypoints: np.ndarray,
    prev_center: Optional[Tuple[float, float]] = None,
    reference_width: float = REFERENCE_WIDTH,
    reference_height: float = REFERENCE_HEIGHT,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Convert one frame of 17x2 COCO keypoints into 40D feature vector.

    40D = 34 normalized 2D coordinates + 6 handcrafted features.

    Normalization:
        per-frame x/y mean-std normalization
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)

    if keypoints.shape != (17, 2):
        raise ValueError(f"Expected 2D keypoints shape (17, 2), got {keypoints.shape}")

    xs = keypoints[:, 0]
    ys = keypoints[:, 1]

    min_x = float(np.min(xs))
    max_x = float(np.max(xs))
    min_y = float(np.min(ys))
    max_y = float(np.max(ys))

    width = max_x - min_x
    height = max_y - min_y

    eps = 1e-6

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    mean_2d = keypoints.mean(axis=0, keepdims=True)
    std_2d = keypoints.std(axis=0, keepdims=True)
    std_2d = np.where(std_2d < eps, 1.0, std_2d)

    normalized_coords = ((keypoints - mean_2d) / std_2d).reshape(-1).astype(np.float32)

    scale = max(width, height) + eps

    aspect_ratio = height / (width + eps)
    norm_width = width / scale
    norm_height = height / scale
    center_x_norm = center_x / reference_width
    center_y_norm = center_y / reference_height

    if prev_center is None:
        velocity = 0.0
    else:
        prev_x, prev_y = prev_center
        velocity = float(np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2))

    handcrafted = np.array(
        [
            aspect_ratio,
            norm_width,
            norm_height,
            center_x_norm,
            center_y_norm,
            velocity,
        ],
        dtype=np.float32,
    )

    features = np.concatenate([normalized_coords, handcrafted], axis=0).astype(np.float32)

    if features.shape[0] != 40:
        raise ValueError(f"2D feature dim must be 40, got {features.shape[0]}")

    return features, (center_x, center_y)


def extract_2d_features_from_df(df: pd.DataFrame) -> np.ndarray:
    cols = get_2d_columns(df)

    values = df[cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    values = values.ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)

    features: List[np.ndarray] = []
    prev_center = None

    for row in values:
        keypoints = row.reshape(17, 2)
        feat, prev_center = extract_one_frame_2d_features(keypoints, prev_center)
        features.append(feat)

    return np.stack(features, axis=0).astype(np.float32)


# =========================
# FEATURE EXTRACTION - 3D
# =========================

def get_3d_columns(df: pd.DataFrame) -> List[str]:
    cols = []

    for i in range(17):
        cols.extend([f"x{i}", f"y{i}", f"z{i}"])

    missing = [col for col in cols if col not in df.columns]

    if missing:
        raise ValueError(f"3D CSV missing columns: {missing[:10]}")

    return cols


def normalize_pose3d_mean_std(pose3d: np.ndarray) -> np.ndarray:
    """
    Normalize one 3D pose using per-axis mean/std.

    Formula:
        pose = (pose - pose.mean(axis=0)) / pose.std(axis=0)
    """
    pose3d = np.asarray(pose3d, dtype=np.float32)

    if pose3d.shape != (17, 3):
        raise ValueError(f"Expected 3D pose shape (17, 3), got {pose3d.shape}")

    eps = 1e-6

    mean = pose3d.mean(axis=0, keepdims=True)
    std = pose3d.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)

    return ((pose3d - mean) / std).astype(np.float32)


def extract_one_frame_3d_features(
    pose3d: np.ndarray,
    prev_center: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert one frame of 17x3 estimated 3D pose into 59D feature vector.

    59D = 51 normalized 3D coordinates + 8 handcrafted features.
    """
    pose = normalize_pose3d_mean_std(pose3d)

    xs = pose[:, 0]
    ys = pose[:, 1]
    zs = pose[:, 2]

    min_x = float(np.min(xs))
    max_x = float(np.max(xs))

    min_y = float(np.min(ys))
    max_y = float(np.max(ys))

    min_z = float(np.min(zs))
    max_z = float(np.max(zs))

    width_x = max_x - min_x
    depth_y = max_y - min_y
    height_z = max_z - min_z

    eps = 1e-6

    height_width_ratio = height_z / (width_x + eps)
    depth_width_ratio = depth_y / (width_x + eps)

    # H36M-like layout assumption used in Phase 2.
    # Joint 10 = head, joint 0 = pelvis.
    head_height = float(zs[10] - zs[0])

    # Joint 8 = thorax, joint 0 = pelvis.
    torso_vector = pose[8, :] - pose[0, :]
    torso_horizontal = float(np.linalg.norm(torso_vector[0:2]))
    torso_vertical = float(abs(torso_vector[2]) + eps)
    torso_tilt = torso_horizontal / torso_vertical

    center = pose.mean(axis=0)

    if prev_center is None:
        velocity = 0.0
    else:
        velocity = float(np.linalg.norm(center - prev_center))

    handcrafted = np.array(
        [
            width_x,
            depth_y,
            height_z,
            height_width_ratio,
            depth_width_ratio,
            head_height,
            torso_tilt,
            velocity,
        ],
        dtype=np.float32,
    )

    coords = pose.reshape(-1).astype(np.float32)
    features = np.concatenate([coords, handcrafted], axis=0).astype(np.float32)

    if features.shape[0] != 59:
        raise ValueError(f"3D feature dim must be 59, got {features.shape[0]}")

    return features, center


def extract_3d_features_from_df(df: pd.DataFrame) -> np.ndarray:
    cols = get_3d_columns(df)

    values = df[cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    values = values.ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)

    features: List[np.ndarray] = []
    prev_center = None

    for row in values:
        pose3d = row.reshape(17, 3)
        feat, prev_center = extract_one_frame_3d_features(pose3d, prev_center)
        features.append(feat)

    return np.stack(features, axis=0).astype(np.float32)


# =========================
# SEQUENCE BUILDING
# =========================

def build_sequences(
    features: np.ndarray,
    sequence_length: int = 60,
    stride: int = 15,
    strategy: str = "sliding",
) -> List[np.ndarray]:
    """
    Build fixed-length sequences from frame-level features.

    strategy = "sliding":
        multiple sequences using sliding window.

    strategy = "single":
        one centered or padded sequence per video.
    """
    features = np.asarray(features, dtype=np.float32)

    if features.ndim != 2:
        raise ValueError(f"Expected features shape (T, D), got {features.shape}")

    t, _ = features.shape

    if t == 0:
        return []

    if t < sequence_length:
        pad_count = sequence_length - t
        pad = np.repeat(features[-1:, :], pad_count, axis=0)
        padded = np.concatenate([features, pad], axis=0)
        return [padded.astype(np.float32)]

    if strategy == "single":
        start = max((t - sequence_length) // 2, 0)
        return [features[start : start + sequence_length].astype(np.float32)]

    if strategy != "sliding":
        raise ValueError(f"Invalid sequence strategy: {strategy}")

    sequences = []

    for start in range(0, t - sequence_length + 1, stride):
        sequences.append(features[start : start + sequence_length].astype(np.float32))

    if not sequences:
        sequences.append(features[:sequence_length].astype(np.float32))

    return sequences


@dataclass
class DatasetStats:
    num_records: int
    num_sequences: int
    class_counts: Dict[str, int]


class Phase3SequenceDataset(Dataset):
    """Dataset for 2D, 3D, concat, or gated fusion experiments."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        split: str,
        task: str,
        feature_mode: str,
        sequence_length: int = 60,
        stride: int = 15,
        sequence_strategy: str = "sliding",
    ) -> None:
        self.metadata = metadata[metadata["split"] == split].reset_index(drop=True)
        self.split = split
        self.task = task.lower().strip()
        self.feature_mode = feature_mode.lower().strip()
        self.sequence_length = sequence_length
        self.stride = stride
        self.sequence_strategy = sequence_strategy
        self.class_names = class_names_for_task(self.task)

        valid_modes = ["2d", "3d", "concat", "gated"]

        if self.feature_mode not in valid_modes:
            raise ValueError(f"feature_mode must be one of {valid_modes}, got {feature_mode}")

        self.samples: List[Tuple[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], int, str]] = []
        self.skipped: List[Tuple[str, str]] = []

        self._build()

    def _build(self) -> None:
        """
        Build sequence samples for Phase 3.

        Important fairness rule:
        All feature modes must use the same aligned frame range.

        For each video:
            min_len = min(number of 2D frames, number of 3D frames)

        Then:
            2D     uses feat2d[:min_len]
            3D     uses feat3d[:min_len]
            Concat uses feat2d[:min_len] + feat3d[:min_len]
            Gated  uses feat2d[:min_len] and feat3d[:min_len]

        This ensures:
            same videos
            same aligned frame range
            same number of generated sequences
        """
        for _, record in self.metadata.iterrows():
            y = label_for_task(record, self.task)

            if y is None:
                continue

            video_key = str(record["video_key"])

            try:
                # Always load both 2D and 3D in Phase 3.
                # Even for 2D-only and 3D-only models, both files are loaded
                # so the sequence length can be aligned by min_len.
                df2d = read_csv_safely(record["path_2d"])
                df3d = read_csv_safely(record["path_3d"])

                feat2d = extract_2d_features_from_df(df2d)
                feat3d = extract_3d_features_from_df(df3d)

                min_len = min(feat2d.shape[0], feat3d.shape[0])

                if min_len <= 0:
                    self.skipped.append(
                        (video_key, "Empty aligned 2D/3D feature sequence")
                    )
                    continue

                # Strict common-set alignment.
                # This is the key fix.
                feat2d_aligned = feat2d[:min_len]
                feat3d_aligned = feat3d[:min_len]

                if self.feature_mode == "2d":
                    sequences = build_sequences(
                        feat2d_aligned,
                        self.sequence_length,
                        self.stride,
                        self.sequence_strategy,
                    )

                    for seq in sequences:
                        self.samples.append((seq, y, video_key))

                elif self.feature_mode == "3d":
                    sequences = build_sequences(
                        feat3d_aligned,
                        self.sequence_length,
                        self.stride,
                        self.sequence_strategy,
                    )

                    for seq in sequences:
                        self.samples.append((seq, y, video_key))

                elif self.feature_mode == "concat":
                    concat_features = np.concatenate(
                        [feat2d_aligned, feat3d_aligned],
                        axis=1,
                    ).astype(np.float32)

                    sequences = build_sequences(
                        concat_features,
                        self.sequence_length,
                        self.stride,
                        self.sequence_strategy,
                    )

                    for seq in sequences:
                        self.samples.append((seq, y, video_key))

                elif self.feature_mode == "gated":
                    seq2d_list = build_sequences(
                        feat2d_aligned,
                        self.sequence_length,
                        self.stride,
                        self.sequence_strategy,
                    )

                    seq3d_list = build_sequences(
                        feat3d_aligned,
                        self.sequence_length,
                        self.stride,
                        self.sequence_strategy,
                    )

                    pair_count = min(len(seq2d_list), len(seq3d_list))

                    for i in range(pair_count):
                        self.samples.append(
                            ((seq2d_list[i], seq3d_list[i]), y, video_key)
                        )

                else:
                    raise ValueError(f"Invalid feature mode: {self.feature_mode}")

            except Exception as exc:
                self.skipped.append((video_key, str(exc)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        x, y, video_key = self.samples[index]

        if self.feature_mode == "gated":
            x2d, x3d = x

            return (
                torch.tensor(x2d, dtype=torch.float32),
                torch.tensor(x3d, dtype=torch.float32),
            ), torch.tensor(y, dtype=torch.long), video_key

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long), video_key

    def get_labels(self) -> np.ndarray:
        return np.array([sample[1] for sample in self.samples], dtype=np.int64)

    def stats(self) -> DatasetStats:
        labels = self.get_labels()
        counts = {}

        for idx, name in enumerate(self.class_names):
            counts[name] = int(np.sum(labels == idx))

        return DatasetStats(
            num_records=len(self.metadata),
            num_sequences=len(self.samples),
            class_counts=counts,
        )


def gated_collate_fn(batch):
    xs2d = torch.stack([item[0][0] for item in batch], dim=0)
    xs3d = torch.stack([item[0][1] for item in batch], dim=0)
    ys = torch.stack([item[1] for item in batch], dim=0)
    video_keys = [item[2] for item in batch]

    return (xs2d, xs3d), ys, video_keys


def single_collate_fn(batch):
    xs = torch.stack([item[0] for item in batch], dim=0)
    ys = torch.stack([item[1] for item in batch], dim=0)
    video_keys = [item[2] for item in batch]

    return xs, ys, video_keys


def build_dataloaders(
    task: str,
    feature_mode: str,
    batch_size: int = 64,
    sequence_length: int = 60,
    stride: int = 15,
    sequence_strategy: str = "sliding",
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[str, DatasetStats]]:
    metadata = load_common_metadata()

    train_ds = Phase3SequenceDataset(
        metadata,
        split="train",
        task=task,
        feature_mode=feature_mode,
        sequence_length=sequence_length,
        stride=stride,
        sequence_strategy=sequence_strategy,
    )

    val_ds = Phase3SequenceDataset(
        metadata,
        split="val",
        task=task,
        feature_mode=feature_mode,
        sequence_length=sequence_length,
        stride=stride,
        sequence_strategy=sequence_strategy,
    )

    test_ds = Phase3SequenceDataset(
        metadata,
        split="test",
        task=task,
        feature_mode=feature_mode,
        sequence_length=sequence_length,
        stride=stride,
        sequence_strategy=sequence_strategy,
    )

    if len(train_ds) == 0:
        raise RuntimeError(f"No training sequences found for task={task}, feature_mode={feature_mode}")

    collate_fn = gated_collate_fn if feature_mode == "gated" else single_collate_fn

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    stats = {
        "train": train_ds.stats(),
        "val": val_ds.stats(),
        "test": test_ds.stats(),
    }

    for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        if ds.skipped:
            print(f"WARNING: {len(ds.skipped)} skipped videos in {split_name} split.")
            print("First skipped examples:")

            for video_key, reason in ds.skipped[:5]:
                print("  ", video_key, "->", reason)

    return train_loader, val_loader, test_loader, train_ds.class_names, stats


# =========================
# TRAINING HELPERS
# =========================

def get_device(use_cuda: bool = True) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def compute_loss_weights(labels: np.ndarray, num_classes: int, device: torch.device) -> Optional[torch.Tensor]:
    unique = np.unique(labels)

    if len(unique) < 2:
        return None

    classes = np.arange(num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    return weights


def build_eval_dict(
    labels: List[int],
    preds: List[int],
    class_names: List[str],
    video_keys: Optional[List[str]] = None,
) -> Dict:
    if len(labels) == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "classification_report_text": "No samples.",
            "classification_report_dict": {},
            "confusion_matrix": [],
            "num_samples": 0,
        }

    labels_np = np.array(labels)
    preds_np = np.array(preds)

    acc = accuracy_score(labels_np, preds_np)
    macro_f1 = f1_score(labels_np, preds_np, average="macro", zero_division=0)

    labels_for_report = list(range(len(class_names)))

    report_text = classification_report(
        labels_np,
        preds_np,
        labels=labels_for_report,
        target_names=class_names,
        zero_division=0,
    )

    report_dict = classification_report(
        labels_np,
        preds_np,
        labels=labels_for_report,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(labels_np, preds_np, labels=labels_for_report)

    result = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
        "confusion_matrix": cm.tolist(),
        "num_samples": int(len(labels)),
    }

    if video_keys is not None:
        result["num_unique_videos"] = int(len(set(video_keys)))

    return result


def evaluate_single_input(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Dict:
    model.eval()

    all_preds: List[int] = []
    all_labels: List[int] = []
    all_video_keys: List[str] = []

    with torch.no_grad():
        for x, y, video_keys in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())
            all_video_keys.extend(video_keys)

    return build_eval_dict(all_labels, all_preds, class_names, all_video_keys)


def evaluate_gated(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Dict:
    model.eval()

    all_preds: List[int] = []
    all_labels: List[int] = []
    all_video_keys: List[str] = []
    gate_values: List[float] = []

    with torch.no_grad():
        for (x2d, x3d), y, video_keys in loader:
            x2d = x2d.to(device)
            x3d = x3d.to(device)
            y = y.to(device)

            output = model(x2d, x3d, return_gate=True)

            if isinstance(output, tuple):
                logits, gate = output
                gate_values.extend(gate.detach().mean(dim=1).cpu().numpy().tolist())
            else:
                logits = output

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())
            all_video_keys.extend(video_keys)

    result = build_eval_dict(all_labels, all_preds, class_names, all_video_keys)

    if gate_values:
        result["mean_gate"] = float(np.mean(gate_values))
        result["std_gate"] = float(np.std(gate_values))
        result["min_gate"] = float(np.min(gate_values))
        result["max_gate"] = float(np.max(gate_values))

    return result


def stats_to_dict(stats: Dict[str, DatasetStats]) -> Dict:
    return {
        split: {
            "num_records": item.num_records,
            "num_sequences": item.num_sequences,
            "class_counts": item.class_counts,
        }
        for split, item in stats.items()
    }


def save_json(data: Dict, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def train_single_input_model(
    model: nn.Module,
    task: str,
    feature_mode: str,
    output_json_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    model_type: str,
    input_dim: int,
    batch_size: int = 64,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    sequence_length: int = 60,
    stride: int = 15,
    sequence_strategy: str = "sliding",
    seed: int = 42,
    use_cuda: bool = True,
) -> Dict:
    set_seed(seed)
    device = get_device(use_cuda)

    train_loader, val_loader, test_loader, class_names, stats = build_dataloaders(
        task=task,
        feature_mode=feature_mode,
        batch_size=batch_size,
        sequence_length=sequence_length,
        stride=stride,
        sequence_strategy=sequence_strategy,
    )

    num_classes = len(class_names)
    model = model.to(device)

    train_labels = train_loader.dataset.get_labels()
    class_weights = compute_loss_weights(train_labels, num_classes, device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
    )

    best_val_f1 = -1.0
    best_epoch = -1
    history = []

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Training {model_type}")
    print(f"Task: {task}")
    print(f"Feature mode: {feature_mode}")
    print(f"Input dim: {input_dim}")
    print(f"Class names: {class_names}")
    print(f"Device: {device}")
    print("Dataset stats:")
    print(stats_to_dict(stats))
    print("=" * 80)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_items += y.size(0)

        train_loss = total_loss / max(total_items, 1)
        val_metrics = evaluate_single_input(model, val_loader, device, class_names)
        scheduler.step(val_metrics["macro_f1"])

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": model_type,
                    "task": task,
                    "feature_mode": feature_mode,
                    "input_dim": input_dim,
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "sequence_length": sequence_length,
                    "stride": stride,
                    "sequence_strategy": sequence_strategy,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_f1,
                    "normalization": "2d_xy_mean_std / 3d_xyz_mean_std depending on mode",
                    "seed": seed,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_single_input(model, test_loader, device, class_names)

    result = {
        "model_type": model_type,
        "task": task,
        "feature_mode": feature_mode,
        "input_dim": input_dim,
        "sequence_length": sequence_length,
        "stride": stride,
        "sequence_strategy": sequence_strategy,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "class_names": class_names,
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_val_f1),
        "final_test_accuracy": test_metrics["accuracy"],
        "final_test_macro_f1": test_metrics["macro_f1"],
        "classification_report_text": test_metrics["classification_report_text"],
        "classification_report_dict": test_metrics["classification_report_dict"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "num_test_samples": test_metrics["num_samples"],
        "num_test_unique_videos": test_metrics.get("num_unique_videos"),
        "dataset_stats": stats_to_dict(stats),
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "notes": "Phase 3 common-set experiment. All compared models should use the same common_metadata.csv split.",
    }

    save_json(result, output_json_path)

    print("=" * 80)
    print("Training finished.")
    print("Best epoch:", best_epoch)
    print("Best val macro F1:", best_val_f1)
    print("Test accuracy:", test_metrics["accuracy"])
    print("Test macro F1:", test_metrics["macro_f1"])
    print("Saved result:", output_json_path)
    print("Saved checkpoint:", checkpoint_path)
    print("=" * 80)

    return result


def train_gated_model(
    model: nn.Module,
    task: str,
    output_json_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    batch_size: int = 64,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    sequence_length: int = 60,
    stride: int = 15,
    sequence_strategy: str = "sliding",
    seed: int = 42,
    use_cuda: bool = True,
) -> Dict:
    set_seed(seed)
    device = get_device(use_cuda)

    train_loader, val_loader, test_loader, class_names, stats = build_dataloaders(
        task=task,
        feature_mode="gated",
        batch_size=batch_size,
        sequence_length=sequence_length,
        stride=stride,
        sequence_strategy=sequence_strategy,
    )

    num_classes = len(class_names)
    model = model.to(device)

    train_labels = train_loader.dataset.get_labels()
    class_weights = compute_loss_weights(train_labels, num_classes, device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
    )

    best_val_f1 = -1.0
    best_epoch = -1
    history = []

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Training Gated Fusion model")
    print(f"Task: {task}")
    print(f"Class names: {class_names}")
    print(f"Device: {device}")
    print("Dataset stats:")
    print(stats_to_dict(stats))
    print("=" * 80)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0

        for (x2d, x3d), y, _ in train_loader:
            x2d = x2d.to(device)
            x3d = x3d.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x2d, x3d)
            loss = criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_items += y.size(0)

        train_loss = total_loss / max(total_items, 1)
        val_metrics = evaluate_gated(model, val_loader, device, class_names)
        scheduler.step(val_metrics["macro_f1"])

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_mean_gate": val_metrics.get("mean_gate"),
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['macro_f1']:.4f} | "
            f"mean_gate={val_metrics.get('mean_gate', 0.0):.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": "GatedFusionCNNLSTM",
                    "task": task,
                    "feature_mode": "gated",
                    "input_dim_2d": 40,
                    "input_dim_3d": 59,
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "sequence_length": sequence_length,
                    "stride": stride,
                    "sequence_strategy": sequence_strategy,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_f1,
                    "normalization": "2d_xy_mean_std + 3d_xyz_mean_std",
                    "seed": seed,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_gated(model, test_loader, device, class_names)

    result = {
        "model_type": "GatedFusionCNNLSTM",
        "task": task,
        "feature_mode": "gated",
        "input_dim_2d": 40,
        "input_dim_3d": 59,
        "sequence_length": sequence_length,
        "stride": stride,
        "sequence_strategy": sequence_strategy,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "class_names": class_names,
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_val_f1),
        "final_test_accuracy": test_metrics["accuracy"],
        "final_test_macro_f1": test_metrics["macro_f1"],
        "classification_report_text": test_metrics["classification_report_text"],
        "classification_report_dict": test_metrics["classification_report_dict"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "num_test_samples": test_metrics["num_samples"],
        "num_test_unique_videos": test_metrics.get("num_unique_videos"),
        "mean_gate": test_metrics.get("mean_gate"),
        "std_gate": test_metrics.get("std_gate"),
        "min_gate": test_metrics.get("min_gate"),
        "max_gate": test_metrics.get("max_gate"),
        "dataset_stats": stats_to_dict(stats),
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "notes": "Gated Fusion learns adaptive weights between 2D and estimated 3D skeleton streams on the same common split.",
    }

    save_json(result, output_json_path)

    print("=" * 80)
    print("Gated training finished.")
    print("Best epoch:", best_epoch)
    print("Best val macro F1:", best_val_f1)
    print("Test accuracy:", test_metrics["accuracy"])
    print("Test macro F1:", test_metrics["macro_f1"])
    print("Mean gate:", test_metrics.get("mean_gate"))
    print("Saved result:", output_json_path)
    print("Saved checkpoint:", checkpoint_path)
    print("=" * 80)

    return result