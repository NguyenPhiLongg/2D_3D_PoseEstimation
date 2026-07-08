import os
import sys
import json
import time
import random
import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


# ============================================================
# PATH SETUP
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE5_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PHASE5_DIR.parent

PHASE1_DIR = PROJECT_ROOT / "phase1_2d_baseline"
PHASE2_DIR = PROJECT_ROOT / "phase2_3d_upgrade"
PHASE3_DIR = PROJECT_ROOT / "phase3_common_set_gated_fusion"
PHASE4_DIR = PROJECT_ROOT / "phase4_quality_aware_fusion"

for p in [PHASE5_DIR, PROJECT_ROOT, PHASE1_DIR, PHASE2_DIR, PHASE3_DIR, PHASE4_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


from phase5_utils import (
    load_config,
    cfg_path,
    ensure_dir,
    print_dict,
    print_dataframe_summary,
)


# ============================================================
# GLOBAL CONSTANTS
# ============================================================

CLASS_NAMES = ["Not_Fall", "Fall"]
LABELS = [0, 1]

EXPECTED_2D_DIM = 40
EXPECTED_3D_DIM = 59
EXPECTED_CONCAT_DIM = 99
EXPECTED_QUALITY_DIM = 33
SEQUENCE_LENGTH = 60


MODEL_SPECS = [
    {
        "model_name": "phase1_2d_common",
        "display_name": "Phase 1 2D Common",
        "checkpoint_key": "phase1_2d_common_binary",
        "model_family": "single_2d",
        "input_key": "x2d_common",
        "uses_quality": False,
    },
    {
        "model_name": "phase2_3d_common",
        "display_name": "Phase 2 3D Common",
        "checkpoint_key": "phase2_3d_common_binary",
        "model_family": "single_3d",
        "input_key": "x3d_common",
        "uses_quality": False,
    },
    {
        "model_name": "phase2_concat_common",
        "display_name": "Phase 2 2D+3D Concat Common",
        "checkpoint_key": "phase2_concat_common_binary",
        "model_family": "single_concat",
        "input_key": "xconcat_common",
        "uses_quality": False,
    },
    {
        "model_name": "phase3_gated_fusion",
        "display_name": "Phase 3 Gated Fusion",
        "checkpoint_key": "phase3_gated_fusion_binary",
        "model_family": "gated",
        "input_key": None,
        "uses_quality": False,
    },
    {
        "model_name": "phase4_quality_gated",
        "display_name": "Phase 4 Quality-Gated",
        "checkpoint_key": "phase4_quality_gated_binary",
        "model_family": "quality_gated",
        "input_key": None,
        "uses_quality": True,
    },
    {
        "model_name": "phase4_quality_concat",
        "display_name": "Phase 4 Quality-Concat",
        "checkpoint_key": "phase4_quality_concat_binary",
        "model_family": "quality_concat",
        "input_key": None,
        "uses_quality": True,
    },
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
# DYNAMIC IMPORT HELPERS
# ============================================================

def load_module_from_file(module_name: str, file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Module not found: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")

    module = importlib.util.module_from_spec(spec)

    # Important for @dataclass inside imported modules.
    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


def find_existing_file(candidates: List[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "None of the candidate files exist:\n"
        + "\n".join(str(p) for p in candidates)
    )


def load_model_classes():
    """
    Load exact model definitions from previous phases.
    This keeps architecture consistent with Phase 1/2/3/4.
    """
    model_2d_path = find_existing_file([
        PHASE1_DIR / "model_2d.py",
        PROJECT_ROOT / "model_2d.py",
    ])

    model_3d_path = find_existing_file([
        PHASE2_DIR / "model_3d.py",
        PROJECT_ROOT / "model_3d.py",
    ])

    gated_path = PHASE3_DIR / "model_gated_fusion.py"
    quality_path = PHASE4_DIR / "model_quality_aware_fusion.py"

    model_2d_module = load_module_from_file("phase5_model_2d_exact", model_2d_path)
    model_3d_module = load_module_from_file("phase5_model_3d_exact", model_3d_path)
    gated_module = load_module_from_file("phase5_model_gated_exact", gated_path)
    quality_module = load_module_from_file("phase5_model_quality_exact", quality_path)

    return {
        "FallCNNLSTM": model_2d_module.FallCNNLSTM,
        "FallCNNLSTM3D": model_3d_module.FallCNNLSTM3D,
        "build_gated_fusion_model": gated_module.build_gated_fusion_model,
        "QualityAwareGatedFusion": quality_module.QualityAwareGatedFusion,
        "ConcatWithQualityFusion": quality_module.ConcatWithQualityFusion,
    }


# ============================================================
# DATA LOADING
# ============================================================

def get_split_dir(config: Dict) -> Path:
    external_sequences_dir = cfg_path(config, config["outputs"]["external_sequences_dir"])
    return external_sequences_dir / "train_val_test_splits"


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing NPZ file: {path}")

    npz = np.load(path, allow_pickle=True)
    return {key: npz[key] for key in npz.files}


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest file: {path}")

    df = pd.read_csv(path)

    required = [
        "dataset",
        "video_id",
        "sequence_key",
        "label",
        "label_name",
        "split",
        "group_key",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    return df.reset_index(drop=True)


def load_splits(config: Dict) -> Dict[str, Any]:
    split_dir = get_split_dir(config)

    paths = {
        "train_npz": split_dir / "external_train_inputs.npz",
        "val_npz": split_dir / "external_val_inputs.npz",
        "test_npz": split_dir / "external_test_inputs.npz",
        "train_manifest": split_dir / "external_train_manifest.csv",
        "val_manifest": split_dir / "external_val_manifest.csv",
        "test_manifest": split_dir / "external_test_manifest.csv",
        "all_split_manifest": split_dir / "external_train_val_test_split.csv",
    }

    arrays = {
        "train": load_npz(paths["train_npz"]),
        "val": load_npz(paths["val_npz"]),
        "test": load_npz(paths["test_npz"]),
    }

    manifests = {
        "train": load_manifest(paths["train_manifest"]),
        "val": load_manifest(paths["val_manifest"]),
        "test": load_manifest(paths["test_manifest"]),
        "all": load_manifest(paths["all_split_manifest"]),
    }

    validate_split_alignment(arrays, manifests)

    return {
        "arrays": arrays,
        "manifests": manifests,
        "paths": paths,
    }


def validate_split_alignment(arrays: Dict[str, Dict[str, np.ndarray]], manifests: Dict[str, pd.DataFrame]):
    for split_name in ["train", "val", "test"]:
        arr = arrays[split_name]
        df = manifests[split_name]

        required_arrays = [
            "x2d_common",
            "x3d_common",
            "xconcat_common",
            "x2d_quality",
            "x3d_quality",
            "quality_raw",
            "y_binary",
            "sequence_keys",
        ]

        for key in required_arrays:
            if key not in arr:
                raise ValueError(f"{split_name} npz missing array: {key}")

        n = len(df)

        if arr["y_binary"].shape[0] != n:
            raise ValueError(
                f"{split_name}: y_binary length {arr['y_binary'].shape[0]} != manifest length {n}"
            )

        npz_keys = [str(x) for x in arr["sequence_keys"].tolist()]
        manifest_keys = df["sequence_key"].astype(str).tolist()

        if npz_keys != manifest_keys:
            raise ValueError(
                f"{split_name}: sequence_keys in NPZ do not match manifest order."
            )

        y_npz = arr["y_binary"].astype(int)
        y_manifest = df["label"].astype(int).to_numpy()

        if not np.array_equal(y_npz, y_manifest):
            raise ValueError(f"{split_name}: y_binary does not match label column.")

        check_shapes(split_name, arr)


def check_shapes(split_name: str, arr: Dict[str, np.ndarray]):
    expected = {
        "x2d_common": (SEQUENCE_LENGTH, EXPECTED_2D_DIM),
        "x3d_common": (SEQUENCE_LENGTH, EXPECTED_3D_DIM),
        "xconcat_common": (SEQUENCE_LENGTH, EXPECTED_CONCAT_DIM),
        "x2d_quality": (SEQUENCE_LENGTH, EXPECTED_2D_DIM),
        "x3d_quality": (SEQUENCE_LENGTH, EXPECTED_3D_DIM),
        "quality_raw": (EXPECTED_QUALITY_DIM,),
    }

    for key, suffix_shape in expected.items():
        if key not in arr:
            raise ValueError(f"{split_name}: missing {key}")

        actual = tuple(arr[key].shape[1:])

        if actual != suffix_shape:
            raise ValueError(
                f"{split_name}: {key} shape wrong. Expected (*,{suffix_shape}), got {arr[key].shape}"
            )

        values = arr[key]

        if np.isnan(values.astype(np.float32)).any():
            raise ValueError(f"{split_name}: NaN found in {key}")

        if np.isinf(values.astype(np.float32)).any():
            raise ValueError(f"{split_name}: Inf found in {key}")


# ============================================================
# DATASET
# ============================================================

class ExternalFineTuneDataset(Dataset):
    def __init__(
        self,
        arrays: Dict[str, np.ndarray],
        manifest_df: pd.DataFrame,
        quality_scaler: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.arrays = arrays
        self.manifest_df = manifest_df.reset_index(drop=True)
        self.quality_scaler = quality_scaler

    def __len__(self):
        return len(self.manifest_df)

    def _quality(self, idx: int) -> np.ndarray:
        q = self.arrays["quality_raw"][idx].astype(np.float32)

        if self.quality_scaler is not None:
            mean = self.quality_scaler["mean"]
            std = self.quality_scaler["std"]
            q = (q - mean) / std

        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        return q.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.manifest_df.iloc[idx]

        return {
            "index": torch.tensor(idx, dtype=torch.long),
            "x2d_common": torch.tensor(self.arrays["x2d_common"][idx], dtype=torch.float32),
            "x3d_common": torch.tensor(self.arrays["x3d_common"][idx], dtype=torch.float32),
            "xconcat_common": torch.tensor(self.arrays["xconcat_common"][idx], dtype=torch.float32),
            "x2d_quality": torch.tensor(self.arrays["x2d_quality"][idx], dtype=torch.float32),
            "x3d_quality": torch.tensor(self.arrays["x3d_quality"][idx], dtype=torch.float32),
            "quality": torch.tensor(self._quality(idx), dtype=torch.float32),
            "label": torch.tensor(int(self.arrays["y_binary"][idx]), dtype=torch.long),
            "sequence_key": str(row["sequence_key"]),
            "dataset": str(row["dataset"]),
            "video_id": str(row["video_id"]),
            "group_key": str(row["group_key"]),
            "label_name": str(row["label_name"]),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensor_keys = [
        "index",
        "x2d_common",
        "x3d_common",
        "xconcat_common",
        "x2d_quality",
        "x3d_quality",
        "quality",
        "label",
    ]

    out = {}

    for key in tensor_keys:
        out[key] = torch.stack([item[key] for item in batch], dim=0)

    for key in ["sequence_key", "dataset", "video_id", "group_key", "label_name"]:
        out[key] = [item[key] for item in batch]

    return out


# ============================================================
# QUALITY SCALER
# ============================================================

def fit_quality_scaler(train_arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    q = train_arrays["quality_raw"].astype(np.float32)

    mean = q.mean(axis=0).astype(np.float32)
    std = q.std(axis=0).astype(np.float32)

    std[std < 1e-6] = 1.0

    if mean.shape[0] != EXPECTED_QUALITY_DIM or std.shape[0] != EXPECTED_QUALITY_DIM:
        raise ValueError(f"Quality scaler wrong shape: mean={mean.shape}, std={std.shape}")

    return {
        "mean": mean,
        "std": std,
    }


def scaler_to_jsonable(scaler: Optional[Dict[str, np.ndarray]]) -> Optional[Dict[str, List[float]]]:
    if scaler is None:
        return None

    return {
        "mean": scaler["mean"].astype(float).tolist(),
        "std": scaler["std"].astype(float).tolist(),
    }


# ============================================================
# CHECKPOINT + MODEL BUILDING
# ============================================================

def resolve_checkpoint_path(config: Dict, checkpoint_key: str) -> Path:
    checkpoints = config.get("checkpoints", {})

    if checkpoint_key not in checkpoints:
        raise KeyError(f"Checkpoint key not found in phase5_config.yaml: {checkpoint_key}")

    path = cfg_path(config, checkpoints[checkpoint_key])

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for {checkpoint_key}:\n{path}"
        )

    return path


def load_checkpoint_payload(path: Path, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict):
        return checkpoint

    return {
        "model_state_dict": checkpoint,
    }


def get_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    return checkpoint


def checkpoint_args(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    args = checkpoint.get("args", {})

    if args is None:
        return {}

    return dict(args)


def build_model_from_spec(
    spec: Dict[str, Any],
    checkpoint: Dict[str, Any],
    model_classes: Dict[str, Any],
) -> nn.Module:
    family = spec["model_family"]

    if family == "single_2d":
        return model_classes["FallCNNLSTM"](
            input_dim=40,
            num_classes=2,
            cnn_channels=int(checkpoint.get("cnn_channels", 128)),
            lstm_hidden=int(checkpoint.get("lstm_hidden", 128)),
            lstm_layers=int(checkpoint.get("lstm_layers", 1)),
            dropout=float(checkpoint.get("dropout", 0.3)),
        )

    if family == "single_3d":
        return model_classes["FallCNNLSTM3D"](
            input_dim=59,
            num_classes=2,
            cnn_channels=int(checkpoint.get("cnn_channels", 128)),
            lstm_hidden=int(checkpoint.get("lstm_hidden", 128)),
            lstm_layers=int(checkpoint.get("lstm_layers", 1)),
            dropout=float(checkpoint.get("dropout", 0.3)),
        )

    if family == "single_concat":
        return model_classes["FallCNNLSTM3D"](
            input_dim=99,
            num_classes=2,
            cnn_channels=int(checkpoint.get("cnn_channels", 128)),
            lstm_hidden=int(checkpoint.get("lstm_hidden", 128)),
            lstm_layers=int(checkpoint.get("lstm_layers", 1)),
            dropout=float(checkpoint.get("dropout", 0.3)),
        )

    if family == "gated":
        return model_classes["build_gated_fusion_model"](
            task="binary",
            cnn_channels=int(checkpoint.get("cnn_channels", 128)),
            lstm_hidden=int(checkpoint.get("lstm_hidden", 128)),
            lstm_layers=int(checkpoint.get("lstm_layers", 1)),
            dropout=float(checkpoint.get("dropout", 0.3)),
            gate_hidden=int(checkpoint.get("gate_hidden", 128)),
            use_vector_gate=bool(checkpoint.get("use_vector_gate", True)),
        )

    args = checkpoint_args(checkpoint)

    if family == "quality_gated":
        return model_classes["QualityAwareGatedFusion"](
            input_dim_2d=40,
            input_dim_3d=59,
            quality_dim=33,
            num_classes=2,
            encoder_dim=int(args.get("encoder_dim", 128)),
            cnn_channels=int(args.get("cnn_channels", 128)),
            lstm_hidden=int(args.get("lstm_hidden", 128)),
            quality_hidden=int(args.get("quality_hidden", 128)),
            fusion_dim=int(args.get("fusion_dim", 128)),
            dropout=float(args.get("dropout", 0.3)),
            gate_type=str(args.get("gate_type", "vector")),
            pooling=str(args.get("pooling", "attention")),
        )

    if family == "quality_concat":
        return model_classes["ConcatWithQualityFusion"](
            input_dim_2d=40,
            input_dim_3d=59,
            quality_dim=33,
            num_classes=2,
            encoder_dim=int(args.get("encoder_dim", 128)),
            cnn_channels=int(args.get("cnn_channels", 128)),
            lstm_hidden=int(args.get("lstm_hidden", 128)),
            quality_hidden=int(args.get("quality_hidden", 128)),
            fusion_dim=int(args.get("fusion_dim", 128)),
            dropout=float(args.get("dropout", 0.3)),
            pooling=str(args.get("pooling", "attention")),
        )

    raise ValueError(f"Unsupported model family: {family}")


def build_and_load_pretrained_model(
    spec: Dict[str, Any],
    config: Dict[str, Any],
    model_classes: Dict[str, Any],
    device: torch.device,
    allow_nonstrict: bool = False,
    from_scratch: bool = False,
) -> Tuple[nn.Module, Dict[str, Any], Optional[Path]]:
    checkpoint_path = resolve_checkpoint_path(config, spec["checkpoint_key"])
    checkpoint = load_checkpoint_payload(checkpoint_path, device=device)

    model = build_model_from_spec(spec, checkpoint, model_classes)
    model = model.to(device)

    if not from_scratch:
        state_dict = get_state_dict(checkpoint)

        if allow_nonstrict:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"WARNING: non-strict load for {spec['model_name']}")
                print("Missing keys:", missing[:20])
                print("Unexpected keys:", unexpected[:20])
        else:
            model.load_state_dict(state_dict, strict=True)

    return model, checkpoint, checkpoint_path


# ============================================================
# FORWARD
# ============================================================

def extract_logits(output: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
    aux = {}

    if isinstance(output, dict):
        if "logits" not in output:
            raise ValueError(f"Model output dict has no 'logits' key. Keys={list(output.keys())}")

        logits = output["logits"]

        for key in ["gate", "mean_gate", "quality_weight"]:
            if key in output:
                aux[key] = output[key]

        return logits, aux

    if isinstance(output, tuple):
        logits = output[0]

        if len(output) > 1:
            aux["gate"] = output[1]

        return logits, aux

    return output, aux


def forward_model(
    model: nn.Module,
    spec: Dict[str, Any],
    batch: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    family = spec["model_family"]

    if family in ["single_2d", "single_3d", "single_concat"]:
        x = batch[spec["input_key"]].to(device)
        return extract_logits(model(x))

    if family == "gated":
        x2d = batch["x2d_common"].to(device)
        x3d = batch["x3d_common"].to(device)

        try:
            return extract_logits(model(x2d, x3d, return_gate=True))
        except TypeError:
            return extract_logits(model(x2d, x3d))

    if family in ["quality_gated", "quality_concat"]:
        x2d = batch["x2d_quality"].to(device)
        x3d = batch["x3d_quality"].to(device)
        quality = batch["quality"].to(device)

        return extract_logits(model(x2d, x3d, quality))

    raise ValueError(f"Unsupported model family: {family}")


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    p, r, f, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    tn, fp, fn, tp = cm.ravel().tolist()

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "accuracy_percent": float(acc) * 100.0,
        "macro_f1_percent": float(macro) * 100.0,
        "not_fall_precision": float(p[0]),
        "not_fall_recall": float(r[0]),
        "not_fall_f1": float(f[0]),
        "not_fall_support": int(s[0]),
        "fall_precision": float(p[1]),
        "fall_recall": float(r[1]),
        "fall_f1": float(f[1]),
        "fall_support": int(s[1]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def class_weights_from_train(train_arrays: Dict[str, np.ndarray], device: torch.device) -> torch.Tensor:
    y = train_arrays["y_binary"].astype(int)

    counts = np.bincount(y, minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0

    weights = counts.sum() / (2.0 * counts)

    return torch.tensor(weights, dtype=torch.float32, device=device)


# ============================================================
# TRAIN / EVAL LOOPS
# ============================================================

def train_one_epoch(
    model: nn.Module,
    spec: Dict[str, Any],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> Dict[str, Any]:
    model.train()

    total_loss = 0.0
    total_count = 0

    all_true = []
    all_pred = []

    for batch in loader:
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits, _ = forward_model(
            model=model,
            spec=spec,
            batch=batch,
            device=device,
        )

        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

        preds = torch.argmax(logits, dim=1)

        all_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
        all_pred.extend(preds.detach().cpu().numpy().astype(int).tolist())

    metrics = compute_metrics(np.asarray(all_true), np.asarray(all_pred))
    metrics["loss"] = total_loss / max(total_count, 1)

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    spec: Dict[str, Any],
    loader: DataLoader,
    criterion: Optional[nn.Module],
    device: torch.device,
    return_predictions: bool = False,
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    model.eval()

    total_loss = 0.0
    total_count = 0

    all_indices = []
    all_true = []
    all_pred = []
    all_prob_not_fall = []
    all_prob_fall = []

    all_sequence_keys = []
    all_datasets = []
    all_video_ids = []
    all_group_keys = []

    all_gate_mean = []

    for batch in loader:
        labels = batch["label"].to(device)

        logits, aux = forward_model(
            model=model,
            spec=spec,
            batch=batch,
            device=device,
        )

        if criterion is not None:
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total_count += labels.size(0)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_indices.extend(batch["index"].detach().cpu().numpy().astype(int).tolist())
        all_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
        all_pred.extend(preds.detach().cpu().numpy().astype(int).tolist())

        all_prob_not_fall.extend(probs[:, 0].detach().cpu().numpy().astype(float).tolist())
        all_prob_fall.extend(probs[:, 1].detach().cpu().numpy().astype(float).tolist())

        all_sequence_keys.extend(batch["sequence_key"])
        all_datasets.extend(batch["dataset"])
        all_video_ids.extend(batch["video_id"])
        all_group_keys.extend(batch["group_key"])

        gate_values = None

        if "mean_gate" in aux:
            gate_tensor = aux["mean_gate"]
            gate_values = gate_tensor.detach().cpu().numpy()
        elif "gate" in aux:
            gate_tensor = aux["gate"]
            gate_values = gate_tensor.detach().cpu().numpy()

        if gate_values is None:
            all_gate_mean.extend([np.nan] * labels.size(0))
        else:
            if gate_values.ndim > 1:
                gate_values = gate_values.mean(axis=1)
            all_gate_mean.extend(gate_values.astype(float).tolist())

    metrics = compute_metrics(np.asarray(all_true), np.asarray(all_pred))
    metrics["loss"] = total_loss / max(total_count, 1) if criterion is not None else np.nan

    pred_df = None

    if return_predictions:
        pred_df = pd.DataFrame({
            "row_index": all_indices,
            "sequence_key": all_sequence_keys,
            "dataset": all_datasets,
            "video_id": all_video_ids,
            "group_key": all_group_keys,
            "y_true": all_true,
            "y_pred": all_pred,
            "prob_not_fall": all_prob_not_fall,
            "prob_fall": all_prob_fall,
            "gate_mean": all_gate_mean,
        })

    return metrics, pred_df


# ============================================================
# OUTPUT HELPERS
# ============================================================

def make_output_dirs(config: Dict[str, Any]) -> Dict[str, Path]:
    base_dir = PHASE5_DIR / "outputs" / "external_finetuning"

    dirs = {
        "base": base_dir,
        "checkpoints": base_dir / "checkpoints",
        "predictions": base_dir / "predictions",
        "logs": base_dir / "logs",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def jsonable_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out = {}

    for key, value in metrics.items():
        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, np.ndarray):
            value = value.tolist()

        out[key] = value

    return out


def save_checkpoint(
    path: Path,
    model: nn.Module,
    spec: Dict[str, Any],
    epoch: int,
    val_metrics: Dict[str, Any],
    test_metrics: Optional[Dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    pretrained_checkpoint_path: Optional[Path],
    quality_scaler: Optional[Dict[str, np.ndarray]],
    args: argparse.Namespace,
):
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_name": spec["model_name"],
        "display_name": spec["display_name"],
        "model_family": spec["model_family"],
        "epoch": int(epoch),
        "val_metrics": jsonable_metrics(val_metrics),
        "test_metrics": jsonable_metrics(test_metrics) if test_metrics is not None else None,
        "pretrained_checkpoint_path": str(pretrained_checkpoint_path) if pretrained_checkpoint_path is not None else None,
        "quality_scaler": scaler_to_jsonable(quality_scaler),
        "args": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "patience": args.patience,
            "seed": args.seed,
            "from_scratch": args.from_scratch,
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def build_loaders_for_model(
    spec: Dict[str, Any],
    split_data: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    quality_scaler: Optional[Dict[str, np.ndarray]],
) -> Dict[str, DataLoader]:
    arrays = split_data["arrays"]
    manifests = split_data["manifests"]

    train_dataset = ExternalFineTuneDataset(
        arrays=arrays["train"],
        manifest_df=manifests["train"],
        quality_scaler=quality_scaler if spec["uses_quality"] else None,
    )

    val_dataset = ExternalFineTuneDataset(
        arrays=arrays["val"],
        manifest_df=manifests["val"],
        quality_scaler=quality_scaler if spec["uses_quality"] else None,
    )

    test_dataset = ExternalFineTuneDataset(
        arrays=arrays["test"],
        manifest_df=manifests["test"],
        quality_scaler=quality_scaler if spec["uses_quality"] else None,
    )

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        ),
    }


def train_one_model(
    spec: Dict[str, Any],
    config: Dict[str, Any],
    split_data: Dict[str, Any],
    model_classes: Dict[str, Any],
    dirs: Dict[str, Path],
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 120)
    print(f"Training model: {spec['display_name']}")
    print("=" * 120)

    quality_scaler = None

    if spec["uses_quality"]:
        quality_scaler = fit_quality_scaler(split_data["arrays"]["train"])
        print("Quality scaler fitted on external TRAIN only.")

    loaders = build_loaders_for_model(
        spec=spec,
        split_data=split_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        quality_scaler=quality_scaler,
    )

    model, pretrained_checkpoint, pretrained_checkpoint_path = build_and_load_pretrained_model(
        spec=spec,
        config=config,
        model_classes=model_classes,
        device=device,
        allow_nonstrict=args.allow_nonstrict,
        from_scratch=args.from_scratch,
    )

    print(f"Initial checkpoint: {pretrained_checkpoint_path}")
    print(f"Training mode      : {'from scratch' if args.from_scratch else 'fine-tune from previous best'}")

    class_weights = class_weights_from_train(split_data["arrays"]["train"], device=device)

    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
        print("Loss: CrossEntropyLoss without class weights.")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Loss: CrossEntropyLoss with class weights: {class_weights.detach().cpu().numpy().tolist()}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(1, args.lr_patience),
    )

    best_val_macro = -1.0
    best_epoch = -1
    best_checkpoint_path = dirs["checkpoints"] / f"best_{spec['model_name']}_external.pt"

    bad_epochs = 0
    epoch_rows = []

    for epoch in range(1, args.epochs + 1):
        print("\n" + "-" * 120)
        print(f"{spec['model_name']} | Epoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(
            model=model,
            spec=spec,
            loader=loaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=args.grad_clip,
        )

        val_metrics, _ = evaluate(
            model=model,
            spec=spec,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
            return_predictions=False,
        )

        scheduler.step(val_metrics["macro_f1"])

        row = {
            "model_name": spec["model_name"],
            "display_name": spec["display_name"],
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_fall_recall": train_metrics["fall_recall"],
            "train_fall_f1": train_metrics["fall_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_fall_recall": val_metrics["fall_recall"],
            "val_fall_f1": val_metrics["fall_f1"],
            "val_not_fall_f1": val_metrics["not_fall_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }

        epoch_rows.append(row)

        print(
            f"Train: loss={train_metrics['loss']:.4f} | "
            f"acc={train_metrics['accuracy_percent']:.2f}% | "
            f"macro_f1={train_metrics['macro_f1_percent']:.2f}% | "
            f"fall_recall={train_metrics['fall_recall']:.4f}"
        )

        print(
            f"Val  : loss={val_metrics['loss']:.4f} | "
            f"acc={val_metrics['accuracy_percent']:.2f}% | "
            f"macro_f1={val_metrics['macro_f1_percent']:.2f}% | "
            f"fall_recall={val_metrics['fall_recall']:.4f} | "
            f"fall_f1={val_metrics['fall_f1']:.4f}"
        )

        improved = val_metrics["macro_f1"] > best_val_macro + 1e-6

        if improved:
            best_val_macro = val_metrics["macro_f1"]
            best_epoch = epoch
            bad_epochs = 0

            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                spec=spec,
                epoch=epoch,
                val_metrics=val_metrics,
                test_metrics=None,
                optimizer=optimizer,
                pretrained_checkpoint_path=pretrained_checkpoint_path,
                quality_scaler=quality_scaler,
                args=args,
            )

            print(f"Saved new best checkpoint: {best_checkpoint_path}")
        else:
            bad_epochs += 1
            print(f"No improvement. bad_epochs={bad_epochs}/{args.patience}")

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    # Load best checkpoint for final test
    best_payload = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    model.eval()

    val_metrics, val_pred_df = evaluate(
        model=model,
        spec=spec,
        loader=loaders["val"],
        criterion=criterion,
        device=device,
        return_predictions=True,
    )

    test_metrics, test_pred_df = evaluate(
        model=model,
        spec=spec,
        loader=loaders["test"],
        criterion=criterion,
        device=device,
        return_predictions=True,
    )

    # Save checkpoint again with test metrics included.
    save_checkpoint(
        path=best_checkpoint_path,
        model=model,
        spec=spec,
        epoch=best_epoch,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        optimizer=optimizer,
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        quality_scaler=quality_scaler,
        args=args,
    )

    val_pred_df.insert(0, "model_name", spec["model_name"])
    test_pred_df.insert(0, "model_name", spec["model_name"])

    val_pred_path = dirs["predictions"] / f"val_predictions_{spec['model_name']}.csv"
    test_pred_path = dirs["predictions"] / f"test_predictions_{spec['model_name']}.csv"

    val_pred_df.to_csv(val_pred_path, index=False, encoding="utf-8-sig")
    test_pred_df.to_csv(test_pred_path, index=False, encoding="utf-8-sig")

    epoch_df = pd.DataFrame(epoch_rows)

    result = {
        "model_name": spec["model_name"],
        "display_name": spec["display_name"],
        "model_family": spec["model_family"],
        "best_epoch": int(best_epoch),
        "best_checkpoint_path": str(best_checkpoint_path),
        "pretrained_checkpoint_path": str(pretrained_checkpoint_path),
        "uses_quality": bool(spec["uses_quality"]),
        "from_scratch": bool(args.from_scratch),
        "val_accuracy": float(val_metrics["accuracy"]),
        "val_macro_f1": float(val_metrics["macro_f1"]),
        "val_fall_recall": float(val_metrics["fall_recall"]),
        "val_fall_f1": float(val_metrics["fall_f1"]),
        "val_not_fall_f1": float(val_metrics["not_fall_f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_weighted_f1": float(test_metrics["weighted_f1"]),
        "test_not_fall_f1": float(test_metrics["not_fall_f1"]),
        "test_fall_precision": float(test_metrics["fall_precision"]),
        "test_fall_recall": float(test_metrics["fall_recall"]),
        "test_fall_f1": float(test_metrics["fall_f1"]),
        "test_tn": int(test_metrics["tn"]),
        "test_fp": int(test_metrics["fp"]),
        "test_fn": int(test_metrics["fn"]),
        "test_tp": int(test_metrics["tp"]),
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "val_predictions_path": str(val_pred_path),
        "test_predictions_path": str(test_pred_path),
    }

    print("\nFinal TEST result")
    print("-" * 120)
    print(f"Accuracy    : {test_metrics['accuracy_percent']:.2f}%")
    print(f"Macro F1    : {test_metrics['macro_f1_percent']:.2f}%")
    print(f"Fall Recall : {test_metrics['fall_recall']:.4f}")
    print(f"Fall F1     : {test_metrics['fall_f1']:.4f}")
    print(f"NotFall F1  : {test_metrics['not_fall_f1']:.4f}")
    print(
        f"Confusion   : TN={test_metrics['tn']} FP={test_metrics['fp']} "
        f"FN={test_metrics['fn']} TP={test_metrics['tp']}"
    )

    return result, epoch_df, test_pred_df


# ============================================================
# SAVE FINAL OUTPUTS
# ============================================================

def save_final_outputs(
    dirs: Dict[str, Path],
    results: List[Dict[str, Any]],
    all_epoch_df: pd.DataFrame,
    all_test_pred_df: pd.DataFrame,
    split_data: Dict[str, Any],
    args: argparse.Namespace,
    started_at: float,
):
    metrics_df = pd.DataFrame(results)

    order_cols = [
        "model_name",
        "display_name",
        "model_family",
        "best_epoch",
        "test_accuracy",
        "test_macro_f1",
        "test_fall_recall",
        "test_fall_f1",
        "test_not_fall_f1",
        "test_tn",
        "test_fp",
        "test_fn",
        "test_tp",
        "val_macro_f1",
        "val_fall_recall",
        "uses_quality",
        "from_scratch",
        "best_checkpoint_path",
        "pretrained_checkpoint_path",
    ]

    order_cols = [c for c in order_cols if c in metrics_df.columns]
    rest_cols = [c for c in metrics_df.columns if c not in order_cols]
    metrics_df = metrics_df[order_cols + rest_cols]

    metrics_csv = dirs["base"] / "external_finetuned_all_models_metrics.csv"
    metrics_json = dirs["base"] / "external_finetuned_all_models_metrics.json"
    epochs_csv = dirs["base"] / "external_finetuned_all_models_per_epoch.csv"
    predictions_csv = dirs["base"] / "external_finetuned_all_models_test_predictions_long.csv"
    report_json = dirs["base"] / "07_finetune_all_models_external_report.json"

    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    all_epoch_df.to_csv(epochs_csv, index=False, encoding="utf-8-sig")
    all_test_pred_df.to_csv(predictions_csv, index=False, encoding="utf-8-sig")

    json_metrics = []

    for item in results:
        fixed = {}

        for k, v in item.items():
            if isinstance(v, np.generic):
                v = v.item()
            if isinstance(v, np.ndarray):
                v = v.tolist()

            fixed[k] = v

        json_metrics.append(fixed)

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(json_metrics, f, ensure_ascii=False, indent=4)

    best_macro = metrics_df.sort_values("test_macro_f1", ascending=False).head(1).to_dict(orient="records")[0]
    best_fall_recall = metrics_df.sort_values("test_fall_recall", ascending=False).head(1).to_dict(orient="records")[0]
    best_fall_f1 = metrics_df.sort_values("test_fall_f1", ascending=False).head(1).to_dict(orient="records")[0]

    train_manifest = split_data["manifests"]["train"]
    val_manifest = split_data["manifests"]["val"]
    test_manifest = split_data["manifests"]["test"]

    report = {
        "phase": "Phase 5 - External Dataset Adaptation",
        "step": "07_finetune_all_models_external",
        "goal": (
            "Fine-tune/train all Phase 1-4 model variants on the same external train split, "
            "select best checkpoint using the same external validation split, and compare all models "
            "on the same external test split."
        ),
        "elapsed_sec": float(time.time() - started_at),
        "num_models": int(len(results)),
        "training_mode": "from_scratch" if args.from_scratch else "fine_tune_from_previous_best",
        "fairness_rules": [
            "All models use the same train/val/test split created by Step 06.",
            "No sequence rebuilding is performed in Step 07.",
            "Quality scaler is fitted only on external train split and reused for val/test.",
            "Best checkpoint is selected by validation Macro F1.",
            "Final comparison uses only the external test split.",
        ],
        "split_summary": {
            "train": {
                "num_sequences": int(len(train_manifest)),
                "label_counts": train_manifest["label_name"].value_counts().to_dict(),
                "num_groups": int(train_manifest["group_key"].nunique()),
            },
            "val": {
                "num_sequences": int(len(val_manifest)),
                "label_counts": val_manifest["label_name"].value_counts().to_dict(),
                "num_groups": int(val_manifest["group_key"].nunique()),
            },
            "test": {
                "num_sequences": int(len(test_manifest)),
                "label_counts": test_manifest["label_name"].value_counts().to_dict(),
                "num_groups": int(test_manifest["group_key"].nunique()),
            },
        },
        "best_by_test_macro_f1": best_macro,
        "best_by_test_fall_recall": best_fall_recall,
        "best_by_test_fall_f1": best_fall_f1,
        "outputs": {
            "metrics_csv": str(metrics_csv),
            "metrics_json": str(metrics_json),
            "epochs_csv": str(epochs_csv),
            "predictions_csv": str(predictions_csv),
            "report_json": str(report_json),
            "checkpoints_dir": str(dirs["checkpoints"]),
            "predictions_dir": str(dirs["predictions"]),
        },
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    return {
        "metrics_csv": str(metrics_csv),
        "metrics_json": str(metrics_json),
        "epochs_csv": str(epochs_csv),
        "predictions_csv": str(predictions_csv),
        "report_json": str(report_json),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 Step 07 - Fine-tune all models on external train split and test fairly."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(PHASE5_DIR / "phase5_config.yaml"),
        help="Path to phase5_config.yaml",
    )

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model filter, e.g. --models phase4_quality_concat phase1_2d_common",
    )

    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Train from random initialization instead of fine-tuning previous best checkpoints.",
    )

    parser.add_argument(
        "--allow-nonstrict",
        action="store_true",
        help="Allow non-strict checkpoint loading. Use only for debugging.",
    )

    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weights in CrossEntropyLoss.",
    )

    args = parser.parse_args()

    started_at = time.time()

    print("\nPhase 5 - Step 07: Fine-tune All Models on External Dataset")
    print("=" * 120)
    print("Correct protocol:")
    print("  1. Load train/val/test split from Step 06")
    print("  2. Fine-tune/train all 6 models on the same external train split")
    print("  3. Select best checkpoint by validation Macro F1")
    print("  4. Test all models on the same external test split")
    print("  5. Compare whether Phase 4 experiments remain valid on adapted external data")
    print("=" * 120)

    set_seed(args.seed)

    config = load_config(args.config)

    runtime_cfg = config.get("runtime", {})
    device_name = str(runtime_cfg.get("device", "cuda"))

    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Training mode: {'from scratch' if args.from_scratch else 'fine-tune previous best checkpoints'}")

    print("\n[1/5] Loading Step 06 train/val/test splits...")
    split_data = load_splits(config)

    for split_name in ["train", "val", "test"]:
        df = split_data["manifests"][split_name]
        print(
            f"{split_name.upper():5s}: "
            f"sequences={len(df)} | "
            f"groups={df['group_key'].nunique()} | "
            f"labels={df['label_name'].value_counts().to_dict()}"
        )

    print("\n[2/5] Loading exact model classes from previous phases...")
    model_classes = load_model_classes()

    specs = MODEL_SPECS

    if args.models:
        selected = set(args.models)
        specs = [spec for spec in specs if spec["model_name"] in selected]

    if not specs:
        raise RuntimeError("No models selected.")

    print("\nModels to train:")
    for spec in specs:
        print(f"- {spec['model_name']}")

    print("\n[3/5] Preparing output folders...")
    dirs = make_output_dirs(config)

    print_dict("Output dirs", {k: str(v) for k, v in dirs.items()})

    print("\n[4/5] Training all selected models...")

    results = []
    epoch_frames = []
    test_prediction_frames = []

    for spec in specs:
        result, epoch_df, test_pred_df = train_one_model(
            spec=spec,
            config=config,
            split_data=split_data,
            model_classes=model_classes,
            dirs=dirs,
            device=device,
            args=args,
        )

        results.append(result)
        epoch_frames.append(epoch_df)
        test_prediction_frames.append(test_pred_df)

    all_epoch_df = pd.concat(epoch_frames, axis=0, ignore_index=True)
    all_test_pred_df = pd.concat(test_prediction_frames, axis=0, ignore_index=True)

    print("\n[5/5] Saving final comparison outputs...")
    outputs = save_final_outputs(
        dirs=dirs,
        results=results,
        all_epoch_df=all_epoch_df,
        all_test_pred_df=all_test_pred_df,
        split_data=split_data,
        args=args,
        started_at=started_at,
    )

    print_dict("Saved outputs", outputs)

    metrics_df = pd.DataFrame(results)
    metrics_df = metrics_df.sort_values("test_macro_f1", ascending=False).reset_index(drop=True)

    print("\nFinal ranking by TEST Macro F1")
    print("=" * 120)

    show_cols = [
        "model_name",
        "best_epoch",
        "test_accuracy",
        "test_macro_f1",
        "test_fall_recall",
        "test_fall_f1",
        "test_not_fall_f1",
        "test_tn",
        "test_fp",
        "test_fn",
        "test_tp",
    ]

    print(metrics_df[show_cols].to_string(index=False))

    best_macro = metrics_df.iloc[0]
    best_fall = metrics_df.sort_values("test_fall_recall", ascending=False).iloc[0]
    best_fall_f1 = metrics_df.sort_values("test_fall_f1", ascending=False).iloc[0]

    print("\nBest by TEST Macro F1:")
    print(f"- {best_macro['model_name']} | Macro F1={best_macro['test_macro_f1']*100:.2f}%")

    print("\nBest by TEST Fall Recall:")
    print(f"- {best_fall['model_name']} | Fall Recall={best_fall['test_fall_recall']:.4f}")

    print("\nBest by TEST Fall F1:")
    print(f"- {best_fall_f1['model_name']} | Fall F1={best_fall_f1['test_fall_f1']:.4f}")

    print("\nDONE: Phase 5 Step 07 completed.")
    print("=" * 120)
    print("Next step:")
    print("  08_compare_external_finetuning_results.py")
    print("=" * 120)


if __name__ == "__main__":
    main()