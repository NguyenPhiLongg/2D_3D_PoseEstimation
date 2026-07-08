import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


"""
All-Phases Robustness Evaluation.

This script evaluates robustness for all fair models:

    Phase 1:
        - Phase 1 - 2D Common

    Phase 2:
        - Phase 2 - 3D Common
        - Phase 2 - 2D+3D Concat Common

    Phase 3:
        - Phase 3 - Gated Fusion

    Phase 4:
        - Phase 4 - Quality-Gated
        - Phase 4 - Quality-Concat

Important design:
    Phase 1/2/3 models are evaluated using Phase 3 feature pipeline:
        phase3_utils.build_dataloaders(...)

    Phase 4 models are evaluated using Phase 4 feature pipeline:
        phase4_utils.create_phase4_dataloaders(...)

This avoids the previous mistake:
    Do not use Phase 4 dataloader to evaluate Phase 1/2/3 checkpoints.

Outputs:
    phase4_quality_aware_fusion/outputs/robustness/
        all_models_robustness_binary.csv
        all_models_robustness_binary.json
        all_models_robustness_binary.md

        all_models_robustness_action.csv
        all_models_robustness_action.json
        all_models_robustness_action.md
"""


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PHASE1_DIR = PROJECT_ROOT / "phase1_2d_baseline"
PHASE2_DIR = PROJECT_ROOT / "phase2_3d_upgrade"
PHASE3_DIR = PROJECT_ROOT / "phase3_common_set_gated_fusion"
PHASE4_DIR = PROJECT_ROOT / "phase4_quality_aware_fusion"

PHASE3_CKPT_DIR = PHASE3_DIR / "checkpoints"
PHASE4_CKPT_DIR = PHASE4_DIR / "checkpoints"

ROBUSTNESS_OUTPUT_DIR = PHASE4_DIR / "outputs" / "robustness"

for path in [
    str(PHASE4_DIR),
    str(PHASE3_DIR),
    str(PHASE2_DIR),
    str(PHASE1_DIR),
    str(PROJECT_ROOT),
]:
    if path not in sys.path:
        sys.path.insert(0, path)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

import phase3_utils as p3
import phase4_utils as p4

from model_2d import FallCNNLSTM
from model_3d import FallCNNLSTM3D
from model_gated_fusion import build_gated_fusion_model

from model_quality_aware_fusion import (
    QualityAwareGatedFusion,
    ConcatWithQualityFusion,
    count_parameters,
)


# ============================================================
# CLEAN REFERENCES
# ============================================================

CLEAN_REFERENCE = {
    "binary": {
        "phase1_2d_common": {
            "accuracy": 0.9346,
            "macro_f1": 0.9234,
        },
        "phase2_3d_common": {
            "accuracy": 0.9272,
            "macro_f1": 0.9140,
        },
        "phase2_concat_common": {
            "accuracy": 0.9393,
            "macro_f1": 0.9299,
        },
        "phase3_gated_fusion": {
            "accuracy": 0.9379,
            "macro_f1": 0.9281,
        },
        "phase4_quality_gated": {
            "accuracy": 0.953457,
            "macro_f1": 0.945028,
        },
        "phase4_quality_concat": {
            "accuracy": 0.961551,
            "macro_f1": 0.954617,
        },
    },
    "action": {
        "phase1_2d_common": {
            "accuracy": 0.9698,
            "macro_f1": 0.9436,
        },
        "phase2_3d_common": {
            "accuracy": 0.9679,
            "macro_f1": 0.9470,
        },
        "phase2_concat_common": {
            "accuracy": 0.9864,
            "macro_f1": 0.9751,
        },
        "phase3_gated_fusion": {
            "accuracy": 0.9781,
            "macro_f1": 0.9618,
        },
        "phase4_quality_gated": {
            "accuracy": 0.979055,
            "macro_f1": 0.959619,
        },
        "phase4_quality_concat": {
            "accuracy": 0.978081,
            "macro_f1": 0.957310,
        },
    },
}


MODEL_SPECS = [
    {
        "model_id": "phase1_2d_common",
        "phase": "Phase 1",
        "model_name": "Phase 1 - 2D Common",
        "pipeline": "phase3",
        "feature_mode": "2d",
        "kind": "single",
        "input_dim": 40,
        "checkpoint_template": PHASE3_CKPT_DIR / "best_model_2d_common_{task}.pt",
    },
    {
        "model_id": "phase2_3d_common",
        "phase": "Phase 2",
        "model_name": "Phase 2 - 3D Common",
        "pipeline": "phase3",
        "feature_mode": "3d",
        "kind": "single",
        "input_dim": 59,
        "checkpoint_template": PHASE3_CKPT_DIR / "best_model_3d_common_{task}.pt",
    },
    {
        "model_id": "phase2_concat_common",
        "phase": "Phase 2",
        "model_name": "Phase 2 - 2D+3D Concat Common",
        "pipeline": "phase3",
        "feature_mode": "concat",
        "kind": "single",
        "input_dim": 99,
        "checkpoint_template": PHASE3_CKPT_DIR / "best_model_concat_common_{task}.pt",
    },
    {
        "model_id": "phase3_gated_fusion",
        "phase": "Phase 3",
        "model_name": "Phase 3 - Gated Fusion",
        "pipeline": "phase3",
        "feature_mode": "gated",
        "kind": "gated",
        "input_dim": None,
        "checkpoint_template": PHASE3_CKPT_DIR / "best_model_gated_fusion_{task}.pt",
    },
    {
        "model_id": "phase4_quality_gated",
        "phase": "Phase 4",
        "model_name": "Phase 4 - Quality-Gated",
        "pipeline": "phase4",
        "feature_mode": "quality_gated",
        "kind": "quality_gated",
        "input_dim": None,
        "checkpoint_template": PHASE4_CKPT_DIR / "best_model_quality_gated_{task}.pt",
    },
    {
        "model_id": "phase4_quality_concat",
        "phase": "Phase 4",
        "model_name": "Phase 4 - Quality-Concat",
        "pipeline": "phase4",
        "feature_mode": "quality_concat",
        "kind": "quality_concat",
        "input_dim": None,
        "checkpoint_template": PHASE4_CKPT_DIR / "best_model_quality_concat_{task}.pt",
    },
]


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, list):
        return [to_jsonable(v) for v in value]

    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    return value


def save_json(data: Dict, path: Path) -> None:
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=4)


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict):
        return checkpoint

    return {
        "model_state_dict": checkpoint,
    }


def get_state_dict(checkpoint: Dict) -> Dict:
    for key in ["model_state_dict", "state_dict", "model", "net"]:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    return checkpoint


def clean_state_dict_keys(state_dict: Dict) -> Dict:
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        cleaned[new_key] = value

    return cleaned


def get_class_names(task: str) -> List[str]:
    if task == "binary":
        return ["Not_Fall", "Fall"]

    if task == "action":
        return ["Sitting", "Sleeping", "Standing", "Walking"]

    raise ValueError(f"Invalid task: {task}")


def get_num_classes(task: str) -> int:
    return len(get_class_names(task))


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_percent(value: float) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value) * 100:.2f}%"


def get_selected_specs(models_arg: str) -> List[Dict]:
    if models_arg == "all":
        return MODEL_SPECS

    selected_ids = [
        item.strip()
        for item in models_arg.split(",")
        if item.strip()
    ]

    selected = [
        spec for spec in MODEL_SPECS
        if spec["model_id"] in selected_ids
    ]

    if not selected:
        raise ValueError(f"No valid models selected from: {models_arg}")

    return selected


# ============================================================
# SCENARIOS
# ============================================================

def get_scenarios() -> List[Dict]:
    return [
        {
            "scenario": "clean",
            "description": "No perturbation.",
            "noise_2d_std": 0.0,
            "noise_3d_std": 0.0,
            "missing_joint_rate": 0.0,
            "frame_drop_rate": 0.0,
        },
        {
            "scenario": "gaussian_2d_0.01",
            "description": "Small Gaussian noise on 2D coordinates.",
            "noise_2d_std": 0.01,
            "noise_3d_std": 0.0,
            "missing_joint_rate": 0.0,
            "frame_drop_rate": 0.0,
        },
        {
            "scenario": "gaussian_2d_0.03",
            "description": "Medium Gaussian noise on 2D coordinates.",
            "noise_2d_std": 0.03,
            "noise_3d_std": 0.0,
            "missing_joint_rate": 0.0,
            "frame_drop_rate": 0.0,
        },
        {
            "scenario": "gaussian_2d_0.05",
            "description": "Strong Gaussian noise on 2D coordinates.",
            "noise_2d_std": 0.05,
            "noise_3d_std": 0.0,
            "missing_joint_rate": 0.0,
            "frame_drop_rate": 0.0,
        },
        {
            "scenario": "gaussian_3d_0.01",
            "description": "Small Gaussian noise on 3D coordinates.",
            "noise_2d_std": 0.0,
            "noise_3d_std": 0.01,
            "missing_joint_rate": 0.0,
            "frame_drop_rate": 0.0,
        },
        {
            "scenario": "gaussian_3d_0.03",
            "description": "Medium Gaussian noise on 3D coordinates.",
            "noise_2d_std": 0.0,
            "noise_3d_std": 0.03,
            "missing_joint_rate": 0.0,
            "frame_drop_rate": 0.0,
        },
        {
            "scenario": "gaussian_3d_0.05",
            "description": "Strong Gaussian noise on 3D coordinates.",
            "noise_2d_std": 0.0,
            "noise_3d_std": 0.05,
            "missing_joint_rate": 0.0,
            "frame_drop_rate": 0.0,
        },
        {
            "scenario": "missing_joint_0.10",
            "description": "Randomly remove 10 percent of joints.",
            "noise_2d_std": 0.0,
            "noise_3d_std": 0.0,
            "missing_joint_rate": 0.10,
            "frame_drop_rate": 0.0,
        },
        {
            "scenario": "missing_joint_0.20",
            "description": "Randomly remove 20 percent of joints.",
            "noise_2d_std": 0.0,
            "noise_3d_std": 0.0,
            "missing_joint_rate": 0.20,
            "frame_drop_rate": 0.0,
        },
        {
            "scenario": "missing_joint_0.30",
            "description": "Randomly remove 30 percent of joints.",
            "noise_2d_std": 0.0,
            "noise_3d_std": 0.0,
            "missing_joint_rate": 0.30,
            "frame_drop_rate": 0.0,
        },
        {
            "scenario": "frame_drop_0.10",
            "description": "Drop 10 percent of frames by repeating previous frame.",
            "noise_2d_std": 0.0,
            "noise_3d_std": 0.0,
            "missing_joint_rate": 0.0,
            "frame_drop_rate": 0.10,
        },
        {
            "scenario": "frame_drop_0.20",
            "description": "Drop 20 percent of frames by repeating previous frame.",
            "noise_2d_std": 0.0,
            "noise_3d_std": 0.0,
            "missing_joint_rate": 0.0,
            "frame_drop_rate": 0.20,
        },
        {
            "scenario": "frame_drop_0.30",
            "description": "Drop 30 percent of frames by repeating previous frame.",
            "noise_2d_std": 0.0,
            "noise_3d_std": 0.0,
            "missing_joint_rate": 0.0,
            "frame_drop_rate": 0.30,
        },
        {
            "scenario": "combined_light",
            "description": "Light combined corruption.",
            "noise_2d_std": 0.02,
            "noise_3d_std": 0.02,
            "missing_joint_rate": 0.10,
            "frame_drop_rate": 0.10,
        },
        {
            "scenario": "combined_heavy",
            "description": "Heavy combined corruption.",
            "noise_2d_std": 0.05,
            "noise_3d_std": 0.05,
            "missing_joint_rate": 0.20,
            "frame_drop_rate": 0.20,
        },
    ]


# ============================================================
# PERTURBATION HELPERS
# ============================================================

def add_gaussian_noise_2d(x2d: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return x2d

    x2d = x2d.clone()
    x2d[:, :, :34] = x2d[:, :, :34] + torch.randn_like(x2d[:, :, :34]) * std

    return x2d


def add_gaussian_noise_3d(x3d: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return x3d

    x3d = x3d.clone()
    x3d[:, :, :51] = x3d[:, :, :51] + torch.randn_like(x3d[:, :, :51]) * std

    return x3d


def apply_missing_joints_2d(
    x2d: torch.Tensor,
    missing_joint_rate: float,
    joint_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if missing_joint_rate <= 0:
        if joint_mask is None:
            batch_size, seq_len, _ = x2d.shape
            joint_mask = torch.zeros(
                batch_size,
                seq_len,
                17,
                dtype=torch.bool,
                device=x2d.device,
            )

        return x2d, joint_mask

    x2d = x2d.clone()
    batch_size, seq_len, _ = x2d.shape

    if joint_mask is None:
        joint_mask = torch.rand(
            batch_size,
            seq_len,
            17,
            device=x2d.device,
        ) < missing_joint_rate

    coords = x2d[:, :, :34].reshape(batch_size, seq_len, 17, 2)
    coords[joint_mask] = 0.0
    x2d[:, :, :34] = coords.reshape(batch_size, seq_len, 34)

    return x2d, joint_mask


def apply_missing_joints_3d(
    x3d: torch.Tensor,
    missing_joint_rate: float,
    joint_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if missing_joint_rate <= 0:
        if joint_mask is None:
            batch_size, seq_len, _ = x3d.shape
            joint_mask = torch.zeros(
                batch_size,
                seq_len,
                17,
                dtype=torch.bool,
                device=x3d.device,
            )

        return x3d, joint_mask

    x3d = x3d.clone()
    batch_size, seq_len, _ = x3d.shape

    if joint_mask is None:
        joint_mask = torch.rand(
            batch_size,
            seq_len,
            17,
            device=x3d.device,
        ) < missing_joint_rate

    coords = x3d[:, :, :51].reshape(batch_size, seq_len, 17, 3)
    coords[joint_mask] = 0.0
    x3d[:, :, :51] = coords.reshape(batch_size, seq_len, 51)

    return x3d, joint_mask


def apply_frame_drop_single(
    x: torch.Tensor,
    frame_drop_rate: float,
    frame_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if frame_drop_rate <= 0:
        if frame_mask is None:
            batch_size, seq_len, _ = x.shape
            frame_mask = torch.zeros(
                batch_size,
                seq_len,
                dtype=torch.bool,
                device=x.device,
            )

        return x, frame_mask

    x = x.clone()
    batch_size, seq_len, _ = x.shape

    if frame_mask is None:
        frame_mask = torch.rand(
            batch_size,
            seq_len,
            device=x.device,
        ) < frame_drop_rate

        frame_mask[:, 0] = False

    for t in range(1, seq_len):
        mask_t = frame_mask[:, t]

        if torch.any(mask_t):
            x[mask_t, t, :] = x[mask_t, t - 1, :]

    return x, frame_mask


def perturb_2d_tensor(x2d: torch.Tensor, cfg: Dict) -> torch.Tensor:
    x2d = add_gaussian_noise_2d(
        x2d,
        std=float(cfg["noise_2d_std"]),
    )

    x2d, _ = apply_missing_joints_2d(
        x2d,
        missing_joint_rate=float(cfg["missing_joint_rate"]),
    )

    x2d, _ = apply_frame_drop_single(
        x2d,
        frame_drop_rate=float(cfg["frame_drop_rate"]),
    )

    return x2d


def perturb_3d_tensor(x3d: torch.Tensor, cfg: Dict) -> torch.Tensor:
    x3d = add_gaussian_noise_3d(
        x3d,
        std=float(cfg["noise_3d_std"]),
    )

    x3d, _ = apply_missing_joints_3d(
        x3d,
        missing_joint_rate=float(cfg["missing_joint_rate"]),
    )

    x3d, _ = apply_frame_drop_single(
        x3d,
        frame_drop_rate=float(cfg["frame_drop_rate"]),
    )

    return x3d


def perturb_2d3d_pair(
    x2d: torch.Tensor,
    x3d: torch.Tensor,
    cfg: Dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x2d = add_gaussian_noise_2d(
        x2d,
        std=float(cfg["noise_2d_std"]),
    )

    x3d = add_gaussian_noise_3d(
        x3d,
        std=float(cfg["noise_3d_std"]),
    )

    missing_rate = float(cfg["missing_joint_rate"])

    if missing_rate > 0:
        batch_size, seq_len, _ = x2d.shape
        joint_mask = torch.rand(
            batch_size,
            seq_len,
            17,
            device=x2d.device,
        ) < missing_rate

        x2d, _ = apply_missing_joints_2d(
            x2d,
            missing_joint_rate=missing_rate,
            joint_mask=joint_mask,
        )

        x3d, _ = apply_missing_joints_3d(
            x3d,
            missing_joint_rate=missing_rate,
            joint_mask=joint_mask,
        )

    frame_rate = float(cfg["frame_drop_rate"])

    if frame_rate > 0:
        batch_size, seq_len, _ = x2d.shape
        frame_mask = torch.rand(
            batch_size,
            seq_len,
            device=x2d.device,
        ) < frame_rate

        frame_mask[:, 0] = False

        x2d, _ = apply_frame_drop_single(
            x2d,
            frame_drop_rate=frame_rate,
            frame_mask=frame_mask,
        )

        x3d, _ = apply_frame_drop_single(
            x3d,
            frame_drop_rate=frame_rate,
            frame_mask=frame_mask,
        )

    return x2d, x3d


def get_quality_std(scaler) -> np.ndarray:
    if hasattr(scaler, "scale_"):
        return np.asarray(scaler.scale_, dtype=np.float32)

    if hasattr(scaler, "std_"):
        return np.asarray(scaler.std_, dtype=np.float32)

    return np.ones(len(p4.QUALITY_FEATURE_COLUMNS), dtype=np.float32)


def add_standardized_quality_delta(
    quality: torch.Tensor,
    quality_columns: List[str],
    quality_std: np.ndarray,
    feature_name: str,
    raw_delta: float,
) -> torch.Tensor:
    if raw_delta == 0:
        return quality

    if feature_name not in quality_columns:
        return quality

    idx = quality_columns.index(feature_name)

    std_value = float(quality_std[idx])

    if std_value < 1e-6:
        std_value = 1.0

    quality[:, idx] = quality[:, idx] + float(raw_delta) / std_value

    return quality


def perturb_quality_tensor(
    quality: torch.Tensor,
    cfg: Dict,
    quality_std: np.ndarray,
) -> torch.Tensor:
    quality = quality.clone()
    quality_columns = list(p4.QUALITY_FEATURE_COLUMNS)

    noise_2d = float(cfg["noise_2d_std"])
    noise_3d = float(cfg["noise_3d_std"])
    missing_rate = float(cfg["missing_joint_rate"])
    frame_rate = float(cfg["frame_drop_rate"])

    if noise_2d > 0:
        quality = add_standardized_quality_delta(
            quality,
            quality_columns,
            quality_std,
            "temporal_jitter",
            noise_2d,
        )

        quality = add_standardized_quality_delta(
            quality,
            quality_columns,
            quality_std,
            "velocity_std",
            noise_2d,
        )

    if noise_3d > 0:
        quality = add_standardized_quality_delta(
            quality,
            quality_columns,
            quality_std,
            "3d_z_instability",
            noise_3d,
        )

        quality = add_standardized_quality_delta(
            quality,
            quality_columns,
            quality_std,
            "3d_velocity_std",
            noise_3d,
        )

    if missing_rate > 0:
        quality = add_standardized_quality_delta(
            quality,
            quality_columns,
            quality_std,
            "missing_joint_ratio",
            missing_rate,
        )

        quality = add_standardized_quality_delta(
            quality,
            quality_columns,
            quality_std,
            "low_conf_03_ratio",
            missing_rate,
        )

        quality = add_standardized_quality_delta(
            quality,
            quality_columns,
            quality_std,
            "low_conf_05_ratio",
            missing_rate,
        )

    if frame_rate > 0:
        quality = add_standardized_quality_delta(
            quality,
            quality_columns,
            quality_std,
            "bbox_center_velocity_std",
            frame_rate * 0.1,
        )

    return quality


# ============================================================
# MODEL BUILDING
# ============================================================

def build_phase3_model(
    spec: Dict,
    task: str,
    device: torch.device,
):
    num_classes = get_num_classes(task)
    input_dim = spec["input_dim"]

    if spec["model_id"] == "phase1_2d_common":
        model = FallCNNLSTM(
            input_dim=40,
            num_classes=num_classes,
            cnn_channels=128,
            lstm_hidden=128,
            lstm_layers=1,
            dropout=0.3,
        )

    elif spec["model_id"] == "phase2_3d_common":
        model = FallCNNLSTM3D(
            input_dim=59,
            num_classes=num_classes,
            cnn_channels=128,
            lstm_hidden=128,
            lstm_layers=1,
            dropout=0.3,
        )

    elif spec["model_id"] == "phase2_concat_common":
        model = FallCNNLSTM3D(
            input_dim=99,
            num_classes=num_classes,
            cnn_channels=128,
            lstm_hidden=128,
            lstm_layers=1,
            dropout=0.3,
        )

    elif spec["model_id"] == "phase3_gated_fusion":
        model = build_gated_fusion_model(
            task=task,
            cnn_channels=128,
            lstm_hidden=128,
            lstm_layers=1,
            dropout=0.3,
            gate_hidden=128,
            use_vector_gate=True,
        )

    else:
        raise ValueError(f"Invalid Phase 3 model_id: {spec['model_id']}")

    checkpoint_path = Path(str(spec["checkpoint_template"]).format(task=task))
    checkpoint = load_checkpoint(checkpoint_path, device)
    state_dict = clean_state_dict_keys(get_state_dict(checkpoint))

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    load_info = {
        "checkpoint_path": str(checkpoint_path),
        "class_name": model.__class__.__name__,
        "input_dim": input_dim,
        "strict_load": True,
        "num_state_keys": len(state_dict),
    }

    return model, load_info


def build_phase4_model(
    spec: Dict,
    task: str,
    device: torch.device,
):
    checkpoint_path = Path(str(spec["checkpoint_template"]).format(task=task))
    checkpoint = load_checkpoint(checkpoint_path, device)

    args_dict = checkpoint.get("args", {})
    dataset_info = checkpoint.get("dataset_info", {})

    quality_dim = int(dataset_info.get("quality_dim", len(p4.QUALITY_FEATURE_COLUMNS)))
    num_classes = get_num_classes(task)

    common_kwargs = {
        "input_dim_2d": 40,
        "input_dim_3d": 59,
        "quality_dim": quality_dim,
        "num_classes": num_classes,
        "encoder_dim": int(args_dict.get("encoder_dim", 128)),
        "cnn_channels": int(args_dict.get("cnn_channels", 128)),
        "lstm_hidden": int(args_dict.get("lstm_hidden", 128)),
        "quality_hidden": int(args_dict.get("quality_hidden", 128)),
        "fusion_dim": int(args_dict.get("fusion_dim", 128)),
        "dropout": float(args_dict.get("dropout", 0.3)),
        "pooling": str(args_dict.get("pooling", "attention")),
    }

    if spec["kind"] == "quality_gated":
        model = QualityAwareGatedFusion(
            **common_kwargs,
            gate_type=str(args_dict.get("gate_type", "vector")),
        )

    elif spec["kind"] == "quality_concat":
        model = ConcatWithQualityFusion(**common_kwargs)

    else:
        raise ValueError(f"Invalid Phase 4 kind: {spec['kind']}")

    state_dict = clean_state_dict_keys(get_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()

    load_info = {
        "checkpoint_path": str(checkpoint_path),
        "class_name": model.__class__.__name__,
        "quality_dim": quality_dim,
        "strict_load": True,
        "num_state_keys": len(state_dict),
    }

    return model, load_info


def build_model(
    spec: Dict,
    task: str,
    device: torch.device,
):
    if spec["pipeline"] == "phase3":
        return build_phase3_model(spec, task, device)

    if spec["pipeline"] == "phase4":
        return build_phase4_model(spec, task, device)

    raise ValueError(f"Invalid pipeline: {spec['pipeline']}")


# ============================================================
# DATALOADER BUILDING
# ============================================================

def build_phase3_test_loader(
    feature_mode: str,
    task: str,
    batch_size: int,
    num_workers: int,
):
    _, _, test_loader, class_names, stats = p3.build_dataloaders(
        task=task,
        feature_mode=feature_mode,
        batch_size=batch_size,
        sequence_length=60,
        stride=15,
        sequence_strategy="sliding",
        num_workers=num_workers,
    )

    info = {
        "task": task,
        "pipeline": "phase3",
        "feature_mode": feature_mode,
        "class_names": class_names,
        "num_test_samples": int(stats["test"].num_sequences),
        "num_test_records": int(stats["test"].num_records),
        "class_counts": stats["test"].class_counts,
    }

    return test_loader, None, info


def build_phase4_test_loader(
    task: str,
    quality_csv: str,
    batch_size: int,
    num_workers: int,
    cache_videos: bool,
):
    _, _, test_loader, scaler, info = p4.create_phase4_dataloaders(
        task=task,
        quality_csv=quality_csv,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_videos=cache_videos,
    )

    info = dict(info)
    info["pipeline"] = "phase4"

    return test_loader, scaler, info


# ============================================================
# EVALUATION HELPERS
# ============================================================

def unpack_phase4_output(output):
    if isinstance(output, dict):
        logits = output["logits"]
        aux = {
            key: value
            for key, value in output.items()
            if key != "logits"
        }

        return logits, aux

    if isinstance(output, tuple):
        logits = output[0]
        aux = output[1] if len(output) > 1 and isinstance(output[1], dict) else {}

        return logits, aux

    return output, {}


def build_metric_result(
    y_true: List[int],
    y_pred: List[int],
    video_keys: List[str],
    class_names: List[str],
    gate_values: Optional[List[float]] = None,
) -> Dict:
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)

    labels = list(range(len(class_names)))

    accuracy = accuracy_score(y_true_np, y_pred_np)
    macro_f1 = f1_score(
        y_true_np,
        y_pred_np,
        average="macro",
        zero_division=0,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        labels=labels,
        average=None,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true_np,
        y_pred_np,
        labels=labels,
    )

    report = classification_report(
        y_true_np,
        y_pred_np,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}

    for idx, class_name in enumerate(class_names):
        per_class[class_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    result = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "accuracy_percent": float(accuracy) * 100.0,
        "macro_f1_percent": float(macro_f1) * 100.0,
        "num_samples": int(len(y_true)),
        "num_unique_videos": int(len(set(video_keys))) if video_keys else None,
        "class_names": class_names,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    if gate_values:
        gate_arr = np.asarray(gate_values, dtype=np.float32)

        result["gate_stats"] = {
            "mean_gate": float(np.mean(gate_arr)),
            "std_gate": float(np.std(gate_arr)),
            "min_gate": float(np.min(gate_arr)),
            "max_gate": float(np.max(gate_arr)),
        }

    return result


@torch.no_grad()
def evaluate_phase3_single(
    model,
    loader,
    spec: Dict,
    device: torch.device,
    task: str,
    cfg: Dict,
    seed: int,
) -> Dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model.eval()

    y_true = []
    y_pred = []
    video_keys_all = []

    feature_mode = spec["feature_mode"]

    for x, y, video_keys in loader:
        x = x.to(device)
        y = y.to(device)

        if feature_mode == "2d":
            x = perturb_2d_tensor(x, cfg)

        elif feature_mode == "3d":
            x = perturb_3d_tensor(x, cfg)

        elif feature_mode == "concat":
            x2d = x[:, :, :40]
            x3d = x[:, :, 40:]

            x2d, x3d = perturb_2d3d_pair(x2d, x3d, cfg)
            x = torch.cat([x2d, x3d], dim=2)

        else:
            raise ValueError(f"Invalid single feature_mode: {feature_mode}")

        logits = model(x)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        video_keys_all.extend(video_keys)

    return build_metric_result(
        y_true=y_true,
        y_pred=y_pred,
        video_keys=video_keys_all,
        class_names=get_class_names(task),
    )


@torch.no_grad()
def evaluate_phase3_gated(
    model,
    loader,
    device: torch.device,
    task: str,
    cfg: Dict,
    seed: int,
) -> Dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model.eval()

    y_true = []
    y_pred = []
    video_keys_all = []
    gate_values = []

    for (x2d, x3d), y, video_keys in loader:
        x2d = x2d.to(device)
        x3d = x3d.to(device)
        y = y.to(device)

        x2d, x3d = perturb_2d3d_pair(x2d, x3d, cfg)

        output = model(x2d, x3d, return_gate=True)

        if isinstance(output, tuple):
            logits, gate = output

            if isinstance(gate, torch.Tensor):
                gate_values.extend(
                    gate.detach().cpu().reshape(-1).numpy().tolist()
                )
        else:
            logits = output

        preds = torch.argmax(logits, dim=1)

        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        video_keys_all.extend(video_keys)

    return build_metric_result(
        y_true=y_true,
        y_pred=y_pred,
        video_keys=video_keys_all,
        class_names=get_class_names(task),
        gate_values=gate_values,
    )


@torch.no_grad()
def evaluate_phase4_model(
    model,
    loader,
    scaler,
    spec: Dict,
    device: torch.device,
    task: str,
    cfg: Dict,
    seed: int,
    perturb_quality: bool,
) -> Dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model.eval()

    y_true = []
    y_pred = []
    video_keys_all = []
    gate_values = []

    quality_std = get_quality_std(scaler)

    for batch in loader:
        x2d = batch["x2d"].to(device)
        x3d = batch["x3d"].to(device)
        quality = batch["quality"].to(device)
        y = batch["label"].to(device)

        video_keys = batch.get("video_key", None)

        if video_keys is None:
            video_keys = batch.get("video_keys", None)

        if video_keys is None:
            video_keys = ["unknown"] * y.shape[0]

        x2d, x3d = perturb_2d3d_pair(x2d, x3d, cfg)

        if perturb_quality:
            quality = perturb_quality_tensor(
                quality=quality,
                cfg=cfg,
                quality_std=quality_std,
            )

        output = model(x2d, x3d, quality)
        logits, aux = unpack_phase4_output(output)

        preds = torch.argmax(logits, dim=1)

        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

        if isinstance(video_keys, torch.Tensor):
            video_keys_all.extend(video_keys.detach().cpu().numpy().tolist())
        else:
            video_keys_all.extend(list(video_keys))

        if isinstance(aux, dict):
            if "gate" in aux and isinstance(aux["gate"], torch.Tensor):
                gate_values.extend(
                    aux["gate"].detach().cpu().reshape(-1).numpy().tolist()
                )

            elif "mean_gate" in aux and isinstance(aux["mean_gate"], torch.Tensor):
                gate_values.extend(
                    aux["mean_gate"].detach().cpu().reshape(-1).numpy().tolist()
                )

    return build_metric_result(
        y_true=y_true,
        y_pred=y_pred,
        video_keys=video_keys_all,
        class_names=get_class_names(task),
        gate_values=gate_values,
    )


def evaluate_one_model(
    model,
    loader,
    scaler,
    spec: Dict,
    device: torch.device,
    task: str,
    cfg: Dict,
    seed: int,
    perturb_quality: bool,
) -> Dict:
    if spec["pipeline"] == "phase3":
        if spec["kind"] == "single":
            return evaluate_phase3_single(
                model=model,
                loader=loader,
                spec=spec,
                device=device,
                task=task,
                cfg=cfg,
                seed=seed,
            )

        if spec["kind"] == "gated":
            return evaluate_phase3_gated(
                model=model,
                loader=loader,
                device=device,
                task=task,
                cfg=cfg,
                seed=seed,
            )

    if spec["pipeline"] == "phase4":
        return evaluate_phase4_model(
            model=model,
            loader=loader,
            scaler=scaler,
            spec=spec,
            device=device,
            task=task,
            cfg=cfg,
            seed=seed,
            perturb_quality=perturb_quality,
        )

    raise ValueError(
        f"Invalid model pipeline/kind: {spec['pipeline']} / {spec['kind']}"
    )


# ============================================================
# MAIN ROBUSTNESS LOOP
# ============================================================

def build_rows_for_model(
    model,
    loader,
    scaler,
    info: Dict,
    spec: Dict,
    args,
) -> Tuple[List[Dict], Dict]:
    rows = []
    detailed = {}

    clean_accuracy = None
    clean_macro_f1 = None

    scenarios = get_scenarios()

    for scenario_index, cfg in enumerate(scenarios):
        scenario_name = cfg["scenario"]

        print(
            f"Evaluating | task={args.task} | model={spec['model_name']} | scenario={scenario_name}"
        )

        result = evaluate_one_model(
            model=model,
            loader=loader,
            scaler=scaler,
            spec=spec,
            device=args.device,
            task=args.task,
            cfg=cfg,
            seed=args.seed + scenario_index,
            perturb_quality=not args.no_perturb_quality,
        )

        if scenario_name == "clean":
            clean_accuracy = result["accuracy"]
            clean_macro_f1 = result["macro_f1"]

        delta_accuracy = result["accuracy"] - clean_accuracy
        delta_macro_f1 = result["macro_f1"] - clean_macro_f1

        reference = CLEAN_REFERENCE.get(args.task, {}).get(spec["model_id"], {})
        reference_accuracy = reference.get("accuracy", np.nan)
        reference_macro_f1 = reference.get("macro_f1", np.nan)

        clean_accuracy_gap = np.nan
        clean_macro_f1_gap = np.nan

        if scenario_name == "clean":
            if not pd.isna(reference_accuracy):
                clean_accuracy_gap = result["accuracy"] - reference_accuracy

            if not pd.isna(reference_macro_f1):
                clean_macro_f1_gap = result["macro_f1"] - reference_macro_f1

        gate_stats = result.get("gate_stats", {})

        row = {
            "task": args.task,
            "phase": spec["phase"],
            "model_id": spec["model_id"],
            "model_name": spec["model_name"],
            "pipeline": spec["pipeline"],
            "feature_mode": spec["feature_mode"],
            "kind": spec["kind"],
            "scenario": scenario_name,
            "description": cfg["description"],
            "noise_2d_std": cfg["noise_2d_std"],
            "noise_3d_std": cfg["noise_3d_std"],
            "missing_joint_rate": cfg["missing_joint_rate"],
            "frame_drop_rate": cfg["frame_drop_rate"],
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "accuracy_percent": result["accuracy_percent"],
            "macro_f1_percent": result["macro_f1_percent"],
            "delta_accuracy_from_clean": delta_accuracy,
            "delta_macro_f1_from_clean": delta_macro_f1,
            "delta_accuracy_percent_from_clean": delta_accuracy * 100.0,
            "delta_macro_f1_percent_from_clean": delta_macro_f1 * 100.0,
            "reference_clean_accuracy": reference_accuracy,
            "reference_clean_macro_f1": reference_macro_f1,
            "reference_clean_accuracy_percent": (
                reference_accuracy * 100.0
                if not pd.isna(reference_accuracy)
                else np.nan
            ),
            "reference_clean_macro_f1_percent": (
                reference_macro_f1 * 100.0
                if not pd.isna(reference_macro_f1)
                else np.nan
            ),
            "clean_accuracy_gap_vs_reference": clean_accuracy_gap,
            "clean_macro_f1_gap_vs_reference": clean_macro_f1_gap,
            "clean_accuracy_gap_percent_vs_reference": (
                clean_accuracy_gap * 100.0
                if not pd.isna(clean_accuracy_gap)
                else np.nan
            ),
            "clean_macro_f1_gap_percent_vs_reference": (
                clean_macro_f1_gap * 100.0
                if not pd.isna(clean_macro_f1_gap)
                else np.nan
            ),
            "num_samples": result["num_samples"],
            "num_unique_videos": result["num_unique_videos"],
            "dataset_num_test_samples": info.get("num_test_samples"),
            "dataset_num_test_records": info.get("num_test_records"),
            "mean_gate": gate_stats.get("mean_gate", np.nan),
            "std_gate": gate_stats.get("std_gate", np.nan),
            "min_gate": gate_stats.get("min_gate", np.nan),
            "max_gate": gate_stats.get("max_gate", np.nan),
        }

        rows.append(row)

        detailed_key = f"{spec['model_id']}__{scenario_name}"
        detailed[detailed_key] = {
            "config": cfg,
            "result": result,
        }

    return rows, detailed


# ============================================================
# REPORTING
# ============================================================

def check_clean_consistency(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df[df["scenario"] == "clean"].copy()

    if clean_df.empty:
        return clean_df

    clean_df["abs_accuracy_gap_percent"] = clean_df[
        "clean_accuracy_gap_percent_vs_reference"
    ].abs()

    clean_df["abs_macro_f1_gap_percent"] = clean_df[
        "clean_macro_f1_gap_percent_vs_reference"
    ].abs()

    return clean_df


def build_markdown_report(df: pd.DataFrame, args) -> str:
    lines = []

    lines.append(f"# All-Models Robustness Report - {args.task}")
    lines.append("")
    lines.append("This report evaluates robustness for Phase 1, Phase 2, Phase 3, and Phase 4 models.")
    lines.append("")
    lines.append("Important:")
    lines.append("")
    lines.append("- Phase 1/2/3 are evaluated with the Phase 3 common-set feature pipeline.")
    lines.append("- Phase 4 models are evaluated with the Phase 4 quality-aware feature pipeline.")
    lines.append("- Perturbations are applied in memory only. Original dataset files are not modified.")
    lines.append("")

    clean_df = check_clean_consistency(df)

    lines.append("## Clean Consistency Check")
    lines.append("")
    lines.append("| Phase | Model | Clean Acc | Ref Acc | Gap Acc | Clean Macro F1 | Ref Macro F1 | Gap Macro F1 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    for _, row in clean_df.iterrows():
        lines.append(
            f"| {row['phase']} "
            f"| {row['model_name']} "
            f"| {row['accuracy_percent']:.2f}% "
            f"| {row['reference_clean_accuracy_percent']:.2f}% "
            f"| {row['clean_accuracy_gap_percent_vs_reference']:.2f} pp "
            f"| {row['macro_f1_percent']:.2f}% "
            f"| {row['reference_clean_macro_f1_percent']:.2f}% "
            f"| {row['clean_macro_f1_gap_percent_vs_reference']:.2f} pp |"
        )

    lines.append("")
    lines.append("If the clean gap is large, the robustness comparison must not be used.")
    lines.append("")

    for scenario, group in df.groupby("scenario"):
        scenario_df = group.copy().sort_values(
            by=["macro_f1", "accuracy"],
            ascending=[False, False],
        )

        lines.append(f"## Scenario: {scenario}")
        lines.append("")
        lines.append("| Rank | Phase | Model | Accuracy | Macro F1 | Delta Accuracy | Delta Macro F1 |")
        lines.append("|---:|---|---|---:|---:|---:|---:|")

        for rank, (_, row) in enumerate(scenario_df.iterrows(), start=1):
            lines.append(
                f"| {rank} "
                f"| {row['phase']} "
                f"| {row['model_name']} "
                f"| {row['accuracy_percent']:.2f}% "
                f"| {row['macro_f1_percent']:.2f}% "
                f"| {row['delta_accuracy_percent_from_clean']:.2f} pp "
                f"| {row['delta_macro_f1_percent_from_clean']:.2f} pp |"
            )

        lines.append("")

    lines.append("## Interpretation Guide")
    lines.append("")
    lines.append("- Higher Macro F1 under corrupted scenarios means better robustness.")
    lines.append("- Smaller negative delta from clean means the model is less sensitive to pose degradation.")
    lines.append("- For binary fall detection, the most important comparison is usually Phase 2 Concat vs Phase 4 Quality-Concat.")
    lines.append("- For action classification, Phase 2 Concat may still remain the best clean model.")
    lines.append("")

    return "\n".join(lines)


def save_outputs(
    rows: List[Dict],
    detailed: Dict,
    load_infos: Dict,
    args,
) -> None:
    ensure_dir(Path(args.output_dir))

    df = pd.DataFrame(rows)

    csv_path = Path(args.output_dir) / f"all_models_robustness_{args.task}.csv"
    json_path = Path(args.output_dir) / f"all_models_robustness_{args.task}.json"
    md_path = Path(args.output_dir) / f"all_models_robustness_{args.task}.md"

    df.to_csv(csv_path, index=False)

    save_json(
        {
            "task": args.task,
            "results": df.to_dict(orient="records"),
            "detailed_results": detailed,
            "model_load_info": load_infos,
            "settings": {
                "quality_csv": args.quality_csv,
                "models": args.models,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "perturb_quality": not args.no_perturb_quality,
                "seed": args.seed,
            },
        },
        json_path,
    )

    report = build_markdown_report(df, args)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\nSaved robustness outputs:")
    print(f"- {csv_path}")
    print(f"- {json_path}")
    print(f"- {md_path}")


def print_clean_check(rows: List[Dict]) -> None:
    df = pd.DataFrame(rows)
    clean_df = check_clean_consistency(df)

    print("\nClean Consistency Check")
    print("=" * 120)

    cols = [
        "phase",
        "model_name",
        "accuracy_percent",
        "reference_clean_accuracy_percent",
        "clean_accuracy_gap_percent_vs_reference",
        "macro_f1_percent",
        "reference_clean_macro_f1_percent",
        "clean_macro_f1_gap_percent_vs_reference",
    ]

    if not clean_df.empty:
        print(clean_df[cols].to_string(index=False))

    print("=" * 120)

    bad = clean_df[
        (clean_df["abs_accuracy_gap_percent"] > 1.0)
        | (clean_df["abs_macro_f1_gap_percent"] > 1.0)
    ]

    if not bad.empty:
        print("\nWARNING: Some clean results differ from reference by more than 1 percentage point.")
        print("Do not use robustness conclusions until this is fixed.")
    else:
        print("\nClean check passed. Robustness results are consistent with saved clean results.")


def print_summary(rows: List[Dict], task: str) -> None:
    df = pd.DataFrame(rows)

    print("\nAll-Models Robustness Summary")
    print("=" * 140)

    for scenario, group in df.groupby("scenario"):
        group = group.copy().sort_values(
            by=["macro_f1", "accuracy"],
            ascending=[False, False],
        )

        print(f"\nScenario: {scenario}")
        print("-" * 140)

        cols = [
            "phase",
            "model_name",
            "accuracy_percent",
            "macro_f1_percent",
            "delta_accuracy_percent_from_clean",
            "delta_macro_f1_percent_from_clean",
        ]

        print(group[cols].to_string(index=False))

    print("=" * 140)


# ============================================================
# ARGUMENTS AND MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate robustness for all Phase 1-4 fair models."
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["binary", "action"],
        help="Task to evaluate.",
    )

    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help=(
            "Use 'all' or comma-separated IDs: "
            "phase1_2d_common,phase2_3d_common,phase2_concat_common,"
            "phase3_gated_fusion,phase4_quality_gated,phase4_quality_concat"
        ),
    )

    parser.add_argument(
        "--quality-csv",
        type=str,
        default=str(p4.DEFAULT_QUALITY_CSV),
        help="Path to Phase 4 quality_sequences.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROBUSTNESS_OUTPUT_DIR),
        help="Output directory.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep 0 on Windows.",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Phase 4 video cache.",
    )

    parser.add_argument(
        "--no-perturb-quality",
        action="store_true",
        help="Do not adjust Phase 4 quality vector during artificial corruption.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    args.device = get_device()

    ensure_dir(Path(args.output_dir))

    print("\nAll-Phases Pose Robustness Evaluation")
    print("=" * 90)
    print(f"Task              : {args.task}")
    print(f"Models            : {args.models}")
    print(f"Quality CSV       : {args.quality_csv}")
    print(f"Output dir        : {args.output_dir}")
    print(f"Device            : {args.device}")
    print(f"Perturb quality   : {not args.no_perturb_quality}")
    print("=" * 90)

    selected_specs = get_selected_specs(args.models)

    phase3_loader_cache = {}
    phase4_loader_cache = None

    all_rows = []
    detailed = {}
    load_infos = {}

    for spec in selected_specs:
        print("\nLoading model")
        print("=" * 90)
        print(f"Model ID          : {spec['model_id']}")
        print(f"Phase             : {spec['phase']}")
        print(f"Model name        : {spec['model_name']}")
        print(f"Pipeline          : {spec['pipeline']}")
        print(f"Feature mode      : {spec['feature_mode']}")
        print(f"Kind              : {spec['kind']}")

        model, load_info = build_model(
            spec=spec,
            task=args.task,
            device=args.device,
        )

        try:
            num_params = count_parameters(model)
        except Exception:
            num_params = sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            )

        print(f"Load info         : {load_info}")
        print(f"Trainable params  : {num_params:,}")
        print("=" * 90)

        if spec["pipeline"] == "phase3":
            feature_mode = spec["feature_mode"]

            if feature_mode not in phase3_loader_cache:
                print(f"\nBuilding Phase 3 dataloader | feature_mode={feature_mode}")

                loader, scaler, info = build_phase3_test_loader(
                    feature_mode=feature_mode,
                    task=args.task,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )

                phase3_loader_cache[feature_mode] = (loader, scaler, info)

            else:
                loader, scaler, info = phase3_loader_cache[feature_mode]

        elif spec["pipeline"] == "phase4":
            if phase4_loader_cache is None:
                print("\nBuilding Phase 4 dataloader")

                loader, scaler, info = build_phase4_test_loader(
                    task=args.task,
                    quality_csv=args.quality_csv,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    cache_videos=not args.no_cache,
                )

                phase4_loader_cache = (loader, scaler, info)

            else:
                loader, scaler, info = phase4_loader_cache

        else:
            raise ValueError(f"Invalid pipeline: {spec['pipeline']}")

        print("\nDataset info")
        print("=" * 90)
        print(f"Pipeline          : {info.get('pipeline')}")
        print(f"Feature mode      : {info.get('feature_mode')}")
        print(f"Task              : {info.get('task')}")
        print(f"Test samples      : {info.get('num_test_samples')}")
        print(f"Test records      : {info.get('num_test_records')}")
        print(f"Class names       : {info.get('class_names')}")
        print("=" * 90)

        load_infos[spec["model_id"]] = {
            "spec": spec,
            "load_info": load_info,
            "num_parameters": int(num_params),
            "dataset_info": info,
        }

        rows, detail = build_rows_for_model(
            model=model,
            loader=loader,
            scaler=scaler,
            info=info,
            spec=spec,
            args=args,
        )

        all_rows.extend(rows)
        detailed.update(detail)

    save_outputs(
        rows=all_rows,
        detailed=detailed,
        load_infos=load_infos,
        args=args,
    )

    print_clean_check(all_rows)
    print_summary(all_rows, args.task)

    print("\nDone.")


if __name__ == "__main__":
    main()