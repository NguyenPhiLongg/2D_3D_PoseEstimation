"""
Train Concat Fusion Common-set model for Phase 3.

Purpose:
- Train early fusion / concat fusion on the SAME common video set used by:
    2D common
    3D common
    Gated fusion

- This is the fair baseline for comparing against Gated Fusion.

Input:
    2D features: 40D
    3D features: 59D

Concat input:
    40D + 59D = 99D

Shape per sequence:
    (60, 99)

Output:
    Binary:
        outputs/training_concat_common/results_concat_common_binary.json
        checkpoints/best_model_concat_common_binary.pt

    Action:
        outputs/training_concat_common/results_concat_common_action.json
        checkpoints/best_model_concat_common_action.pt

Run:
    python phase3_common_set_gated_fusion/train_concat_common.py --task binary
    python phase3_common_set_gated_fusion/train_concat_common.py --task action
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from phase3_utils import (
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    class_names_for_task,
    train_single_input_model,
)


# =========================
# IMPORT PHASE 2 MODEL
# =========================

PHASE2_DIR = PROJECT_ROOT / "phase2_3d_upgrade"

if str(PHASE2_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE2_DIR))

try:
    from model_3d import FallCNNLSTM3D
except Exception as exc:
    raise ImportError(
        f"Cannot import FallCNNLSTM3D from {PHASE2_DIR}. "
        "Please check phase2_3d_upgrade/model_3d.py"
    ) from exc


# =========================
# PATH HELPERS
# =========================

def get_output_paths(task: str) -> tuple[Path, Path]:
    task = task.lower().strip()

    output_dir = OUTPUT_DIR / "training_concat_common"
    output_dir.mkdir(parents=True, exist_ok=True)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if task == "binary":
        output_json = output_dir / "results_concat_common_binary.json"
        checkpoint = CHECKPOINT_DIR / "best_model_concat_common_binary.pt"

    elif task == "action":
        output_json = output_dir / "results_concat_common_action.json"
        checkpoint = CHECKPOINT_DIR / "best_model_concat_common_action.pt"

    else:
        raise ValueError(f"Invalid task: {task}. Expected 'binary' or 'action'.")

    return output_json, checkpoint


# =========================
# MODEL BUILDER
# =========================

def build_model(
    task: str,
    input_dim: int = 99,
    cnn_channels: int = 128,
    lstm_hidden: int = 128,
    lstm_layers: int = 1,
    dropout: float = 0.3,
) -> FallCNNLSTM3D:
    """
    Build concat fusion CNN1D + BiLSTM model.

    The architecture is the same CNN1D + BiLSTM backbone used in Phase 2.
    The only difference is the input dimension.

    Concat input:
        2D feature = 40D
        3D feature = 59D
        Total      = 99D
    """
    class_names = class_names_for_task(task)
    num_classes = len(class_names)

    model = FallCNNLSTM3D(
        input_dim=input_dim,
        num_classes=num_classes,
        cnn_channels=cnn_channels,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        dropout=dropout,
    )

    return model


# =========================
# ARGUMENTS
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train concat fusion CNN1D-BiLSTM model on Phase 3 common set."
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["binary", "action"],
        help="Training task: binary or action.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs. Default: 30",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size. Default: 64",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate. Default: 1e-3",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay. Default: 1e-4",
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=60,
        help="Sequence length. Default: 60",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=15,
        help="Sliding window stride. Default: 15",
    )

    parser.add_argument(
        "--sequence-strategy",
        type=str,
        default="sliding",
        choices=["sliding", "single"],
        help="Sequence building strategy. Default: sliding",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )

    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Force CPU training.",
    )

    parser.add_argument(
        "--cnn-channels",
        type=int,
        default=128,
        help="CNN channels. Default: 128",
    )

    parser.add_argument(
        "--lstm-hidden",
        type=int,
        default=128,
        help="LSTM hidden size. Default: 128",
    )

    parser.add_argument(
        "--lstm-layers",
        type=int,
        default=1,
        help="Number of LSTM layers. Default: 1",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate. Default: 0.3",
    )

    return parser.parse_args()


# =========================
# MAIN
# =========================

def main() -> None:
    args = parse_args()

    task = args.task.lower().strip()

    output_json, checkpoint = get_output_paths(task)

    model = build_model(
        task=task,
        input_dim=99,
        cnn_channels=args.cnn_channels,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    )

    print("=" * 80)
    print("PHASE 3 - TRAIN CONCAT FUSION COMMON-SET MODEL")
    print("=" * 80)
    print("Task:", task)
    print("Feature mode: concat")
    print("Input dimension: 99D")
    print("2D feature dimension: 40D")
    print("3D feature dimension: 59D")
    print("Project root:", PROJECT_ROOT)
    print("Output JSON:", output_json)
    print("Checkpoint:", checkpoint)
    print("=" * 80)

    train_single_input_model(
        model=model,
        task=task,
        feature_mode="concat",
        output_json_path=output_json,
        checkpoint_path=checkpoint,
        model_type="CNN1D_BiLSTM_CONCAT_2D3D_COMMON",
        input_dim=99,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        sequence_length=args.sequence_length,
        stride=args.stride,
        sequence_strategy=args.sequence_strategy,
        seed=args.seed,
        use_cuda=not args.no_cuda,
    )


if __name__ == "__main__":
    main()