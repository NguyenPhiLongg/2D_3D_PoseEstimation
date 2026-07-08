"""
Train Gated Fusion model for Phase 3.

Purpose:
- Train the proposed Gated Fusion model on the SAME common video set used by:
    2D common
    3D common
    Concat fusion common

- This model is used to test whether adaptive fusion is better than fixed concat fusion.

Input:
    2D sequence:
        shape = (60, 40)

    3D sequence:
        shape = (60, 59)

Gated Fusion idea:
    2D input -> 2D encoder -> h2d
    3D input -> 3D encoder -> h3d

    gate = sigmoid(MLP([h2d, h3d]))

    fused = gate * h2d + (1 - gate) * h3d

    classifier(fused) -> prediction

Output:
    Binary:
        outputs/training_gated_fusion/results_gated_fusion_binary.json
        checkpoints/best_model_gated_fusion_binary.pt

    Action:
        outputs/training_gated_fusion/results_gated_fusion_action.json
        checkpoints/best_model_gated_fusion_action.pt

Run:
    python phase3_common_set_gated_fusion/train_gated_fusion.py --task binary
    python phase3_common_set_gated_fusion/train_gated_fusion.py --task action
"""

from __future__ import annotations

import argparse
from pathlib import Path

from model_gated_fusion import build_gated_fusion_model
from phase3_utils import (
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    train_gated_model,
)


# =========================
# PATH HELPERS
# =========================

def get_output_paths(task: str) -> tuple[Path, Path]:
    task = task.lower().strip()

    output_dir = OUTPUT_DIR / "training_gated_fusion"
    output_dir.mkdir(parents=True, exist_ok=True)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if task == "binary":
        output_json = output_dir / "results_gated_fusion_binary.json"
        checkpoint = CHECKPOINT_DIR / "best_model_gated_fusion_binary.pt"

    elif task == "action":
        output_json = output_dir / "results_gated_fusion_action.json"
        checkpoint = CHECKPOINT_DIR / "best_model_gated_fusion_action.pt"

    else:
        raise ValueError(f"Invalid task: {task}. Expected 'binary' or 'action'.")

    return output_json, checkpoint


# =========================
# ARGUMENTS
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Gated Fusion 2D+3D model on Phase 3 common set."
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
        help="CNN channels for both 2D and 3D encoders. Default: 128",
    )

    parser.add_argument(
        "--lstm-hidden",
        type=int,
        default=128,
        help="LSTM hidden size for both 2D and 3D encoders. Default: 128",
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

    parser.add_argument(
        "--gate-hidden",
        type=int,
        default=128,
        help="Hidden dimension of gate network. Default: 128",
    )

    parser.add_argument(
        "--scalar-gate",
        action="store_true",
        help=(
            "Use one scalar gate value for the whole representation. "
            "Default is vector gate, which learns one gate value per hidden dimension."
        ),
    )

    return parser.parse_args()


# =========================
# MAIN
# =========================

def main() -> None:
    args = parse_args()

    task = args.task.lower().strip()

    output_json, checkpoint = get_output_paths(task)

    use_vector_gate = not args.scalar_gate

    model = build_gated_fusion_model(
        task=task,
        cnn_channels=args.cnn_channels,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        gate_hidden=args.gate_hidden,
        use_vector_gate=use_vector_gate,
    )

    total_params, trainable_params = model.get_num_parameters()

    print("=" * 80)
    print("PHASE 3 - TRAIN GATED FUSION MODEL")
    print("=" * 80)
    print("Task:", task)
    print("Feature mode: gated")
    print("2D input dimension: 40D")
    print("3D input dimension: 59D")
    print("Fusion type: Adaptive Gated Fusion")
    print("Gate type:", "vector gate" if use_vector_gate else "scalar gate")
    print("Project root:", PROJECT_ROOT)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)
    print("Output JSON:", output_json)
    print("Checkpoint:", checkpoint)
    print("=" * 80)

    train_gated_model(
        model=model,
        task=task,
        output_json_path=output_json,
        checkpoint_path=checkpoint,
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