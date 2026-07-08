import os
import time
import argparse
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from phase4_utils import (
    DEFAULT_QUALITY_CSV,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_OUTPUT_DIR,
    QUALITY_FEATURE_COLUMNS,
    set_seed,
    get_device,
    ensure_dir,
    save_json,
    create_phase4_dataloaders,
    compute_loss_weights,
    get_dataset_labels,
    train_one_epoch,
    evaluate_model,
    save_checkpoint,
    load_checkpoint,
    summarize_training_history,
    print_dataset_info,
    get_task_class_names,
    to_jsonable,
)

from model_quality_aware_fusion import (
    QualityAwareGatedFusion,
    ConcatWithQualityFusion,
    count_parameters,
)


"""
Phase 4 - Train Quality-Aware Fusion Model.

This script trains the proposed Phase 4 model:

    Quality-Aware 2D-3D Pose Fusion

Inputs:
    - data/6_pose_quality_features/quality_sequences.csv
    - data/5_extracted_2d_confidence/
    - data/4_normalized_3d/
    - Phase 3 train/val/test split

Main model:
    QualityAwareGatedFusion

The model receives:
    x2d     : [B, T, 40]
    x3d     : [B, T, 59]
    quality : [B, Q]

The gate is computed from:
    3 train/val/test split

Main model:
    QualityAwareGatedFusion

The model receives:
    x2d     : [B, T, 40]
    x3d     : [B, T, 59]
    quality : [B, Q]

 h2d, h3d, hq

where hq is an encoded quality vector.

Commands:
    Binary:
        python phase4_quality_aware_fusion/train_quality_gated_fusion.py --task binary

    Action:
        python phase4_quality_aware_fusion/train_quality_gated_fusion.py --task action

Optional ablation:
    Quality concat without gate:
        python phase4_quality_aware_fusion/train_quality_gated_fusion.py --task binary --model-type quality_concat
"""


def build_model(args: argparse.Namespace, quality_dim: int, num_classes: int):
    if args.model_type == "quality_gated":
        model = QualityAwareGatedFusion(
            input_dim_2d=40,
            input_dim_3d=59,
            quality_dim=quality_dim,
            num_classes=num_classes,
            encoder_dim=args.encoder_dim,
            cnn_channels=args.cnn_channels,
            lstm_hidden=args.lstm_hidden,
            quality_hidden=args.quality_hidden,
            fusion_dim=args.fusion_dim,
            dropout=args.dropout,
            gate_type=args.gate_type,
            pooling=args.pooling,
        )
        return model

    if args.model_type == "quality_concat":
        model = ConcatWithQualityFusion(
            input_dim_2d=40,
            input_dim_3d=59,
            quality_dim=quality_dim,
            num_classes=num_classes,
            encoder_dim=args.encoder_dim,
            cnn_channels=args.cnn_channels,
            lstm_hidden=args.lstm_hidden,
            quality_hidden=args.quality_hidden,
            fusion_dim=args.fusion_dim,
            dropout=args.dropout,
            pooling=args.pooling,
        )
        return model

    raise ValueError(f"Invalid model_type: {args.model_type}")


def make_output_names(args: argparse.Namespace) -> Dict[str, str]:
    ensure_dir(args.output_dir)
    ensure_dir(args.checkpoint_dir)

    if args.model_type == "quality_gated":
        prefix = "quality_gated"
    elif args.model_type == "quality_concat":
        prefix = "quality_concat"
    else:
        prefix = args.model_type

    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f"best_model_{prefix}_{args.task}.pt",
    )

    result_path = os.path.join(
        args.output_dir,
        f"results_{prefix}_{args.task}.json",
    )

    scaler_path = os.path.join(
        args.output_dir,
        f"quality_scaler_{prefix}_{args.task}.json",
    )

    history_path = os.path.join(
        args.output_dir,
        f"history_{prefix}_{args.task}.json",
    )

    return {
        "prefix": prefix,
        "checkpoint_path": checkpoint_path,
        "result_path": result_path,
        "scaler_path": scaler_path,
        "history_path": history_path,
    }


def print_train_header(args: argparse.Namespace, device: torch.device) -> None:
    print("\nPhase 4 - Quality-Aware Fusion Training")
    print("=" * 80)
    print(f"Task              : {args.task}")
    print(f"Model type        : {args.model_type}")
    print(f"Quality CSV       : {args.quality_csv}")
    print(f"Device            : {device}")
    print(f"Epochs            : {args.epochs}")
    print(f"Batch size        : {args.batch_size}")
    print(f"Learning rate     : {args.learning_rate}")
    print(f"Weight decay      : {args.weight_decay}")
    print(f"Dropout           : {args.dropout}")
    print(f"Gate type         : {args.gate_type}")
    print(f"Pooling           : {args.pooling}")
    print(f"Seed              : {args.seed}")
    print("=" * 80)


def train_model(args: argparse.Namespace) -> Dict:
    started_at = time.time()

    set_seed(args.seed)

    device = get_device()

    print_train_header(args, device)

    train_loader, val_loader, test_loader, quality_scaler, dataset_info = create_phase4_dataloaders(
        task=args.task,
        quality_csv=args.quality_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        quality_columns=QUALITY_FEATURE_COLUMNS,
        cache_videos=not args.no_cache,
    )

    print_dataset_info(dataset_info)

    num_classes = len(get_task_class_names(args.task))
    quality_dim = dataset_info["quality_dim"]

    model = build_model(
        args=args,
        quality_dim=quality_dim,
        num_classes=num_classes,
    )

    model = model.to(device)

    print("\nModel info")
    print("=" * 80)
    print(f"Model class       : {model.__class__.__name__}")
    print(f"Trainable params  : {count_parameters(model):,}")
    print("=" * 80)

    train_labels = get_dataset_labels(train_loader)
    class_weights = compute_loss_weights(
        labels=train_labels,
        num_classes=num_classes,
    ).to(device)

    print("\nClass weights")
    print("=" * 80)
    print(class_weights.detach().cpu().numpy().tolist())
    print("=" * 80)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    output_names = make_output_names(args)

    save_json(
        quality_scaler.to_dict(),
        output_names["scaler_path"],
    )

    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_val_result = None
    history: List[Dict] = []
    epochs_without_improvement = 0

    print("\nStart training")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_grad_norm=args.max_grad_norm,
        )

        val_result = evaluate_model(
            model=model,
            loader=val_loader,
            device=device,
            task=args.task,
        )

        val_macro_f1 = float(val_result["macro_f1"])
        val_accuracy = float(val_result["accuracy"])

        scheduler.step(val_macro_f1)

        current_lr = optimizer.param_groups[0]["lr"]

        improved = val_macro_f1 > best_val_macro_f1

        if improved:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_val_result = val_result
            epochs_without_improvement = 0

            checkpoint_extra = {
                "epoch": epoch,
                "task": args.task,
                "model_type": args.model_type,
                "best_val_macro_f1": best_val_macro_f1,
                "dataset_info": dataset_info,
                "quality_columns": QUALITY_FEATURE_COLUMNS,
                "quality_scaler": quality_scaler.to_dict(),
                "args": vars(args),
            }

            save_checkpoint(
                model=model,
                path=output_names["checkpoint_path"],
                extra=checkpoint_extra,
            )
        else:
            epochs_without_improvement += 1

        epoch_time = time.time() - epoch_started

        history_item = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "learning_rate": float(current_lr),
            "epoch_time_seconds": float(epoch_time),
            "improved": bool(improved),
        }

        if "gate_stats" in val_result:
            history_item["val_gate_stats"] = val_result["gate_stats"]

        history.append(history_item)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"val_acc={val_accuracy:.4f} | "
            f"val_f1={val_macro_f1:.4f} | "
            f"best_f1={best_val_macro_f1:.4f} | "
            f"lr={current_lr:.6f} | "
            f"time={epoch_time:.1f}s"
        )

        if args.early_stopping_patience > 0:
            if epochs_without_improvement >= args.early_stopping_patience:
                print(
                    f"\nEarly stopping triggered at epoch {epoch}. "
                    f"No improvement for {epochs_without_improvement} epochs."
                )
                break

    save_json(
        {
            "history": history,
            "summary": summarize_training_history(history),
        },
        output_names["history_path"],
    )

    print("\nLoading best checkpoint for test evaluation")
    print("=" * 80)
    checkpoint = load_checkpoint(
        model=model,
        path=output_names["checkpoint_path"],
        device=device,
    )

    test_result = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        task=args.task,
    )

    elapsed_seconds = time.time() - started_at

    train_summary = summarize_training_history(history)

    result = {
        "phase": "phase4_quality_aware_fusion",
        "task": args.task,
        "model_type": args.model_type,
        "model_name": model.__class__.__name__,
        "input": "2D + 3D + quality",
        "input_dim_2d": 40,
        "input_dim_3d": 59,
        "quality_dim": quality_dim,
        "quality_columns": QUALITY_FEATURE_COLUMNS,
        "num_classes": num_classes,
        "class_names": get_task_class_names(args.task),
        "dataset_info": dataset_info,
        "training_summary": train_summary,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_macro_f1),
        "best_val_result": best_val_result,
        "test_result": test_result,
        "checkpoint_path": output_names["checkpoint_path"],
        "result_path": output_names["result_path"],
        "history_path": output_names["history_path"],
        "scaler_path": output_names["scaler_path"],
        "num_parameters": int(count_parameters(model)),
        "elapsed_seconds": float(elapsed_seconds),
        "args": vars(args),
    }

    if "gate_stats" in test_result:
        result["gate_stats"] = test_result["gate_stats"]

    save_json(
        result,
        output_names["result_path"],
    )

    print("\nTraining finished.")
    print("=" * 80)
    print(f"Best epoch         : {best_epoch}")
    print(f"Best val macro F1  : {best_val_macro_f1:.6f}")
    print(f"Test accuracy      : {test_result['accuracy']:.6f}")
    print(f"Test macro F1      : {test_result['macro_f1']:.6f}")

    if "gate_stats" in test_result:
        print(f"Mean gate          : {test_result['gate_stats']['mean_gate']:.6f}")
        print(f"Std gate           : {test_result['gate_stats']['std_gate']:.6f}")

    print(f"Saved checkpoint   : {output_names['checkpoint_path']}")
    print(f"Saved result       : {output_names['result_path']}")
    print("=" * 80)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Phase 4 Quality-Aware 2D-3D Fusion model."
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["binary", "action"],
        help="Task to train: binary or action.",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="quality_gated",
        choices=["quality_gated", "quality_concat"],
        help="Model type. Default is quality_gated.",
    )

    parser.add_argument(
        "--quality-csv",
        type=str,
        default=DEFAULT_QUALITY_CSV,
        help="Path to quality_sequences.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save training results.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory to save checkpoints.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay.",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate.",
    )

    parser.add_argument(
        "--encoder-dim",
        type=int,
        default=128,
        help="Encoder output dimension.",
    )

    parser.add_argument(
        "--cnn-channels",
        type=int,
        default=128,
        help="CNN channels.",
    )

    parser.add_argument(
        "--lstm-hidden",
        type=int,
        default=128,
        help="LSTM hidden dimension.",
    )

    parser.add_argument(
        "--quality-hidden",
        type=int,
        default=128,
        help="Quality encoder hidden dimension.",
    )

    parser.add_argument(
        "--fusion-dim",
        type=int,
        default=128,
        help="Fusion dimension.",
    )

    parser.add_argument(
        "--gate-type",
        type=str,
        default="vector",
        choices=["vector", "scalar"],
        help="Gate type.",
    )

    parser.add_argument(
        "--pooling",
        type=str,
        default="attention",
        choices=["attention", "mean", "last", "mean_max"],
        help="Temporal pooling method.",
    )

    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau factor.",
    )

    parser.add_argument(
        "--lr-patience",
        type=int,
        default=4,
        help="ReduceLROnPlateau patience.",
    )

    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate.",
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience. Set 0 to disable.",
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm. Set 0 to disable.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader num_workers. Keep 0 on Windows for safety.",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable per-video feature caching. Use this if RAM is limited.",
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
    train_model(args)


if __name__ == "__main__":
    main()