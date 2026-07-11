from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import confusion_matrix


SCRIPT_PATH = Path(__file__).resolve()
BASELINE_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = BASELINE_ROOT.parents[1]

sys.path.insert(0, str(BASELINE_ROOT))

from utils.io_utils import ensure_dir, save_json, read_json
from utils.models import build_model
from utils.metrics_utils import compute_binary_metrics


# ======================================================================================
# FILE 05: TRAIN CNN, LSTM, AND SEQUENTIAL CONVLSTM
# ======================================================================================
#
# Original Yadav et al. pipeline:
#   skeleton coordinates + guided features
#   -> CNN
#   -> LSTM
#   -> ConvLSTM
#
# Their proposed ConvLSTM follows sequential fusion:
#   CNN feature extraction/filtering
#   -> LSTM temporal modeling
#   -> Fully Connected
#   -> Softmax classification
#
# Adapted project:
#   Input X shape:
#       (num_sequences, sequence_length, feature_dim)
#
#   Current expected shape:
#       (32199, 30, 65)
#
#   Feature dim:
#       51 raw skeleton coordinate features
#       14 guided features
#
# Output:
#   outputs/checkpoints/best_yadav2021_cnn.pt
#   outputs/checkpoints/best_yadav2021_lstm.pt
#   outputs/checkpoints/best_yadav2021_convlstm.pt
#
#   outputs/metrics/05_train_history_*.csv
#   outputs/metrics/05_train_cnn_lstm_convlstm_summary.csv
#   outputs/reports/05_train_cnn_lstm_convlstm_report.json
#
# Notes:
#   - Split is video-level from file 04.
#   - Model selection uses best validation macro-F1.
#   - Feature scaling is train-only min-max to avoid test leakage.
# ======================================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_feature_columns(feature_json_path: Path) -> Dict:
    if not feature_json_path.exists():
        raise FileNotFoundError(f"Feature columns JSON not found: {feature_json_path}")

    with open(feature_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_split_indices(sequence_split_df: pd.DataFrame, split_name: str) -> np.ndarray:
    part = sequence_split_df[sequence_split_df["split"] == split_name].copy()

    if len(part) == 0:
        raise ValueError(f"No sequences found for split: {split_name}")

    return part["sequence_index"].astype(int).to_numpy()


def compute_train_minmax_stats(X_train: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute train-only min-max stats per feature dimension.

    X_train shape:
        (N, T, F)

    Stats shape:
        (F,)
    """
    flat = X_train.reshape(-1, X_train.shape[-1])

    feature_min = np.nanmin(flat, axis=0).astype(np.float32)
    feature_max = np.nanmax(flat, axis=0).astype(np.float32)

    denom = feature_max - feature_min
    denom = np.where(np.abs(denom) < 1e-6, 1.0, denom).astype(np.float32)

    return {
        "feature_min": feature_min,
        "feature_max": feature_max,
        "feature_denom": denom,
    }


def apply_train_minmax(X: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    feature_min = stats["feature_min"].reshape(1, 1, -1)
    feature_denom = stats["feature_denom"].reshape(1, 1, -1)

    X_scaled = (X - feature_min) / feature_denom
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    return X_scaled.astype(np.float32)


def save_minmax_stats(stats: Dict[str, np.ndarray], path: Path) -> None:
    ensure_dir(path.parent)
    np.savez(
        path,
        feature_min=stats["feature_min"],
        feature_max=stats["feature_max"],
        feature_denom=stats["feature_denom"],
    )


def build_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    sequence_indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    idx_tensor = torch.tensor(sequence_indices, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor, idx_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_class_weights(y_train: np.ndarray, mode: str, device: torch.device) -> torch.Tensor | None:
    mode = mode.lower().strip()

    if mode == "none":
        return None

    if mode != "balanced":
        raise ValueError(f"Unknown class weight mode: {mode}")

    counts = np.bincount(y_train.astype(int), minlength=2).astype(np.float32)
    total = float(np.sum(counts))

    weights = total / (2.0 * np.maximum(counts, 1.0))

    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> Dict:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for xb, yb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = int(yb.size(0))
        total_loss += float(loss.item()) * batch_size

        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == yb).sum().item())
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    y_true_all = []
    y_pred_all = []
    prob_fall_all = []
    sequence_idx_all = []

    for xb, yb, idxb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        batch_size = int(yb.size(0))
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        y_true_all.append(yb.detach().cpu().numpy())
        y_pred_all.append(preds.detach().cpu().numpy())
        prob_fall_all.append(probs[:, 1].detach().cpu().numpy())
        sequence_idx_all.append(idxb.detach().cpu().numpy())

    y_true = np.concatenate(y_true_all).astype(int)
    y_pred = np.concatenate(y_pred_all).astype(int)
    prob_fall = np.concatenate(prob_fall_all).astype(np.float32)
    sequence_indices = np.concatenate(sequence_idx_all).astype(int)

    metrics = compute_binary_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["num_samples"] = int(total_samples)

    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "prob_fall": prob_fall,
        "sequence_indices": sequence_indices,
    }


def save_predictions(
    result: Dict,
    sequence_split_df: pd.DataFrame,
    output_path: Path,
) -> None:
    pred_df = pd.DataFrame({
        "sequence_index": result["sequence_indices"],
        "y_true": result["y_true"],
        "y_pred": result["y_pred"],
        "prob_fall": result["prob_fall"],
    })

    meta_cols = [
        "sequence_index",
        "sequence_id",
        "video_id",
        "split",
        "label",
        "label_name",
        "start_frame_index",
        "end_frame_index",
        "original_num_frames",
        "is_padded",
    ]

    available_meta_cols = [c for c in meta_cols if c in sequence_split_df.columns]

    pred_df = pred_df.merge(
        sequence_split_df[available_meta_cols],
        on="sequence_index",
        how="left",
    )

    pred_df["true_label_name"] = pred_df["y_true"].map({0: "Not_Fall", 1: "Fall"})
    pred_df["pred_label_name"] = pred_df["y_pred"].map({0: "Not_Fall", 1: "Fall"})

    ensure_dir(output_path.parent)
    pred_df.to_csv(output_path, index=False)


def save_history_plot(history_df: pd.DataFrame, model_name: str, output_dir: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt

        ensure_dir(output_dir)

        fig_path = output_dir / f"05_training_curve_{model_name}.png"

        plt.figure(figsize=(9, 5))
        plt.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
        plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
        plt.plot(history_df["epoch"], history_df["val_macro_f1"], label="val_macro_f1")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(f"Yadav2021-style {model_name.upper()} Training Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=160)
        plt.close()

        return str(fig_path)

    except Exception as e:
        print(f"Warning: could not save plot for {model_name}: {e}")
        return None


def train_model(
    model_name: str,
    input_dim: int,
    num_classes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    y_train: np.ndarray,
    sequence_split_df: pd.DataFrame,
    device: torch.device,
    args,
    output_dirs: Dict[str, Path],
) -> Dict:
    print("=" * 100)
    print(f"Training model: {model_name.upper()}")
    print("=" * 100)

    model = build_model(
        model_name=model_name,
        input_dim=input_dim,
        num_classes=num_classes,
        cnn_channels=args.cnn_channels,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)

    class_weights = compute_class_weights(
        y_train=y_train,
        mode=args.class_weight_mode,
        device=device,
    )

    if class_weights is not None:
        print(f"Class weights: {class_weights.detach().cpu().numpy().tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(3, args.early_stopping_patience // 3),
    )

    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_checkpoint_path = output_dirs["checkpoints"] / f"best_yadav2021_{model_name}.pt"

    history_rows = []
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )

        val_result = evaluate_loader(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        val_metrics = val_result["metrics"]
        val_macro_f1 = float(val_metrics["macro_f1"])

        scheduler.step(val_macro_f1)

        current_lr = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": int(epoch),
            "model_name": model_name,
            "learning_rate": current_lr,
            "train_loss": float(train_stats["loss"]),
            "train_accuracy": float(train_stats["accuracy"]),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_fall_recall": float(val_metrics["fall_recall"]),
            "val_fall_f1": float(val_metrics["fall_f1"]),
            "val_specificity": float(val_metrics["specificity_not_fall_recall"]),
            "val_tn": int(val_metrics["tn"]),
            "val_fp": int(val_metrics["fp"]),
            "val_fn": int(val_metrics["fn"]),
            "val_tp": int(val_metrics["tp"]),
        }

        history_rows.append(row)

        improved = val_macro_f1 > best_val_macro_f1

        if improved:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "input_dim": int(input_dim),
                "num_classes": int(num_classes),
                "epoch": int(epoch),
                "best_val_macro_f1": float(best_val_macro_f1),
                "args": vars(args),
            }

            ensure_dir(best_checkpoint_path.parent)
            torch.save(checkpoint, best_checkpoint_path)

        else:
            patience_counter += 1

        print(
            f"[{model_name.upper()}] "
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_stats['loss']:.4f} | "
            f"train_acc={train_stats['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"val_fall_recall={val_metrics['fall_recall']:.4f} | "
            f"best_epoch={best_epoch}"
        )

        if patience_counter >= args.early_stopping_patience:
            print(f"Early stopping triggered for {model_name} at epoch {epoch}.")
            break

    elapsed_seconds = time.time() - start_time

    history_df = pd.DataFrame(history_rows)
    history_csv = output_dirs["metrics"] / f"05_train_history_{model_name}.csv"
    ensure_dir(history_csv.parent)
    history_df.to_csv(history_csv, index=False)

    plot_path = save_history_plot(
        history_df=history_df,
        model_name=model_name,
        output_dir=output_dirs["plots"],
    )

    # Load best checkpoint before final split evaluation.
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    train_result = evaluate_loader(model, train_loader, criterion, device)
    val_result = evaluate_loader(model, val_loader, criterion, device)
    test_result = evaluate_loader(model, test_loader, criterion, device)

    save_predictions(
        result=train_result,
        sequence_split_df=sequence_split_df,
        output_path=output_dirs["predictions"] / f"05_predictions_train_{model_name}.csv",
    )

    save_predictions(
        result=val_result,
        sequence_split_df=sequence_split_df,
        output_path=output_dirs["predictions"] / f"05_predictions_val_{model_name}.csv",
    )

    save_predictions(
        result=test_result,
        sequence_split_df=sequence_split_df,
        output_path=output_dirs["predictions"] / f"05_predictions_test_{model_name}.csv",
    )

    summary = {
        "model_name": model_name,
        "checkpoint_path": str(best_checkpoint_path),
        "history_csv": str(history_csv),
        "training_curve_png": plot_path,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_macro_f1),
        "elapsed_seconds": float(elapsed_seconds),

        "train_accuracy": float(train_result["metrics"]["accuracy"]),
        "train_macro_f1": float(train_result["metrics"]["macro_f1"]),
        "train_fall_recall": float(train_result["metrics"]["fall_recall"]),
        "train_fall_f1": float(train_result["metrics"]["fall_f1"]),
        "train_specificity": float(train_result["metrics"]["specificity_not_fall_recall"]),
        "train_tn": int(train_result["metrics"]["tn"]),
        "train_fp": int(train_result["metrics"]["fp"]),
        "train_fn": int(train_result["metrics"]["fn"]),
        "train_tp": int(train_result["metrics"]["tp"]),

        "val_accuracy": float(val_result["metrics"]["accuracy"]),
        "val_macro_f1": float(val_result["metrics"]["macro_f1"]),
        "val_fall_recall": float(val_result["metrics"]["fall_recall"]),
        "val_fall_f1": float(val_result["metrics"]["fall_f1"]),
        "val_specificity": float(val_result["metrics"]["specificity_not_fall_recall"]),
        "val_tn": int(val_result["metrics"]["tn"]),
        "val_fp": int(val_result["metrics"]["fp"]),
        "val_fn": int(val_result["metrics"]["fn"]),
        "val_tp": int(val_result["metrics"]["tp"]),

        "test_accuracy": float(test_result["metrics"]["accuracy"]),
        "test_macro_f1": float(test_result["metrics"]["macro_f1"]),
        "test_fall_recall": float(test_result["metrics"]["fall_recall"]),
        "test_fall_f1": float(test_result["metrics"]["fall_f1"]),
        "test_specificity": float(test_result["metrics"]["specificity_not_fall_recall"]),
        "test_tn": int(test_result["metrics"]["tn"]),
        "test_fp": int(test_result["metrics"]["fp"]),
        "test_fn": int(test_result["metrics"]["fn"]),
        "test_tp": int(test_result["metrics"]["tp"]),
    }

    print("=" * 100)
    print(f"Completed model: {model_name.upper()}")
    print(f"Best epoch:      {best_epoch}")
    print(f"Best val F1:     {best_val_macro_f1:.4f}")
    print(f"Test acc:        {summary['test_accuracy']:.4f}")
    print(f"Test macro F1:   {summary['test_macro_f1']:.4f}")
    print(f"Test Fall recall:{summary['test_fall_recall']:.4f}")
    print(f"Checkpoint:      {best_checkpoint_path}")
    print("=" * 100)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Train CNN, LSTM, and sequential ConvLSTM for Yadav2021-style guided feature baseline."
    )

    parser.add_argument(
        "--x-path",
        type=str,
        default=str(BASELINE_ROOT / "data" / "sequences" / "X_sequences.npy"),
    )

    parser.add_argument(
        "--y-path",
        type=str,
        default=str(BASELINE_ROOT / "data" / "sequences" / "y_sequences.npy"),
    )

    parser.add_argument(
        "--sequence-split-csv",
        type=str,
        default=str(BASELINE_ROOT / "data" / "splits" / "yadav2021_sequence_level_split.csv"),
    )

    parser.add_argument(
        "--feature-columns-json",
        type=str,
        default=str(BASELINE_ROOT / "data" / "sequences" / "feature_columns.json"),
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["cnn", "lstm", "convlstm"],
        choices=["cnn", "lstm", "convlstm"],
        help="Models to train.",
    )

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.30)

    parser.add_argument("--cnn-channels", type=int, default=128)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)

    parser.add_argument(
        "--class-weight-mode",
        type=str,
        default="balanced",
        choices=["none", "balanced"],
        help="Use balanced class weights or normal cross-entropy.",
    )

    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    device = get_device(force_cpu=args.force_cpu)

    x_path = Path(args.x_path)
    y_path = Path(args.y_path)
    sequence_split_csv = Path(args.sequence_split_csv)
    feature_columns_json = Path(args.feature_columns_json)

    output_dirs = {
        "checkpoints": BASELINE_ROOT / "outputs" / "checkpoints",
        "metrics": BASELINE_ROOT / "outputs" / "metrics",
        "predictions": BASELINE_ROOT / "outputs" / "predictions",
        "reports": BASELINE_ROOT / "outputs" / "reports",
        "plots": BASELINE_ROOT / "outputs" / "plots",
    }

    for p in output_dirs.values():
        ensure_dir(p)

    print("=" * 100)
    print("FILE 05 - Train CNN, LSTM, and Sequential ConvLSTM")
    print("=" * 100)
    print(f"Device:                {device}")
    print(f"X path:                {x_path}")
    print(f"y path:                {y_path}")
    print(f"Sequence split CSV:    {sequence_split_csv}")
    print(f"Feature columns JSON:  {feature_columns_json}")
    print(f"Models:                {args.models}")
    print(f"Epochs:                {args.epochs}")
    print(f"Batch size:            {args.batch_size}")
    print(f"LR:                    {args.learning_rate}")
    print(f"Class weight mode:     {args.class_weight_mode}")
    print("=" * 100)

    if not x_path.exists():
        raise FileNotFoundError(f"X not found: {x_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"y not found: {y_path}")

    if not sequence_split_csv.exists():
        raise FileNotFoundError(f"sequence split CSV not found: {sequence_split_csv}")

    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    sequence_split_df = pd.read_csv(sequence_split_csv)

    feature_info = load_feature_columns(feature_columns_json)

    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: X={len(X)}, y={len(y)}")

    if "sequence_index" not in sequence_split_df.columns:
        raise ValueError("sequence_split_df must contain sequence_index column.")

    train_idx = get_split_indices(sequence_split_df, "train")
    val_idx = get_split_indices(sequence_split_df, "val")
    test_idx = get_split_indices(sequence_split_df, "test")

    print(f"Loaded X shape:        {X.shape}")
    print(f"Loaded y shape:        {y.shape}")
    print(f"Train sequences:       {len(train_idx)}")
    print(f"Val sequences:         {len(val_idx)}")
    print(f"Test sequences:        {len(test_idx)}")
    print(f"Input dim:             {X.shape[-1]}")
    print("=" * 100)

    # Train-only min-max scaling for all 65 features.
    minmax_stats = compute_train_minmax_stats(X[train_idx])
    stats_path = output_dirs["metrics"] / "05_train_feature_minmax_stats.npz"
    save_minmax_stats(minmax_stats, stats_path)

    X_scaled = apply_train_minmax(X, minmax_stats)

    train_loader = build_dataloader(
        X=X_scaled[train_idx],
        y=y[train_idx],
        sequence_indices=train_idx,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = build_dataloader(
        X=X_scaled[val_idx],
        y=y[val_idx],
        sequence_indices=val_idx,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_loader = build_dataloader(
        X=X_scaled[test_idx],
        y=y[test_idx],
        sequence_indices=test_idx,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    summaries = []

    for model_name in args.models:
        summary = train_model(
            model_name=model_name,
            input_dim=int(X.shape[-1]),
            num_classes=2,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            y_train=y[train_idx],
            sequence_split_df=sequence_split_df,
            device=device,
            args=args,
            output_dirs=output_dirs,
        )

        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_csv = output_dirs["metrics"] / "05_train_cnn_lstm_convlstm_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    report = {
        "status": "completed",
        "pipeline_note": "Train CNN, LSTM, and sequential ConvLSTM on Yadav-style skeleton coordinates plus guided features. Best checkpoint is selected by validation macro-F1.",
        "device": str(device),
        "x_path": str(x_path),
        "y_path": str(y_path),
        "sequence_split_csv": str(sequence_split_csv),
        "feature_columns_json": str(feature_columns_json),
        "feature_minmax_stats_npz": str(stats_path),
        "x_shape": list(X.shape),
        "y_shape": list(y.shape),
        "num_train_sequences": int(len(train_idx)),
        "num_val_sequences": int(len(val_idx)),
        "num_test_sequences": int(len(test_idx)),
        "input_dim": int(X.shape[-1]),
        "num_classes": 2,
        "class_names": ["Not_Fall", "Fall"],
        "args": vars(args),
        "feature_info": feature_info,
        "summary_csv": str(summary_csv),
        "models": summaries,
    }

    report_path = output_dirs["reports"] / "05_train_cnn_lstm_convlstm_report.json"
    save_json(report, report_path)

    print("=" * 100)
    print("FILE 05 completed.")
    print("=" * 100)
    print(f"Summary CSV: {summary_csv}")
    print(f"Report:      {report_path}")
    print("=" * 100)

    print(summary_df[
        [
            "model_name",
            "best_epoch",
            "best_val_macro_f1",
            "test_accuracy",
            "test_macro_f1",
            "test_fall_recall",
            "test_fall_f1",
            "test_specificity",
            "test_tn",
            "test_fp",
            "test_fn",
            "test_tp",
        ]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
