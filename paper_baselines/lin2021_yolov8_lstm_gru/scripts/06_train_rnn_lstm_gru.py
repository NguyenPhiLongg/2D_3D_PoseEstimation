from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


SCRIPT_PATH = Path(__file__).resolve()
BASELINE_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = BASELINE_ROOT.parents[1]


# ======================================================================================
# FILE 06: TRAIN RNN / LSTM / GRU
# ======================================================================================
#
# This file follows the temporal model training stage of the Lin-style pipeline.
#
# Input from File 05:
#   train NPZ:
#       X = (N, 100, 30)
#       y = (N,)
#
#   val NPZ:
#       X = (N, 100, 30)
#       y = (N,)
#
# Models:
#   - Simple RNN
#   - LSTM
#   - GRU
#
# This file does NOT use:
#   - YOLOv8 model during training
#   - CNN
#   - 3D pose
#   - quality-aware fusion
#
# It only trains recurrent models on skeleton sequences, following the paper-style
# RNN/LSTM/GRU temporal classification pipeline.
# ======================================================================================


LABEL_NAMES = {
    0: "Not_Fall",
    1: "Fall",
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_arg)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")

    data = np.load(path, allow_pickle=True)

    required = ["X", "y", "sequence_ids", "video_ids"]

    for key in required:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in {path}")

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    sequence_ids = data["sequence_ids"].astype(str)
    video_ids = data["video_ids"].astype(str)

    if X.ndim != 3:
        raise ValueError(f"Expected X shape (N, T, D), got {X.shape}")

    if len(y) != X.shape[0]:
        raise ValueError(f"y length does not match X. y={len(y)}, X={X.shape[0]}")

    return {
        "X": X,
        "y": y,
        "sequence_ids": sequence_ids,
        "video_ids": video_ids,
    }


class SkeletonSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RecurrentClassifier(nn.Module):
    def __init__(
        self,
        model_type: str,
        input_size: int = 30,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()

        model_type = model_type.lower()

        if model_type not in ["rnn", "lstm", "gru"]:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        recurrent_dropout = dropout if num_layers > 1 else 0.0

        if model_type == "rnn":
            self.recurrent = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
                bidirectional=bidirectional,
                nonlinearity="tanh",
            )

        elif model_type == "lstm":
            self.recurrent = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
                bidirectional=bidirectional,
            )

        else:
            self.recurrent = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
                bidirectional=bidirectional,
            )

        direction_factor = 2 if bidirectional else 1

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * direction_factor, num_classes)

    def forward(self, x):
        if self.model_type == "lstm":
            _, (h_n, _) = self.recurrent(x)
        else:
            _, h_n = self.recurrent(x)

        if self.bidirectional:
            # Last layer forward and backward hidden states.
            last_forward = h_n[-2]
            last_backward = h_n[-1]
            features = torch.cat([last_forward, last_backward], dim=1)
        else:
            features = h_n[-1]

        features = self.dropout(features)
        logits = self.classifier(features)

        return logits


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=2).astype(np.float32)
    total = counts.sum()

    weights = np.zeros_like(counts, dtype=np.float32)

    for cls in range(len(counts)):
        if counts[cls] > 0:
            weights[cls] = total / (len(counts) * counts[cls])
        else:
            weights[cls] = 0.0

    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),

        "not_fall_precision": float(precision_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),
        "not_fall_recall": float(recall_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),
        "not_fall_f1": float(f1_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),

        "fall_precision": float(precision_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "fall_recall": float(recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "fall_f1": float(f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return metrics


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
) -> Tuple[float, Dict]:
    model.train()

    total_loss = 0.0
    total_items = 0

    all_true = []
    all_pred = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        batch_size = X_batch.size(0)

        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

        preds = torch.argmax(logits, dim=1)

        all_true.append(y_batch.detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())

    avg_loss = total_loss / max(total_items, 1)

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(avg_loss)

    return avg_loss, metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict, np.ndarray, np.ndarray]:
    model.eval()

    total_loss = 0.0
    total_items = 0

    all_true = []
    all_pred = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        batch_size = X_batch.size(0)

        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

        preds = torch.argmax(logits, dim=1)

        all_true.append(y_batch.detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())

    avg_loss = total_loss / max(total_items, 1)

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(avg_loss)

    return avg_loss, metrics, y_true, y_pred


def save_checkpoint(
    model: nn.Module,
    optimizer,
    checkpoint_path: Path,
    model_type: str,
    epoch: int,
    args,
    val_metrics: Dict,
):
    ensure_dir(checkpoint_path.parent)

    checkpoint = {
        "model_type": model_type,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "val_metrics": val_metrics,
        "label_names": LABEL_NAMES,
    }

    torch.save(checkpoint, checkpoint_path)


def train_one_model(
    model_type: str,
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    args,
    device: torch.device,
) -> Dict:
    model_type = model_type.lower()

    X_train = train_data["X"]
    y_train = train_data["y"]

    X_val = val_data["X"]
    y_val = val_data["y"]

    input_size = int(X_train.shape[2])

    train_dataset = SkeletonSequenceDataset(X_train, y_train)
    val_dataset = SkeletonSequenceDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = RecurrentClassifier(
        model_type=model_type,
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=2,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)

    if args.no_class_weight:
        criterion = nn.CrossEntropyLoss()
        class_weights_list = None
    else:
        class_weights = compute_class_weights(y_train).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        class_weights_list = [float(x) for x in class_weights.detach().cpu().numpy().tolist()]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    metrics_dir = Path(args.metrics_dir)

    ensure_dir(checkpoint_dir)
    ensure_dir(metrics_dir)

    best_checkpoint_path = checkpoint_dir / f"best_{model_type}.pt"
    last_checkpoint_path = checkpoint_dir / f"last_{model_type}.pt"
    history_path = metrics_dir / f"{model_type}_training_history.csv"

    best_val_macro_f1 = -1.0
    best_epoch = -1
    patience_counter = 0

    history_rows = []

    start_time = time.time()

    print("=" * 100)
    print(f"Training model: {model_type.upper()}")
    print("=" * 100)
    print(f"Device:          {device}")
    print(f"Train shape:     {X_train.shape}")
    print(f"Val shape:       {X_val.shape}")
    print(f"Input size:      {input_size}")
    print(f"Hidden size:     {args.hidden_size}")
    print(f"Layers:          {args.num_layers}")
    print(f"Dropout:         {args.dropout}")
    print(f"Bidirectional:   {args.bidirectional}")
    print(f"Class weights:   {class_weights_list}")
    print("=" * 100)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_metrics, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        row = {
            "model_type": model_type,
            "epoch": int(epoch),

            "train_loss": float(train_loss),
            "train_accuracy": float(train_metrics["accuracy"]),
            "train_macro_f1": float(train_metrics["macro_f1"]),
            "train_fall_recall": float(train_metrics["fall_recall"]),
            "train_fall_f1": float(train_metrics["fall_f1"]),

            "val_loss": float(val_loss),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_fall_recall": float(val_metrics["fall_recall"]),
            "val_fall_f1": float(val_metrics["fall_f1"]),
            "val_tn": int(val_metrics["tn"]),
            "val_fp": int(val_metrics["fp"]),
            "val_fn": int(val_metrics["fn"]),
            "val_tp": int(val_metrics["tp"]),
        }

        history_rows.append(row)

        is_best = val_metrics["macro_f1"] > best_val_macro_f1

        if is_best:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = int(epoch)
            patience_counter = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_path=best_checkpoint_path,
                model_type=model_type,
                epoch=epoch,
                args=args,
                val_metrics=val_metrics,
            )
        else:
            patience_counter += 1

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_path=last_checkpoint_path,
            model_type=model_type,
            epoch=epoch,
            args=args,
            val_metrics=val_metrics,
        )

        print(
            f"[{model_type.upper()}] "
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_fall_recall={val_metrics['fall_recall']:.4f} "
            f"{'BEST' if is_best else ''}"
        )

        pd.DataFrame(history_rows).to_csv(history_path, index=False)

        if patience_counter >= args.patience:
            print(
                f"Early stopping for {model_type.upper()} at epoch {epoch}. "
                f"Best epoch = {best_epoch}, best val macro F1 = {best_val_macro_f1:.4f}"
            )
            break

    elapsed_seconds = time.time() - start_time

    history_df = pd.DataFrame(history_rows)

    best_row = history_df.loc[history_df["val_macro_f1"].idxmax()].to_dict()

    summary = {
        "model_type": model_type,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_macro_f1),
        "best_val_accuracy": float(best_row["val_accuracy"]),
        "best_val_fall_recall": float(best_row["val_fall_recall"]),
        "best_val_fall_f1": float(best_row["val_fall_f1"]),
        "best_val_tn": int(best_row["val_tn"]),
        "best_val_fp": int(best_row["val_fp"]),
        "best_val_fn": int(best_row["val_fn"]),
        "best_val_tp": int(best_row["val_tp"]),
        "epochs_ran": int(len(history_df)),
        "elapsed_seconds": float(elapsed_seconds),
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "history_csv": str(history_path),
    }

    return summary


def parse_models(models_text: str) -> List[str]:
    text = models_text.strip().lower()

    if text == "all":
        return ["rnn", "lstm", "gru"]

    models = [
        m.strip().lower()
        for m in text.split(",")
        if m.strip() != ""
    ]

    allowed = {"rnn", "lstm", "gru"}

    bad = [
        m for m in models
        if m not in allowed
    ]

    if bad:
        raise ValueError(f"Unsupported model names: {bad}. Allowed: rnn,lstm,gru,all")

    return models


def main():
    parser = argparse.ArgumentParser(
        description="Train RNN, LSTM, and GRU models for Lin-style YOLOv8 skeleton baseline."
    )

    parser.add_argument(
        "--train-npz",
        type=str,
        default=str(BASELINE_ROOT / "data" / "splits" / "lin2021_yolov8_train_sequences.npz"),
        help="Train NPZ from File 05.",
    )

    parser.add_argument(
        "--val-npz",
        type=str,
        default=str(BASELINE_ROOT / "data" / "splits" / "lin2021_yolov8_val_sequences.npz"),
        help="Validation NPZ from File 05.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "checkpoints"),
        help="Directory to save checkpoints.",
    )

    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "metrics"),
        help="Directory to save training metrics.",
    )

    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "reports"),
        help="Directory to save report JSON.",
    )

    parser.add_argument(
        "--models",
        type=str,
        default="rnn,lstm,gru",
        help="Models to train: rnn,lstm,gru or all.",
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Use bidirectional recurrent models. Default is false to stay simpler and closer to the baseline setting.",
    )

    parser.add_argument(
        "--no-class-weight",
        action="store_true",
        help="Disable balanced class weights.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = get_device(args.device)

    train_npz = Path(args.train_npz)
    val_npz = Path(args.val_npz)
    checkpoint_dir = Path(args.checkpoint_dir)
    metrics_dir = Path(args.metrics_dir)
    reports_dir = Path(args.reports_dir)

    ensure_dir(checkpoint_dir)
    ensure_dir(metrics_dir)
    ensure_dir(reports_dir)

    train_data = load_npz(train_npz)
    val_data = load_npz(val_npz)

    models_to_train = parse_models(args.models)

    print("=" * 100)
    print("FILE 06 - Train RNN / LSTM / GRU")
    print("=" * 100)
    print(f"Baseline root:    {BASELINE_ROOT}")
    print(f"Train NPZ:        {train_npz}")
    print(f"Val NPZ:          {val_npz}")
    print(f"Train X shape:    {train_data['X'].shape}")
    print(f"Val X shape:      {val_data['X'].shape}")
    print(f"Device:           {device}")
    print(f"Models:           {models_to_train}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch size:       {args.batch_size}")
    print("=" * 100)

    summaries = []

    for model_type in models_to_train:
        summary = train_one_model(
            model_type=model_type,
            train_data=train_data,
            val_data=val_data,
            args=args,
            device=device,
        )

        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_csv = metrics_dir / "lin2021_yolov8_training_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    best_model_row = summary_df.sort_values(
        ["best_val_macro_f1", "best_val_fall_recall"],
        ascending=False,
    ).iloc[0].to_dict()

    report = {
        "status": "completed",
        "pipeline_note": "RNN, LSTM, and GRU are trained on fixed-length normalized skeleton sequences. This corresponds to the recurrent temporal classification stage of the Lin-style pipeline.",
        "train_npz": str(train_npz),
        "val_npz": str(val_npz),
        "train_shape": list(train_data["X"].shape),
        "val_shape": list(val_data["X"].shape),
        "device": str(device),
        "models_trained": models_to_train,
        "training_args": vars(args),
        "summaries": summaries,
        "summary_csv": str(summary_csv),
        "best_model_by_val_macro_f1": {
            "model_type": str(best_model_row["model_type"]),
            "best_epoch": int(best_model_row["best_epoch"]),
            "best_val_macro_f1": float(best_model_row["best_val_macro_f1"]),
            "best_val_accuracy": float(best_model_row["best_val_accuracy"]),
            "best_val_fall_recall": float(best_model_row["best_val_fall_recall"]),
            "best_val_fall_f1": float(best_model_row["best_val_fall_f1"]),
            "best_checkpoint": str(best_model_row["best_checkpoint"]),
        },
    }

    report_path = reports_dir / "06_train_rnn_lstm_gru_report.json"
    save_json(report, report_path)

    print("=" * 100)
    print("FILE 06 completed.")
    print("=" * 100)
    print("Training summary:")
    print(summary_df.to_string(index=False))
    print("-" * 100)
    print("Best model by validation Macro F1:")
    print(report["best_model_by_val_macro_f1"])
    print("-" * 100)
    print(f"Summary CSV: {summary_csv}")
    print(f"Report:      {report_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
