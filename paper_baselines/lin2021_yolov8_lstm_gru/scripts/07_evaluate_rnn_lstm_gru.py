from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


SCRIPT_PATH = Path(__file__).resolve()
BASELINE_ROOT = SCRIPT_PATH.parents[1]

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


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")

    data = np.load(path, allow_pickle=True)

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
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_ids: np.ndarray, video_ids: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.sequence_ids = sequence_ids.astype(str)
        self.video_ids = video_ids.astype(str)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.sequence_ids[idx], self.video_ids[idx]


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
            last_forward = h_n[-2]
            last_backward = h_n[-1]
            features = torch.cat([last_forward, last_backward], dim=1)
        else:
            features = h_n[-1]

        features = self.dropout(features)
        logits = self.classifier(features)

        return logits


def safe_torch_load(checkpoint_path: Path, device: torch.device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_model_from_checkpoint(checkpoint_path: Path, model_type: str, input_size: int, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = safe_torch_load(checkpoint_path, device)

    ckpt_args = checkpoint.get("args", {})

    hidden_size = int(ckpt_args.get("hidden_size", 128))
    num_layers = int(ckpt_args.get("num_layers", 2))
    dropout = float(ckpt_args.get("dropout", 0.3))
    bidirectional = bool(ckpt_args.get("bidirectional", False))

    model = RecurrentClassifier(
        model_type=model_type,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=2,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    model_info = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "val_metrics_at_checkpoint": checkpoint.get("val_metrics", {}),
    }

    return model, model_info


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
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


def parse_models(models_text: str) -> List[str]:
    text = models_text.strip().lower()

    if text == "all":
        return ["rnn", "lstm", "gru"]

    models = [m.strip().lower() for m in text.split(",") if m.strip()]

    allowed = {"rnn", "lstm", "gru"}
    bad = [m for m in models if m not in allowed]

    if bad:
        raise ValueError(f"Unsupported model names: {bad}. Allowed: rnn,lstm,gru,all")

    return models


@torch.no_grad()
def predict_sequences(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()

    rows = []

    for X_batch, y_batch, sequence_ids, video_ids in loader:
        X_batch = X_batch.to(device)

        logits = model(X_batch)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_np = y_batch.cpu().numpy().astype(int)
        pred_np = preds.cpu().numpy().astype(int)
        probs_np = probs.cpu().numpy()

        for i in range(len(y_np)):
            true_label = int(y_np[i])
            pred_label = int(pred_np[i])

            rows.append({
                "sequence_id": str(sequence_ids[i]),
                "video_id": str(video_ids[i]),
                "true_label": true_label,
                "true_label_name": LABEL_NAMES.get(true_label, "Unknown"),
                "pred_label": pred_label,
                "pred_label_name": LABEL_NAMES.get(pred_label, "Unknown"),
                "prob_not_fall": float(probs_np[i, 0]),
                "prob_fall": float(probs_np[i, 1]),
                "correct": int(true_label == pred_label),
            })

    return pd.DataFrame(rows)


def aggregate_to_video_level(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for video_id, group in pred_df.groupby("video_id"):
        true_labels = sorted(group["true_label"].astype(int).unique().tolist())

        if len(true_labels) != 1:
            raise ValueError(f"Video {video_id} has multiple true labels: {true_labels}")

        true_label = int(true_labels[0])

        vote_counts = group["pred_label"].astype(int).value_counts().to_dict()
        pred_majority = int(group["pred_label"].astype(int).mode().iloc[0])

        mean_prob_not_fall = float(group["prob_not_fall"].mean())
        mean_prob_fall = float(group["prob_fall"].mean())
        pred_mean_prob = 1 if mean_prob_fall >= mean_prob_not_fall else 0

        rows.append({
            "video_id": str(video_id),
            "true_label": true_label,
            "true_label_name": LABEL_NAMES.get(true_label, "Unknown"),

            "pred_label_majority": pred_majority,
            "pred_label_majority_name": LABEL_NAMES.get(pred_majority, "Unknown"),
            "correct_majority": int(true_label == pred_majority),

            "pred_label_mean_prob": int(pred_mean_prob),
            "pred_label_mean_prob_name": LABEL_NAMES.get(pred_mean_prob, "Unknown"),
            "correct_mean_prob": int(true_label == pred_mean_prob),

            "mean_prob_not_fall": mean_prob_not_fall,
            "mean_prob_fall": mean_prob_fall,
            "num_sequences": int(len(group)),
            "vote_not_fall": int(vote_counts.get(0, 0)),
            "vote_fall": int(vote_counts.get(1, 0)),
        })

    return pd.DataFrame(rows)


def evaluate_one_model(model_type: str, checkpoint_path: Path, test_data: Dict[str, np.ndarray], args, device: torch.device) -> Dict:
    input_size = int(test_data["X"].shape[2])

    model, model_info = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        input_size=input_size,
        device=device,
    )

    dataset = SkeletonSequenceDataset(
        X=test_data["X"],
        y=test_data["y"],
        sequence_ids=test_data["sequence_ids"],
        video_ids=test_data["video_ids"],
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    seq_pred_df = predict_sequences(model, loader, device)

    y_true_seq = seq_pred_df["true_label"].astype(int).to_numpy()
    y_pred_seq = seq_pred_df["pred_label"].astype(int).to_numpy()
    sequence_metrics = compute_metrics(y_true_seq, y_pred_seq)

    video_pred_df = aggregate_to_video_level(seq_pred_df)

    y_true_video = video_pred_df["true_label"].astype(int).to_numpy()

    y_pred_video_majority = video_pred_df["pred_label_majority"].astype(int).to_numpy()
    video_majority_metrics = compute_metrics(y_true_video, y_pred_video_majority)

    y_pred_video_mean_prob = video_pred_df["pred_label_mean_prob"].astype(int).to_numpy()
    video_mean_prob_metrics = compute_metrics(y_true_video, y_pred_video_mean_prob)

    predictions_dir = Path(args.predictions_dir)
    metrics_dir = Path(args.metrics_dir)

    ensure_dir(predictions_dir)
    ensure_dir(metrics_dir)

    seq_pred_csv = predictions_dir / f"{model_type}_test_sequence_predictions.csv"
    video_pred_csv = predictions_dir / f"{model_type}_test_video_predictions.csv"

    seq_pred_df.to_csv(seq_pred_csv, index=False)
    video_pred_df.to_csv(video_pred_csv, index=False)

    full_metrics = {
        "model_type": model_type,
        "model_info": model_info,
        "sequence_metrics": sequence_metrics,
        "video_majority_metrics": video_majority_metrics,
        "video_mean_probability_metrics": video_mean_prob_metrics,
        "sequence_predictions_csv": str(seq_pred_csv),
        "video_predictions_csv": str(video_pred_csv),
    }

    full_metrics_json = metrics_dir / f"{model_type}_test_metrics.json"
    save_json(full_metrics, full_metrics_json)

    result = {
        "model_type": model_type,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(model_info["checkpoint_epoch"]),

        "sequence_accuracy": float(sequence_metrics["accuracy"]),
        "sequence_macro_f1": float(sequence_metrics["macro_f1"]),
        "sequence_fall_recall": float(sequence_metrics["fall_recall"]),
        "sequence_fall_f1": float(sequence_metrics["fall_f1"]),
        "sequence_not_fall_f1": float(sequence_metrics["not_fall_f1"]),
        "sequence_tn": int(sequence_metrics["tn"]),
        "sequence_fp": int(sequence_metrics["fp"]),
        "sequence_fn": int(sequence_metrics["fn"]),
        "sequence_tp": int(sequence_metrics["tp"]),

        "video_majority_accuracy": float(video_majority_metrics["accuracy"]),
        "video_majority_macro_f1": float(video_majority_metrics["macro_f1"]),
        "video_majority_fall_recall": float(video_majority_metrics["fall_recall"]),
        "video_majority_fall_f1": float(video_majority_metrics["fall_f1"]),
        "video_majority_not_fall_f1": float(video_majority_metrics["not_fall_f1"]),
        "video_majority_tn": int(video_majority_metrics["tn"]),
        "video_majority_fp": int(video_majority_metrics["fp"]),
        "video_majority_fn": int(video_majority_metrics["fn"]),
        "video_majority_tp": int(video_majority_metrics["tp"]),

        "video_mean_prob_accuracy": float(video_mean_prob_metrics["accuracy"]),
        "video_mean_prob_macro_f1": float(video_mean_prob_metrics["macro_f1"]),
        "video_mean_prob_fall_recall": float(video_mean_prob_metrics["fall_recall"]),
        "video_mean_prob_fall_f1": float(video_mean_prob_metrics["fall_f1"]),
        "video_mean_prob_not_fall_f1": float(video_mean_prob_metrics["not_fall_f1"]),
        "video_mean_prob_tn": int(video_mean_prob_metrics["tn"]),
        "video_mean_prob_fp": int(video_mean_prob_metrics["fp"]),
        "video_mean_prob_fn": int(video_mean_prob_metrics["fn"]),
        "video_mean_prob_tp": int(video_mean_prob_metrics["tp"]),

        "num_test_sequences": int(len(seq_pred_df)),
        "num_test_videos": int(video_pred_df["video_id"].nunique()),

        "sequence_predictions_csv": str(seq_pred_csv),
        "video_predictions_csv": str(video_pred_csv),
        "full_metrics_json": str(full_metrics_json),
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RNN, LSTM, and GRU checkpoints on the test set."
    )

    parser.add_argument(
        "--test-npz",
        type=str,
        default=str(BASELINE_ROOT / "data" / "splits" / "lin2021_yolov8_test_sequences.npz"),
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "checkpoints"),
    )

    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "metrics"),
    )

    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "predictions"),
    )

    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(BASELINE_ROOT / "outputs" / "reports"),
    )

    parser.add_argument(
        "--models",
        type=str,
        default="rnn,lstm,gru",
        help="Models to evaluate: rnn,lstm,gru or all.",
    )

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    device = get_device(args.device)

    test_npz = Path(args.test_npz)
    checkpoint_dir = Path(args.checkpoint_dir)
    metrics_dir = Path(args.metrics_dir)
    predictions_dir = Path(args.predictions_dir)
    reports_dir = Path(args.reports_dir)

    ensure_dir(metrics_dir)
    ensure_dir(predictions_dir)
    ensure_dir(reports_dir)

    test_data = load_npz(test_npz)
    models_to_eval = parse_models(args.models)

    print("=" * 100)
    print("FILE 07 - Evaluate RNN / LSTM / GRU on test set")
    print("=" * 100)
    print(f"Test NPZ:        {test_npz}")
    print(f"Test X shape:    {test_data['X'].shape}")
    print(f"Device:          {device}")
    print(f"Models:          {models_to_eval}")
    print(f"Checkpoint dir:  {checkpoint_dir}")
    print("=" * 100)

    results = []
    failed = []

    for model_type in models_to_eval:
        checkpoint_path = checkpoint_dir / f"best_{model_type}.pt"

        print(f"Evaluating {model_type.upper()} from {checkpoint_path}")

        try:
            result = evaluate_one_model(
                model_type=model_type,
                checkpoint_path=checkpoint_path,
                test_data=test_data,
                args=args,
                device=device,
            )

            results.append(result)

            print(
                f"  OK -> "
                f"seq_macro_f1={result['sequence_macro_f1']:.4f}, "
                f"seq_fall_recall={result['sequence_fall_recall']:.4f}, "
                f"video_mean_macro_f1={result['video_mean_prob_macro_f1']:.4f}, "
                f"video_mean_fall_recall={result['video_mean_prob_fall_recall']:.4f}"
            )

        except Exception as e:
            failed.append({
                "model_type": model_type,
                "checkpoint": str(checkpoint_path),
                "error": repr(e),
            })

            print(f"  FAILED: {repr(e)}")

    if len(results) == 0:
        raise RuntimeError(f"No model was evaluated successfully. Failed: {failed}")

    summary_df = pd.DataFrame(results)

    summary_csv = metrics_dir / "lin2021_yolov8_test_evaluation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    best_seq = summary_df.sort_values(
        ["sequence_macro_f1", "sequence_fall_recall"],
        ascending=False,
    ).iloc[0].to_dict()

    best_video = summary_df.sort_values(
        ["video_mean_prob_macro_f1", "video_mean_prob_fall_recall"],
        ascending=False,
    ).iloc[0].to_dict()

    report = {
        "status": "completed",
        "pipeline_note": "This file evaluates the held-out test split. Sequence-level metrics are primary; video-level metrics are reported by aggregating sequence predictions per video.",
        "test_npz": str(test_npz),
        "test_shape": list(test_data["X"].shape),
        "device": str(device),
        "models_evaluated": models_to_eval,
        "num_success": len(results),
        "num_failed": len(failed),
        "summary_csv": str(summary_csv),
        "results": results,
        "failed": failed,
        "best_model_by_sequence_macro_f1": {
            "model_type": str(best_seq["model_type"]),
            "sequence_macro_f1": float(best_seq["sequence_macro_f1"]),
            "sequence_accuracy": float(best_seq["sequence_accuracy"]),
            "sequence_fall_recall": float(best_seq["sequence_fall_recall"]),
            "sequence_fall_f1": float(best_seq["sequence_fall_f1"]),
            "checkpoint": str(best_seq["checkpoint"]),
        },
        "best_model_by_video_mean_prob_macro_f1": {
            "model_type": str(best_video["model_type"]),
            "video_mean_prob_macro_f1": float(best_video["video_mean_prob_macro_f1"]),
            "video_mean_prob_accuracy": float(best_video["video_mean_prob_accuracy"]),
            "video_mean_prob_fall_recall": float(best_video["video_mean_prob_fall_recall"]),
            "video_mean_prob_fall_f1": float(best_video["video_mean_prob_fall_f1"]),
            "checkpoint": str(best_video["checkpoint"]),
        },
    }

    report_path = reports_dir / "07_evaluate_rnn_lstm_gru_report.json"
    save_json(report, report_path)

    print("=" * 100)
    print("FILE 07 completed.")
    print("=" * 100)
    print("Test evaluation summary:")
    print(summary_df.to_string(index=False))
    print("-" * 100)
    print("Best by sequence Macro F1:")
    print(report["best_model_by_sequence_macro_f1"])
    print("-" * 100)
    print("Best by video mean-probability Macro F1:")
    print(report["best_model_by_video_mean_prob_macro_f1"])
    print("-" * 100)
    print(f"Summary CSV: {summary_csv}")
    print(f"Report:      {report_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
