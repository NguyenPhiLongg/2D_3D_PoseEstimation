import os
import argparse
import random
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from model_2d import FallCNNLSTM


# =========================
# CONFIG
# =========================

DATA_PATH = "data/master_dataset.csv"
CHECKPOINT_DIR = "phase1_2d_baseline/checkpoints"
RESULT_DIR = "phase1_2d_baseline/outputs/training_2d"

SEQUENCE_LENGTH = 60
STRIDE = 15
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# =========================
# SEED
# =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


# =========================
# FEATURE ENGINEERING
# =========================

def get_2d_keypoint_columns():
    keypoint_cols = []

    for i in range(17):
        keypoint_cols.extend([f"x{i}", f"y{i}"])

    return keypoint_cols


def add_pose_features(video_df):
    """
    Create 2D pose features.

    New normalization:
        For each frame, normalize 2D pose by mean/std separately for x and y.

        pose = (pose - pose.mean(axis=joint_dim)) / pose.std(axis=joint_dim)

    Input:
        17 COCO keypoints: x0, y0, ..., x16, y16

    Output feature dim:
        34 normalized keypoint features
        + 6 handcrafted features
        = 40 features
    """

    keypoint_cols = get_2d_keypoint_columns()

    coords = video_df[keypoint_cols].values.astype(np.float32)

    # Shape: (num_frames, 17, 2)
    pose_2d = coords.reshape(len(video_df), 17, 2)

    xs = pose_2d[:, :, 0]
    ys = pose_2d[:, :, 1]

    min_x = xs.min(axis=1, keepdims=True)
    max_x = xs.max(axis=1, keepdims=True)
    min_y = ys.min(axis=1, keepdims=True)
    max_y = ys.max(axis=1, keepdims=True)

    width = max_x - min_x
    height = max_y - min_y

    eps = 1e-6

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    # =========================
    # NEW NORMALIZATION
    # =========================
    # Mean/std over 17 joints, separately for x and y
    mean_2d = pose_2d.mean(axis=1, keepdims=True)   # shape: (T, 1, 2)
    std_2d = pose_2d.std(axis=1, keepdims=True)     # shape: (T, 1, 2)

    std_2d = np.where(std_2d < eps, 1.0, std_2d)

    normalized_pose_2d = (pose_2d - mean_2d) / std_2d
    normalized_coords = normalized_pose_2d.reshape(len(video_df), -1).astype(np.float32)

    # =========================
    # HANDCRAFTED FEATURES
    # =========================

    aspect_ratio = height / (width + eps)

    scale = np.maximum(width, height) + eps
    norm_width = width / (scale + eps)
    norm_height = height / (scale + eps)

    # Keep center information roughly normalized by common video size
    center_x_norm = center_x / 1920.0
    center_y_norm = center_y / 1080.0

    # Body center velocity
    center = np.concatenate([center_x, center_y], axis=1)
    velocity = np.zeros((len(center), 1), dtype=np.float32)

    if len(center) > 1:
        diff = center[1:] - center[:-1]
        velocity[1:, 0] = np.linalg.norm(diff, axis=1)

    handcrafted = np.concatenate(
        [
            aspect_ratio,
            norm_width,
            norm_height,
            center_x_norm,
            center_y_norm,
            velocity
        ],
        axis=1
    ).astype(np.float32)

    features = np.concatenate([normalized_coords, handcrafted], axis=1).astype(np.float32)

    return features


def create_sequences_from_video(video_df, label_column, sequence_length=60, stride=15):
    video_df = video_df.sort_values("frame").reset_index(drop=True)

    label = int(video_df[label_column].iloc[0])
    features = add_pose_features(video_df)

    sequences = []
    labels = []

    n_frames = len(features)

    if n_frames == 0:
        return sequences, labels

    if n_frames < sequence_length:
        pad_len = sequence_length - n_frames
        pad = np.repeat(features[-1:], pad_len, axis=0)
        padded_features = np.concatenate([features, pad], axis=0)

        sequences.append(padded_features.astype(np.float32))
        labels.append(label)

    else:
        for start in range(0, n_frames - sequence_length + 1, stride):
            end = start + sequence_length
            sequences.append(features[start:end].astype(np.float32))
            labels.append(label)

    return sequences, labels


# =========================
# DATASET
# =========================

class FallSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(np.array(sequences), dtype=torch.float32)
        self.labels = torch.tensor(np.array(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# =========================
# TRAIN / EVAL
# =========================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, f1


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, f1, all_labels, all_preds


def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        default="binary",
        choices=["binary", "action"],
        help="binary = Fall vs Not_Fall, action = Sitting/Sleeping/Standing/Walking"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS
    )

    args = parser.parse_args()

    task = args.task
    epochs = args.epochs

    if task == "binary":
        label_column = "binary_label"
        num_classes = 2
        class_names = ["Not_Fall", "Fall"]
        best_model_path = os.path.join(CHECKPOINT_DIR, "best_model_2d_binary.pt")
        result_json_path = os.path.join(RESULT_DIR, "results_2d_binary.json")

    else:
        label_column = "action_label"
        num_classes = 4
        class_names = ["Sitting", "Sleeping", "Standing", "Walking"]
        best_model_path = os.path.join(CHECKPOINT_DIR, "best_model_2d_action.pt")
        result_json_path = os.path.join(RESULT_DIR, "results_2d_action.json")

    print("=" * 60)
    print(f"TRAINING 2D TASK: {task.upper()}")
    print("NORMALIZATION: per-frame mean/std over x,y")
    print("=" * 60)

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)

    if task == "action":
        # Only use Not_Fall videos for action classification
        df = df[df["action_label"] != -1].reset_index(drop=True)
        print("Filtered action dataset shape:", df.shape)

    print("\nFrame-level label distribution:")
    print(df[label_column].value_counts().sort_index())
    print(df[label_column].value_counts(normalize=True).sort_index())

    # Split by video, not by frame
    video_label_df = df[["source_file", label_column]].drop_duplicates().reset_index(drop=True)

    train_files, test_files = train_test_split(
        video_label_df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=video_label_df[label_column]
    )

    train_file_set = set(train_files["source_file"])
    test_file_set = set(test_files["source_file"])

    train_df = df[df["source_file"].isin(train_file_set)].reset_index(drop=True)
    test_df = df[df["source_file"].isin(test_file_set)].reset_index(drop=True)

    print("\nVideo-level split:")
    print("Train videos:", len(train_file_set))
    print("Test videos:", len(test_file_set))

    print("\nTrain video label distribution:")
    print(train_files[label_column].value_counts().sort_index())
    print(train_files[label_column].value_counts(normalize=True).sort_index())

    print("\nTest video label distribution:")
    print(test_files[label_column].value_counts().sort_index())
    print(test_files[label_column].value_counts(normalize=True).sort_index())

    # Create sequences
    print("\nCreating train sequences...")
    train_sequences = []
    train_labels = []

    for source_file, video_df in train_df.groupby("source_file"):
        seqs, labs = create_sequences_from_video(
            video_df,
            label_column=label_column,
            sequence_length=SEQUENCE_LENGTH,
            stride=STRIDE
        )

        train_sequences.extend(seqs)
        train_labels.extend(labs)

    print("Creating test sequences...")
    test_sequences = []
    test_labels = []

    for source_file, video_df in test_df.groupby("source_file"):
        seqs, labs = create_sequences_from_video(
            video_df,
            label_column=label_column,
            sequence_length=SEQUENCE_LENGTH,
            stride=STRIDE
        )

        test_sequences.extend(seqs)
        test_labels.extend(labs)

    print("\nSequence-level data:")
    print("Train sequences:", len(train_sequences))
    print("Test sequences:", len(test_sequences))

    if len(train_sequences) == 0 or len(test_sequences) == 0:
        raise RuntimeError("No sequences were created. Check dataset and labels.")

    input_dim = train_sequences[0].shape[1]
    print("Input dim:", input_dim)

    train_dataset = FallSequenceDataset(train_sequences, train_labels)
    test_dataset = FallSequenceDataset(test_sequences, test_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device)

    model = FallCNNLSTM(
        input_dim=input_dim,
        num_classes=num_classes,
        cnn_channels=128,
        lstm_hidden=128,
        lstm_layers=1,
        dropout=0.3
    ).to(device)

    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = class_counts.sum() / (num_classes * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("\nClass counts:", class_counts)
    print("Class weights:", class_weights.detach().cpu().numpy())

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    best_f1 = -1.0
    best_acc = -1.0
    best_epoch = 0
    best_report = None
    best_cm = None
    history = []

    print("\nStart training...")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        test_loss, test_acc, test_f1, y_true, y_pred = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        scheduler.step(test_f1)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "train_macro_f1": float(train_f1),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "test_macro_f1": float(test_f1)
        }

        history.append(row)

        print(
            f"Epoch [{epoch:02d}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} || "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}"
        )

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_acc = test_acc
            best_epoch = epoch

            best_report = classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                zero_division=0,
                output_dict=True
            )

            best_report_text = classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                zero_division=0
            )

            best_cm = confusion_matrix(y_true, y_pred)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "sequence_length": SEQUENCE_LENGTH,
                    "stride": STRIDE,
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "task": task,
                    "best_f1": float(best_f1),
                    "best_accuracy": float(best_acc),
                    "best_epoch": int(best_epoch),
                    "normalization": "per_frame_xy_mean_std",
                    "classification_report": best_report,
                    "classification_report_text": best_report_text,
                    "confusion_matrix": best_cm.tolist()
                },
                best_model_path
            )

            print(f"Saved best model to {best_model_path}")

    print("\nTraining finished.")
    print("Best Test Accuracy:", best_acc)
    print("Best Test Macro F1:", best_f1)
    print("Best Epoch:", best_epoch)

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)

    model = FallCNNLSTM(
        input_dim=checkpoint["input_dim"],
        num_classes=checkpoint["num_classes"],
        cnn_channels=128,
        lstm_hidden=128,
        lstm_layers=1,
        dropout=0.3
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )

    cm = confusion_matrix(y_true, y_pred)

    print("\nFinal Best Model Evaluation")
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)
    print("Test Macro F1:", test_f1)

    print("\nClassification Report:")
    print(report_text)

    print("\nConfusion Matrix:")
    print(cm)

    result_data = {
        "task": task,
        "model_type": "CNN1D_BiLSTM_2D",
        "normalization": "per_frame_xy_mean_std",
        "data_path": DATA_PATH,
        "train_videos": len(train_file_set),
        "test_videos": len(test_file_set),
        "train_sequences": len(train_sequences),
        "test_sequences": len(test_sequences),
        "input_dim": int(input_dim),
        "sequence_length": SEQUENCE_LENGTH,
        "stride": STRIDE,
        "batch_size": BATCH_SIZE,
        "epochs": epochs,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "best_epoch": int(best_epoch),
        "best_accuracy": float(best_acc),
        "best_f1": float(best_f1),
        "final_test_loss": float(test_loss),
        "final_test_accuracy": float(test_acc),
        "final_test_macro_f1": float(test_f1),
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "checkpoint_path": best_model_path,
        "history": history
    }

    save_json(result_data, result_json_path)

    print("\nSaved result json to:", result_json_path)
    print("Saved checkpoint to:", best_model_path)


if __name__ == "__main__":
    main()