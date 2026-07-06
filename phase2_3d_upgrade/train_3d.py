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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from model_3d import FallCNNLSTM3D


# =========================
# CONFIG
# =========================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "4_normalized_3d")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "phase2_3d_upgrade", "checkpoints")
RESULT_DIR = os.path.join(PROJECT_ROOT, "phase2_3d_upgrade", "outputs", "training_3d")

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
# DATA UTILS
# =========================

def get_3d_columns():
    cols = []

    for i in range(17):
        cols.extend([f"x{i}", f"y{i}", f"z{i}"])

    return cols


def find_csv_files():
    files = []

    for file in os.listdir(DATA_DIR):
        if file.lower().endswith(".csv"):
            files.append(os.path.join(DATA_DIR, file))

    return sorted(files)


def read_label_from_csv(csv_path, task):
    df = pd.read_csv(csv_path, nrows=1)

    if task == "binary":
        # label: 0 = Not_Fall, 1 = Fall
        return int(df["label"].iloc[0])

    if task == "action":
        # 3D dataset currently uses:
        # action_label: 0 = Fall, 1 = Sitting, 2 = Sleeping, 3 = Standing, 4 = Walking
        # For fair comparison with Phase 1 action model, remove Fall and remap:
        # Sitting=0, Sleeping=1, Standing=2, Walking=3
        action_label = int(df["action_label"].iloc[0])

        if action_label == 0:
            return None

        return action_label - 1

    raise ValueError(f"Unknown task: {task}")


def split_files_by_video(csv_files, task):
    usable_files = []
    labels = []

    for path in csv_files:
        label = read_label_from_csv(path, task)

        if label is None:
            continue

        usable_files.append(path)
        labels.append(label)

    train_files, test_files, train_labels, test_labels = train_test_split(
        usable_files,
        labels,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    return train_files, test_files, train_labels, test_labels


# =========================
# FEATURE ENGINEERING
# =========================

def add_pose_features_3d(video_df):
    """
    Create 3D pose features in the same spirit as Phase 1 2D features.

    Phase 1 2D:
        normalized xy coordinates + handcrafted body features

    Phase 2 3D:
        normalized xyz coordinates + handcrafted 3D body features

    Input normalized 3D CSV already has:
        pelvis-centered and body-scale-normalized 3D pose.
    """

    keypoint_cols = get_3d_columns()

    coords = video_df[keypoint_cols].values.astype(np.float32)
    pose = coords.reshape(len(video_df), 17, 3)

    xs = pose[:, :, 0]
    ys = pose[:, :, 1]
    zs = pose[:, :, 2]

    min_x = xs.min(axis=1, keepdims=True)
    max_x = xs.max(axis=1, keepdims=True)

    min_y = ys.min(axis=1, keepdims=True)
    max_y = ys.max(axis=1, keepdims=True)

    min_z = zs.min(axis=1, keepdims=True)
    max_z = zs.max(axis=1, keepdims=True)

    width_x = max_x - min_x
    depth_y = max_y - min_y
    height_z = max_z - min_z

    eps = 1e-6

    # Important 3D pose descriptors
    height_width_ratio = height_z / (width_x + eps)
    depth_width_ratio = depth_y / (width_x + eps)

    # Head relative height to pelvis.
    # Joint 10 = head, joint 0 = pelvis.
    head_height = zs[:, 10:11] - zs[:, 0:1]

    # Torso tilt:
    # Joint 0 = pelvis, joint 8 = thorax.
    torso_vector = pose[:, 8, :] - pose[:, 0, :]
    torso_horizontal = np.linalg.norm(torso_vector[:, 0:2], axis=1, keepdims=True)
    torso_vertical = np.abs(torso_vector[:, 2:3]) + eps
    torso_tilt = torso_horizontal / torso_vertical

    # Skeleton centroid velocity.
    center = pose.mean(axis=1)
    velocity = np.zeros((len(center), 1), dtype=np.float32)

    if len(center) > 1:
        diff = center[1:] - center[:-1]
        velocity[1:, 0] = np.linalg.norm(diff, axis=1)

    handcrafted = np.concatenate(
        [
            width_x,
            depth_y,
            height_z,
            height_width_ratio,
            depth_width_ratio,
            head_height,
            torso_tilt,
            velocity
        ],
        axis=1
    ).astype(np.float32)

    features = np.concatenate([coords, handcrafted], axis=1).astype(np.float32)

    return features


def create_sequences_from_video(video_df, label, sequence_length=60, stride=15):
    video_df = video_df.sort_values("frame").reset_index(drop=True)

    features = add_pose_features_3d(video_df)

    sequences = []
    labels = []

    n_frames = len(features)

    if n_frames == 0:
        return sequences, labels

    if n_frames < sequence_length:
        pad_len = sequence_length - n_frames
        pad = np.repeat(features[-1:], pad_len, axis=0)
        padded_features = np.concatenate([features, pad], axis=0)

        sequences.append(padded_features)
        labels.append(label)

    else:
        for start in range(0, n_frames - sequence_length + 1, stride):
            end = start + sequence_length
            sequences.append(features[start:end])
            labels.append(label)

    return sequences, labels


def load_sequences_from_files(csv_files, task):
    all_sequences = []
    all_labels = []

    failed_files = 0
    skipped_files = 0

    for csv_path in csv_files:
        try:
            label = read_label_from_csv(csv_path, task)

            if label is None:
                skipped_files += 1
                continue

            df = pd.read_csv(csv_path)

            seqs, labs = create_sequences_from_video(
                df,
                label=label,
                sequence_length=SEQUENCE_LENGTH,
                stride=STRIDE
            )

            all_sequences.extend(seqs)
            all_labels.extend(labs)

        except Exception as e:
            failed_files += 1
            print("Failed file:", csv_path)
            print("Reason:", e)

    return all_sequences, all_labels, failed_files, skipped_files


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

    avg_loss = total_loss / len(loader.dataset)
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

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, f1, all_labels, all_preds


def save_results_json(output_path, data):
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
        num_classes = 2
        class_names = ["Not_Fall", "Fall"]
        best_model_path = os.path.join(CHECKPOINT_DIR, "best_model_3d_binary.pt")
        result_path = os.path.join(RESULT_DIR, "results_3d_binary_cnn_lstm.json")

    else:
        # Fair comparison with Phase 1 action:
        # only Not_Fall actions: Sitting, Sleeping, Standing, Walking
        num_classes = 4
        class_names = ["Sitting", "Sleeping", "Standing", "Walking"]
        best_model_path = os.path.join(CHECKPOINT_DIR, "best_model_3d_action.pt")
        result_path = os.path.join(RESULT_DIR, "results_3d_action_cnn_lstm.json")

    print("=" * 60)
    print(f"TRAINING 3D TASK: {task.upper()}")
    print("MODEL: CNN-LSTM same architecture style as Phase 1 2D")
    print("=" * 60)

    print("Input folder:", DATA_DIR)

    csv_files = find_csv_files()

    if len(csv_files) == 0:
        raise RuntimeError(f"No CSV files found in {DATA_DIR}")

    print("CSV files found:", len(csv_files))

    train_files, test_files, train_file_labels, test_file_labels = split_files_by_video(
        csv_files,
        task
    )

    print("\nVideo-level split:")
    print("Train videos:", len(train_files))
    print("Test videos:", len(test_files))

    print("\nTrain video label distribution:")
    print(pd.Series(train_file_labels).value_counts().sort_index())
    print(pd.Series(train_file_labels).value_counts(normalize=True).sort_index())

    print("\nTest video label distribution:")
    print(pd.Series(test_file_labels).value_counts().sort_index())
    print(pd.Series(test_file_labels).value_counts(normalize=True).sort_index())

    print("\nCreating train sequences...")
    train_sequences, train_labels, train_failed, train_skipped = load_sequences_from_files(
        train_files,
        task
    )

    print("Creating test sequences...")
    test_sequences, test_labels, test_failed, test_skipped = load_sequences_from_files(
        test_files,
        task
    )

    if len(train_sequences) == 0 or len(test_sequences) == 0:
        raise RuntimeError("No sequences created. Check normalized 3D CSV files.")

    input_dim = train_sequences[0].shape[1]

    print("\nSequence-level data:")
    print("Train sequences:", len(train_sequences))
    print("Test sequences:", len(test_sequences))
    print("Input dim:", input_dim)
    print("Train failed files:", train_failed, "| Train skipped files:", train_skipped)
    print("Test failed files:", test_failed, "| Test skipped files:", test_skipped)

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

    model = FallCNNLSTM3D(
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

    best_f1 = 0.0
    best_epoch = 0

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

        print(
            f"Epoch [{epoch:02d}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} || "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}"
        )

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "sequence_length": SEQUENCE_LENGTH,
                    "stride": STRIDE,
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "task": task,
                    "best_f1": best_f1,
                    "best_epoch": best_epoch,
                    "model_type": "CNN_LSTM_3D_same_as_2D_phase1"
                },
                best_model_path
            )

            print(f"Saved best model to {best_model_path}")

    print("\nTraining finished.")
    print("Best Test F1:", best_f1)
    print("Best Epoch:", best_epoch)

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)

    model = FallCNNLSTM3D(
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

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    print("\nFinal Best Model Evaluation")
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)
    print("Test Macro F1:", test_f1)

    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    print(cm)

    result_data = {
        "task": task,
        "model_type": "CNN_LSTM_3D_same_as_2D_phase1",
        "input_dir": DATA_DIR,
        "num_csv_files": len(csv_files),
        "train_videos": len(train_files),
        "test_videos": len(test_files),
        "train_sequences": len(train_sequences),
        "test_sequences": len(test_sequences),
        "input_dim": input_dim,
        "sequence_length": SEQUENCE_LENGTH,
        "stride": STRIDE,
        "batch_size": BATCH_SIZE,
        "epochs": epochs,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "final_test_loss": test_loss,
        "final_test_accuracy": test_acc,
        "final_test_macro_f1": test_f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "checkpoint_path": best_model_path
    }

    save_results_json(result_path, result_data)

    print("\nSaved result json to:", result_path)
    print("Saved checkpoint to:", best_model_path)


if __name__ == "__main__":
    main()