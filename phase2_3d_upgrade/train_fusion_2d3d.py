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

DATA_2D_DIR = os.path.join(PROJECT_ROOT, "data", "2_extracted_2d")
DATA_3D_DIR = os.path.join(PROJECT_ROOT, "data", "4_normalized_3d")

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "phase2_3d_upgrade", "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "phase2_3d_upgrade", "outputs", "training_fusion_2d3d")

SEQUENCE_LENGTH = 60
STRIDE = 15
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
# COLUMNS
# =========================

def get_2d_columns():
    cols = []

    for i in range(17):
        cols.extend([f"x{i}", f"y{i}"])

    return cols


def get_3d_columns():
    cols = []

    for i in range(17):
        cols.extend([f"x{i}", f"y{i}", f"z{i}"])

    return cols


# =========================
# FILE UTILS
# =========================

def find_3d_csv_files():
    files = []

    for file in os.listdir(DATA_3D_DIR):
        if file.lower().endswith(".csv"):
            files.append(os.path.join(DATA_3D_DIR, file))

    return sorted(files)


def get_2d_path_from_3d_path(path_3d):
    file_name = os.path.basename(path_3d)
    return os.path.join(DATA_2D_DIR, file_name)


def read_label_from_3d_csv(path_3d, task):
    df = pd.read_csv(path_3d, nrows=1)

    if task == "binary":
        return int(df["label"].iloc[0])

    if task == "action":
        action_label = int(df["action_label"].iloc[0])

        # 3D extracted data:
        # 0 = Fall
        # 1 = Sitting
        # 2 = Sleeping
        # 3 = Standing
        # 4 = Walking
        #
        # For fair comparison with Phase 1 action:
        # remove Fall and remap:
        # Sitting  -> 0
        # Sleeping -> 1
        # Standing -> 2
        # Walking  -> 3
        if action_label == 0:
            return None

        return action_label - 1

    raise ValueError(f"Unknown task: {task}")


# =========================
# 2D FEATURE ENGINEERING
# =========================

def add_pose_features_2d(df):
    """
    Create 2D pose features for fusion.

    New normalization:
        Per-frame mean/std normalization over 17 joints,
        separately for x and y.

    Input:
        x0, y0, ..., x16, y16

    Output:
        34 normalized 2D keypoint features
        + 6 handcrafted 2D features
        = 40 features
    """

    keypoint_cols = get_2d_columns()

    coords = df[keypoint_cols].values.astype(np.float32)

    # Shape: (num_frames, 17, 2)
    pose_2d = coords.reshape(len(df), 17, 2)

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
    # MEAN/STD NORMALIZATION
    # =========================
    # Mean/std over 17 joints, separately for x and y
    mean_2d = pose_2d.mean(axis=1, keepdims=True)   # shape: (T, 1, 2)
    std_2d = pose_2d.std(axis=1, keepdims=True)     # shape: (T, 1, 2)

    std_2d = np.where(std_2d < eps, 1.0, std_2d)

    normalized_pose_2d = (pose_2d - mean_2d) / std_2d
    normalized_coords = normalized_pose_2d.reshape(len(df), -1).astype(np.float32)

    # =========================
    # HANDCRAFTED 2D FEATURES
    # =========================

    aspect_ratio = height / (width + eps)

    scale = np.maximum(width, height) + eps
    norm_width = width / (scale + eps)
    norm_height = height / (scale + eps)

    # Keep rough body center information
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


# =========================
# 3D FEATURE ENGINEERING
# =========================

def add_pose_features_3d(df):
    """
    Create 3D pose features for fusion.

    Important:
        The 3D CSV in data/4_normalized_3d should already be normalized
        by normalize_3d_dataset.py.

    Input:
        x0, y0, z0, ..., x16, y16, z16

    Output:
        51 normalized 3D keypoint features
        + 8 handcrafted 3D features
        = 59 features
    """

    keypoint_cols = get_3d_columns()

    coords = df[keypoint_cols].values.astype(np.float32)
    pose = coords.reshape(len(df), 17, 3)

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

    height_width_ratio = height_z / (width_x + eps)
    depth_width_ratio = depth_y / (width_x + eps)

    # Joint 10 = head, joint 0 = pelvis
    head_height = zs[:, 10:11] - zs[:, 0:1]

    # Joint 8 = thorax, joint 0 = pelvis
    torso_vector = pose[:, 8, :] - pose[:, 0, :]
    torso_horizontal = np.linalg.norm(torso_vector[:, 0:2], axis=1, keepdims=True)
    torso_vertical = np.abs(torso_vector[:, 2:3]) + eps
    torso_tilt = torso_horizontal / torso_vertical

    # Skeleton centroid velocity
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


# =========================
# FUSION FEATURE LOADING
# =========================

def load_fusion_features(path_3d):
    """
    Load matching 2D and 3D CSV files, align by frame if possible,
    otherwise align by row order.

    Fusion feature:
        2D features = 40
        3D features = 59
        total = 99
    """

    path_2d = get_2d_path_from_3d_path(path_3d)

    if not os.path.exists(path_2d):
        raise FileNotFoundError(f"Missing matching 2D CSV: {path_2d}")

    df2 = pd.read_csv(path_2d)
    df3 = pd.read_csv(path_3d)

    cols_2d = get_2d_columns()
    cols_3d = get_3d_columns()

    required_2d = ["frame"] + cols_2d
    required_3d = ["frame"] + cols_3d

    for col in required_2d:
        if col not in df2.columns:
            raise ValueError(f"Missing 2D column {col} in {path_2d}")

    for col in required_3d:
        if col not in df3.columns:
            raise ValueError(f"Missing 3D column {col} in {path_3d}")

    df2 = df2.copy()
    df3 = df3.copy()

    df2 = df2.sort_values("frame").reset_index(drop=True)
    df3 = df3.sort_values("frame").reset_index(drop=True)

    rename_2d = {col: f"2d_{col}" for col in cols_2d}
    rename_3d = {col: f"3d_{col}" for col in cols_3d}

    df2 = df2.rename(columns=rename_2d)
    df3 = df3.rename(columns=rename_3d)

    cols_2d_prefixed = [f"2d_{col}" for col in cols_2d]
    cols_3d_prefixed = [f"3d_{col}" for col in cols_3d]

    # =========================================================
    # Try 1: 2D frame + 1 match 3D frame
    # =========================================================
    df2["frame_key"] = df2["frame"].astype(int) + 1
    df3["frame_key"] = df3["frame"].astype(int)

    merged = pd.merge(
        df3[["frame_key"] + cols_3d_prefixed],
        df2[["frame_key"] + cols_2d_prefixed],
        on="frame_key",
        how="inner"
    )

    align_method = "frame_plus_1"

    # =========================================================
    # Try 2: exact frame match
    # =========================================================
    if len(merged) < max(5, int(len(df3) * 0.3)):
        df2["frame_key"] = df2["frame"].astype(int)
        df3["frame_key"] = df3["frame"].astype(int)

        merged = pd.merge(
            df3[["frame_key"] + cols_3d_prefixed],
            df2[["frame_key"] + cols_2d_prefixed],
            on="frame_key",
            how="inner"
        )

        align_method = "exact_frame"

    # =========================================================
    # Try 3: fallback by row order
    # Use this when frame IDs do not overlap.
    # =========================================================
    if len(merged) == 0:
        min_len = min(len(df2), len(df3))

        if min_len == 0:
            raise RuntimeError(f"No usable rows between 2D and 3D for {path_3d}")

        df2_cut = df2.iloc[:min_len].reset_index(drop=True)
        df3_cut = df3.iloc[:min_len].reset_index(drop=True)

        merged = pd.DataFrame()

        merged["frame_key"] = np.arange(min_len)

        for col in cols_3d_prefixed:
            merged[col] = df3_cut[col].values

        for col in cols_2d_prefixed:
            merged[col] = df2_cut[col].values

        align_method = "row_order_fallback"

    df3_aligned = merged[["frame_key"] + cols_3d_prefixed].copy()
    df2_aligned = merged[["frame_key"] + cols_2d_prefixed].copy()

    df3_aligned = df3_aligned.rename(
        columns={"frame_key": "frame", **{f"3d_{col}": col for col in cols_3d}}
    )

    df2_aligned = df2_aligned.rename(
        columns={"frame_key": "frame", **{f"2d_{col}": col for col in cols_2d}}
    )

    features_2d = add_pose_features_2d(df2_aligned)
    features_3d = add_pose_features_3d(df3_aligned)

    fusion_features = np.concatenate([features_2d, features_3d], axis=1).astype(np.float32)

    return fusion_features


# =========================
# SEQUENCE CREATION
# =========================

def create_sequences(features, label):
    sequences = []
    labels = []

    n_frames = len(features)

    if n_frames == 0:
        return sequences, labels

    if n_frames < SEQUENCE_LENGTH:
        pad_len = SEQUENCE_LENGTH - n_frames
        pad = np.repeat(features[-1:], pad_len, axis=0)
        padded = np.concatenate([features, pad], axis=0)

        sequences.append(padded.astype(np.float32))
        labels.append(label)

        return sequences, labels

    for start in range(0, n_frames - SEQUENCE_LENGTH + 1, STRIDE):
        end = start + SEQUENCE_LENGTH
        sequences.append(features[start:end].astype(np.float32))
        labels.append(label)

    return sequences, labels


def split_files_by_video(files, task):
    usable_files = []
    labels = []

    for path in files:
        label = read_label_from_3d_csv(path, task)

        if label is None:
            continue

        path_2d = get_2d_path_from_3d_path(path)

        if not os.path.exists(path_2d):
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


def load_sequences_from_files(files, task):
    all_sequences = []
    all_labels = []

    failed = 0

    for path in files:
        try:
            label = read_label_from_3d_csv(path, task)

            if label is None:
                continue

            features = load_fusion_features(path)
            seqs, labs = create_sequences(features, label)

            all_sequences.extend(seqs)
            all_labels.extend(labs)

        except Exception as e:
            failed += 1
            print("Failed:", path)
            print("Reason:", e)

    return all_sequences, all_labels, failed


# =========================
# DATASET
# =========================

class FusionSequenceDataset(Dataset):
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


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
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
        help="binary = Fall/Not_Fall, action = Sitting/Sleeping/Standing/Walking"
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
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model_fusion_2d3d_binary.pt")
        result_path = os.path.join(OUTPUT_DIR, "results_fusion_2d3d_binary.json")
    else:
        num_classes = 4
        class_names = ["Sitting", "Sleeping", "Standing", "Walking"]
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model_fusion_2d3d_action.pt")
        result_path = os.path.join(OUTPUT_DIR, "results_fusion_2d3d_action.json")

    print("=" * 80)
    print(f"TRAINING FUSION 2D + 3D: {task.upper()}")
    print("MODEL: CNN1D + BiLSTM")
    print("2D normalization: per-frame x/y mean-std")
    print("3D normalization: precomputed normalized 3D CSV")
    print("=" * 80)

    print("2D dir:", DATA_2D_DIR)
    print("3D dir:", DATA_3D_DIR)

    all_files = find_3d_csv_files()

    if len(all_files) == 0:
        raise RuntimeError(f"No 3D CSV files found in {DATA_3D_DIR}")

    print("3D CSV files found:", len(all_files))

    train_files, test_files, train_file_labels, test_file_labels = split_files_by_video(
        all_files,
        task
    )

    print("\nVideo-level split:")
    print("Train videos:", len(train_files))
    print("Test videos:", len(test_files))

    print("\nTrain label distribution:")
    print(pd.Series(train_file_labels).value_counts().sort_index())

    print("\nTest label distribution:")
    print(pd.Series(test_file_labels).value_counts().sort_index())

    print("\nCreating train sequences...")
    train_sequences, train_labels, train_failed = load_sequences_from_files(train_files, task)

    print("Creating test sequences...")
    test_sequences, test_labels, test_failed = load_sequences_from_files(test_files, task)

    if len(train_sequences) == 0 or len(test_sequences) == 0:
        raise RuntimeError("No fusion sequences created.")

    input_dim = train_sequences[0].shape[1]

    print("\nSequence-level data:")
    print("Train sequences:", len(train_sequences))
    print("Test sequences:", len(test_sequences))
    print("Input dim:", input_dim)
    print("Expected input dim: 99")
    print("Train failed files:", train_failed)
    print("Test failed files:", test_failed)

    if input_dim != 99:
        print("WARNING: Input dim is not 99. Please check 2D/3D feature dimensions.")

    train_dataset = FusionSequenceDataset(train_sequences, train_labels)
    test_dataset = FusionSequenceDataset(test_sequences, test_labels)

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

    print("Class counts:", class_counts)
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
    best_report_text = ""
    best_report_dict = {}
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
            "epoch": int(epoch),
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

            best_report_text = classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                zero_division=0
            )

            best_report_dict = classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                zero_division=0,
                output_dict=True
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
                    "model_type": "CNN_LSTM_FUSION_2D_3D",
                    "normalization": "2d_per_frame_xy_mean_std__3d_precomputed_normalized",
                    "classification_report_text": best_report_text,
                    "classification_report": best_report_dict,
                    "confusion_matrix": best_cm.tolist()
                },
                checkpoint_path
            )

            print("Saved best model to:", checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

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
    print("Best epoch:", best_epoch)
    print("Best accuracy:", best_acc)
    print("Best macro F1:", best_f1)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)
    print("Test Macro F1:", test_f1)

    print("\nClassification Report:")
    print(report_text)

    print("\nConfusion Matrix:")
    print(cm)

    result_data = {
        "task": task,
        "model_type": "CNN_LSTM_FUSION_2D_3D",
        "normalization": "2d_per_frame_xy_mean_std__3d_precomputed_normalized",
        "input_dim": int(input_dim),
        "sequence_length": SEQUENCE_LENGTH,
        "stride": STRIDE,
        "batch_size": BATCH_SIZE,
        "epochs": epochs,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "best_epoch": int(best_epoch),
        "best_f1": float(best_f1),
        "best_accuracy": float(best_acc),
        "final_test_loss": float(test_loss),
        "final_test_accuracy": float(test_acc),
        "final_test_macro_f1": float(test_f1),
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "checkpoint_path": checkpoint_path,
        "train_sequences": len(train_sequences),
        "test_sequences": len(test_sequences),
        "train_failed_files": train_failed,
        "test_failed_files": test_failed,
        "history": history
    }

    save_json(result_path, result_data)

    print("\nSaved result json to:", result_path)
    print("Saved checkpoint to:", checkpoint_path)


if __name__ == "__main__":
    main()