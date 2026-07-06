import os
import argparse
import pandas as pd
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_3D_DIR = os.path.join(PROJECT_ROOT, "data", "3_extracted_3d")
OUTPUT_3D_DIR = os.path.join(PROJECT_ROOT, "data", "4_normalized_3d")


def get_3d_columns():
    cols = []

    for joint_idx in range(17):
        cols.extend([
            f"x{joint_idx}",
            f"y{joint_idx}",
            f"z{joint_idx}"
        ])

    return cols


def row_to_pose3d(row):
    pose = np.zeros((17, 3), dtype=np.float32)

    for joint_idx in range(17):
        pose[joint_idx, 0] = row[f"x{joint_idx}"]
        pose[joint_idx, 1] = row[f"y{joint_idx}"]
        pose[joint_idx, 2] = row[f"z{joint_idx}"]

    return pose


def pose3d_to_row_values(pose):
    values = {}

    for joint_idx in range(17):
        values[f"x{joint_idx}"] = float(pose[joint_idx, 0])
        values[f"y{joint_idx}"] = float(pose[joint_idx, 1])
        values[f"z{joint_idx}"] = float(pose[joint_idx, 2])

    return values


def compute_body_scale(pose):
    """
    Compute body scale for normalization.

    H36M joints:
    0: pelvis
    1, 4: hips
    2, 5: knees
    3, 6: ankles
    7: spine
    8: thorax
    9: neck
    10: head
    11, 14: shoulders
    12, 15: elbows
    13, 16: wrists

    We use several body distances and take the median.
    This is more stable than using only one bone.
    """

    pairs = [
        (0, 7),    # pelvis to spine
        (7, 8),    # spine to thorax
        (8, 9),    # thorax to neck
        (9, 10),   # neck to head
        (0, 1),    # pelvis to right hip
        (0, 4),    # pelvis to left hip
        (1, 2),    # right thigh
        (2, 3),    # right shin
        (4, 5),    # left thigh
        (5, 6),    # left shin
        (8, 11),   # thorax to left shoulder
        (8, 14),   # thorax to right shoulder
        (11, 12),  # left upper arm
        (12, 13),  # left lower arm
        (14, 15),  # right upper arm
        (15, 16),  # right lower arm
    ]

    distances = []

    for a, b in pairs:
        dist = np.linalg.norm(pose[a] - pose[b])

        if np.isfinite(dist) and dist > 1e-6:
            distances.append(dist)

    if len(distances) == 0:
        return 1.0

    scale = float(np.median(distances))

    if scale < 1e-6:
        scale = 1.0

    return scale


def normalize_pose3d(pose):
    """
    Normalize one 3D pose frame using per-axis z-score normalization.

    Advisor's suggestion:
        pose = (pose - pose.mean()) / pose.std()

    Here we apply mean/std separately for x, y, z dimensions:
        mean shape: (1, 3)
        std shape:  (1, 3)

    Input:
        pose shape = (17, 3)

    Output:
        normalized pose shape = (17, 3)
    """

    pose = np.asarray(pose, dtype=np.float32)

    if pose.shape != (17, 3):
        raise ValueError(f"Expected pose shape (17, 3), got {pose.shape}")

    eps = 1e-6

    # Mean/std over 17 joints, separately for x, y, z
    mean = pose.mean(axis=0, keepdims=True)
    std = pose.std(axis=0, keepdims=True)

    std = np.where(std < eps, 1.0, std)

    normalized = (pose - mean) / std

    return normalized.astype(np.float32)


def normalize_one_csv(input_path, output_path):
    df = pd.read_csv(input_path)

    required_cols = get_3d_columns()

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {input_path}")

    normalized_rows = []

    for _, row in df.iterrows():
        pose = row_to_pose3d(row)
        normalized_pose = normalize_pose3d(pose)

        new_row = pose3d_to_row_values(normalized_pose)

        # Keep metadata columns
        for col in df.columns:
            if col not in required_cols:
                new_row[col] = row[col]

        normalized_rows.append(new_row)

    out_df = pd.DataFrame(normalized_rows)

    # Put columns in a clean order
    meta_cols = [col for col in df.columns if col not in required_cols]
    out_df = out_df[required_cols + meta_cols]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)

    return len(out_df)


def find_csv_files():
    csv_files = []

    for file in os.listdir(INPUT_3D_DIR):
        if file.lower().endswith(".csv"):
            csv_files.append(file)

    return sorted(csv_files)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of CSV files for quick testing"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing normalized CSV files"
    )

    args = parser.parse_args()

    os.makedirs(OUTPUT_3D_DIR, exist_ok=True)

    csv_files = find_csv_files()

    if args.limit is not None:
        csv_files = csv_files[:args.limit]

    print("Input 3D dir:", INPUT_3D_DIR)
    print("Output normalized 3D dir:", OUTPUT_3D_DIR)
    print("CSV files found:", len(csv_files))

    processed = 0
    skipped = 0
    failed = 0
    total_rows = 0

    for file in csv_files:
        input_path = os.path.join(INPUT_3D_DIR, file)
        output_path = os.path.join(OUTPUT_3D_DIR, file)

        if os.path.exists(output_path) and not args.overwrite:
            skipped += 1
            print("Skipped existing:", file)
            continue

        try:
            rows = normalize_one_csv(input_path, output_path)
            processed += 1
            total_rows += rows

            print(f"Done: {file} | rows={rows}")

        except Exception as e:
            failed += 1
            print("Error:", file)
            print("Reason:", e)

    print()
    print("Finished normalizing 3D dataset.")
    print("Processed:", processed)
    print("Skipped existing:", skipped)
    print("Failed:", failed)
    print("Total rows:", total_rows)
    print("Saved to:", OUTPUT_3D_DIR)


if __name__ == "__main__":
    main()