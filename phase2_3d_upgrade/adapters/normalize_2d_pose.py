import numpy as np


def normalize_2d_pose(keypoints_2d, image_width=None, image_height=None):
    """
    Normalize 2D keypoints for 2D-to-3D pose lifting.

    Input:
        keypoints_2d: numpy array with shape (17, 2)

    Output:
        normalized_keypoints: numpy array with shape (17, 2)

    Method:
        - Center pose around pelvis/root joint.
        - Scale by max body size.
        - Optional image size normalization is not required here because
          the pose is normalized based on body scale.
    """

    keypoints_2d = np.asarray(keypoints_2d, dtype=np.float32)

    if keypoints_2d.shape != (17, 2):
        raise ValueError(f"Expected keypoints shape (17, 2), got {keypoints_2d.shape}")

    eps = 1e-6

    valid_mask = (keypoints_2d[:, 0] > 1) & (keypoints_2d[:, 1] > 1)

    if valid_mask.sum() < 5:
        return np.zeros_like(keypoints_2d, dtype=np.float32)

    root = keypoints_2d[0].copy()

    centered = keypoints_2d - root

    valid_points = centered[valid_mask]

    min_xy = valid_points.min(axis=0)
    max_xy = valid_points.max(axis=0)

    body_width = max_xy[0] - min_xy[0]
    body_height = max_xy[1] - min_xy[1]

    scale = max(body_width, body_height) + eps

    normalized = centered / scale

    normalized[~valid_mask] = 0.0

    return normalized.astype(np.float32)


def flatten_pose_sequence(sequence_2d):
    """
    Convert sequence shape from (T, 17, 2) to (T, 34).
    """

    sequence_2d = np.asarray(sequence_2d, dtype=np.float32)

    if sequence_2d.ndim != 3 or sequence_2d.shape[1:] != (17, 2):
        raise ValueError(f"Expected sequence shape (T, 17, 2), got {sequence_2d.shape}")

    return sequence_2d.reshape(sequence_2d.shape[0], 34).astype(np.float32)