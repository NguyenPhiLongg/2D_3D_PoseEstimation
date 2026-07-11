from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


EPS = 1e-8


def safe_norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec) + EPS)


def angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle ABC in degrees.

    a, b, c are 3D points.
    b is the center joint.
    """
    ba = a - b
    bc = c - b

    denom = safe_norm(ba) * safe_norm(bc)

    if denom <= EPS:
        return 0.0

    cos_value = float(np.dot(ba, bc) / denom)
    cos_value = max(-1.0, min(1.0, cos_value))

    return float(math.degrees(math.acos(cos_value)))


def compute_bbox_3d(points: np.ndarray) -> Dict[str, float]:
    """
    points shape: (J, 3)
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points shape (J, 3), got {points.shape}")

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    min_x = float(np.min(x))
    max_x = float(np.max(x))

    min_y = float(np.min(y))
    max_y = float(np.max(y))

    min_z = float(np.min(z))
    max_z = float(np.max(z))

    width = max_x - min_x
    height = max_y - min_y
    depth = max_z - min_z

    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "min_z": min_z,
        "max_z": max_z,
        "width": float(width),
        "height": float(height),
        "depth": float(depth),
    }


def minmax_normalize_array(x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Min-max normalization following the paper's preprocessing idea.

    x shape can be:
        (T, D)
        (N, T, D)
    """
    x = x.astype(np.float32)

    x_min = np.nanmin(x, axis=0, keepdims=True)
    x_max = np.nanmax(x, axis=0, keepdims=True)

    denom = x_max - x_min
    denom = np.where(np.abs(denom) < EPS, 1.0, denom)

    out = (x - x_min) / denom

    stats = {
        "min": x_min.astype(np.float32),
        "max": x_max.astype(np.float32),
    }

    return out.astype(np.float32), stats


def normalize_pose_by_bbox(points: np.ndarray) -> np.ndarray:
    """
    Normalize one 3D pose using its 3D bounding box.

    points shape: (J, 3)

    The paper uses 3D joint normalization and 3D bounding box.
    This adapted implementation centers the skeleton and divides by body scale.
    """
    points = points.astype(np.float32)

    center = np.mean(points, axis=0, keepdims=True)

    bbox = compute_bbox_3d(points)

    scale = max(
        abs(bbox["width"]),
        abs(bbox["height"]),
        abs(bbox["depth"]),
        EPS,
    )

    return ((points - center) / scale).astype(np.float32)


def finite_or_zero(value: float) -> float:
    if value is None:
        return 0.0

    if math.isnan(value) or math.isinf(value):
        return 0.0

    return float(value)
