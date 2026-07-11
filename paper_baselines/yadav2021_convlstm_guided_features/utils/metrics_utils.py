from __future__ import annotations

from typing import Dict

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),

        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),

        "fall_precision": float(precision_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "fall_recall": float(recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "fall_f1": float(f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),

        "not_fall_precision": float(precision_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),
        "not_fall_recall": float(recall_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),
        "not_fall_f1": float(f1_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),

        "sensitivity_fall_recall": float(sensitivity),
        "specificity_not_fall_recall": float(specificity),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
