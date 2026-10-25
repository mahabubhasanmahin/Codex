"""Classification, calibration and group-bootstrap uncertainty metrics."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


def classification_metrics(target: np.ndarray, probability: np.ndarray, classes: list[str], bins: int = 15) -> dict[str, Any]:
    """Report undefined per-class ROC AUC as null rather than inventing a score."""
    if not len(target):
        return {'samples': 0, 'accuracy': None, 'balanced_accuracy': None, 'macro_f1': None,
                'macro_auc_roc': None, 'auc_defined_classes': 0, 'ece': None, 'per_class_f1': {}, 'confusion_matrix': []}
    predicted = probability.argmax(1)
    labels = np.arange(len(classes))
    auc = []
    for label in labels:
        binary = target == label
        if binary.any() and not binary.all():
            auc.append(float(roc_auc_score(binary, probability[:, label])))
    confidence = probability.max(1)
    correct = predicted == target
    ece = 0.0
    for index in range(bins):
        selected = (confidence >= index / bins) & ((confidence < (index + 1) / bins) if index < bins - 1 else (confidence <= 1))
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return dict(samples=len(target), accuracy=float(accuracy_score(target, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(target, predicted)),
        macro_f1=float(f1_score(target, predicted, labels=labels, average='macro', zero_division=0)),
        macro_auc_roc=float(np.mean(auc)) if auc else None, auc_defined_classes=len(auc), ece=float(ece),
        per_class_f1=dict(zip(classes, f1_score(target, predicted, labels=labels, average=None, zero_division=0).tolist())),
        confusion_matrix=confusion_matrix(target, predicted, labels=labels).tolist())


def bootstrap_f1(target: np.ndarray, probability: np.ndarray, groups: list[str], seed: int,
                 repetitions: int = 1000) -> tuple[float | None, float | None]:
    """Resample independent groups, retaining every image within each sampled group."""
    if not len(target):
        return None, None
    if repetitions < 1:
        raise ValueError('bootstrap_repetitions must be positive')
    group_array = np.array(groups)
    unique = np.unique(group_array)
    indices = {group: np.flatnonzero(group_array == group) for group in unique}
    rng = np.random.default_rng(seed)
    predicted = probability.argmax(1)
    scores = []
    for _ in range(repetitions):
        sampled = np.concatenate([indices[group] for group in rng.choice(unique, len(unique), replace=True)])
        scores.append(f1_score(target[sampled], predicted[sampled], labels=np.arange(probability.shape[1]), average='macro', zero_division=0))
    return tuple(float(value) for value in np.quantile(scores, [.025, .975]))
