from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int


def detection_metrics(truth: np.ndarray, estimate: np.ndarray) -> DetectionMetrics:
    t = set(map(int, np.asarray(truth).ravel()))
    e = set(map(int, np.asarray(estimate).ravel()))
    tp = len(t & e)
    fp = len(e - t)
    fn = len(t - e)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-18)
    return DetectionMetrics(precision, recall, f1, tp, fp, fn)
