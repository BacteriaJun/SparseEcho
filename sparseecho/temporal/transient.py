from __future__ import annotations

import numpy as np


def apply_first_order_memory(samples: np.ndarray, coefficient: float) -> np.ndarray:
    """Simple first-order settling/memory model used by the public harness."""
    x = np.asarray(samples)
    eps = float(coefficient)
    if not 0.0 <= eps < 1.0:
        raise ValueError("coefficient must be in [0,1)")
    y = np.array(x, copy=True)
    if x.shape[0] > 1 and eps:
        y[1:] = (1.0 - eps) * x[1:] + eps * x[:-1]
    return y


def deconvolve_first_order_memory(
    samples: np.ndarray, coefficient: float, valid: np.ndarray | None = None
) -> np.ndarray:
    """Causal inverse for the first-order public switch-response model.

    Missing slots break the recursion; following samples restart conservatively until a valid
    predecessor is available. Erasure repair is handled separately.
    """
    y = np.asarray(samples)
    eps = float(coefficient)
    if not 0.0 <= eps < 1.0:
        raise ValueError("coefficient must be in [0,1)")
    if valid is None:
        valid_a = np.ones(y.shape[0], dtype=bool)
    else:
        valid_a = np.asarray(valid, dtype=bool)
    x = np.array(y, copy=True)
    if eps == 0.0:
        return x
    for i in range(1, y.shape[0]):
        if valid_a[i] and valid_a[i - 1]:
            x[i] = (y[i] - eps * x[i - 1]) / (1.0 - eps)
    return x
