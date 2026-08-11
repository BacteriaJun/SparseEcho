from __future__ import annotations

import numpy as np


def fir_from_memory_coefficient(coefficient: float) -> tuple[float, float]:
    eps = float(coefficient)
    if not 0.0 <= eps < 1.0:
        raise ValueError("coefficient must be in [0,1)")
    return (1.0 - eps, eps)


def apply_fir_memory(samples: np.ndarray, taps: tuple[float, ...] | np.ndarray) -> np.ndarray:
    """Apply a causal finite settling kernel continuously across the supplied samples."""
    x = np.asarray(samples)
    h = np.asarray(taps, dtype=np.float64)
    if h.ndim != 1 or h.size == 0 or abs(float(h[0])) < 1e-12:
        raise ValueError("taps must be a one-dimensional causal kernel with non-zero first tap")
    y = np.zeros_like(x, dtype=np.result_type(x, np.complex128))
    for k, tap in enumerate(h):
        if k == 0:
            y += tap * x
        elif k < x.shape[0]:
            y[k:] += tap * x[:-k]
    return y


def deconvolve_fir_memory(
    samples: np.ndarray,
    taps: tuple[float, ...] | np.ndarray,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Causal inverse for a calibrated finite settling kernel.

    The state is continuous over the supplied array.  A missing sample invalidates only recursions
    that depend on it; those positions fall back to the observed value and can be repaired later.
    """
    y = np.asarray(samples)
    h = np.asarray(taps, dtype=np.float64)
    if h.ndim != 1 or h.size == 0 or abs(float(h[0])) < 1e-12:
        raise ValueError("taps must be a one-dimensional causal kernel with non-zero first tap")
    if valid is None:
        valid_a = np.ones(y.shape[0], dtype=bool)
    else:
        valid_a = np.asarray(valid, dtype=bool)
        if valid_a.shape != (y.shape[0],):
            raise ValueError("valid mask shape mismatch")
    x = np.array(y, dtype=np.result_type(y, np.complex128), copy=True)
    for i in range(y.shape[0]):
        if not valid_a[i]:
            continue
        correction = 0.0
        usable = True
        for k in range(1, min(h.size, i + 1)):
            if not valid_a[i - k]:
                usable = False
                break
            correction = correction + h[k] * x[i - k]
        if usable:
            x[i] = (y[i] - correction) / h[0]
    return x


def apply_first_order_memory(samples: np.ndarray, coefficient: float) -> np.ndarray:
    """Compatibility wrapper for the historical two-tap settling model."""
    return apply_fir_memory(samples, fir_from_memory_coefficient(coefficient))


def deconvolve_first_order_memory(
    samples: np.ndarray, coefficient: float, valid: np.ndarray | None = None
) -> np.ndarray:
    """Compatibility wrapper for the historical two-tap inverse."""
    return deconvolve_fir_memory(samples, fir_from_memory_coefficient(coefficient), valid)
