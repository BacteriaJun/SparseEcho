from __future__ import annotations

import numpy as np


def moment_weights(times: np.ndarray, *, degree: int, target_time: float = 0.0) -> np.ndarray:
    """Minimum-norm weights that reproduce polynomials through *degree* at target_time.

    The constraints are sum_r w_r (t_r-target)^p = delta[p,0]. This is the generic VTQC
    primitive used when a single Gray-time sweep exceeds the configured dynamic envelope.
    """
    t = np.asarray(times, dtype=np.float64) - float(target_time)
    if t.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if t.size < degree + 1:
        raise ValueError("insufficient samples for requested polynomial degree")
    a = np.vstack([t**p for p in range(degree + 1)])
    b = np.zeros(degree + 1, dtype=np.float64)
    b[0] = 1.0
    # Minimum-norm solution of A w = b.
    return a.T @ np.linalg.solve(a @ a.T, b)


def synthesize_virtual_sample(
    samples: np.ndarray, times: np.ndarray, *, degree: int, target_time: float = 0.0
) -> np.ndarray:
    w = moment_weights(times, degree=degree, target_time=target_time)
    return np.tensordot(w, np.asarray(samples), axes=(0, 0))


def symmetric_four_pass_weights(inner: float, outer: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form cubic-exact weights for times [-outer,-inner,+inner,+outer]."""
    x = float(abs(inner))
    z = float(abs(outer))
    if not (0 <= x < z):
        raise ValueError("require 0 <= inner < outer")
    alpha = z * z / (2.0 * (z * z - x * x))
    beta = -(x * x) / (2.0 * (z * z - x * x))
    times = np.array([-z, -x, x, z], dtype=np.float64)
    weights = np.array([beta, alpha, alpha, beta], dtype=np.float64)
    return times, weights
