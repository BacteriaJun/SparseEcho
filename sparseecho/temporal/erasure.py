from __future__ import annotations

import numpy as np

from sparseecho.transforms import hadamard_matrix


def repair_erased_gray_slots(
    slots: np.ndarray,
    valid: np.ndarray,
    *,
    local_order: np.ndarray,
    candidate_buckets: np.ndarray,
    degree: int = 1,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Repair missing physical slots using a small candidate-conditioned temporal model.

    The fit uses only already-observed slots and only local bucket candidates. It never scans the
    global identity population and adds no physical interrogation states.
    """
    y = np.asarray(slots)
    valid_a = np.asarray(valid, dtype=bool)
    if valid_a.all() or candidate_buckets.size == 0:
        return np.array(y, copy=True)
    n = y.shape[0]
    bits = int(np.log2(n))
    if 1 << bits != n:
        raise ValueError("slot count must be a power of two")
    h = hadamard_matrix(bits)
    order = np.asarray(local_order, dtype=np.int64)
    t = (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) / max(n - 1, 1)

    columns: list[np.ndarray] = []
    for bucket in np.unique(candidate_buckets.astype(np.int64)):
        character = h[order, bucket]
        for p in range(degree + 1):
            columns.append(character * (t**p))
    design = np.stack(columns, axis=1).astype(np.complex128)
    obs = np.flatnonzero(valid_a)
    a = design[obs]
    rhs = y[obs]
    gram = a.conj().T @ a
    scale = max(float(np.trace(gram).real) / max(gram.shape[0], 1), 1e-12)
    coeff = np.linalg.solve(gram + ridge * scale * np.eye(gram.shape[0]), a.conj().T @ rhs)
    prediction = design @ coeff
    repaired = np.array(y, copy=True)
    repaired[~valid_a] = prediction[~valid_a]
    return repaired
