from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationSnapshot:
    """Calibration values applied to exactly one reconstruction attempt."""

    epoch: str
    switch_fir_taps: tuple[float, ...] = (0.90, 0.10)
    receiver_complex_gain: np.ndarray | None = None
    issued_monotonic_ns: int | None = None
    valid_until_monotonic_ns: int | None = None

    def validate(self, n_rx: int, *, now_monotonic_ns: int | None = None) -> None:
        taps = np.asarray(self.switch_fir_taps, dtype=float)
        if taps.ndim != 1 or taps.size == 0 or abs(float(taps[0])) < 1e-12:
            raise ValueError("switch_fir_taps must be a causal kernel with a non-zero first tap")
        if self.receiver_complex_gain is not None:
            gains = np.asarray(self.receiver_complex_gain)
            if gains.shape != (n_rx,):
                raise ValueError("receiver_complex_gain must have shape (n_rx,)")
            if not np.all(np.isfinite(gains)):
                raise ValueError("receiver_complex_gain contains a non-finite value")
            if np.any(np.abs(gains) < 1e-12):
                raise ValueError("receiver_complex_gain contains a zero-magnitude channel")
        if now_monotonic_ns is not None:
            if self.issued_monotonic_ns is not None and now_monotonic_ns < self.issued_monotonic_ns:
                raise ValueError("calibration snapshot is not active yet")
            if self.valid_until_monotonic_ns is not None and now_monotonic_ns > self.valid_until_monotonic_ns:
                raise ValueError("calibration snapshot has expired")


class StaticCalibrationProvider:
    """Small deployment adapter for a fixed calibration snapshot."""

    def __init__(self, snapshot: CalibrationSnapshot | None = None) -> None:
        self._snapshot = snapshot or CalibrationSnapshot(epoch="reference")

    def snapshot(self, *, n_rx: int, now_monotonic_ns: int | None = None) -> CalibrationSnapshot:
        self._snapshot.validate(n_rx, now_monotonic_ns=now_monotonic_ns)
        return self._snapshot
