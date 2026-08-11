from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import CalibrationSnapshot


class JsonCalibrationStore:
    """Read a calibration snapshot from a data-only JSON file.

    Complex receiver gains are represented as parallel gain/phase arrays. The store never imports
    Python objects or executable calibration code from the deployment file.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def snapshot(self, *, n_rx: int, now_monotonic_ns: int | None = None) -> CalibrationSnapshot:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        magnitude = np.asarray(payload.get("receiver_gain", [1.0] * n_rx), dtype=float)
        phase_deg = np.asarray(payload.get("receiver_phase_deg", [0.0] * n_rx), dtype=float)
        if magnitude.shape != (n_rx,) or phase_deg.shape != (n_rx,):
            raise ValueError("calibration receiver arrays do not match n_rx")
        gains = magnitude * np.exp(1j * np.deg2rad(phase_deg))
        snap = CalibrationSnapshot(
            epoch=str(payload["epoch"]),
            switch_fir_taps=tuple(float(v) for v in payload.get("switch_fir_taps", [0.90, 0.10])),
            receiver_complex_gain=gains,
            issued_monotonic_ns=(
                None if payload.get("issued_monotonic_ns") is None else int(payload["issued_monotonic_ns"])
            ),
            valid_until_monotonic_ns=(
                None
                if payload.get("valid_until_monotonic_ns") is None
                else int(payload["valid_until_monotonic_ns"])
            ),
        )
        snap.validate(n_rx, now_monotonic_ns=now_monotonic_ns)
        return snap
