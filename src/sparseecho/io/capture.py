from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sparseecho.planning import QueryPlan


@dataclass(frozen=True)
class SlotCapture:
    slots: np.ndarray
    valid: np.ndarray
    n_rx: int
    pass_count: int
    aperture_scale: float
    plan_fingerprint: str | None = None
    sequence_id: int = 0
    monotonic_start_ns: int = 0
    monotonic_end_ns: int = 0
    slot_timestamps_ns: np.ndarray | None = None
    metadata: dict | None = None


def write_capture_directory(
    directory: str | Path,
    slots: np.ndarray,
    valid: np.ndarray,
    plan: QueryPlan,
    *,
    pass_count: int = 1,
    aperture_scale: float = 1.0,
    sequence_id: int = 0,
    monotonic_start_ns: int | None = None,
    monotonic_end_ns: int | None = None,
    slot_timestamps_ns: np.ndarray | None = None,
    calibration_epoch: str | None = None,
    extra_metadata: dict | None = None,
) -> Path:
    """Persist a capture using the public, data-only replay contract."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    x = np.asarray(slots, dtype=np.complex64)
    mask = np.asarray(valid, dtype=np.uint8)
    expected = plan.physical_slots * int(pass_count)
    if x.ndim != 2 or x.shape[0] != expected:
        raise ValueError("slot array does not match query plan/pass count")
    if mask.shape != (x.shape[0],):
        raise ValueError("valid mask shape mismatch")

    if monotonic_start_ns is None:
        monotonic_start_ns = time.monotonic_ns()
    if monotonic_end_ns is None:
        monotonic_end_ns = monotonic_start_ns + max(x.shape[0] - 1, 1)
    if monotonic_end_ns <= monotonic_start_ns:
        raise ValueError("capture interval must be positive")

    x.tofile(root / "capture.c64")
    mask.tofile(root / "valid.u8")
    timestamp_file = None
    if slot_timestamps_ns is not None:
        ts = np.asarray(slot_timestamps_ns, dtype=np.int64)
        if ts.shape != (x.shape[0],):
            raise ValueError("slot_timestamps_ns shape mismatch")
        timestamp_file = "timestamps.i64"
        ts.tofile(root / timestamp_file)

    metadata = {
        "format": "sparseecho-slot-complex64-v2",
        "dtype": "complex64",
        "layout": "slot-major, receiver-minor",
        "physical_slots": int(x.shape[0]),
        "n_rx": int(x.shape[1]),
        "pass_count": int(pass_count),
        "aperture_scale": float(aperture_scale),
        "sequence_id": int(sequence_id),
        "monotonic_start_ns": int(monotonic_start_ns),
        "monotonic_end_ns": int(monotonic_end_ns),
        "plan_fingerprint": plan.fingerprint(),
        "calibration_epoch": calibration_epoch,
        "valid_mask": "valid.u8",
        "capture": "capture.c64",
        "timestamps": timestamp_file,
        "plan": plan.to_dict(pass_count=pass_count, aperture_scale=aperture_scale),
    }
    if extra_metadata:
        metadata["extra"] = extra_metadata
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return root


def open_capture_directory(directory: str | Path) -> SlotCapture:
    root = Path(directory)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    slots = int(metadata["physical_slots"])
    n_rx = int(metadata["n_rx"])
    pass_count = int(metadata.get("pass_count", 1))
    aperture_scale = float(metadata.get("aperture_scale", 1.0))
    data = np.memmap(root / metadata["capture"], dtype=np.complex64, mode="r", shape=(slots, n_rx))
    valid = np.memmap(root / metadata["valid_mask"], dtype=np.uint8, mode="r", shape=(slots,)).astype(bool)
    timestamp_file = metadata.get("timestamps")
    timestamps = None
    if timestamp_file:
        timestamps = np.memmap(root / timestamp_file, dtype=np.int64, mode="r", shape=(slots,))
    return SlotCapture(
        data,
        valid,
        n_rx,
        pass_count,
        aperture_scale,
        plan_fingerprint=metadata.get("plan_fingerprint"),
        sequence_id=int(metadata.get("sequence_id", 0)),
        monotonic_start_ns=int(metadata.get("monotonic_start_ns", 0)),
        monotonic_end_ns=int(metadata.get("monotonic_end_ns", 0)),
        slot_timestamps_ns=timestamps,
        metadata=metadata,
    )
