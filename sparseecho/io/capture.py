from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sparseecho.planning import QueryPlan


@dataclass(frozen=True)
class SlotCapture:
    slots: np.ndarray
    valid: np.ndarray
    n_rx: int


def write_capture_directory(
    directory: str | Path,
    slots: np.ndarray,
    valid: np.ndarray,
    plan: QueryPlan,
    *,
    extra_metadata: dict | None = None,
) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    x = np.asarray(slots, dtype=np.complex64)
    mask = np.asarray(valid, dtype=np.uint8)
    if x.ndim != 2 or x.shape[0] != plan.physical_slots:
        raise ValueError("slot array does not match query plan")
    if mask.shape != (x.shape[0],):
        raise ValueError("valid mask shape mismatch")
    x.tofile(root / "capture.c64")
    mask.tofile(root / "valid.u8")
    metadata = {
        "format": "sparseecho-slot-complex64",
        "dtype": "complex64",
        "layout": "slot-major, receiver-minor",
        "physical_slots": int(x.shape[0]),
        "n_rx": int(x.shape[1]),
        "valid_mask": "valid.u8",
        "capture": "capture.c64",
        "plan": plan.to_dict(),
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
    data = np.memmap(root / metadata["capture"], dtype=np.complex64, mode="r", shape=(slots, n_rx))
    valid = np.memmap(root / metadata["valid_mask"], dtype=np.uint8, mode="r", shape=(slots,)).astype(bool)
    return SlotCapture(data, valid, n_rx)
