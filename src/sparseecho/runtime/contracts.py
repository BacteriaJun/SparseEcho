from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np

from sparseecho.model import ReconstructionResult
from sparseecho.planning import AcquisitionDecision, AcquisitionRequest


@dataclass(frozen=True)
class CaptureFrame:
    """One acquisition frame at the public runtime boundary."""

    slots: np.ndarray
    valid: np.ndarray
    plan_fingerprint: str
    sequence_id: int
    monotonic_start_ns: int
    monotonic_end_ns: int
    aperture_scale: float = 1.0
    pass_count: int = 1
    slot_timestamps_ns: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_rx(self) -> int:
        x = np.asarray(self.slots)
        return 1 if x.ndim == 1 else int(x.shape[1])

    @property
    def physical_slots(self) -> int:
        return int(np.asarray(self.slots).shape[0])


@dataclass(frozen=True)
class RuntimeLimits:
    max_invalid_fraction: float = 0.10
    max_consecutive_invalid: int = 32
    max_slot_timing_error_fraction: float = 0.05
    max_reacquisitions: int = 2
    min_receivers: int = 2
    max_receivers: int = 64

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_invalid_fraction < 1.0:
            raise ValueError("max_invalid_fraction must be in [0, 1)")
        if self.max_consecutive_invalid < 0:
            raise ValueError("max_consecutive_invalid must be non-negative")
        if self.max_slot_timing_error_fraction < 0.0:
            raise ValueError("max_slot_timing_error_fraction must be non-negative")
        if self.max_reacquisitions < 0:
            raise ValueError("max_reacquisitions must be non-negative")
        if self.min_receivers <= 0 or self.max_receivers < self.min_receivers:
            raise ValueError("receiver limits are invalid")


class RuntimeState(str, Enum):
    IDLE = "idle"
    ACQUIRING = "acquiring"
    RECONSTRUCTING = "reconstructing"
    REACQUIRE_PENDING = "reacquire_pending"
    COMPLETED = "completed"
    FAULTED = "faulted"


class FaultCode(str, Enum):
    PLAN_MISMATCH = "plan_mismatch"
    SEQUENCE_REGRESSION = "sequence_regression"
    SLOT_COUNT_MISMATCH = "slot_count_mismatch"
    NONFINITE_INPUT = "nonfinite_input"
    INVALID_TIMING = "invalid_timing"
    TIMING_JITTER = "timing_jitter"
    TOO_MANY_ERASURES = "too_many_erasures"
    ERASE_BURST = "erasure_burst"
    RECEIVER_COUNT = "receiver_count"
    CALIBRATION = "calibration"
    ACQUISITION = "acquisition"
    RECONSTRUCTION = "reconstruction"
    REACQUIRE_EXHAUSTED = "reacquire_exhausted"


@dataclass(frozen=True)
class RuntimeFault:
    code: FaultCode
    detail: str
    recoverable: bool


@dataclass(frozen=True)
class RuntimeAttempt:
    request: AcquisitionRequest
    decision: AcquisitionDecision | None
    frame_sequence_id: int | None
    calibration_epoch: str | None
    reconstruction_ns: int | None
    invalid_fraction: float | None
    result: ReconstructionResult | None
    fault: RuntimeFault | None


@dataclass(frozen=True)
class RuntimeOutcome:
    state: RuntimeState
    result: ReconstructionResult | None
    attempts: tuple[RuntimeAttempt, ...]
    fault: RuntimeFault | None = None
