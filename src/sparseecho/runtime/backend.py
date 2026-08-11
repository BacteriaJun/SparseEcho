from __future__ import annotations

from typing import Protocol

from sparseecho.calibration import CalibrationSnapshot
from sparseecho.planning import AcquisitionRequest, QueryPlan

from .contracts import CaptureFrame


class AcquisitionBackend(Protocol):
    """Deployment-owned acquisition implementation."""

    def acquire(self, plan: QueryPlan, request: AcquisitionRequest) -> CaptureFrame: ...


class CalibrationProvider(Protocol):
    def snapshot(self, *, n_rx: int, now_monotonic_ns: int | None = None) -> CalibrationSnapshot: ...


class TelemetrySink(Protocol):
    def emit(self, event: dict) -> None: ...
