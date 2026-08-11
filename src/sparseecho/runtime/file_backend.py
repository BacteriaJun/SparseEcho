from __future__ import annotations

from collections import deque
from pathlib import Path

from sparseecho.io import open_capture_directory
from sparseecho.planning import AcquisitionRequest, QueryPlan

from .contracts import CaptureFrame


class CaptureDirectoryBackend:
    """Replay one or more capture directories through the runtime acquisition contract.

    Multiple directories can represent bounded reacquisition attempts recorded by an external
    system. This backend is intended for integration tests and deterministic incident replay.
    """

    def __init__(self, directories: list[str | Path] | tuple[str | Path, ...]) -> None:
        if not directories:
            raise ValueError("at least one capture directory is required")
        self._directories = deque(Path(p) for p in directories)

    def acquire(self, plan: QueryPlan, request: AcquisitionRequest) -> CaptureFrame:
        if not self._directories:
            raise RuntimeError("no replay capture remains for requested reacquisition")
        capture = open_capture_directory(self._directories.popleft())
        fingerprint = capture.plan_fingerprint or plan.fingerprint()
        return CaptureFrame(
            slots=capture.slots,
            valid=capture.valid,
            plan_fingerprint=fingerprint,
            sequence_id=capture.sequence_id,
            monotonic_start_ns=capture.monotonic_start_ns,
            monotonic_end_ns=capture.monotonic_end_ns,
            aperture_scale=capture.aperture_scale,
            pass_count=capture.pass_count,
            slot_timestamps_ns=capture.slot_timestamps_ns,
            metadata=capture.metadata or {},
        )
