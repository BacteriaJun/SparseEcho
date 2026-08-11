from __future__ import annotations

import time

import numpy as np

from sparseecho.calibration import StaticCalibrationProvider
from sparseecho.engine import SparseEchoEngine
from sparseecho.planning import AcquisitionRequest, AdaptiveAcquisitionController

from .backend import AcquisitionBackend, CalibrationProvider, TelemetrySink
from .contracts import (
    FaultCode,
    RuntimeAttempt,
    RuntimeFault,
    RuntimeLimits,
    RuntimeOutcome,
    RuntimeState,
)
from .telemetry import NullTelemetrySink
from .validation import validate_frame


class ReconstructionRuntime:
    """Reference closed-loop runtime around the SFPTI engine.

    The class deliberately owns no hardware driver. A deployment supplies an AcquisitionBackend;
    the runtime owns validation, calibration snapshotting, bounded reacquisition and telemetry.
    """

    def __init__(
        self,
        engine: SparseEchoEngine,
        backend: AcquisitionBackend,
        *,
        calibration: CalibrationProvider | None = None,
        controller: AdaptiveAcquisitionController | None = None,
        telemetry: TelemetrySink | None = None,
        limits: RuntimeLimits | None = None,
    ) -> None:
        self.engine = engine
        self.backend = backend
        self.calibration = calibration or StaticCalibrationProvider()
        self.controller = controller or AdaptiveAcquisitionController(engine.aperture_budget)
        self.telemetry = telemetry or NullTelemetrySink()
        self.limits = limits or RuntimeLimits()
        self._last_sequence_id: int | None = None

    def _emit(self, event: str, **fields: object) -> None:
        self.telemetry.emit({"event": event, "plan_fingerprint": self.engine.plan.fingerprint(), **fields})

    def run_once(self) -> RuntimeOutcome:
        request = AcquisitionRequest()
        attempts: list[RuntimeAttempt] = []

        for attempt_index in range(self.limits.max_reacquisitions + 1):
            self._emit(
                "acquisition_requested",
                attempt=attempt_index,
                aperture_scale=request.aperture_scale,
                pass_count=request.pass_count,
                reason=request.reason,
            )
            try:
                frame = self.backend.acquire(self.engine.plan, request)
            except Exception as exc:  # backend is a deployment boundary
                fault = RuntimeFault(FaultCode.ACQUISITION, str(exc), True)
                attempts.append(RuntimeAttempt(request, None, None, None, None, None, None, fault))
                self._emit("runtime_fault", code=fault.code.value, detail=fault.detail)
                return RuntimeOutcome(RuntimeState.FAULTED, None, tuple(attempts), fault)

            fault = validate_frame(
                frame,
                self.engine.plan,
                self.limits,
                previous_sequence_id=self._last_sequence_id,
            )
            invalid_fraction = float(1.0 - np.mean(np.asarray(frame.valid, dtype=bool)))
            if fault is not None:
                attempts.append(
                    RuntimeAttempt(
                        request,
                        None,
                        frame.sequence_id,
                        None,
                        None,
                        invalid_fraction,
                        None,
                        fault,
                    )
                )
                self._emit("frame_rejected", code=fault.code.value, detail=fault.detail)
                if fault.recoverable and attempt_index < self.limits.max_reacquisitions:
                    request = AcquisitionRequest(
                        aperture_scale=request.aperture_scale,
                        pass_count=request.pass_count,
                        reason=f"retry after {fault.code.value}",
                    )
                    continue
                return RuntimeOutcome(RuntimeState.FAULTED, None, tuple(attempts), fault)

            self._last_sequence_id = frame.sequence_id
            try:
                calibration = self.calibration.snapshot(
                    n_rx=frame.n_rx,
                    now_monotonic_ns=frame.monotonic_end_ns,
                )
            except Exception as exc:
                fault = RuntimeFault(FaultCode.CALIBRATION, str(exc), False)
                attempts.append(
                    RuntimeAttempt(request, None, frame.sequence_id, None, None, invalid_fraction, None, fault)
                )
                self._emit("runtime_fault", code=fault.code.value, detail=fault.detail)
                return RuntimeOutcome(RuntimeState.FAULTED, None, tuple(attempts), fault)

            started = time.perf_counter_ns()
            try:
                result = self.engine.process_capture(
                    frame.slots,
                    frame.valid,
                    pass_count=frame.pass_count,
                    receiver_calibration=calibration.receiver_complex_gain,
                    switch_fir_taps=calibration.switch_fir_taps,
                )
            except Exception as exc:
                fault = RuntimeFault(FaultCode.RECONSTRUCTION, str(exc), False)
                attempts.append(
                    RuntimeAttempt(
                        request,
                        None,
                        frame.sequence_id,
                        calibration.epoch,
                        time.perf_counter_ns() - started,
                        invalid_fraction,
                        None,
                        fault,
                    )
                )
                self._emit("runtime_fault", code=fault.code.value, detail=fault.detail)
                return RuntimeOutcome(RuntimeState.FAULTED, None, tuple(attempts), fault)
            elapsed = time.perf_counter_ns() - started
            decision = self.controller.decide(result)
            attempts.append(
                RuntimeAttempt(
                    request,
                    decision,
                    frame.sequence_id,
                    calibration.epoch,
                    elapsed,
                    invalid_fraction,
                    result,
                    None,
                )
            )
            self._emit(
                "reconstruction_completed",
                attempt=attempt_index,
                sequence_id=frame.sequence_id,
                calibration_epoch=calibration.epoch,
                reconstruction_ns=elapsed,
                identities=int(result.identities.size),
                decision=decision.action,
                fiber_tail=decision.tail_energy,
            )

            if decision.action == "accept":
                return RuntimeOutcome(RuntimeState.COMPLETED, result, tuple(attempts), None)
            if attempt_index >= self.limits.max_reacquisitions:
                fault = RuntimeFault(
                    FaultCode.REACQUIRE_EXHAUSTED,
                    "fiber-tail budget still exceeds limit after bounded reacquisition",
                    False,
                )
                return RuntimeOutcome(RuntimeState.FAULTED, result, tuple(attempts), fault)
            request = AcquisitionRequest(
                aperture_scale=request.aperture_scale * decision.aperture_scale_factor,
                pass_count=1,
                reason=decision.reason,
            )

        raise RuntimeError("unreachable runtime loop exit")
