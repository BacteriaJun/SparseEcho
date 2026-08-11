from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from sparseecho.engine import SparseEchoEngine
from sparseecho.model import ReconstructionResult
from sparseecho.planning import (
    AcquisitionDecision,
    AcquisitionRequest,
    AdaptiveAcquisitionController,
    QueryPlan,
)


AcquireFunction = Callable[[QueryPlan, AcquisitionRequest], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class SessionResult:
    initial: ReconstructionResult
    final: ReconstructionResult
    decisions: tuple[AcquisitionDecision, ...]
    final_request: AcquisitionRequest


class AdaptiveReconstructionSession:
    """Reacquire at a shorter aperture until the measured fiber tail closes.

    ``aperture_scale`` is relative to nominal view duration. Hardware may realize it by increasing
    query cadence or by reconfiguring an upstream coarse-Doppler stage so that the residual phase
    span over one view is reduced. The session never assumes that repeated long passes improve a
    rapidly rotating complex signal.
    """

    def __init__(
        self,
        engine: SparseEchoEngine,
        controller: AdaptiveAcquisitionController | None = None,
        *,
        max_reacquisitions: int = 2,
    ) -> None:
        self.engine = engine
        self.controller = controller or AdaptiveAcquisitionController(engine.aperture_budget)
        self.max_reacquisitions = int(max_reacquisitions)

    def run(self, acquire: AcquireFunction) -> SessionResult:
        request = AcquisitionRequest()
        slots, valid = acquire(self.engine.plan, request)
        initial = self.engine.process_capture(slots, valid, pass_count=request.pass_count)
        current = initial
        decisions: list[AcquisitionDecision] = []

        for _ in range(self.max_reacquisitions + 1):
            decision = self.controller.decide(current)
            decisions.append(decision)
            if decision.action == "accept":
                break
            if len(decisions) > self.max_reacquisitions:
                break
            request = AcquisitionRequest(
                aperture_scale=request.aperture_scale * decision.aperture_scale_factor,
                pass_count=1,
                reason=decision.reason,
            )
            slots, valid = acquire(self.engine.plan, request)
            current = self.engine.process_capture(slots, valid, pass_count=request.pass_count)

        return SessionResult(initial, current, tuple(decisions), request)
