"""Minimal acquisition adapter showing the deployment-facing runtime contract.

The example uses the independent forward model as a stand-in for an external acquisition process.
A production backend would replace only Backend.acquire().
"""

from __future__ import annotations

import time

from sparseecho import (
    CalibrationSnapshot,
    CaptureFrame,
    EngineConfig,
    ReconstructionRuntime,
    SparseEchoEngine,
    StaticCalibrationProvider,
    compile_query_plan,
)
from validation.independent_forward import ForwardConfig, generate_capture


class Backend:
    def __init__(self) -> None:
        self.sequence_id = 0

    def acquire(self, plan, request) -> CaptureFrame:
        self.sequence_id += 1
        config = ForwardConfig(max_phase_span_cycles=0.55 * request.aperture_scale)
        capture = generate_capture(plan.to_dict(), config, seed=100 + self.sequence_id)
        return CaptureFrame(
            slots=capture.slots,
            valid=capture.valid,
            plan_fingerprint=plan.fingerprint(),
            sequence_id=self.sequence_id,
            monotonic_start_ns=int(capture.timestamps_ns[0]),
            monotonic_end_ns=int(capture.timestamps_ns[-1]),
            aperture_scale=request.aperture_scale,
            pass_count=request.pass_count,
            slot_timestamps_ns=capture.timestamps_ns,
            metadata={"source": "example-backend"},
        )


def main() -> None:
    plan = compile_query_plan(n_views=16)
    engine = SparseEchoEngine(EngineConfig(n_views=16), plan)
    calibration = StaticCalibrationProvider(
        CalibrationSnapshot(epoch="example", switch_fir_taps=(0.90, 0.10))
    )
    runtime = ReconstructionRuntime(engine, Backend(), calibration=calibration)
    outcome = runtime.run_once()
    print(outcome.state.value, None if outcome.result is None else outcome.result.identities.size)


if __name__ == "__main__":
    main()
