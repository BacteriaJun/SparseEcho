import numpy as np

from sparseecho import CaptureFrame, ReconstructionRuntime, RuntimeLimits, compile_query_plan
from sparseecho.calibration import StaticCalibrationProvider
from sparseecho.model import ReconstructionResult, ViewDiagnostics
from sparseecho.planning import AcquisitionDecision, ApertureBudget


class StubEngine:
    def __init__(self):
        self.plan = compile_query_plan(n_views=16)
        self.aperture_budget = ApertureBudget()
        self.calls = 0

    def process_capture(self, *args, **kwargs):
        self.calls += 1
        span = 0.6 if self.calls == 1 else 0.1
        views = tuple(
            ViewDiagnostics(i, 1, 0.0, 1.0, 2.0, 0.0, span, self.aperture_budget.tail_energy(span))
            for i in range(16)
        )
        return ReconstructionResult(np.array([7], dtype=np.uint32), tuple(), views)


class StubController:
    def __init__(self):
        self.calls = 0

    def decide(self, result):
        self.calls += 1
        if self.calls == 1:
            return AcquisitionDecision("compress", 0.5, 0.6, 0.4, 1e-3, "tail above budget")
        return AcquisitionDecision("accept", 1.0, 0.1, 0.4, 1e-8, "within budget")


class Backend:
    def __init__(self):
        self.sequence = 0

    def acquire(self, plan, request):
        self.sequence += 1
        n = plan.physical_slots
        ts = np.arange(n, dtype=np.int64) * 1000 + self.sequence * 10_000_000
        return CaptureFrame(
            np.zeros((n, 4), dtype=np.complex64),
            np.ones(n, dtype=bool),
            plan.fingerprint(),
            self.sequence,
            int(ts[0]),
            int(ts[-1]),
            request.aperture_scale,
            request.pass_count,
            ts,
        )


def test_closed_loop_runtime_reacquires_once():
    engine = StubEngine()
    runtime = ReconstructionRuntime(
        engine,
        Backend(),
        calibration=StaticCalibrationProvider(),
        controller=StubController(),
        limits=RuntimeLimits(max_reacquisitions=2),
    )
    outcome = runtime.run_once()
    assert outcome.fault is None
    assert outcome.result is not None
    assert len(outcome.attempts) == 2
    assert outcome.attempts[1].request.aperture_scale == 0.5
