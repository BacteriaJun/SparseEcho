from __future__ import annotations

import argparse
import json
from pathlib import Path

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


class ScalingBackend:
    def __init__(self, phase_span: float, seed: int) -> None:
        self.phase_span = float(phase_span)
        self.seed = int(seed)
        self.sequence = 0

    def acquire(self, plan, request):
        self.sequence += 1
        config = ForwardConfig(max_phase_span_cycles=self.phase_span * request.aperture_scale)
        capture = generate_capture(plan.to_dict(), config, seed=self.seed + self.sequence)
        return CaptureFrame(
            capture.slots,
            capture.valid,
            plan.fingerprint(),
            self.sequence,
            int(capture.timestamps_ns[0]),
            int(capture.timestamps_ns[-1]),
            request.aperture_scale,
            request.pass_count,
            capture.timestamps_ns,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-span", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--output", default="results/adaptive_aperture.latest.json")
    args = parser.parse_args()
    plan = compile_query_plan(n_views=16)
    engine = SparseEchoEngine(EngineConfig(n_views=16), plan)
    runtime = ReconstructionRuntime(
        engine,
        ScalingBackend(args.phase_span, args.seed),
        calibration=StaticCalibrationProvider(CalibrationSnapshot(epoch="benchmark")),
    )
    outcome = runtime.run_once()
    payload = {
        "state": outcome.state.value,
        "attempts": [
            {
                "aperture_scale": a.request.aperture_scale,
                "decision": None if a.decision is None else a.decision.action,
                "tail": None if a.decision is None else a.decision.tail_energy,
            }
            for a in outcome.attempts
        ],
        "n_identities": None if outcome.result is None else int(outcome.result.identities.size),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if outcome.fault is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
