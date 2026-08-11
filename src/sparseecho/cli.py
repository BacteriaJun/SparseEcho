from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparseecho import (
    ApertureBudget,
    CaptureDirectoryBackend,
    EngineConfig,
    JsonCalibrationStore,
    JsonlTelemetrySink,
    ReconstructionRuntime,
    RuntimeLimits,
    SparseEchoEngine,
    compile_query_plan,
)
from sparseecho.config import SceneConfig
from sparseecho.io import open_capture_directory, write_capture_directory
from sparseecho.metrics import detection_metrics
from sparseecho.planning import AdaptiveAcquisitionController
from sparseecho.simulator import simulate_capture


def _demo(args: argparse.Namespace) -> int:
    plan = compile_query_plan(n_views=args.views, ordering=args.ordering)
    scene = SceneConfig(
        n_active=args.active,
        n_rx=args.receivers,
        weakest_fiber_snr_db=args.snr,
        max_phase_span_cycles_per_view=args.phase_span,
        slot_erasure_rate=args.erasure,
        switch_fir_taps=(1.0 - args.switch_memory, args.switch_memory),
    )
    capture = simulate_capture(plan, scene, seed=args.seed)
    engine = SparseEchoEngine(
        EngineConfig(
            n_views=args.views,
            switch_fir_taps=(1.0 - args.switch_memory, args.switch_memory),
        ),
        plan,
    )
    result = engine.process_capture(capture.slots, capture.valid)
    metrics = detection_metrics(capture.truth.identities, result.identities)
    print(
        json.dumps(
            {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "physical_slots": plan.physical_slots,
                "result": result.to_dict(),
            },
            indent=2,
        )
    )
    if args.write_capture:
        write_capture_directory(
            args.write_capture,
            capture.slots,
            capture.valid,
            plan,
            extra_metadata={"matched_synthetic_seed": args.seed},
        )
    return 0


def _replay(args: argparse.Namespace) -> int:
    capture = open_capture_directory(args.directory)
    plan = compile_query_plan(n_views=args.views, ordering=args.ordering)
    engine = SparseEchoEngine(EngineConfig(n_views=args.views), plan)
    result = engine.process_capture(capture.slots, capture.valid, pass_count=capture.pass_count)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _plan(args: argparse.Namespace) -> int:
    plan = compile_query_plan(n_views=args.views, ordering=args.ordering)
    payload = plan.to_dict(pass_count=args.passes, aperture_scale=args.aperture_scale)
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))
    return 0


def _budget(args: argparse.Namespace) -> int:
    budget = ApertureBudget(
        local_bits=args.local_bits,
        modeled_shell_order=args.shell_order,
        leakage_budget=args.leakage,
    )
    print(
        json.dumps(
            {
                "local_bits": args.local_bits,
                "shell_order": args.shell_order,
                "leakage_budget": args.leakage,
                "max_phase_span_cycles": budget.max_phase_span_cycles(),
                "max_view_seconds": budget.max_view_seconds(args.residual_doppler_hz),
            },
            indent=2,
        )
    )
    return 0


def _recommend(args: argparse.Namespace) -> int:
    capture = open_capture_directory(args.directory)
    plan = compile_query_plan(n_views=args.views, ordering=args.ordering)
    engine = SparseEchoEngine(EngineConfig(n_views=args.views), plan)
    initial = engine.process_capture(capture.slots, capture.valid, pass_count=capture.pass_count)
    decision = AdaptiveAcquisitionController(engine.aperture_budget).decide(initial)
    print(json.dumps(decision.__dict__, indent=2))
    return 0



def _runtime_replay(args: argparse.Namespace) -> int:
    plan = compile_query_plan(n_views=args.views, ordering=args.ordering)
    engine = SparseEchoEngine(EngineConfig(n_views=args.views), plan)
    backend = CaptureDirectoryBackend(args.directories)
    calibration = JsonCalibrationStore(args.calibration) if args.calibration else None
    telemetry = JsonlTelemetrySink(args.telemetry) if args.telemetry else None
    runtime = ReconstructionRuntime(
        engine,
        backend,
        calibration=calibration,
        telemetry=telemetry,
        limits=RuntimeLimits(max_reacquisitions=max(len(args.directories) - 1, 0)),
    )
    outcome = runtime.run_once()
    payload = {
        "state": outcome.state.value,
        "fault": None if outcome.fault is None else {"code": outcome.fault.code.value, "detail": outcome.fault.detail},
        "attempts": len(outcome.attempts),
        "result": None if outcome.result is None else outcome.result.to_dict(),
    }
    print(json.dumps(payload, indent=2))
    return 0 if outcome.fault is None else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sparseecho")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the matched-model smoke path")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--active", type=int, default=32)
    demo.add_argument("--receivers", type=int, default=8)
    demo.add_argument("--views", type=int, default=16)
    demo.add_argument("--ordering", choices=["gray", "binary", "random"], default="gray")
    demo.add_argument("--snr", type=float, default=9.0)
    demo.add_argument("--phase-span", type=float, default=0.30)
    demo.add_argument("--erasure", type=float, default=0.01)
    demo.add_argument("--switch-memory", type=float, default=0.10)
    demo.add_argument("--write-capture")
    demo.set_defaults(func=_demo)

    replay = sub.add_parser("replay", help="reconstruct a complex64 capture directory")
    replay.add_argument("directory")
    replay.add_argument("--views", type=int, default=16)
    replay.add_argument("--ordering", choices=["gray", "binary", "random"], default="gray")
    replay.set_defaults(func=_replay)

    plan = sub.add_parser("plan", help="emit a query plan")
    plan.add_argument("--views", type=int, default=16)
    plan.add_argument("--ordering", choices=["gray", "binary", "random"], default="gray")
    plan.add_argument("--passes", type=int, choices=[1, 2, 4], default=1)
    plan.add_argument("--aperture-scale", type=float, default=1.0)
    plan.add_argument("--output")
    plan.set_defaults(func=_plan)

    budget = sub.add_parser("budget", help="compute a temporal-fiber aperture budget")
    budget.add_argument("--residual-doppler-hz", type=float, required=True)
    budget.add_argument("--local-bits", type=int, default=8)
    budget.add_argument("--shell-order", type=int, default=3)
    budget.add_argument("--leakage", type=float, default=1e-5)
    budget.set_defaults(func=_budget)

    runtime_replay = sub.add_parser("runtime-replay", help="replay one or more captures through the closed-loop runtime")
    runtime_replay.add_argument("directories", nargs="+")
    runtime_replay.add_argument("--views", type=int, default=16)
    runtime_replay.add_argument("--ordering", choices=["gray", "binary", "random"], default="gray")
    runtime_replay.add_argument("--calibration")
    runtime_replay.add_argument("--telemetry")
    runtime_replay.set_defaults(func=_runtime_replay)

    recommend = sub.add_parser("recommend", help="recommend an aperture-duration update from a capture")
    recommend.add_argument("directory")
    recommend.add_argument("--views", type=int, default=16)
    recommend.add_argument("--ordering", choices=["gray", "binary", "random"], default="gray")
    recommend.set_defaults(func=_recommend)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
