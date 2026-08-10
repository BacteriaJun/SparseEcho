from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparseecho import ApertureBudget, EngineConfig, SparseEchoEngine, compile_query_plan
from sparseecho.config import SceneConfig
from sparseecho.io import open_capture_directory, write_capture_directory
from sparseecho.metrics import detection_metrics
from sparseecho.simulator import simulate_capture


def _demo(args: argparse.Namespace) -> int:
    engine_cfg = EngineConfig()
    plan = compile_query_plan()
    scene_cfg = SceneConfig(
        n_active=args.active,
        n_rx=args.receivers,
        weakest_fiber_snr_db=args.snr,
        max_phase_span_cycles_per_view=args.phase_span,
        slot_erasure_rate=args.erasure,
        switch_memory_coefficient=args.switch_memory,
    )
    capture = simulate_capture(plan, scene_cfg, seed=args.seed)
    engine = SparseEchoEngine(engine_cfg, plan)
    result = engine.process_capture(capture.slots, capture.valid)
    metrics = detection_metrics(capture.truth.identities, result.identities)
    payload = {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "recovered": result.to_dict(),
        "physical_slots": plan.physical_slots,
    }
    print(json.dumps(payload, indent=2))
    if args.write_capture:
        write_capture_directory(
            args.write_capture,
            capture.slots,
            capture.valid,
            plan,
            extra_metadata={"synthetic_seed": args.seed},
        )
    return 0


def _replay(args: argparse.Namespace) -> int:
    capture = open_capture_directory(args.directory)
    plan = compile_query_plan()
    if capture.slots.shape[0] != plan.physical_slots:
        raise SystemExit("capture slot count does not match the default 1.0 query plan")
    engine = SparseEchoEngine(plan=plan)
    result = engine.process_capture(capture.slots, capture.valid)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _plan(args: argparse.Namespace) -> int:
    plan = compile_query_plan()
    payload = plan.to_dict()
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
    payload = {
        "local_bits": args.local_bits,
        "shell_order": args.shell_order,
        "leakage_budget": args.leakage,
        "max_phase_span_cycles": budget.max_phase_span_cycles(),
        "max_view_seconds": budget.max_view_seconds(args.residual_doppler_hz),
    }
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sparseecho")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the physical-query synthetic acceptance path")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--active", type=int, default=32)
    demo.add_argument("--receivers", type=int, default=8)
    demo.add_argument("--snr", type=float, default=6.0, help="weakest fiber-matched bucket SNR in dB")
    demo.add_argument("--phase-span", type=float, default=0.40, help="max residual cycles per local view")
    demo.add_argument("--erasure", type=float, default=0.02)
    demo.add_argument("--switch-memory", type=float, default=0.10)
    demo.add_argument("--write-capture", type=str)
    demo.set_defaults(func=_demo)

    replay = sub.add_parser("replay", help="reconstruct a slot-level complex64 capture directory")
    replay.add_argument("directory")
    replay.set_defaults(func=_replay)

    plan = sub.add_parser("plan", help="emit the deterministic 1.0 query plan")
    plan.add_argument("--output")
    plan.set_defaults(func=_plan)

    budget = sub.add_parser("budget", help="compute a Gray-Doppler fiber aperture budget")
    budget.add_argument("--residual-doppler-hz", type=float, required=True)
    budget.add_argument("--local-bits", type=int, default=8)
    budget.add_argument("--shell-order", type=int, default=3)
    budget.add_argument("--leakage", type=float, default=1e-5)
    budget.set_defaults(func=_budget)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
