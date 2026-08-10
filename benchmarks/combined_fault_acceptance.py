from __future__ import annotations

import argparse
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from sparseecho import EngineConfig, SparseEchoEngine, compile_query_plan
from sparseecho.config import SceneConfig
from sparseecho.metrics import detection_metrics
from sparseecho.simulator import simulate_capture


def run_seed(seed: int) -> dict:
    plan = compile_query_plan()
    engine = SparseEchoEngine(EngineConfig(), plan)
    scene = SceneConfig()
    started = time.perf_counter()
    capture = simulate_capture(plan, scene, seed=seed)
    result = engine.process_capture(capture.slots, capture.valid)
    metric = detection_metrics(capture.truth.identities, result.identities)
    return {
        "seed": seed,
        "precision": metric.precision,
        "recall": metric.recall,
        "f1": metric.f1,
        "true_positive": metric.true_positive,
        "false_positive": metric.false_positive,
        "false_negative": metric.false_negative,
        "recovered": int(result.identities.size),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    seed_values = list(range(args.start_seed, args.start_seed + args.seeds))
    started = time.perf_counter()
    if args.workers <= 1:
        rows = [run_seed(seed) for seed in seed_values]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_seed, seed): seed for seed in seed_values}
            for future in as_completed(futures):
                rows.append(future.result())
        rows.sort(key=lambda row: row["seed"])

    precision = np.array([row["precision"] for row in rows])
    recall = np.array([row["recall"] for row in rows])
    f1 = np.array([row["f1"] for row in rows])
    payload = {
        "profile": "sfpti-1.0-combined-fault",
        "software_version": "1.0",
        "scenario": {
            "identity_space_bits": 32,
            "active": 32,
            "receivers": 8,
            "hash_views": 14,
            "states_per_view": 256,
            "physical_slots": 3584,
            "near_far_power_db": 30.0,
            "weakest_fiber_matched_snr_db": 6.0,
            "max_residual_phase_span_cycles_per_view": 0.40,
            "max_quadratic_phase_cycles": 0.05,
            "max_fractional_amplitude_drift": 0.04,
            "slot_erasure_rate": 0.02,
            "switch_memory_coefficient": 0.10,
        },
        "summary": {
            "seeds": len(rows),
            "mean_precision": float(np.mean(precision)),
            "min_precision": float(np.min(precision)),
            "mean_recall": float(np.mean(recall)),
            "min_recall": float(np.min(recall)),
            "mean_f1": float(np.mean(f1)),
            "passes_release_gate": int(np.sum((precision >= 0.99) & (recall >= 0.95))),
            "release_gate": {"precision": 0.99, "recall": 0.95},
            "wall_seconds": time.perf_counter() - started,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "runs": rows,
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
