from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np

from sparseecho import EngineConfig, SparseEchoEngine, compile_query_plan
from sparseecho.metrics import detection_metrics
from validation.independent_forward import ForwardConfig, generate_capture


def run_independent_case(
    *,
    seed: int,
    forward: ForwardConfig | None = None,
    engine_config: EngineConfig | None = None,
    ordering: str = "gray",
) -> dict:
    cfg = engine_config or EngineConfig()
    plan = compile_query_plan(
        identity_bits=cfg.identity_bits,
        local_bits=cfg.local_bits,
        n_views=cfg.n_views,
        seed=cfg.hash_seed,
        ordering=ordering,
    )
    fcfg = forward or ForwardConfig()
    capture = generate_capture(plan.to_dict(), fcfg, seed=seed)
    engine = SparseEchoEngine(cfg, plan)
    started = time.perf_counter()
    result = engine.process_capture(
        capture.slots,
        capture.valid,
        receiver_calibration=capture.truth.receiver_calibration,
    )
    elapsed = time.perf_counter() - started
    metrics = detection_metrics(capture.truth.identities, result.identities)
    return {
        "seed": int(seed),
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "true_positive": metrics.true_positive,
        "false_positive": metrics.false_positive,
        "false_negative": metrics.false_negative,
        "recovered": int(result.identities.size),
        "candidate_count": len(result.candidates),
        "mean_local_proposals": float(np.mean([v.detected_buckets for v in result.views])),
        "elapsed_seconds": elapsed,
        "forward": asdict(fcfg),
    }


def summarize(runs: list[dict]) -> dict:
    return {
        "runs": len(runs),
        "mean_precision": float(np.mean([r["precision"] for r in runs])),
        "min_precision": float(np.min([r["precision"] for r in runs])),
        "mean_recall": float(np.mean([r["recall"] for r in runs])),
        "min_recall": float(np.min([r["recall"] for r in runs])),
        "mean_f1": float(np.mean([r["f1"] for r in runs])),
        "false_positive_total": int(sum(r["false_positive"] for r in runs)),
        "false_negative_total": int(sum(r["false_negative"] for r in runs)),
        "mean_elapsed_seconds": float(np.mean([r["elapsed_seconds"] for r in runs])),
    }
