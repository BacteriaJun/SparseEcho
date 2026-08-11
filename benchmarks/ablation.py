from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from sparseecho import EngineConfig
from validation.independent_forward import ForwardConfig

from .common import run_independent_case


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ablation.latest.json")
    args = parser.parse_args()
    cases = [
        ("default", EngineConfig(), "gray"),
        ("no-spatial", replace(EngineConfig(), spatial_consistency_threshold=0.0, max_pre_candidates=2000), "gray"),
        ("rank1", replace(EngineConfig(), spatial_subspace_rank=1, spatial_consistency_threshold=0.47), "gray"),
        ("views-12", replace(EngineConfig(), n_views=12, min_view_support=10), "gray"),
        ("views-14", replace(EngineConfig(), n_views=14, min_view_support=11), "gray"),
        ("binary", EngineConfig(), "binary"),
        ("random", EngineConfig(), "random"),
        ("fiber-blind", replace(EngineConfig(), view_max_abs_phase_span_cycles=0.0), "gray"),
    ]
    runs = []
    for name, config, ordering in cases:
        run = run_independent_case(seed=0, forward=ForwardConfig(), engine_config=config, ordering=ordering)
        runs.append({"case": name, "ordering": ordering, **{k: run[k] for k in ("precision", "recall", "f1", "candidate_count", "elapsed_seconds")}})
    payload = {"profile": "architecture-ablation", "runs": runs}
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
