from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from validation.independent_forward import ForwardConfig

from .common import run_independent_case


def _point(name: str, value: object, cfg: ForwardConfig) -> dict:
    run = run_independent_case(seed=0, forward=cfg)
    return {"value": value, "precision": run["precision"], "recall": run["recall"], "f1": run["f1"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/robustness_sweep.latest.json")
    args = parser.parse_args()
    base = ForwardConfig()
    groups = {
        "snr_db": [_point("snr", v, replace(base, weakest_bucket_snr_db=v)) for v in (6.0, 9.0, 12.0, 15.0)],
        "active_sources": [_point("k", v, replace(base, n_active=v)) for v in (8, 16, 32, 48)],
        "near_far_db": [_point("near_far", v, replace(base, near_far_power_db=v)) for v in (0.0, 15.0, 30.0, 40.0)],
        "phase_span_cycles": [_point("phase", v, replace(base, max_phase_span_cycles=v)) for v in (0.1, 0.4, 0.8)],
        "burst_erasure": [_point("erasure", v, replace(base, erasure_mode="burst", erasure_rate=v)) for v in (0.0, 0.02, 0.05)],
    }
    payload = {"profile": "independent-forward-robustness-map", "groups": groups}
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: len(v) for k, v in groups.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
