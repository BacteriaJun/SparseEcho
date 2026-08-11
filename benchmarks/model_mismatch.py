from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from validation.independent_forward import ForwardConfig

from .common import run_independent_case


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/model_mismatch.latest.json")
    args = parser.parse_args()
    base = ForwardConfig(switch_model="first_order")
    points = []
    for coefficient in (0.08, 0.10, 0.12, 0.20):
        run = run_independent_case(seed=0, forward=replace(base, first_order_memory=coefficient))
        points.append({"true_coefficient": coefficient, **{k: run[k] for k in ("precision", "recall", "f1")}})
    payload = {"inverse_fir": [0.90, 0.10], "points": points}
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
