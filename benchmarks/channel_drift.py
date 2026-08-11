from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from validation.independent_forward import ForwardConfig

from .common import run_independent_case


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/channel_drift.latest.json")
    args = parser.parse_args()
    base = ForwardConfig()
    points = []
    for rho in (1.0, 0.98, 0.90, 0.80):
        run = run_independent_case(seed=0, forward=replace(base, cross_view_direction_rho=rho))
        points.append({"rho": rho, **{k: run[k] for k in ("precision", "recall", "f1")}})
    payload = {"points": points}
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
