from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import run_independent_case, summarize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--output", default="results/independent_acceptance.latest.json")
    args = parser.parse_args()
    runs = [run_independent_case(seed=seed) for seed in range(args.seeds)]
    payload = {"profile": "independent-forward-acceptance", "software_version": "1.1", "summary": summarize(runs), "runs": runs}
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
