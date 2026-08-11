from __future__ import annotations

import argparse
import json

import numpy as np

from sparseecho.recovery import GrayFiberViewDecoder
from sparseecho.transforms import gray_order


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--receivers", type=int, default=8)
    args = parser.parse_args()
    rng = np.random.default_rng(0xCF4A)
    decoder = GrayFiberViewDecoder(local_bits=8, max_components=48, query_order=gray_order(8))
    counts = []
    for _ in range(args.trials):
        noise = (rng.normal(size=(256, args.receivers)) + 1j * rng.normal(size=(256, args.receivers))) / np.sqrt(2.0)
        counts.append(len(decoder.decode(noise).components))
    payload = {
        "trials": args.trials,
        "mean_proposals": float(np.mean(counts)),
        "max_proposals": int(np.max(counts)),
        "budget_saturations": int(np.count_nonzero(np.asarray(counts) >= 48)),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
