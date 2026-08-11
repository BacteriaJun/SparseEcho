from __future__ import annotations

import argparse

from sparseecho import CaptureDirectoryBackend, EngineConfig, ReconstructionRuntime, SparseEchoEngine, compile_query_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", nargs="+")
    args = parser.parse_args()
    plan = compile_query_plan(n_views=16)
    engine = SparseEchoEngine(EngineConfig(n_views=16), plan)
    runtime = ReconstructionRuntime(engine, CaptureDirectoryBackend(args.capture))
    outcome = runtime.run_once()
    print(outcome.state.value)
    if outcome.result is not None:
        print(outcome.result.identities.tolist())
    return 0 if outcome.fault is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
