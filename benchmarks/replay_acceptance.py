from __future__ import annotations

import json
import tempfile
from pathlib import Path

from sparseecho import EngineConfig, SparseEchoEngine, compile_query_plan
from sparseecho.config import SceneConfig
from sparseecho.io import open_capture_directory, write_capture_directory
from sparseecho.metrics import detection_metrics
from sparseecho.simulator import simulate_capture


def main() -> int:
    plan = compile_query_plan()
    scene = SceneConfig()
    capture = simulate_capture(plan, scene, seed=1)
    with tempfile.TemporaryDirectory() as tmp:
        root = write_capture_directory(
            Path(tmp) / "capture",
            capture.slots,
            capture.valid,
            plan,
            extra_metadata={"synthetic_seed": 1, "purpose": "release replay acceptance"},
        )
        replay = open_capture_directory(root)
        result = SparseEchoEngine(EngineConfig(), plan).process_capture(replay.slots, replay.valid)
        metric = detection_metrics(capture.truth.identities, result.identities)
        payload = {
            "profile": "sfpti-1.0-complex64-replay",
            "software_version": "1.0",
            "capture_bytes": int((root / "capture.c64").stat().st_size),
            "physical_slots": plan.physical_slots,
            "receivers": replay.n_rx,
            "decoder_ground_truth_access": False,
            "precision": metric.precision,
            "recall": metric.recall,
            "f1": metric.f1,
            "true_positive": metric.true_positive,
            "false_positive": metric.false_positive,
            "false_negative": metric.false_negative,
        }
    output = Path("results/replay_acceptance.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
