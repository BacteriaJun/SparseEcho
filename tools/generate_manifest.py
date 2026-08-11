from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_record(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def main() -> int:
    files = []
    for top in (
        "src", "validation", "benchmarks", "tests", "examples", "configs",
        "schemas", "include", "proto", "docs", "samples", "tools",
    ):
        root = ROOT / top
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
                continue
            files.append(file_record(path))

    for name in (
        "README.md", "README_RELEASES.md", "CHANGELOG.md", "CONTRIBUTING.md",
        "SECURITY.md", "LICENSE", "NOTICE", "REFERENCES.md", "pyproject.toml",
        "Makefile", ".gitignore", "VERSION",
    ):
        path = ROOT / name
        if path.exists():
            files.append(file_record(path))

    result_files = {}
    for name in (
        "independent_acceptance.json", "cfar_null.json", "model_mismatch.json",
        "channel_drift.json", "robustness_sweep.json", "ablation.json",
        "adaptive_aperture.json", "replay_acceptance.json", "test_acceptance.txt",
    ):
        path = ROOT / "results" / name
        if path.exists():
            result_files[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}

    artifacts = {}
    for path in sorted((ROOT / "dist").glob("*.whl")):
        artifacts[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}

    payload = {
        "project": "SparseEcho",
        "version": "1.1",
        "algorithm": "Spectrally Factorized Physical-Time Inversion (SFPTI)",
        "runtime_boundary": "query-plan / coherent-slot-baseband",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "source_files": sorted(files, key=lambda item: item["path"]),
        "result_files": result_files,
        "release_artifacts": artifacts,
        "public_runtime": {
            "python_protocol": "src/sparseecho/runtime",
            "c_header": "include/sparseecho/runtime.h",
            "protobuf": "proto/sparseecho_runtime.proto",
            "schemas": "schemas/",
        },
    }
    out = ROOT / "results" / "release_manifest.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
