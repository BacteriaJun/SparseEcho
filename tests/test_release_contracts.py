from pathlib import Path


def test_non_python_contract_files_are_shipped():
    required = [
        "include/sparseecho/runtime.h",
        "proto/sparseecho_runtime.proto",
        "schemas/capture-manifest.schema.json",
        "schemas/query-plan.schema.json",
        "schemas/reconstruction-result.schema.json",
        "schemas/runtime-event.schema.json",
        "configs/reference-runtime.toml",
        "README_RELEASES.md",
    ]
    for name in required:
        assert Path(name).is_file(), name
