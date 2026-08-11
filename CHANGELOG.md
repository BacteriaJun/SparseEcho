# Changelog

## 1.1

1.1 is the operationalization release for the current SFPTI generation.

Key changes:

- closed acquisition/reconstruction/reacquisition loop through a backend contract;
- runtime validation of plan identity, timing, erasure rate and calibration epoch;
- calibration snapshots and per-capture switch/receiver correction inputs;
- structured runtime fault policy and JSONL telemetry journal;
- stable capture, query-plan and result schemas for non-Python integration;
- C ABI and protobuf interface contracts for deployment adapters;
- continuous switch-state treatment across view boundaries;
- CFAR local proposal generation with an explicit compute budget;
- low-rank cross-view receiver-subspace consistency;
- independent forward validation, mismatch sweeps and architecture ablations;
- aperture-budget feedback connected to runtime reacquisition;
- release checks that separate matched regression from independent validation.

The public package still terminates at the acquisition/control boundary. Endpoint hardware, deployment geometry and field calibration assets are not part of this repository.
