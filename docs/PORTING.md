# Porting the reference runtime

The Python package is the public reference implementation. A deployment port is expected to preserve the data and decision contracts rather than reproduce Python internals line for line.

## Required compatibility points

A compatible implementation should preserve:

- query-plan generation and SHA-256 fingerprinting;
- slot-major complex sample order;
- validity semantics (invalid is erasure, not measured zero);
- monotonic capture metadata and optional per-slot timestamps;
- calibration epoch and causal switch-response input;
- bounded reacquisition decisions;
- reconstruction result identity, support and consistency fields;
- runtime fault vocabulary or an unambiguous mapping to it.

## Numerical representation

The file and C contracts use complex float32 for transport. The Python inverse promotes to complex128 internally where conditioning benefits from it. A production implementation can choose another internal type, but replay comparisons should specify the transport and internal precision used.

## Concurrency

The reference `SparseEchoEngine` should be treated as immutable after construction. A runtime instance carries sequence state and therefore should not be shared across unrelated acquisition streams without an external stream key or lock.

## Memory ownership and ABI evolution

The C header intentionally does not define allocation functions. Buffer ownership belongs to the embedding process. A production binding should make ownership explicit at the FFI boundary and avoid copying shared-memory or device-backed buffers unless numerical alignment requires it.

Top-level C records carry `struct_size` and ABI major/minor fields. Consumers should reject incompatible major revisions, accept known prefixes of newer minor revisions, and never infer record layout from a package version string.

## Clock domain

The runtime contract expects one monotonic clock domain per capture stream. If acquisition hardware timestamps originate in another clock domain, translate or annotate them before runtime validation. Do not mix wall-clock timestamps into `monotonic_*` fields.

## Calibration

Calibration discovery is deployment-specific. The public runtime only consumes a snapshot. A port should record the epoch used for each result so raw captures can be reconstructed against the same calibration later.

## Process boundary

`proto/sparseecho_runtime.proto` includes optional acquisition and reconstruction service contracts. They are transport contracts, not a requirement to deploy gRPC. A production system may keep the same messages while transporting raw sample buffers out-of-band.
