# Architecture

SparseEcho 1.1 is organized as a deployment-facing reconstruction component. The simulator is not part of the runtime data plane.

## Process boundary

A typical deployment has at least three responsibilities:

```text
acquisition process          SparseEcho runtime             consuming system
-------------------          ------------------             ----------------
apply query state     ───▶   validate frame
capture complex slots        snapshot calibration
mark invalid slots           reconstruct support      ───▶  consume identities
provide timestamps           evaluate fiber budget          consume diagnostics
                    ◀───     request reacquisition           store telemetry
```

The acquisition process can be in another language or host. The public contract is represented by the Python protocol, C header, protobuf messages and JSON capture schema.

## Runtime state machine

`ReconstructionRuntime` owns a bounded operating cycle:

```text
IDLE
  ↓
ACQUIRING
  ↓
frame validation
  ├─ hard contract fault ──▶ FAULTED
  ├─ recoverable frame fault ──▶ reacquire (bounded)
  ↓
RECONSTRUCTING
  ↓
fiber-tail decision
  ├─ accept ──▶ COMPLETED
  └─ compress aperture ──▶ REACQUIRE_PENDING ──▶ ACQUIRING
```

The runtime does not keep retrying indefinitely. `RuntimeLimits` bounds reacquisition count and input quality.

## Reconstruction data plane

```text
CaptureFrame
  ↓
continuous calibrated switch inverse
  ↓
receiver calibration
  ↓
per-view slot synthesis
  ↓
local WHT
  ↓
CFAR fiber-aware proposals
  ↓
candidate-conditioned erasure repair
  ↓
final local spectra
  ↓
redundant structured support assembly
  ↓
receiver-subspace consistency
  ↓
32-bit sparse support + diagnostics
```

Global recovery never constructs a `2^32` population vector or codebook.

## Planning plane

`planning/` owns information that must exist before acquisition:

- linear hash views;
- global parity challenge masks;
- physical execution ordering;
- plan fingerprint;
- pass structure;
- temporal-fiber aperture budget.

The fingerprint is a SHA-256 digest of the logical single-pass plan. Capture frames with a different fingerprint are rejected before reconstruction.

## Calibration plane

A calibration provider snapshots the calibration epoch at each attempt. The public reference values are:

- causal switch-response FIR taps;
- receiver complex gain/phase.

The runtime records the epoch used for each reconstruction attempt. Deployment-specific calibration discovery remains outside the repository.

## Observability

`JsonlTelemetrySink` records acquisition requests, frame rejection, calibration epoch, reconstruction latency, identity count, tail energy and runtime decisions. Telemetry is append-only JSONL so it can be ingested by an external logging system without coupling SparseEcho to one operations stack.

## Non-Python integration

`include/sparseecho/runtime.h` and `proto/sparseecho_runtime.proto` intentionally contain only the stable data boundary. They do not claim a C implementation is shipped. A production service can replace the Python reference while preserving plan/capture/result compatibility.
