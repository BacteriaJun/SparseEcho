# Integration

SparseEcho integrates at the query-plan / coherent-baseband boundary. The deployment owns the mechanism that applies a query state and acquires the corresponding complex receiver vector.

## Acquisition backend

The Python reference uses this protocol:

```python
class AcquisitionBackend(Protocol):
    def acquire(self, plan: QueryPlan, request: AcquisitionRequest) -> CaptureFrame:
        ...
```

`AcquisitionRequest` carries an aperture scale and pass count. `CaptureFrame` returns:

- plan SHA-256 fingerprint;
- monotonically increasing sequence ID;
- monotonic start/end time;
- complex slot array;
- validity mask;
- optional per-slot timestamps;
- actual aperture scale and pass count.

The runtime validates these fields before the inverse is entered.

## Timing requirements

The GTF path assumes a sufficiently uniform physical cadence inside a view. If per-slot timestamps are supplied, the runtime checks:

- strict monotonicity;
- maximum fractional cadence error;
- positive capture duration.

A deployment with intentionally irregular timing should provide a corresponding forward/inverse timing model rather than disabling validation.

## Calibration

The open calibration contract contains a named epoch, causal switch-response taps and optional receiver complex gains. `JsonCalibrationStore` reads a data-only JSON representation; production systems may provide the same snapshot from another service.

Calibration expiry is checked against the capture's monotonic end time when an expiry is provided.

## Reacquisition

The default controller does not extend a high-dynamics aperture by blindly collecting more repeated passes. When the measured temporal-fiber tail exceeds budget it requests a shorter physical aperture. The acquisition backend decides how to realize that request within the surrounding system.

The 2/4-pass VTQC primitives remain available for integrations whose timing model justifies them, but they are not the default runtime response.

## C and RPC boundaries

- `include/sparseecho/runtime.h` defines buffer-oriented structs for an in-process or FFI integration.
- `proto/sparseecho_runtime.proto` defines transport-neutral acquisition/result messages.
- `schemas/` defines file/message manifests used by replay and archival tooling.

None of these files contain endpoint circuitry, carrier configuration, deployment geometry or platform control.
