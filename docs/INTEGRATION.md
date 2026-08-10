# Acquisition integration

SparseEcho does not require the synthetic channel. The production boundary is a compiled query plan plus slot-major complex receiver vectors.

## 1. Compile the query plan

```python
from sparseecho import compile_query_plan

plan = compile_query_plan()
print(plan.physical_slots)  # 3584 in the public profile
```

Each `QuerySlot` identifies the view, physical rank, local query state and global 32-bit challenge mask. An external acquisition controller is responsible for applying the requested state at the corresponding slot boundary.

The public plan is deterministic. A deployment can persist `plan.to_dict()` beside a capture so that acquisition and reconstruction use the same schedule description.

## 2. Receiver contract

For each physical state, provide one coherent complex value per receiver channel:

```text
slots.shape == (plan.physical_slots, n_rx)
slots.dtype  == complex64 or complex128
```

A separate Boolean vector marks whether each state is usable:

```text
valid.shape == (plan.physical_slots,)
```

An invalid state is not interpreted as a measured zero. It is an erasure and is routed through the erasure-aware recovery path.

## 3. Calibration responsibility

Before reconstruction, the integration layer should establish the quantities that are hardware-specific but required by the public core:

- receiver complex gain/phase consistency across a local aperture;
- slot timing and state-settling validity;
- a measured switching-memory model when transient compensation is enabled;
- a coarse physical gate narrow enough that the configured residual phase-span budget is meaningful;
- saturation/invalid-state flags.

SparseEcho deliberately does not prescribe how those quantities are obtained. Their calibration depends on the endpoint, propagation path and receiver implementation that sit outside the public repository.

## 4. Switching-memory model

The reference implementation contains a calibrated first-order memory operator:

```text
y[j] = (1 - eps) x[j] + eps x[j-1]
```

It exists to make switching transients explicit in the reconstruction contract. `eps` must be treated as a calibration parameter; a real front end should not assume that its transient is exactly first order. A different measured response can be adapted before `SparseEchoEngine.process_capture()` or implemented as another calibration operator.

## 5. Timing and aperture budget

SFPTI treats execution time as part of the inverse model. The planner can bound a local Gray sweep from a residual-Doppler estimate and a tolerated unmodeled fiber tail:

```python
from sparseecho import ApertureBudget

budget = ApertureBudget(local_bits=8, modeled_shell_order=3, leakage_budget=1e-5)
max_view_seconds = budget.max_view_seconds(residual_doppler_hz=120.0)
```

This budget belongs between coarse physical acquisition and the sparse identity inversion. The coarse estimator is deployment-specific and is not implemented by guessing a carrier or geometry in the public package.

## 6. Binary replay

The provided capture directory format separates acquisition from reconstruction:

```text
capture/
  capture.c64     slot-major interleaved complex64
  valid.u8        one validity byte per physical slot
  metadata.json   dimensions and query-plan metadata
```

Use:

```python
from sparseecho.io import open_capture_directory

capture = open_capture_directory("capture")
result = engine.process_capture(capture.slots, capture.valid)
```

`capture.c64` is memory mapped; the replay path does not need a simulator object or simulator truth.

## 7. Boundary of responsibility

A complete deployed system may have additional control, geometry, calibration, authorization and platform layers around this interface. They are intentionally outside the public source tree. SparseEcho's responsibility begins once physical query states can be scheduled and coherent slot-level observations can be presented at this contract.
