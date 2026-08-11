# Validation and benchmarks

SparseEcho 1.1 separates algorithm regression from model-mismatch evidence and runtime integration evidence.

## Evidence tiers

1. `tests/` — algebra, contract, runtime and matched-smoke regression.
2. `sparseecho.simulator` — matched examples only.
3. `validation/independent_forward.py` — separately implemented synthetic forward path with deliberate mismatch.
4. capture/runtime replay — file, calibration, timing and state-machine integration.
5. hardware-in-the-loop / field capture — not included in the public release.

## Independent-forward acceptance

The independent forward path consumes serialized challenge masks and does not import SparseEcho temporal, transform, simulator or recovery modules. The release profile includes:

```text
active sources                 32
receivers                       8
near/far                       30 dB
weakest bucket SNR             15 dB
phase span                      0.4 cycle/view
cross-view direction rho       0.98
erasure                         2% burst
switch forward                  two weak settling poles + direct path
switch inverse                  FIR (0.90, 0.10)
receiver noise correlation      0.20
temporal noise AR               0.15
receiver calibration drift      enabled
quadratic/sinusoidal phase      enabled
```

Four fixed seeds regenerated from the final 1.1 source tree:

| metric | value |
|---|---:|
| mean precision | 1.000000 |
| minimum precision | 1.000000 |
| mean recall | 0.984375 |
| minimum recall | 0.968750 |
| false positives | 0 |
| false negatives | 2 / 128 |

The result is stored in `results/independent_acceptance.json`.

## Local CFAR null behavior

100 pure complex-Gaussian local apertures:

| metric | value |
|---|---:|
| proposal budget | 48 |
| mean proposals | 0.09 |
| maximum proposals | 2 |
| budget saturations | 0 |

This confirms that `view_max_components` is a compute guard rather than the effective statistical stopping rule.

## Switch-model mismatch

Inverse correction is fixed at `(0.90, 0.10)`. The coefficient sweep uses a separately implemented first-order state response.

| true memory coefficient | precision | recall |
|---:|---:|---:|
| 0.08 | 1.000 | 1.000 |
| 0.10 | 1.000 | 1.000 |
| 0.12 | 1.000 | 1.000 |
| 0.20 | 1.000 | 1.000 |
| 0.35 | 1.000 | 0.844 |

The primary release acceptance is a stronger model-family mismatch because its forward path uses a different two-pole settling model.

## Cross-view receiver-direction drift

The independent forward rotates each source's receiver-space direction while preserving near/far amplitude as a separate state.

| direction correlation `rho` | precision | recall |
|---:|---:|---:|
| 1.00 | 1.000 | 1.000 |
| 0.98 | 1.000 | 1.000 |
| 0.90 | 1.000 | 0.875 |
| 0.80 | 1.000 | 0.406 |

Strong receiver-subspace rotation is the clearest current physical-model sensitivity.

## Robustness map

`results/robustness_sweep.json` contains one fixed seed per point. Selected current points:

| dimension | mild/default | harder point |
|---|---|---|
| weakest bucket SNR | 15 dB → P/R 1.000/1.000 | 6 dB → 1.000/0.938 |
| active sources | K=32 → 1.000/1.000 | K=48 → 1.000/0.833 |
| near/far | 30 dB → 1.000/1.000 | 40 dB → 1.000/0.812 |
| phase span | 0.4 cycle → 1.000/1.000 | 0.8 cycle → 1.000/0.875 |
| sinusoidal phase | 0.025 cycle → 1.000/1.000 | 0.08 cycle → 1.000/0.906 |
| burst erasure | 2% → 1.000/1.000 | 5% → 1.000/1.000 |
| receiver noise correlation | 0.2 → 1.000/1.000 | 0.6 → 1.000/1.000 |
| calibration phase drift | 1° RMS → 1.000/1.000 | 3° RMS → 1.000/1.000 |

For `K=8`, the recorded point has precision `0.889` and recall `1.000`; sparse occupancy can therefore expose false-positive behavior under thresholds tuned for the reference occupancy.

One seed per point is intentionally not presented as a confidence interval.

## Runtime aperture closure

`results/adaptive_aperture.json` exercises the closed-loop runtime with an independent forward backend. Starting from a 0.65-cycle nominal span:

```text
attempt 1 aperture scale      1.0000
tail energy                  2.64e-4
decision                     compress

attempt 2 aperture scale      0.5763
tail energy                  8.31e-6
decision                     accept
final identities             32
```

The controller changes the requested physical aperture; it does not silently change decoder thresholds.

## Binary replay

`samples/independent_reference/` contains a small independent-forward `complex64` capture, validity vector, per-slot timestamps and a data-only calibration file. Reopening those files through the public I/O path gives:

```text
physical slots      4096
receivers               8
precision           1.000
recall              1.000
false IDs               0
```

This sample is synthetic and is labeled as such. It exists to validate the transport/runtime boundary without a simulator object.

## Interpretation

The public evidence supports a deployment-facing reconstruction architecture and a reproducible operating envelope. It does not substitute for platform-specific hardware-in-the-loop testing, field calibration or propagation validation.
