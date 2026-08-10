# Release acceptance

The acceptance material shipped with SparseEcho 1.0 is divided into three evidence classes. They are deliberately kept separate so that mathematical self-checks, normalized physical-query stress tests and binary I/O replay are not presented as the same kind of validation.

## 1. Combined-fault reconstruction acceptance

`results/combined_fault_acceptance.json` was produced by the source tree in this release. Each seed generates the physical query slots first, then reconstructs from the resulting complex observations and slot-validity mask. The engine is not given the true support or the active-count value.

Default acceptance profile:

| Property | Value |
|---|---:|
| Virtual address space | 32 bit (`2^32`) |
| Active support | 32 |
| Complex receiver channels | 8 |
| Hash views | 14 |
| Query states per view | 256 |
| Physical query states | 3584 |
| Near/far power spread | 30 dB |
| Weakest fiber-matched bucket SNR | 6 dB |
| Maximum residual phase span per view | 0.40 cycle |
| Quadratic phase curvature | up to 0.05 cycle |
| Fractional amplitude drift | up to 0.04 |
| Slot erasure rate | 2% |
| First-order switch-memory coefficient | 0.10 |

Ten independent release seeds were executed:

| Metric | Result |
|---|---:|
| Seeds passing release gate | **10 / 10** |
| Mean precision | **1.000000** |
| Minimum precision | **1.000000** |
| Mean recall | **0.990625** |
| Minimum recall | **0.968750** |
| Mean F1 | **0.995238** |
| True sources across all seeds | 320 |
| Recovered true sources | 317 |
| Final false IDs | **0** |
| Missed sources | 3 |

The release gate is precision >= 0.99 and recall >= 0.95 for every recorded seed.

The Python reference implementation is intentionally clarity-first. Per-seed runtime in the JSON is recorded for reproducibility but is **not** presented as a throughput benchmark: release seeds were run in bounded parallel batches and therefore experienced different local CPU contention.

## 2. Theory acceptance

`results/theory_acceptance.json` checks identities used by SFPTI rather than detection performance.

For the default 8-bit local aperture:

- centered Gray physical rank has exactly **8** nonzero Walsh coefficients, matching the injective-support lower bound;
- direct WHT evaluation and the closed-form exact Gray-Doppler factorization agree to approximately machine precision in the recorded cases;
- at 0.40-cycle residual phase span, modeled-through-order-3 unmodeled fiber energy is approximately **7.39e-6**;
- an order-3 tail budget of `1e-5` gives a maximum phase span of approximately **0.41623 cycle** for the default local aperture.

These values test the implementation of the mathematical identities in `docs/SFPTI.md`; they are not empirical field-range measurements.

## 3. `complex64` replay acceptance

`results/replay_acceptance.json` exercises the public binary acquisition boundary:

1. physical query observations are generated;
2. observations are written as slot-major `complex64` records plus a validity mask;
3. the original in-memory generation path is discarded;
4. the files are reopened with memory mapping;
5. `SparseEchoEngine` reconstructs from those files without ground-truth access.

The recorded replay recovered 32/32 active addresses with no false ID.

The replay data is a **synthetic physical-channel capture**, not a field capture. It validates the binary contract and the separation between acquisition and reconstruction, not a particular carrier, range, endpoint or deployment geometry.

## Interpretation boundary

The public harness is normalized at the slot-level complex-baseband boundary. In particular, `weakest_fiber_matched_snr_db` is an effective post-query/bucket quantity, not a claim about optical/RF link budget, transmitted power, path loss or detector sensitivity. Deployment-specific front-end calibration belongs outside this public tree.
