# SparseEcho 1.0

**Hardware-neutral reconstruction of sparse, time-varying modulated scatterer fields from sequential complex-baseband measurements.**

SparseEcho implements **Spectrally Factorized Physical-Time Inversion (SFPTI)**. The central design problem is not merely recovering a sparse vector; it is making a sparse-transform algorithm usable when its mathematical queries must be executed by a real, sequential, time-varying physical system.

The public tree ends at the **query-scheduling / slot-level complex-baseband boundary**. It does not contain endpoint implementation, deployment geometry, calibrated propagation transfer functions, field calibration material, platform control, scheduling policy above the reconstruction layer, or operational thresholds tied to a particular deployment.

That boundary is deliberate. The interfaces in this repository are shaped by an existing deployment class in which query states are physically switched and received coherently rather than evaluated from a stored vector. This is why switch settling, missing slots, near/far mixtures, residual motion and multi-receiver ingest are first-class API concerns rather than simulator decorations.

## SFPTI in one diagram

```text
sparse-transform query demand
          |
          v
  query-plan compiler
          |
          |  Gray-ranked physical execution
          v
sequential complex-baseband slots
          |
          +--> switch-response calibration
          +--> erasure mask / repair
          |
          v
 Gray Temporal Fiber (GTF)
          |
          |  time variation becomes a structured
          |  Walsh-domain spectral fiber
          v
 local 2^r hash apertures
          |
          v
 sparse local recovery + list consistency
          |
          v
32-bit sparse address/support reconstruction
```

The default public profile uses 14 independent 8-bit hash views over a 32-bit address space. Each view executes one 256-state Gray sweep:

```text
14 views × 256 states = 3584 physical query states
```

No `2^32` codebook is allocated and no population-scale loop is performed.

## The named algorithm

### Spectrally Factorized Physical-Time Inversion — SFPTI

SFPTI co-designs the **execution order of physical transform queries** with the target transform domain. Instead of treating sequential measurement time as unstructured distortion, it chooses a schedule for which physical time has a low-generator Walsh representation. Smooth temporal nuisance therefore becomes a structured spectral fiber rather than arbitrary transform leakage.

The current realization is **Gray Temporal Fiber (GTF)**.

For an `r`-bit local query `u`, reflected Gray execution gives the centered physical rank

\[
\tau(u)=-\frac12\sum_{i=0}^{r-1}2^i\chi_{s_i}(u),
\]

where `s_i` are `r` linearly independent Walsh masks. An injective `2^r`-slot time map cannot have fewer than `r` Walsh characters, so Gray rank attains the minimum possible support size; it simultaneously changes only one query bit between adjacent physical states.

For constant residual Doppler,

\[
h(u)=e^{j\Omega\tau(u)},
\]

GTF factorizes exactly:

\[
e^{j\Omega\tau(u)}=
\prod_{i=0}^{r-1}
\left(\cos\theta_i-j\chi_{s_i}(u)\sin\theta_i\right),
\qquad
\theta_i=\frac{\Omega 2^i}{2}.
\]

The Walsh coefficient at the XOR of a generator subset `S` is therefore

\[
c_S=(-j)^{|S|}
\prod_{i\in S}\sin\theta_i
\prod_{i\notin S}\cos\theta_i.
\]

This gives an exact shell-energy law and lets the planner translate a residual-Doppler estimate into a physical aperture budget. See [`docs/SFPTI.md`](docs/SFPTI.md).

## Why physical time is part of the inverse problem

A conventional sparse transform assumes that requested coordinates are sampled from one mathematical object. A switched physical system instead returns

```text
y(a1, t1), y(a2, t2), ...
```

while the underlying channel evolves between `t1`, `t2`, ... . SparseEcho treats this mismatch explicitly.

The fast path uses a single GTF sweep and models temporal evolution as a spectral fiber. If the configured fiber-tail budget is exceeded, **Virtual-Time Query Compilation (VTQC)** provides a multi-pass fallback using moment-constrained interpolation to synthesize measurements at a common virtual time.

## Engineering contract

The reconstruction engine consumes one complex vector per physical query state:

```text
shape: (physical_slots, receivers)
dtype: complex64 / complex128
layout: slot-major, receiver-minor
```

An optional validity mask marks corrupted or absent states. The public reference front end supports:

- complex multi-receiver ingest;
- calibrated first-order switching-memory deconvolution;
- slot erasure masks and candidate-conditioned repair;
- continuous residual phase evolution inside each local aperture;
- near/far source power spread;
- raw `complex64` memory-mapped replay.

The synthetic channel is only one producer of this contract. The main engine does not depend on simulator objects or simulator truth. See [`docs/INTEGRATION.md`](docs/INTEGRATION.md) for the acquisition boundary.

## Installation

```bash
python -m pip install .
```

Development install:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

## Quick start

Run the physical-query synthetic path:

```bash
sparseecho demo --seed 7
```

Write the same acquisition to the binary replay contract:

```bash
sparseecho demo --seed 7 --write-capture ./capture-demo
sparseecho replay ./capture-demo
```

Emit the deterministic 1.0 query plan:

```bash
sparseecho plan --output query-plan.json
```

Compute a physical aperture limit from an SFPTI fiber-tail budget:

```bash
sparseecho budget \
  --residual-doppler-hz 120 \
  --shell-order 3 \
  --leakage 1e-5
```

Python:

```python
from sparseecho import EngineConfig, SparseEchoEngine, compile_query_plan
from sparseecho.io import open_capture_directory

plan = compile_query_plan()
engine = SparseEchoEngine(EngineConfig(), plan)
capture = open_capture_directory("capture-demo")
result = engine.process_capture(capture.slots, capture.valid)
print(result.identities)
```

## Release profile

The default 1.0 plan is intentionally finite and inspectable:

| Property | Default |
|---|---:|
| Address space | 32 bit (`2^32` virtual addresses) |
| Local hash aperture | 8 bit / 256 states |
| Hash views | 14 |
| Physical query states | 3584 |
| Receiver channels in acceptance harness | 8 |
| Named inversion method | SFPTI |
| Single-pass temporal realization | GTF |
| High-dynamics fallback | VTQC |

The release benchmark files in `results/` record measured software-channel behavior for the exact source tree shipped here. They are not field-range claims and are not substitutes for a deployment-specific link budget.

### Measured release acceptance

The combined-fault acceptance profile simultaneously applies 30 dB near/far spread, 6 dB weakest fiber-matched bucket SNR, 0.40-cycle residual phase span per local view, 2% slot erasure and a 0.10 first-order switch-memory coefficient. Across the ten release seeds:

| Metric | Measured |
|---|---:|
| Release-gate passes | **10 / 10** |
| Mean precision | **1.000000** |
| Minimum precision | **1.000000** |
| Mean recall | **0.990625** |
| Minimum recall | **0.968750** |
| Final false IDs | **0** |

A separate `complex64` memory-mapped replay recovers 32/32 active addresses with no false ID. See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) and the machine-readable files in [`results/`](results/).

## Public-core boundary

SparseEcho deliberately does **not** publish or assume a particular endpoint, carrier, wavelength, antenna/optical geometry, mobility platform or sensing payload. The following are outside this repository:

- endpoint circuit/mechanical implementation;
- propagation-specific illumination and collection geometry;
- calibrated hardware transfer-function packs;
- platform state/control interfaces;
- deployment-specific thresholds and scheduling policy;
- field datasets whose metadata would bind the reconstruction kernel to a particular deployment.

The repository therefore describes what is required to reproduce and test the **reconstruction substrate**, not the surrounding system.

## Novelty boundary

SparseEcho does not claim invention of Gray code, Walsh-Hadamard transforms, sparse WHT recovery, linear hashing, peeling, least-squares repair, challenge/response sensing, or Hadamard measurement ordering. Those are established tools.

The project name **SFPTI** refers to the co-designed construction in which a sequential physical query schedule is chosen so that **physical time itself has structured transform support**, allowing time-varying nuisance to be represented, budgeted and inverted as a spectral fiber. GTF is the concrete Walsh/Gray realization shipped here.

The included prior-art notes are engineering documentation, not a patentability or freedom-to-operate opinion. See [`docs/PRIOR_ART_BOUNDARY.md`](docs/PRIOR_ART_BOUNDARY.md).

## Repository map

```text
sparseecho/
  planning/       query plans, linear hash views, aperture budgets
  transforms/     Walsh/Gray primitives
  temporal/       GTF, exact Doppler fibers, VTQC, erasure/transient handling
  recovery/       local fiber recovery and global list consistency
  io/             raw complex64 slot replay
  simulator/      public physical-query channel harness
  metrics/        acceptance metrics

docs/
  SFPTI.md
  ARCHITECTURE.md
  ENGINEERING_BOUNDARY.md
  INTEGRATION.md
  BENCHMARKS.md
  PRIOR_ART_BOUNDARY.md
benchmarks/
examples/
tests/
results/
```

## License

Apache-2.0. See [`LICENSE`](LICENSE).
