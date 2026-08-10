# Architecture

SparseEcho 1.0 is organized around the measurement contract rather than around a simulator.

## Data plane

```text
QueryPlan
   |
   +--> view 0: 256 Gray-ordered states
   +--> view 1: 256 Gray-ordered states
   ...
   +--> view 13
   |
   v
slot-major complex baseband + validity mask
   |
   v
switch-response inverse
   |
   v
initial local WHT / GTF decode
   |
   +--> candidate-conditioned erasure repair when needed
   |
   v
final local spectra
   |
   v
structured/list support assembly
   |
   v
multi-view spatial consistency
   |
   v
32-bit sparse support
```

## Planner

`planning/` owns information that must be known before physical acquisition:

- linear hash views;
- global challenge masks;
- Gray execution order;
- physical slot count;
- residual-Doppler aperture budgets.

The default profile contains 14 views × 256 states = 3584 physical states.

## Temporal layer

`temporal/` is the SFPTI-specific layer:

- exact Gray-Doppler fiber coefficients;
- exact shell-energy distribution;
- residual phase-span diagnostics;
- Virtual-Time Query Compilation primitives;
- switch-response correction;
- candidate-conditioned erasure repair.

## Recovery layer

`recovery/` deliberately separates local and global complexity.

Local recovery operates on a 256-bin transform aperture and may use dense local operations. Global recovery never constructs a `2^32` vector or codebook; it combines a small list of local bucket candidates across fixed linear projections.

## Capture contract

`io/` treats a physical capture as:

```text
capture.c64  complex64[physical_slots, n_rx]
valid.u8     uint8[physical_slots]
metadata.json
```

The binary path is memory-mapped. Synthetic input uses the same in-memory shape and does not receive a privileged decoder API.

## Failure surfaces

The public engine exposes, rather than hides, the main assumptions:

- each local view must remain inside a configured residual-dynamics envelope;
- enough structured views must retain the weak component to form a global candidate;
- receiver-space fingerprints should remain sufficiently stable across the acquisition aperture for the default consistency filter;
- the reference switch-response inverse assumes a calibrated first-order model;
- severe erasure and settling combinations may require a deployment-specific adapter or additional view redundancy.

These are engineering surfaces, not claims of universal identifiability.
