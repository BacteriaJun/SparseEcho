# Prior-art boundary

SparseEcho uses established mathematical and engineering tools. This file is intentionally explicit about which concepts are *not* presented as project inventions.

## Established tools used here

- Walsh-Hadamard transforms and FWHT;
- binary reflected Gray ordering;
- sparse Walsh-Hadamard / sparse Fourier recovery ideas;
- linear hashing, list recovery and peeling/CLEAN-style subtraction;
- least-squares reconstruction from partial observations;
- moment-constrained interpolation;
- reader/query-controlled physical sensing in the broad sense;
- Hadamard measurement ordering and orthogonal physical sensing.

The repository references representative prior work in `REFERENCES.md`.

## Named SparseEcho construction

**Spectrally Factorized Physical-Time Inversion (SFPTI)** names the co-designed construction implemented here:

1. a sparse-transform algorithm requires transform-domain queries;
2. those queries must be executed sequentially in a time-varying physical system;
3. the physical execution order is selected so that the time coordinate itself has minimal/low-generator support in the target character domain;
4. temporal nuisance is consequently transformed into a structured spectral fiber rather than arbitrary transform leakage;
5. the fiber admits explicit residual-Doppler factorization, shell-energy budgeting and local parameter diagnostics;
6. the same structure informs scheduling, recovery and high-dynamics fallback.

**Gray Temporal Fiber (GTF)** is the Walsh/Gray realization in this release.

## Scope of the novelty statement

The above is an engineering/research characterization of the source tree. It is not a patentability opinion, a freedom-to-operate analysis, or a representation that no earlier publication or patent contains an equivalent construction. Independent formal prior-art review is required before making legal novelty claims.
