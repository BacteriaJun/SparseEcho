# Public engineering boundary

SparseEcho is published as an application-neutral reconstruction core.

The code is intentionally complete enough to connect to a real acquisition path: it compiles physical query states, accepts slot-level complex baseband, carries a validity mask, applies a calibrated switching-memory model, handles erasures and reconstructs support without simulator truth.

The public repository intentionally stops before system-specific implementation details.

## Included

- deterministic physical query plans;
- complex coherent multi-receiver slot contract;
- temporal nuisance modeling and SFPTI/GTF inversion;
- bounded local transform recovery;
- global population-free support assembly;
- replayable raw binary captures;
- normalized stress models and acceptance scripts.

## Outside the public tree

- endpoint circuitry or mechanical implementation;
- endpoint size, power or packaging targets;
- carrier/wavelength-specific modulation hardware;
- illumination/collection geometry;
- platform state/control interfaces;
- sensing payload integration;
- deployment-specific propagation calibration;
- field calibration packs and site geometry;
- mission/operational scheduling above the reconstruction layer;
- deployment-specific safety, permission and authorization layers.

This separation is not a claim that those layers do not exist. It defines the reusable boundary of the public reconstruction substrate.

## Why some hardware effects remain visible

A purely offline algorithm could hide switching transients, missing physical states and timing order behind a synthetic matrix. SparseEcho does not, because these effects materially change whether the mathematical transform requested by the recovery algorithm is physically realizable.

The presence of such interfaces should be read as an engineering constraint on the public core, not as documentation of a particular endpoint or deployment.
