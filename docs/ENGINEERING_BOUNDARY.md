# Public engineering boundary

SparseEcho is published as an application-neutral reconstruction component extracted from a larger engineering system.

## Included

- deterministic query planning and plan fingerprints;
- acquisition/reconstruction/reacquisition state machine;
- slot-level coherent complex-baseband contract;
- calibration snapshot interface;
- timing and erasure validation;
- SFPTI/GTF temporal inversion;
- bounded local proposal generation;
- population-free global support reconstruction;
- runtime faults and telemetry;
- file, C and protobuf integration contracts;
- independent synthetic validation and deterministic replay.

## Outside the public tree

- endpoint implementation and packaging;
- carrier- or wavelength-specific front ends;
- deployment geometry and site models;
- platform motion/control interfaces;
- payload integration;
- field calibration assets;
- operational scheduling above reconstruction;
- deployment authorization, safety and policy layers.

The interfaces visible in this repository exist because switching, timing, calibration and acquisition quality materially affect the inverse. They should not be read as a description of a particular endpoint.
