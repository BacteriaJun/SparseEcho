# Changelog

## 1.0 — Initial public release

Initial public release of the SparseEcho reconstruction core.

The release centers on **Spectrally Factorized Physical-Time Inversion (SFPTI)** and ships:

- Gray Temporal Fiber (GTF) query scheduling and exact residual-Doppler fiber models;
- a physical query compiler for 32-bit address spaces without population-scale dictionaries;
- hardware-neutral complex-baseband slot ingestion;
- switch-settling and slot-erasure handling;
- multi-view support reconstruction and consistency filtering;
- Virtual-Time Query Compilation (VTQC) as a high-dynamics fallback;
- synthetic physical-channel and binary-replay acceptance paths;
- deterministic tests for the core algebraic invariants and scheduling properties.
