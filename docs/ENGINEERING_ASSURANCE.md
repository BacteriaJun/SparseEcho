# Engineering assurance

SparseEcho separates algorithm regression, model-mismatch validation, integration tests and deployment evidence.

## Evidence tiers

1. **Algebra and unit tests** — transform, schedule and fiber identities.
2. **Matched simulator** — deterministic regression and examples.
3. **Independent forward** — release validation with a separately implemented forward path and deliberate mismatch.
4. **Binary/runtime replay** — file contract, calibration and state-machine integration.
5. **Hardware-in-the-loop / field capture** — deployment evidence; not included in the public release.

A result from a lower tier is not described as evidence for a higher tier.

## Release gates

The source release is checked for:

- package compilation;
- deterministic unit and runtime tests;
- plan fingerprint stability;
- malformed-frame rejection;
- continuous switch correction across view boundaries;
- CFAR null behavior;
- independent-forward import separation;
- runtime bounded-reacquisition behavior;
- capture replay compatibility;
- stale internal/application-specific names in the public tree.

## Model mismatch

The validation forward path can vary switch dynamics, cross-view receiver direction, erasure process, receiver correlation, temporal noise, calibration drift and non-polynomial phase. Stored sweeps are operating-envelope maps, not confidence intervals.

The largest current sensitivity remains strong cross-view receiver-subspace rotation. This is documented as an operating limit rather than hidden behind the nominal profile.

## Threshold classes

Runtime thresholds are kept in three categories:

- analytic budgets, such as temporal-fiber leakage and CFAR false-alarm target;
- resource limits, such as local proposal and pre-candidate budgets;
- deployment-tuned thresholds, such as view support and receiver-subspace consistency.

Only the first two categories are intended to transfer directly between integrations.
