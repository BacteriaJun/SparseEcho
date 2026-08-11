# Contributing

Changes should preserve the public hardware-neutral boundary and include a reproducible test for any behavior that affects runtime decisions.

Before submitting a change:

```bash
make check
```

For algorithm changes, include the relevant baseline or failure-region result. For runtime changes, include at least one contract/state-machine test. Avoid embedding deployment-specific geometry, endpoint implementation details or field calibration constants in the public tree.
