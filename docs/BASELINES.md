# Baselines and architecture controls

The baseline suite isolates contributions of the current 1.1 architecture. Values below are one fixed seed from the independent forward model and should be read as contribution controls, not confidence intervals. Raw data is in `results/ablation.json`.

## Spatial evidence

| case | precision | recall | candidates |
|---|---:|---:|---:|
| default rank-2 | 1.000 | 1.000 | 461 |
| rank-1 | 1.000 | 1.000 | 461 |
| no spatial gate | 0.113 | 1.000 | 284 |

Removing receiver-subspace evidence preserves recall in this case but creates a large false-positive set. Spatial consistency is therefore a first-class observable in the current architecture, not a cosmetic post-filter.

## View count

| views | precision | recall | candidates |
|---:|---:|---:|---:|
| 12 | 0.800 | 1.000 | 4489 |
| 14 | 1.000 | 1.000 | 5201 |
| 16 | 1.000 | 1.000 | 461 |

The candidate count is not monotonic because structured recovery and validation groups change with the available view set. The table should not be interpreted as a universal scaling law.

## Execution order

| ordering | precision | recall |
|---|---:|---:|
| Gray | 1.000 | 1.000 |
| natural binary | 1.000 | 0.938 |
| random | 1.000 | 0.688 |

Gray and natural binary both give minimal `r`-generator time support for an `r`-bit local aperture. Gray remains the default because adjacent physical query states differ by one bit. Random ordering destroys most of the low-generator temporal structure.

## Fiber-aware local model

| local model | precision | recall |
|---|---:|---:|
| fiber-aware | 1.000 | 1.000 |
| zero-span / fiber-blind | 1.000 | 0.781 |

This control isolates the contribution of temporal-fiber modeling in the recorded scene.

## External sparse-WHT baselines

SparseEcho references SPRIGHT and deterministic sparse-WHT work in `REFERENCES.md`, but the repository does not ship a source-compatible implementation of those algorithms. The internal fiber-blind control is therefore not labeled as SPRIGHT.
