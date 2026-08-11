# Operations

## Runtime limits

`RuntimeLimits` provides hard bounds at the public integration boundary:

- maximum invalid-slot fraction;
- maximum consecutive invalid run;
- maximum slot-timing error;
- maximum reacquisition attempts;
- allowed receiver-count range.

A plan mismatch, sequence regression or malformed timing record is a hard fault. Excess erasure or timing jitter is recoverable only within the bounded reacquisition policy.

## Fault codes

The reference runtime emits structured codes including:

```text
plan_mismatch
sequence_regression
slot_count_mismatch
invalid_timing
timing_jitter
too_many_erasures
erasure_burst
receiver_count
calibration
acquisition
reconstruction
reacquire_exhausted
```

The fault vocabulary is intentionally small. Deployment layers can map these codes into their own alarm, retry or isolation policies.

## Telemetry

The JSONL sink records one event per line. It is suitable for deterministic incident replay and simple log shipping. At minimum, retain:

- plan fingerprint;
- capture sequence ID;
- calibration epoch;
- invalid fraction;
- reconstruction latency;
- identity count;
- measured fiber tail;
- acquisition decision.

Do not store deployment secrets or executable calibration objects in telemetry metadata.

## Deterministic replay

Raw capture replay is a first-class operations path. Archive `metadata.json`, `capture.c64`, `valid.u8` and optional `timestamps.i64` together with the calibration epoch and software build identifier. This allows reconstruction behavior to be reproduced without the original acquisition process.

## Health interpretation

A successful reconstruction is not equivalent to a validated physical deployment. Treat repeated reacquisition, proposal-budget saturation, rising cross-view drift and persistent erasures as health signals even when a support result is produced.
