from __future__ import annotations

import numpy as np

from sparseecho.planning import QueryPlan

from .contracts import CaptureFrame, FaultCode, RuntimeFault, RuntimeLimits


def _max_false_run(valid: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(valid, dtype=bool):
        if value:
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def validate_frame(
    frame: CaptureFrame,
    plan: QueryPlan,
    limits: RuntimeLimits,
    *,
    previous_sequence_id: int | None = None,
) -> RuntimeFault | None:
    x = np.asarray(frame.slots)
    valid = np.asarray(frame.valid, dtype=bool)
    expected = plan.physical_slots * int(frame.pass_count)
    if frame.plan_fingerprint != plan.fingerprint():
        return RuntimeFault(FaultCode.PLAN_MISMATCH, "capture plan fingerprint differs from runtime plan", False)
    if previous_sequence_id is not None and frame.sequence_id <= previous_sequence_id:
        return RuntimeFault(FaultCode.SEQUENCE_REGRESSION, "capture sequence_id is not increasing", False)
    if x.ndim not in (1, 2) or x.shape[0] != expected or valid.shape != (expected,):
        return RuntimeFault(FaultCode.SLOT_COUNT_MISMATCH, "capture dimensions do not match the compiled plan", False)
    if frame.pass_count not in (1, 2, 4) or not np.isfinite(frame.aperture_scale) or frame.aperture_scale <= 0.0:
        return RuntimeFault(FaultCode.INVALID_TIMING, "pass_count or aperture_scale is invalid", False)
    valid_samples = x[valid] if x.ndim == 1 else x[valid, :]
    if not np.all(np.isfinite(valid_samples)):
        return RuntimeFault(FaultCode.NONFINITE_INPUT, "valid capture slots contain non-finite samples", False)
    if not limits.min_receivers <= frame.n_rx <= limits.max_receivers:
        return RuntimeFault(FaultCode.RECEIVER_COUNT, "receiver count is outside configured runtime limits", False)
    if frame.monotonic_end_ns <= frame.monotonic_start_ns:
        return RuntimeFault(FaultCode.INVALID_TIMING, "capture monotonic interval is not positive", False)
    invalid_fraction = float(1.0 - np.mean(valid))
    if invalid_fraction > limits.max_invalid_fraction:
        return RuntimeFault(
            FaultCode.TOO_MANY_ERASURES,
            f"invalid slot fraction {invalid_fraction:.4f} exceeds {limits.max_invalid_fraction:.4f}",
            True,
        )
    run = _max_false_run(valid)
    if run > limits.max_consecutive_invalid:
        return RuntimeFault(
            FaultCode.ERASE_BURST,
            f"invalid slot run {run} exceeds {limits.max_consecutive_invalid}",
            True,
        )
    if frame.slot_timestamps_ns is not None:
        ts = np.asarray(frame.slot_timestamps_ns, dtype=np.int64)
        if ts.shape != (expected,) or np.any(np.diff(ts) <= 0):
            return RuntimeFault(FaultCode.INVALID_TIMING, "slot timestamps are missing, duplicated or non-monotonic", False)
        if ts[0] < frame.monotonic_start_ns or ts[-1] > frame.monotonic_end_ns:
            return RuntimeFault(FaultCode.INVALID_TIMING, "slot timestamps fall outside the capture interval", False)
        dt = np.diff(ts).astype(float)
        nominal = float(np.median(dt))
        if nominal <= 0:
            return RuntimeFault(FaultCode.INVALID_TIMING, "slot cadence is not positive", False)
        jitter = float(np.max(np.abs(dt - nominal)) / nominal)
        if jitter > limits.max_slot_timing_error_fraction:
            return RuntimeFault(
                FaultCode.TIMING_JITTER,
                f"slot timing error fraction {jitter:.4f} exceeds {limits.max_slot_timing_error_fraction:.4f}",
                True,
            )
    return None
