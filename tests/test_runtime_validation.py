import numpy as np

from sparseecho import CaptureFrame, RuntimeLimits, compile_query_plan
from sparseecho.runtime import FaultCode, validate_frame


def _frame(plan, *, fingerprint=None, valid=None, timestamps=None, seq=1):
    n = plan.physical_slots
    if valid is None:
        valid = np.ones(n, dtype=bool)
    if timestamps is None:
        timestamps = np.arange(n, dtype=np.int64) * 1000 + 1_000_000
    return CaptureFrame(
        slots=np.zeros((n, 4), dtype=np.complex64),
        valid=valid,
        plan_fingerprint=fingerprint or plan.fingerprint(),
        sequence_id=seq,
        monotonic_start_ns=int(timestamps[0]),
        monotonic_end_ns=int(timestamps[-1]),
        slot_timestamps_ns=timestamps,
    )


def test_runtime_accepts_well_formed_frame():
    plan = compile_query_plan(n_views=16)
    assert validate_frame(_frame(plan), plan, RuntimeLimits()) is None


def test_runtime_rejects_plan_mismatch():
    plan = compile_query_plan(n_views=16)
    fault = validate_frame(_frame(plan, fingerprint="0" * 64), plan, RuntimeLimits())
    assert fault is not None and fault.code is FaultCode.PLAN_MISMATCH


def test_runtime_detects_burst_erasure():
    plan = compile_query_plan(n_views=16)
    valid = np.ones(plan.physical_slots, dtype=bool)
    valid[100:150] = False
    fault = validate_frame(_frame(plan, valid=valid), plan, RuntimeLimits(max_consecutive_invalid=16))
    assert fault is not None and fault.code is FaultCode.ERASE_BURST


def test_runtime_detects_timing_jitter():
    plan = compile_query_plan(n_views=16)
    ts = np.arange(plan.physical_slots, dtype=np.int64) * 1000 + 1_000_000
    ts[100:] += 500
    fault = validate_frame(_frame(plan, timestamps=ts), plan, RuntimeLimits(max_slot_timing_error_fraction=0.1))
    assert fault is not None and fault.code is FaultCode.TIMING_JITTER
