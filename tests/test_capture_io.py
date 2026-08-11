import numpy as np

from sparseecho import compile_query_plan
from sparseecho.io import open_capture_directory, write_capture_directory


def test_capture_v2_roundtrip(tmp_path):
    plan = compile_query_plan(n_views=16)
    n = plan.physical_slots
    slots = np.zeros((n, 3), dtype=np.complex64)
    valid = np.ones(n, dtype=bool)
    ts = np.arange(n, dtype=np.int64) * 1000 + 10_000
    write_capture_directory(
        tmp_path,
        slots,
        valid,
        plan,
        sequence_id=7,
        monotonic_start_ns=int(ts[0]),
        monotonic_end_ns=int(ts[-1]),
        slot_timestamps_ns=ts,
        calibration_epoch="lab-a",
    )
    cap = open_capture_directory(tmp_path)
    assert cap.plan_fingerprint == plan.fingerprint()
    assert cap.sequence_id == 7
    assert cap.slots.shape == (n, 3)
    np.testing.assert_array_equal(cap.slot_timestamps_ns, ts)
