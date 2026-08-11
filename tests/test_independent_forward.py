from pathlib import Path

import numpy as np

from sparseecho import compile_query_plan
from validation.independent_forward import ForwardConfig, generate_capture


def test_independent_forward_does_not_import_inverse_modules():
    source = Path("validation/independent_forward.py").read_text(encoding="utf-8")
    for token in (
        "sparseecho.temporal",
        "sparseecho.transforms",
        "sparseecho.simulator",
        "sparseecho.recovery",
    ):
        assert token not in source


def test_independent_forward_shape_and_timestamps():
    plan = compile_query_plan(n_views=16)
    capture = generate_capture(
        plan.to_dict(),
        ForwardConfig(n_active=4, n_rx=3, weakest_bucket_snr_db=30.0, erasure_rate=0.0),
        seed=3,
    )
    assert capture.slots.shape == (plan.physical_slots, 3)
    assert capture.valid.shape == (plan.physical_slots,)
    assert np.all(np.diff(capture.timestamps_ns) > 0)
    assert capture.truth.identities.size == 4
