import json

import numpy as np

from sparseecho.calibration import JsonCalibrationStore


def test_json_calibration_store(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "epoch": "2026-08-a",
                "switch_fir_taps": [0.88, 0.09, 0.03],
                "receiver_gain": [1.0, 0.98],
                "receiver_phase_deg": [0.0, 2.0],
            }
        ),
        encoding="utf-8",
    )
    snap = JsonCalibrationStore(path).snapshot(n_rx=2)
    assert snap.epoch == "2026-08-a"
    assert snap.switch_fir_taps == (0.88, 0.09, 0.03)
    assert snap.receiver_complex_gain.shape == (2,)
    assert np.isfinite(snap.receiver_complex_gain).all()
