from sparseecho import EngineConfig, SparseEchoEngine, compile_query_plan
from sparseecho.config import SceneConfig
from sparseecho.metrics import detection_metrics
from sparseecho.simulator import simulate_capture


def test_matched_smoke_recovers_small_scene():
    plan = compile_query_plan(n_views=16)
    scene = SceneConfig(
        n_active=8,
        n_rx=8,
        near_far_power_db=10.0,
        weakest_fiber_snr_db=20.0,
        max_phase_span_cycles_per_view=0.20,
        slot_erasure_rate=0.0,
    )
    capture = simulate_capture(plan, scene, seed=5)
    result = SparseEchoEngine(EngineConfig(n_views=16), plan).process_capture(capture.slots, capture.valid)
    metrics = detection_metrics(capture.truth.identities, result.identities)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
