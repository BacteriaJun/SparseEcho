from __future__ import annotations

import numpy as np

import sparseecho
from sparseecho import ApertureBudget, EngineConfig, SparseEchoEngine, compile_query_plan
from sparseecho.config import SceneConfig
from sparseecho.io import open_capture_directory, write_capture_directory
from sparseecho.metrics import detection_metrics
from sparseecho.simulator import simulate_capture
from sparseecho.temporal import (
    apply_first_order_memory,
    deconvolve_first_order_memory,
    doppler_fiber_coefficients,
    doppler_shell_energies,
    moment_weights,
    repair_erased_gray_slots,
)
from sparseecho.transforms import (
    centered_gray_rank,
    fwht,
    gray_code,
    gray_inverse,
    gray_order,
    gray_time_generators,
    hadamard_matrix,
    walsh_character,
)


def test_version_is_1_0_0() -> None:
    assert sparseecho.__version__ == "1.0"


def test_default_query_plan_has_3584_states() -> None:
    plan = compile_query_plan()
    assert plan.identity_bits == 32
    assert plan.local_bits == 8
    assert plan.n_views == 14
    assert plan.slots_per_view == 256
    assert plan.physical_slots == 3584


def test_gray_order_is_bijective_and_one_bit_transition() -> None:
    order = gray_order(8)
    assert np.unique(order).size == 256
    for a, b in zip(order[:-1], order[1:], strict=False):
        x = int(a) ^ int(b)
        assert x and (x & (x - 1)) == 0
    assert all(gray_inverse(int(gray_code(i))) == i for i in range(256))


def test_centered_gray_time_has_minimal_r_support() -> None:
    r = 8
    u = np.arange(1 << r, dtype=np.uint32)
    tau = centered_gray_rank(u, r)
    spectrum = fwht(tau, normalize=True)
    assert np.count_nonzero(np.abs(spectrum) > 1e-12) == r
    assert len(np.unique(tau)) == 1 << r
    assert len(np.unique(gray_time_generators(r))) == r


def test_exact_doppler_fiber_matches_direct_transform() -> None:
    r = 8
    n = 1 << r
    span = 0.4
    omega = 2 * np.pi * span / (n - 1)
    order = gray_order(r)
    t = np.arange(n) - (n - 1) / 2
    samples = np.exp(1j * omega * t)
    by_query = np.zeros(n, dtype=np.complex128)
    by_query[order] = samples
    direct = fwht(by_query, normalize=True)
    model = doppler_fiber_coefficients(span, r)
    assert np.linalg.norm(direct - model) < 1e-11


def test_doppler_shell_energy_is_probability_distribution() -> None:
    energies = doppler_shell_energies(0.4, 8)
    assert np.all(energies >= 0)
    assert abs(float(np.sum(energies)) - 1.0) < 1e-12
    assert float(np.sum(energies[4:])) < 1e-5


def test_aperture_budget_inverts_tail_law() -> None:
    budget = ApertureBudget(local_bits=8, modeled_shell_order=3, leakage_budget=1e-5)
    span = budget.max_phase_span_cycles()
    assert 0.40 < span < 0.43
    assert budget.tail_energy(span) <= 1.00001e-5


def test_hash_query_character_identity() -> None:
    plan = compile_query_plan()
    rng = np.random.default_rng(1)
    q = int(rng.integers(0, 2**32, dtype=np.uint64))
    for view in plan.views[:3]:
        z = int(view.hash_ids(np.array([q], dtype=np.uint32))[0])
        for u in [0, 1, 7, 31, 93, 255]:
            a = view.challenge(u)
            left = walsh_character(a, q)
            right = walsh_character(u, z)
            assert float(left) == float(right)


def test_virtual_time_moment_weights_reproduce_cubic() -> None:
    times = np.array([-2.0, -0.5, 0.5, 2.0])
    weights = moment_weights(times, degree=3)
    for degree in range(4):
        values = times**degree
        expected = 1.0 if degree == 0 else 0.0
        assert abs(float(weights @ values) - expected) < 1e-12


def test_first_order_switch_model_roundtrip() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(32, 3)) + 1j * rng.normal(size=(32, 3))
    y = apply_first_order_memory(x, 0.1)
    recovered = deconvolve_first_order_memory(y, 0.1)
    assert np.linalg.norm(recovered - x) / np.linalg.norm(x) < 1e-12


def test_erasure_repair_single_bucket() -> None:
    r = 5
    n = 1 << r
    order = gray_order(r)
    h = hadamard_matrix(r)
    bucket = 19
    t = (np.arange(n) - (n - 1) / 2) / (n - 1)
    slots = (h[order, bucket] * (1 + 0.15j * t))[:, None]
    valid = np.ones(n, dtype=bool)
    valid[[3, 9, 20]] = False
    observed = slots.copy()
    observed[~valid] = 0
    repaired = repair_erased_gray_slots(
        observed,
        valid,
        local_order=order,
        candidate_buckets=np.array([bucket], dtype=np.uint16),
        degree=1,
    )
    assert np.max(np.abs(repaired[~valid] - slots[~valid])) < 1e-3


def test_high_snr_engine_recovers_sparse_support() -> None:
    plan = compile_query_plan()
    scene = SceneConfig(
        n_active=8,
        n_rx=4,
        near_far_power_db=12,
        weakest_fiber_snr_db=18,
        max_phase_span_cycles_per_view=0.15,
        max_quadratic_phase_cycles=0.01,
        slot_erasure_rate=0.0,
        switch_memory_coefficient=0.05,
    )
    capture = simulate_capture(plan, scene, seed=9)
    config = EngineConfig(
        switch_memory_coefficient=0.05,
        view_max_components=28,
        spatial_consistency_threshold=0.45,
    )
    result = SparseEchoEngine(config, plan).process_capture(capture.slots, capture.valid)
    metrics = detection_metrics(capture.truth.identities, result.identities)
    assert metrics.precision >= 0.99
    assert metrics.recall >= 0.95


def test_binary_replay_path(tmp_path) -> None:
    plan = compile_query_plan()
    scene = SceneConfig(
        n_active=4,
        n_rx=3,
        near_far_power_db=6,
        weakest_fiber_snr_db=24,
        max_phase_span_cycles_per_view=0.05,
        max_quadratic_phase_cycles=0.0,
        slot_erasure_rate=0.0,
        switch_memory_coefficient=0.0,
    )
    capture = simulate_capture(plan, scene, seed=17)
    root = write_capture_directory(tmp_path / "capture", capture.slots, capture.valid, plan)
    replay = open_capture_directory(root)
    result = SparseEchoEngine(EngineConfig(switch_memory_coefficient=0.0), plan).process_capture(
        replay.slots, replay.valid
    )
    metrics = detection_metrics(capture.truth.identities, result.identities)
    assert replay.slots.dtype == np.complex64
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
