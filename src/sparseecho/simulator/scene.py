from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sparseecho.config import SceneConfig
from sparseecho.planning import QueryPlan
from sparseecho.temporal import apply_fir_memory
from sparseecho.transforms import hadamard_matrix


@dataclass(frozen=True)
class SyntheticTruth:
    identities: np.ndarray
    local_buckets: np.ndarray
    base_channels: np.ndarray


@dataclass(frozen=True)
class SyntheticCapture:
    slots: np.ndarray
    valid: np.ndarray
    truth: SyntheticTruth


def _unique_u32(rng: np.random.Generator, count: int) -> np.ndarray:
    values: set[int] = set()
    while len(values) < count:
        values.add(int(rng.integers(0, 2**32, dtype=np.uint64)))
    return np.asarray(sorted(values), dtype=np.uint32)


def simulate_capture(plan: QueryPlan, config: SceneConfig, *, seed: int) -> SyntheticCapture:
    """Matched synthetic scene used by examples and unit tests.

    Release robustness benchmarks use ``validation.independent_forward`` instead.
    """
    if plan.identity_bits != 32 or plan.local_bits != 8:
        raise ValueError("matched scene currently targets the 32-bit / 8-bit profile")
    rng = np.random.default_rng(seed)
    k = int(config.n_active)
    n_rx = int(config.n_rx)
    b = plan.slots_per_view
    hmat = hadamard_matrix(plan.local_bits)
    order = plan.local_order.astype(np.int64)

    identities = _unique_u32(rng, k)
    local_buckets = np.stack([view.hash_ids(identities) for view in plan.views], axis=0)

    spatial = (rng.normal(size=(k, n_rx)) + 1j * rng.normal(size=(k, n_rx))) / np.sqrt(2.0)
    spatial /= np.maximum(np.linalg.norm(spatial, axis=1, keepdims=True), 1e-12)
    power_db = np.linspace(0.0, -float(config.near_far_power_db), k)
    rng.shuffle(power_db)
    amplitude = 10.0 ** (power_db / 20.0)
    base_channels = spatial * amplitude[:, None]
    base_channels /= np.min(np.linalg.norm(base_channels, axis=1))

    snr_linear = 10.0 ** (float(config.weakest_fiber_snr_db) / 10.0)
    weakest_energy = float(np.min(np.sum(np.abs(base_channels) ** 2, axis=1)))
    sigma = np.sqrt(b * weakest_energy / (n_rx * snr_linear))

    ideal_all = np.zeros((plan.physical_slots, n_rx), dtype=np.complex128)
    valid_all = np.ones(plan.physical_slots, dtype=bool)
    t = np.arange(b, dtype=np.float64)
    centered = (t - (b - 1) / 2.0) / max(b - 1, 1)

    for view_index, view in enumerate(plan.views):
        z = local_buckets[view_index].astype(np.int64)
        signs = hmat[order[:, None], z[None, :]]
        span = rng.uniform(
            -config.max_phase_span_cycles_per_view,
            config.max_phase_span_cycles_per_view,
            size=k,
        )
        curvature = rng.uniform(
            -config.max_quadratic_phase_cycles,
            config.max_quadratic_phase_cycles,
            size=k,
        )
        phase_offset = rng.uniform(-np.pi, np.pi, size=k)
        phase = (
            phase_offset[:, None]
            + 2.0 * np.pi * span[:, None] * centered[None, :]
            + 2.0 * np.pi * curvature[:, None] * centered[None, :] ** 2
        )
        amp_drift = rng.uniform(
            -config.max_fractional_amplitude_drift,
            config.max_fractional_amplitude_drift,
            size=k,
        )
        gain = 1.0 + amp_drift[:, None] * centered[None, :]
        temporal = gain * np.exp(1j * phase)
        sl = plan.slice_for_view(view_index)
        ideal_all[sl] = np.einsum("bk,kb,kr->br", signs, temporal, base_channels, optimize=True)
        valid_all[sl] = rng.random(b) >= float(config.slot_erasure_rate)

    settled = apply_fir_memory(ideal_all, config.switch_fir_taps)
    noise = sigma * (rng.normal(size=settled.shape) + 1j * rng.normal(size=settled.shape)) / np.sqrt(2.0)
    observed = settled + noise
    return SyntheticCapture(
        slots=observed.astype(np.complex64),
        valid=valid_all,
        truth=SyntheticTruth(identities, local_buckets, base_channels.astype(np.complex64)),
    )
