from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForwardConfig:
    n_active: int = 32
    n_rx: int = 8
    near_far_power_db: float = 30.0
    weakest_bucket_snr_db: float = 15.0
    max_phase_span_cycles: float = 0.40
    max_quadratic_phase_cycles: float = 0.06
    max_sinusoidal_phase_cycles: float = 0.025
    max_fractional_amplitude_drift: float = 0.05
    cross_view_direction_rho: float = 0.98
    receiver_noise_correlation: float = 0.20
    temporal_noise_ar: float = 0.15
    receiver_gain_std: float = 0.02
    receiver_phase_drift_deg_std: float = 1.0
    erasure_mode: str = "burst"
    erasure_rate: float = 0.02
    mean_burst_slots: float = 4.0
    switch_model: str = "two_pole"
    first_order_memory: float = 0.10
    two_pole_alpha: tuple[float, float] = (0.55, 0.82)
    timing_jitter_fraction: float = 0.002


@dataclass(frozen=True)
class ForwardTruth:
    identities: np.ndarray
    receiver_calibration: np.ndarray


@dataclass(frozen=True)
class ForwardCapture:
    slots: np.ndarray
    valid: np.ndarray
    timestamps_ns: np.ndarray
    truth: ForwardTruth


def _unique_u32(rng: np.random.Generator, count: int) -> np.ndarray:
    values: set[int] = set()
    while len(values) < count:
        values.add(int(rng.integers(0, 2**32, dtype=np.uint64)))
    return np.asarray(sorted(values), dtype=np.uint32)


def _parity(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.uint64).copy()
    x ^= x >> 32
    x ^= x >> 16
    x ^= x >> 8
    x ^= x >> 4
    x &= 0xF
    lut = np.asarray([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0], dtype=np.uint8)
    return lut[x.astype(np.int64)]


def _burst_mask(rng: np.random.Generator, n: int, rate: float, mean_burst: float) -> np.ndarray:
    valid = np.ones(n, dtype=bool)
    target = int(round(rate * n))
    erased = 0
    while erased < target:
        start = int(rng.integers(0, n))
        length = max(1, int(rng.geometric(1.0 / max(mean_burst, 1.0))))
        stop = min(n, start + length)
        before = np.count_nonzero(~valid[start:stop])
        valid[start:stop] = False
        erased += (stop - start) - before
    return valid


def _settle(x: np.ndarray, config: ForwardConfig) -> np.ndarray:
    if config.switch_model == "first_order":
        eps = float(config.first_order_memory)
        y = np.empty_like(x, dtype=np.complex128)
        state = np.zeros(x.shape[1], dtype=np.complex128)
        for i in range(x.shape[0]):
            state = (1.0 - eps) * x[i] + eps * state
            y[i] = state
        return y
    if config.switch_model != "two_pole":
        raise ValueError("switch_model must be 'first_order' or 'two_pole'")
    a1, a2 = (float(v) for v in config.two_pole_alpha)
    s1 = np.zeros(x.shape[1], dtype=np.complex128)
    s2 = np.zeros(x.shape[1], dtype=np.complex128)
    y = np.empty_like(x, dtype=np.complex128)
    # Two weak settling tails around a dominant direct state. This is intentionally not the
    # inverse's two-tap FIR, but it preserves the current query state instead of low-pass filtering
    # the whole Walsh sequence.
    direct, tail1, tail2 = 0.88, 0.08, 0.04
    for i in range(x.shape[0]):
        s1 = a1 * s1 + (1.0 - a1) * x[i]
        s2 = a2 * s2 + (1.0 - a2) * s1
        y[i] = direct * x[i] + tail1 * s1 + tail2 * s2
    return y


def _correlated_noise(
    rng: np.random.Generator,
    shape: tuple[int, int],
    sigma: float,
    receiver_correlation: float,
    temporal_ar: float,
) -> np.ndarray:
    n, n_rx = shape
    rho = float(np.clip(receiver_correlation, 0.0, 0.99))
    cov = (1.0 - rho) * np.eye(n_rx) + rho * np.ones((n_rx, n_rx))
    chol = np.linalg.cholesky(cov)
    white = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2.0)
    noise = white @ chol.T
    ar = float(np.clip(temporal_ar, -0.95, 0.95))
    if ar != 0.0:
        gain = np.sqrt(max(1.0 - ar * ar, 1e-9))
        for i in range(1, n):
            noise[i] = ar * noise[i - 1] + gain * noise[i]
    return sigma * noise


def generate_capture(plan: dict, config: ForwardConfig, *, seed: int) -> ForwardCapture:
    """Generate physical slots from serialized challenge masks.

    This module deliberately does not import SparseEcho's temporal, transform, simulator or
    recovery code. It is suitable for model-mismatch and replay validation, not as a field model.
    """
    rng = np.random.default_rng(seed)
    challenges = np.asarray(plan["challenges"], dtype=np.uint64)
    n_views = int(plan["n_views"])
    slots_per_view = int(plan["slots_per_view"])
    if challenges.size != n_views * slots_per_view:
        raise ValueError("independent forward expects a single-pass serialized query plan")
    if config.erasure_mode not in ("iid", "burst"):
        raise ValueError("erasure_mode must be 'iid' or 'burst'")

    k = int(config.n_active)
    n_rx = int(config.n_rx)
    identities = _unique_u32(rng, k)

    base_direction = (rng.normal(size=(k, n_rx)) + 1j * rng.normal(size=(k, n_rx))) / np.sqrt(2.0)
    base_direction /= np.maximum(np.linalg.norm(base_direction, axis=1, keepdims=True), 1e-12)
    power_db = np.linspace(0.0, -float(config.near_far_power_db), k)
    rng.shuffle(power_db)
    amplitude = 10.0 ** (power_db / 20.0)
    amplitude_scale = amplitude / max(float(np.min(amplitude)), 1e-12)

    rx_gain = 1.0 + rng.normal(0.0, config.receiver_gain_std, size=n_rx)
    rx_phase = np.deg2rad(rng.normal(0.0, config.receiver_phase_drift_deg_std, size=n_rx))
    receiver_calibration = rx_gain * np.exp(1j * rx_phase)

    ideal = np.zeros((challenges.size, n_rx), dtype=np.complex128)
    direction = base_direction.copy()
    rho = float(np.clip(config.cross_view_direction_rho, 0.0, 1.0))
    t_local = np.linspace(-0.5, 0.5, slots_per_view)

    for view in range(n_views):
        sl = slice(view * slots_per_view, (view + 1) * slots_per_view)
        if view > 0 and rho < 1.0:
            innovation = (rng.normal(size=(k, n_rx)) + 1j * rng.normal(size=(k, n_rx))) / np.sqrt(2.0)
            innovation /= np.maximum(np.linalg.norm(innovation, axis=1, keepdims=True), 1e-12)
            direction = rho * direction + np.sqrt(max(1.0 - rho * rho, 0.0)) * innovation
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
        channel = direction * amplitude_scale[:, None]

        q = challenges[sl]
        parity = _parity(np.bitwise_and(q[:, None], identities.astype(np.uint64)[None, :]))
        signs = 1.0 - 2.0 * parity.astype(float)
        span = rng.uniform(-config.max_phase_span_cycles, config.max_phase_span_cycles, size=k)
        quad = rng.uniform(-config.max_quadratic_phase_cycles, config.max_quadratic_phase_cycles, size=k)
        sine = rng.uniform(-config.max_sinusoidal_phase_cycles, config.max_sinusoidal_phase_cycles, size=k)
        phase0 = rng.uniform(-np.pi, np.pi, size=k)
        amp_drift = rng.uniform(
            -config.max_fractional_amplitude_drift,
            config.max_fractional_amplitude_drift,
            size=k,
        )
        phase = (
            phase0[:, None]
            + 2.0 * np.pi * span[:, None] * t_local[None, :]
            + 2.0 * np.pi * quad[:, None] * t_local[None, :] ** 2
            + 2.0 * np.pi * sine[:, None] * np.sin(2.0 * np.pi * t_local[None, :])
        )
        temporal = (1.0 + amp_drift[:, None] * t_local[None, :]) * np.exp(1j * phase)
        ideal[sl] = np.einsum("bk,kb,kr->br", signs, temporal, channel, optimize=True)

    ideal *= receiver_calibration[None, :]
    settled = _settle(ideal, config)
    weakest_energy = float(np.min(amplitude_scale**2))
    snr_linear = 10.0 ** (float(config.weakest_bucket_snr_db) / 10.0)
    sigma = np.sqrt(slots_per_view * weakest_energy / max(n_rx * snr_linear, 1e-12))
    observed = settled + _correlated_noise(
        rng,
        settled.shape,
        sigma,
        config.receiver_noise_correlation,
        config.temporal_noise_ar,
    )

    if config.erasure_mode == "iid":
        valid = rng.random(challenges.size) >= float(config.erasure_rate)
    else:
        valid = _burst_mask(rng, challenges.size, float(config.erasure_rate), config.mean_burst_slots)

    nominal_step_ns = 10_000
    jitter = rng.normal(0.0, config.timing_jitter_fraction, size=challenges.size - 1)
    steps = np.maximum(1, np.rint(nominal_step_ns * (1.0 + jitter)).astype(np.int64))
    timestamps = np.empty(challenges.size, dtype=np.int64)
    timestamps[0] = 1_000_000_000
    timestamps[1:] = timestamps[0] + np.cumsum(steps)

    return ForwardCapture(
        observed.astype(np.complex64),
        valid,
        timestamps,
        ForwardTruth(identities, receiver_calibration.astype(np.complex64)),
    )
