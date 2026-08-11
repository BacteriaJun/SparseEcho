from __future__ import annotations

import itertools

import numpy as np

from sparseecho.transforms import gray_time_generators


def phase_increment_from_span(phase_span_cycles: float, bits: int) -> float:
    """Radians per physical slot for a requested first-to-last phase span."""
    n = 1 << bits
    return 2.0 * np.pi * float(phase_span_cycles) / (n - 1)


def doppler_fiber_coefficients(phase_span_cycles: float, bits: int) -> np.ndarray:
    """Exact Walsh coefficients of exp(j*omega*tau_gray(u)).

    Coefficients are indexed by the local Walsh offset delta. The coefficient energy sums to one
    (up to floating-point error) because Walsh characters are orthonormal under the normalized
    finite-group inner product and the input has unit magnitude.
    """
    omega = phase_increment_from_span(phase_span_cycles, bits)
    generators = gray_time_generators(bits)
    theta = omega * (2.0 ** np.arange(bits)) / 2.0
    c = np.cos(theta)
    s = np.sin(theta)
    n = 1 << bits
    out = np.zeros(n, dtype=np.complex128)
    for subset in range(n):
        coeff = 1.0 + 0.0j
        delta = 0
        for i in range(bits):
            if (subset >> i) & 1:
                coeff *= -1j * s[i]
                delta ^= int(generators[i])
            else:
                coeff *= c[i]
        out[delta] = coeff
    return out


def doppler_shell_energies(phase_span_cycles: float, bits: int) -> np.ndarray:
    """Exact temporal-shell energy distribution for constant residual Doppler.

    The shell order is the number of Gray-time generators participating in the Walsh offset.
    Energies form a Poisson-binomial distribution with p_i = sin^2(theta_i).
    """
    omega = phase_increment_from_span(phase_span_cycles, bits)
    theta = omega * (2.0 ** np.arange(bits)) / 2.0
    p = np.sin(theta) ** 2
    pmf = np.array([1.0], dtype=np.float64)
    for prob in p:
        nxt = np.zeros(pmf.size + 1, dtype=np.float64)
        nxt[:-1] += pmf * (1.0 - prob)
        nxt[1:] += pmf * prob
        pmf = nxt
    return pmf


def shell_masks(bits: int, order: int) -> np.ndarray:
    """Walsh offsets belonging to exactly *order* Gray-time generators."""
    generators = gray_time_generators(bits)
    masks: list[int] = []
    for combo in itertools.combinations(range(bits), order):
        value = 0
        for i in combo:
            value ^= int(generators[i])
        masks.append(value)
    return np.array(masks, dtype=np.uint32)


def cumulative_shell_masks(bits: int, max_order: int) -> np.ndarray:
    values = [shell_masks(bits, order) for order in range(max_order + 1)]
    return np.unique(np.concatenate(values)).astype(np.uint32)


def estimate_phase_span_from_first_shell(
    spectrum: np.ndarray,
    anchor: int,
    *,
    bits: int,
    max_abs_span_cycles: float = 0.8,
) -> float:
    """Estimate residual phase span from anchor/first-shell complex ratios.

    For a single component and constant residual Doppler,

        C_i / C_0 = -j tan(theta_i), theta_i = omega * 2^i / 2.

    Multiple receiver channels are combined coherently through a vector inner product. The
    estimate is intended as a local initializer and diagnostic, not as a population search.
    """
    x = np.asarray(spectrum)
    if x.ndim == 1:
        x = x[:, None]
    c0 = x[int(anchor)]
    den = float(np.vdot(c0, c0).real) + 1e-18
    estimates: list[float] = []
    weights: list[float] = []
    for i, mask in enumerate(gray_time_generators(bits)):
        ci = x[int(anchor) ^ int(mask)]
        ratio = np.vdot(c0, ci) / den
        theta = np.arctan(-float(np.imag(ratio)))
        omega = 2.0 * theta / (2.0**i)
        span = omega * ((1 << bits) - 1) / (2.0 * np.pi)
        if np.isfinite(span) and abs(span) <= max_abs_span_cycles * 1.5:
            estimates.append(float(span))
            weights.append(float(min(np.linalg.norm(ci), np.linalg.norm(c0))) + 1e-12)
    if not estimates:
        return 0.0
    values = np.asarray(estimates)
    weights_a = np.asarray(weights)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) + 1e-9
    keep = np.abs(values - median) <= max(3.0 * mad, 0.03)
    if np.count_nonzero(keep) < 2:
        keep[:] = True
    estimate = float(np.average(values[keep], weights=weights_a[keep]))
    return float(np.clip(estimate, -max_abs_span_cycles, max_abs_span_cycles))


def temporal_fiber_coefficients_for_order(
    phase_span_cycles: float, order: np.ndarray
) -> np.ndarray:
    """Walsh coefficients for constant phase rate under an arbitrary execution order.

    This is used for ordering baselines and hardware plans that are not reflected Gray order.
    """
    from sparseecho.transforms import fwht

    physical_order = np.asarray(order, dtype=np.int64)
    n = int(physical_order.size)
    if n < 1 or n & (n - 1):
        raise ValueError("order length must be a power of two")
    if np.unique(physical_order).size != n:
        raise ValueError("order must be a permutation")
    omega = 2.0 * np.pi * float(phase_span_cycles) / max(n - 1, 1)
    t = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    samples = np.exp(1j * omega * t)
    by_query = np.zeros(n, dtype=np.complex128)
    by_query[physical_order] = samples
    return fwht(by_query, normalize=True)
