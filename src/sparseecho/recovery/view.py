from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from sparseecho.temporal import doppler_fiber_coefficients, temporal_fiber_coefficients_for_order
from sparseecho.transforms import fwht


@dataclass(frozen=True)
class ViewComponent:
    bucket: int
    phase_span_cycles: float
    score: float


@dataclass(frozen=True)
class ViewDecodeResult:
    buckets: np.ndarray
    components: tuple[ViewComponent, ...]
    noise_floor: float
    detection_threshold: float
    residual_energy: float


def _gamma_quantile_wilson_hilferty(shape: float, probability: float) -> float:
    """Approximate a Gamma(shape, scale=1) quantile without a SciPy dependency."""
    p = min(max(float(probability), 1e-10), 1.0 - 1e-10)
    k = float(shape)
    z = NormalDist().inv_cdf(p)
    base = 1.0 - 1.0 / (9.0 * k) + z / (3.0 * np.sqrt(k))
    return max(k * base**3, 1e-12)


class GrayFiberViewDecoder:
    """Matched Gray-fiber proposal detector for one local hash aperture.

    This stage only proposes local buckets.  It never scans the global identity space.  Detection
    uses a family-wise CFAR threshold over the configured bucket/phase hypotheses; ``max_components``
    is a hard compute guard, not the statistical stopping rule.
    """

    def __init__(
        self,
        *,
        local_bits: int = 8,
        max_components: int = 48,
        max_abs_phase_span_cycles: float = 0.65,
        phase_grid: int = 33,
        false_alarm_rate: float = 0.02,
        noise_quantile: float = 0.40,
        query_order: np.ndarray | None = None,
    ) -> None:
        self.local_bits = int(local_bits)
        self.bucket_count = 1 << self.local_bits
        self.max_components = int(max_components)
        self.max_abs_phase_span_cycles = float(max_abs_phase_span_cycles)
        self.false_alarm_rate = float(false_alarm_rate)
        self.noise_quantile = float(noise_quantile)
        if not 0.0 < self.false_alarm_rate < 1.0:
            raise ValueError("false_alarm_rate must be in (0,1)")
        if not 0.0 < self.noise_quantile < 0.5:
            raise ValueError("noise_quantile must be in (0,0.5)")
        if self.max_abs_phase_span_cycles == 0.0:
            self._spans = np.array([0.0], dtype=np.float64)
        else:
            self._spans = np.linspace(
                -self.max_abs_phase_span_cycles,
                self.max_abs_phase_span_cycles,
                int(phase_grid),
            )
        if query_order is None:
            self._atoms = np.stack(
                [doppler_fiber_coefficients(span, self.local_bits) for span in self._spans], axis=0
            )
        else:
            order = np.asarray(query_order, dtype=np.int64)
            self._atoms = np.stack(
                [temporal_fiber_coefficients_for_order(span, order) for span in self._spans], axis=0
            )
        # For XOR correlation c[b] = sum_j conj(atom[j]) x[j xor b],
        # H(c)=H(conj(atom))*H(x).  Cache the atom transforms once.
        self._atom_walsh = fwht(np.conjugate(self._atoms), axis=1, normalize=False)
        self._xor_index = np.arange(self.bucket_count, dtype=np.int64)

    def _noise_statistics(self, energy: np.ndarray, n_rx: int) -> tuple[float, float, float]:
        q = float(np.quantile(energy, self.noise_quantile))
        gamma_q = _gamma_quantile_wilson_hilferty(float(n_rx), self.noise_quantile)
        per_rx_variance = q / gamma_q
        noise_mean_energy = float(n_rx) * per_rx_variance
        # Phase hypotheses are correlated samples of one smooth fiber family. Use a bucket-wise
        # Bonferroni tail and verify null behavior separately in benchmarks/cfar_null.py.
        trials = self.bucket_count
        tail = self.false_alarm_rate / float(trials)
        gamma_threshold = _gamma_quantile_wilson_hilferty(float(n_rx), 1.0 - tail)
        threshold = per_rx_variance * gamma_threshold
        return noise_mean_energy, per_rx_variance, threshold

    def _matched_channels(self, residual: np.ndarray) -> np.ndarray:
        hx = fwht(residual, axis=0, normalize=False)
        products = self._atom_walsh[:, :, None] * hx[None, :, :]
        return fwht(products, axis=1, normalize=False) / float(self.bucket_count)

    def decode(self, spectrum: np.ndarray) -> ViewDecodeResult:
        x = np.asarray(spectrum)
        if x.shape[0] != self.bucket_count:
            raise ValueError("spectrum length does not match local aperture")
        if x.ndim == 1:
            x = x[:, None]
        residual = x.astype(np.complex128, copy=True)
        base_energy = np.sum(np.abs(x) ** 2, axis=1)
        noise_floor, _, threshold = self._noise_statistics(base_energy, x.shape[1])
        components: list[ViewComponent] = []
        used_buckets: set[int] = set()

        for _ in range(self.max_components):
            channels = self._matched_channels(residual)
            scores = np.sum(np.abs(channels) ** 2, axis=2)
            if used_buckets:
                scores[:, np.fromiter(used_buckets, dtype=np.int64)] = -np.inf
            flat = int(np.argmax(scores))
            span_index, bucket = np.unravel_index(flat, scores.shape)
            score = float(scores[span_index, bucket])
            if score < threshold:
                break
            channel = channels[span_index, bucket]
            residual[self._xor_index ^ int(bucket)] -= self._atoms[span_index, :, None] * channel[None, :]
            used_buckets.add(int(bucket))
            components.append(
                ViewComponent(
                    bucket=int(bucket),
                    phase_span_cycles=float(self._spans[span_index]),
                    score=score,
                )
            )

        return ViewDecodeResult(
            buckets=np.asarray([c.bucket for c in components], dtype=np.uint16),
            components=tuple(components),
            noise_floor=noise_floor,
            detection_threshold=threshold,
            residual_energy=float(np.sum(np.abs(residual) ** 2)),
        )
