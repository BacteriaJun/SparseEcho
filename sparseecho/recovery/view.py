from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sparseecho.temporal import doppler_fiber_coefficients


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
    residual_energy: float


class GrayFiberViewDecoder:
    """Local Gray-temporal-fiber CLEAN decoder.

    This solver operates only inside a small hash aperture (normally 256 buckets). It does not
    traverse the global identity address space.
    """

    def __init__(
        self,
        *,
        local_bits: int = 8,
        max_components: int = 64,
        max_abs_phase_span_cycles: float = 0.65,
        phase_grid: int = 33,
        stop_noise_multiple: float = 0.45,
    ) -> None:
        self.local_bits = int(local_bits)
        self.bucket_count = 1 << self.local_bits
        self.max_components = int(max_components)
        self.max_abs_phase_span_cycles = float(max_abs_phase_span_cycles)
        self.stop_noise_multiple = float(stop_noise_multiple)
        self._spans = np.linspace(
            -self.max_abs_phase_span_cycles,
            self.max_abs_phase_span_cycles,
            int(phase_grid),
        )
        self._atoms = np.stack(
            [doppler_fiber_coefficients(span, self.local_bits) for span in self._spans], axis=0
        )
        self._xor_index = np.arange(self.bucket_count, dtype=np.int64)

    def decode(self, spectrum: np.ndarray) -> ViewDecodeResult:
        x = np.asarray(spectrum)
        if x.shape[0] != self.bucket_count:
            raise ValueError("spectrum length does not match local aperture")
        if x.ndim == 1:
            x = x[:, None]
        residual = x.astype(np.complex128, copy=True)
        base_energy = np.sum(np.abs(x) ** 2, axis=1)
        noise_floor = float(np.median(np.sort(base_energy)[: self.bucket_count // 2]) + 1e-18)
        components: list[ViewComponent] = []

        for _ in range(self.max_components):
            energy = np.sum(np.abs(residual) ** 2, axis=1)
            bucket = int(np.argmax(energy))
            peak = float(energy[bucket])
            if peak < self.stop_noise_multiple * noise_floor:
                break

            shifted = residual[self._xor_index ^ bucket]
            channel_by_span = self._atoms.conj() @ shifted
            score_by_span = np.sum(np.abs(channel_by_span) ** 2, axis=1)
            best = int(np.argmax(score_by_span))
            channel = channel_by_span[best]
            residual[self._xor_index ^ bucket] -= self._atoms[best, :, None] * channel[None, :]
            components.append(
                ViewComponent(
                    bucket=bucket,
                    phase_span_cycles=float(self._spans[best]),
                    score=float(score_by_span[best]),
                )
            )

        unique: list[int] = []
        seen: set[int] = set()
        for component in components:
            if component.bucket not in seen:
                seen.add(component.bucket)
                unique.append(component.bucket)
        return ViewDecodeResult(
            buckets=np.asarray(unique, dtype=np.uint16),
            components=tuple(components),
            noise_floor=noise_floor,
            residual_energy=float(np.sum(np.abs(residual) ** 2)),
        )
