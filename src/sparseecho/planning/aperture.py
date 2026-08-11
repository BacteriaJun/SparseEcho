from __future__ import annotations

from dataclasses import dataclass

from sparseecho.temporal.fiber import doppler_shell_energies


@dataclass(frozen=True)
class ApertureBudget:
    local_bits: int = 8
    modeled_shell_order: int = 3
    leakage_budget: float = 1e-5

    def tail_energy(self, phase_span_cycles: float) -> float:
        energies = doppler_shell_energies(phase_span_cycles, self.local_bits)
        return float(sum(energies[self.modeled_shell_order + 1 :]))

    def max_phase_span_cycles(self, *, upper: float = 2.0, iterations: int = 70) -> float:
        lo, hi = 0.0, float(upper)
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            if self.tail_energy(mid) <= self.leakage_budget:
                lo = mid
            else:
                hi = mid
        return lo

    def max_view_seconds(self, residual_doppler_hz: float) -> float:
        f = abs(float(residual_doppler_hz))
        if f == 0:
            return float("inf")
        return self.max_phase_span_cycles() / f
