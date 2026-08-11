from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sparseecho.planning.aperture import ApertureBudget

if TYPE_CHECKING:
    from sparseecho.model import ReconstructionResult


@dataclass(frozen=True)
class AcquisitionRequest:
    """One acquisition request at a specified relative aperture duration."""

    aperture_scale: float = 1.0
    pass_count: int = 1
    reason: str = "initial acquisition"


@dataclass(frozen=True)
class AcquisitionDecision:
    action: str
    aperture_scale_factor: float
    estimated_phase_span_cycles: float
    allowed_phase_span_cycles: float
    tail_energy: float
    reason: str


class AdaptiveAcquisitionController:
    """Close the fiber-tail budget by changing physical aperture duration.

    A shorter aperture is the default high-dynamics response. Repeating the same query sequence
    over a longer interval can attenuate complex exponentials, so multi-pass virtual-time synthesis
    remains an explicit engine primitive rather than the automatic fallback.
    """

    def __init__(
        self,
        budget: ApertureBudget | None = None,
        *,
        safety_factor: float = 0.90,
        min_scale_factor: float = 0.20,
        diagnostic_quantile: float = 0.90,
    ) -> None:
        self.budget = budget or ApertureBudget()
        self.safety_factor = float(safety_factor)
        self.min_scale_factor = float(min_scale_factor)
        self.diagnostic_quantile = float(diagnostic_quantile)

    def decide(self, result: "ReconstructionResult") -> AcquisitionDecision:
        spans = np.asarray(
            [abs(view.estimated_phase_span_cycles) for view in result.views], dtype=float
        )
        span = float(np.quantile(spans, self.diagnostic_quantile)) if spans.size else 0.0
        tail = self.budget.tail_energy(span)
        allowed = self.budget.max_phase_span_cycles()
        if tail <= self.budget.leakage_budget or span <= 0.0:
            return AcquisitionDecision(
                "accept",
                1.0,
                span,
                allowed,
                tail,
                "measured fiber tail is within budget",
            )
        factor = self.safety_factor * allowed / span
        factor = max(self.min_scale_factor, min(1.0, factor))
        return AcquisitionDecision(
            "compress",
            factor,
            span,
            allowed,
            tail,
            "measured fiber tail exceeds budget; reacquire at a shorter physical aperture",
        )
