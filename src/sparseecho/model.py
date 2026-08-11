from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sparseecho.recovery import SupportCandidate


@dataclass(frozen=True)
class ViewDiagnostics:
    view_index: int
    detected_buckets: int
    erasure_fraction: float
    noise_floor: float
    detection_threshold: float
    residual_energy: float
    estimated_phase_span_cycles: float
    fiber_tail_energy: float


@dataclass(frozen=True)
class ReconstructionResult:
    identities: np.ndarray
    candidates: tuple[SupportCandidate, ...]
    views: tuple[ViewDiagnostics, ...]
    pass_count: int = 1

    def to_dict(self) -> dict:
        return {
            "identities": [int(v) for v in self.identities],
            "n_identities": int(self.identities.size),
            "candidate_count": len(self.candidates),
            "pass_count": int(self.pass_count),
            "views": [
                {
                    "view_index": d.view_index,
                    "detected_buckets": d.detected_buckets,
                    "erasure_fraction": d.erasure_fraction,
                    "noise_floor": d.noise_floor,
                    "detection_threshold": d.detection_threshold,
                    "residual_energy": d.residual_energy,
                    "estimated_phase_span_cycles": d.estimated_phase_span_cycles,
                    "fiber_tail_energy": d.fiber_tail_energy,
                }
                for d in self.views
            ],
        }
