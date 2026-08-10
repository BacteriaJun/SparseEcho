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
    residual_energy: float


@dataclass(frozen=True)
class ReconstructionResult:
    identities: np.ndarray
    candidates: tuple[SupportCandidate, ...]
    views: tuple[ViewDiagnostics, ...]

    def to_dict(self) -> dict:
        return {
            "identities": [int(v) for v in self.identities],
            "n_identities": int(self.identities.size),
            "candidate_count": len(self.candidates),
            "views": [
                {
                    "view_index": d.view_index,
                    "detected_buckets": d.detected_buckets,
                    "erasure_fraction": d.erasure_fraction,
                    "noise_floor": d.noise_floor,
                    "residual_energy": d.residual_energy,
                }
                for d in self.views
            ],
        }
