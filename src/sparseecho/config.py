from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    identity_bits: int = 32
    local_bits: int = 8
    n_views: int = 16
    hash_seed: int = 0x5EED

    # Calibrated causal settling kernel.  It is applied continuously across view boundaries.
    switch_fir_taps: tuple[float, ...] = (0.90, 0.10)

    erasure_repair_degree: int = 1
    erasure_repair_ridge: float = 1e-3
    erasure_repair_candidate_limit: int = 56

    view_max_components: int = 48
    view_phase_grid: int = 33
    view_max_abs_phase_span_cycles: float = 0.65
    view_false_alarm_rate: float = 0.20
    view_noise_quantile: float = 0.15
    diagnostic_phase_quantile: float = 0.90
    diagnostic_min_score_ratio: float = 2.0

    min_view_support: int = 13
    spatial_subspace_rank: int = 2
    spatial_consistency_threshold: float = 0.70
    low_support_spatial_margin: float = 0.04
    max_pre_candidates: int = 40000

    modeled_shell_order: int = 3
    fiber_leakage_budget: float = 1e-5

    def __post_init__(self) -> None:
        if not 1 <= self.identity_bits <= 64:
            raise ValueError("identity_bits must be in [1,64]")
        if not 1 <= self.local_bits <= 16:
            raise ValueError("local_bits must be in [1,16]")
        if self.n_views < 1:
            raise ValueError("n_views must be positive")
        taps = tuple(float(v) for v in self.switch_fir_taps)
        if not taps or abs(taps[0]) < 1e-12:
            raise ValueError("switch_fir_taps must have a non-zero first tap")
        if not 0.0 < self.view_false_alarm_rate < 1.0:
            raise ValueError("view_false_alarm_rate must be in (0,1)")
        if not 0.0 < self.view_noise_quantile < 0.5:
            raise ValueError("view_noise_quantile must be in (0,0.5)")
        if self.view_max_components < 1 or self.view_phase_grid < 1:
            raise ValueError("view proposal budgets must be positive")
        if self.min_view_support < 1:
            raise ValueError("min_view_support must be positive")
        if self.spatial_subspace_rank < 1:
            raise ValueError("spatial_subspace_rank must be positive")
        if not 0.0 <= self.spatial_consistency_threshold <= 1.0:
            raise ValueError("spatial_consistency_threshold must be in [0,1]")
        if self.max_pre_candidates < 1:
            raise ValueError("max_pre_candidates must be positive")
        if self.modeled_shell_order < 0 or self.fiber_leakage_budget <= 0.0:
            raise ValueError("fiber budget must be positive and shell order non-negative")


@dataclass(frozen=True)
class SceneConfig:
    """Small matched-model scene used by unit tests and examples, not release validation."""

    n_active: int = 32
    n_rx: int = 8
    near_far_power_db: float = 30.0
    weakest_fiber_snr_db: float = 6.0
    max_phase_span_cycles_per_view: float = 0.40
    max_quadratic_phase_cycles: float = 0.05
    max_fractional_amplitude_drift: float = 0.04
    slot_erasure_rate: float = 0.02
    switch_fir_taps: tuple[float, ...] = (0.90, 0.10)
