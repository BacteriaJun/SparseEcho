from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    identity_bits: int = 32
    local_bits: int = 8
    n_views: int = 14
    hash_seed: int = 0x5EED

    switch_memory_coefficient: float = 0.10
    erasure_repair_degree: int = 1
    erasure_repair_ridge: float = 1e-3

    view_max_components: int = 64
    view_phase_grid: int = 33
    view_max_abs_phase_span_cycles: float = 0.65
    view_stop_noise_multiple: float = 0.45

    min_view_support: int = 13
    spatial_consistency_threshold: float = 0.47
    max_pre_candidates: int = 20000

    high_confidence_component_ratio: float = 10.0
    medium_confidence_component_ratio: float = 5.0
    low_occupancy_cutoff: int = 8
    low_occupancy_min_view_support: int = 14
    low_occupancy_spatial_consistency: float = 0.70


@dataclass(frozen=True)
class SceneConfig:
    n_active: int = 32
    n_rx: int = 8
    near_far_power_db: float = 30.0
    weakest_fiber_snr_db: float = 6.0
    max_phase_span_cycles_per_view: float = 0.40
    max_quadratic_phase_cycles: float = 0.05
    max_fractional_amplitude_drift: float = 0.04
    slot_erasure_rate: float = 0.02
    switch_memory_coefficient: float = 0.10
