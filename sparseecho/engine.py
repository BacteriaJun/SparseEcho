from __future__ import annotations

import numpy as np

from sparseecho.config import EngineConfig
from sparseecho.model import ReconstructionResult, ViewDiagnostics
from sparseecho.planning import QueryPlan, compile_query_plan
from sparseecho.recovery import GrayFiberViewDecoder, recover_global_support
from sparseecho.temporal import deconvolve_first_order_memory, repair_erased_gray_slots
from sparseecho.transforms import fwht


class SparseEchoEngine:
    """State-free reconstruction engine for one compiled acquisition aperture."""

    def __init__(self, config: EngineConfig | None = None, plan: QueryPlan | None = None) -> None:
        self.config = config or EngineConfig()
        self.plan = plan or compile_query_plan(
            identity_bits=self.config.identity_bits,
            local_bits=self.config.local_bits,
            n_views=self.config.n_views,
            seed=self.config.hash_seed,
        )
        self.view_decoder = GrayFiberViewDecoder(
            local_bits=self.config.local_bits,
            max_components=self.config.view_max_components,
            max_abs_phase_span_cycles=self.config.view_max_abs_phase_span_cycles,
            phase_grid=self.config.view_phase_grid,
            stop_noise_multiple=self.config.view_stop_noise_multiple,
        )

    def _spectrum_from_slots(self, slots: np.ndarray) -> np.ndarray:
        by_query = np.zeros_like(slots, dtype=np.complex128)
        by_query[self.plan.local_order.astype(np.int64)] = slots
        return fwht(by_query, axis=0, normalize=True)

    def process_capture(self, slots: np.ndarray, valid: np.ndarray | None = None) -> ReconstructionResult:
        x = np.asarray(slots)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[0] != self.plan.physical_slots:
            raise ValueError(
                f"expected {self.plan.physical_slots} physical slots, received {x.shape[0]}"
            )
        if valid is None:
            valid_a = np.ones(x.shape[0], dtype=bool)
        else:
            valid_a = np.asarray(valid, dtype=bool)
            if valid_a.shape != (x.shape[0],):
                raise ValueError("valid mask shape mismatch")

        spectra: list[np.ndarray] = []
        detected: list[np.ndarray] = []
        diagnostics: list[ViewDiagnostics] = []
        high_confidence_counts: list[int] = []
        medium_confidence_counts: list[int] = []

        for v in range(self.plan.n_views):
            sl = self.plan.slice_for_view(v)
            view_slots = np.array(x[sl], dtype=np.complex128, copy=True)
            view_valid = valid_a[sl]
            view_slots[~view_valid] = 0.0
            view_slots = deconvolve_first_order_memory(
                view_slots,
                self.config.switch_memory_coefficient,
                view_valid,
            )
            initial_spectrum = self._spectrum_from_slots(view_slots)
            initial = self.view_decoder.decode(initial_spectrum)

            if not np.all(view_valid) and initial.buckets.size:
                view_slots = repair_erased_gray_slots(
                    view_slots,
                    view_valid,
                    local_order=self.plan.local_order,
                    candidate_buckets=initial.buckets[:50],
                    degree=self.config.erasure_repair_degree,
                    ridge=self.config.erasure_repair_ridge,
                )

            spectrum = self._spectrum_from_slots(view_slots)
            final = self.view_decoder.decode(spectrum)
            spectra.append(spectrum)
            detected.append(final.buckets)
            per_bucket_ratio: dict[int, float] = {}
            for component in final.components:
                ratio = float(component.score / max(final.noise_floor, 1e-18))
                per_bucket_ratio[component.bucket] = max(per_bucket_ratio.get(component.bucket, 0.0), ratio)
            high_confidence_counts.append(
                sum(value >= self.config.high_confidence_component_ratio for value in per_bucket_ratio.values())
            )
            medium_confidence_counts.append(
                sum(value >= self.config.medium_confidence_component_ratio for value in per_bucket_ratio.values())
            )
            diagnostics.append(
                ViewDiagnostics(
                    view_index=v,
                    detected_buckets=int(final.buckets.size),
                    erasure_fraction=float(1.0 - np.mean(view_valid)),
                    noise_floor=float(final.noise_floor),
                    residual_energy=float(final.residual_energy),
                )
            )

        high_hint = float(np.median(high_confidence_counts)) if high_confidence_counts else 0.0
        medium_hint = float(np.median(medium_confidence_counts)) if medium_confidence_counts else 0.0
        occupancy_hint = high_hint if high_hint <= self.config.low_occupancy_cutoff else medium_hint
        if high_hint <= self.config.low_occupancy_cutoff:
            min_view_support = max(
                self.config.min_view_support, self.config.low_occupancy_min_view_support
            )
            spatial_threshold = max(
                self.config.spatial_consistency_threshold,
                self.config.low_occupancy_spatial_consistency,
            )
        else:
            min_view_support = self.config.min_view_support
            spatial_threshold = self.config.spatial_consistency_threshold

        primary = recover_global_support(
            detected,
            spectra,
            list(self.plan.views),
            min_view_support=min_view_support,
            spatial_consistency_threshold=spatial_threshold,
            max_pre_candidates=self.config.max_pre_candidates,
            early_group=(5, 6, 7),
            allow_structured_erasure=False,
        )

        results = [primary]
        target_floor = max(0, int(np.floor(occupancy_hint - 0.25)))
        # In populated apertures no single early validation triplet is allowed to become a
        # support gate. Two disjoint list gates are evaluated and unioned before any slower
        # structured-erasure fallback. This trades bounded local compute for removal of a
        # view-level single point of failure; neither path traverses the global address space.
        if high_hint > self.config.low_occupancy_cutoff:
            secondary = recover_global_support(
                detected,
                spectra,
                list(self.plan.views),
                min_view_support=min_view_support,
                spatial_consistency_threshold=spatial_threshold,
                max_pre_candidates=self.config.max_pre_candidates,
                early_group=(8, 9, 10),
                allow_structured_erasure=False,
            )
            results.append(secondary)

        merged_ids = sorted({int(v) for result in results for v in result.identities})
        # A structured-erasure fallback is only paid when both fast list gates still undershoot the
        # occupancy evidence extracted from the local fiber components.
        if high_hint > self.config.low_occupancy_cutoff and len(merged_ids) < target_floor:
            tertiary = recover_global_support(
                detected,
                spectra,
                list(self.plan.views),
                min_view_support=min_view_support,
                spatial_consistency_threshold=spatial_threshold,
                max_pre_candidates=self.config.max_pre_candidates,
                early_group=(8, 9, 10),
                allow_structured_erasure=True,
            )
            results.append(tertiary)
            merged_ids = sorted({int(v) for result in results for v in result.identities})

        record_map = {}
        for result in results:
            for record in result.candidates:
                previous = record_map.get(record.identity)
                if previous is None or (record.view_support, record.spatial_consistency) > (
                    previous.view_support, previous.spatial_consistency
                ):
                    record_map[record.identity] = record

        return ReconstructionResult(
            identities=np.asarray(merged_ids, dtype=np.uint32),
            candidates=tuple(record_map[key] for key in sorted(record_map)),
            views=tuple(diagnostics),
        )
