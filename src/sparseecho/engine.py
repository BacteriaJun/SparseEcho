from __future__ import annotations

import numpy as np

from sparseecho.config import EngineConfig
from sparseecho.model import ReconstructionResult, ViewDiagnostics
from sparseecho.planning import ApertureBudget, QueryPlan, compile_query_plan
from sparseecho.recovery import GrayFiberViewDecoder, recover_global_support
from sparseecho.temporal import deconvolve_fir_memory, moment_weights, repair_erased_gray_slots
from sparseecho.transforms import fwht


class SparseEchoEngine:
    """Reconstruct one compiled sparse-query aperture from slot-level complex baseband."""

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
            false_alarm_rate=self.config.view_false_alarm_rate,
            noise_quantile=self.config.view_noise_quantile,
            query_order=self.plan.local_order,
        )
        self.aperture_budget = ApertureBudget(
            local_bits=self.config.local_bits,
            modeled_shell_order=self.config.modeled_shell_order,
            leakage_budget=self.config.fiber_leakage_budget,
        )

    def _spectrum_from_slots(self, slots: np.ndarray) -> np.ndarray:
        by_query = np.zeros_like(slots, dtype=np.complex128)
        by_query[self.plan.local_order.astype(np.int64)] = slots
        return fwht(by_query, axis=0, normalize=True)

    def _synthesize_view(
        self,
        samples: np.ndarray,
        valid: np.ndarray,
        *,
        pass_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collapse 1/2/4 physical passes to one forward-order virtual view."""
        b = self.plan.slots_per_view
        if pass_count == 1:
            return np.asarray(samples), np.asarray(valid, dtype=bool)
        orders = self.plan.pass_orders(pass_count)
        x = np.asarray(samples).reshape(pass_count, b, -1)
        mask = np.asarray(valid, dtype=bool).reshape(pass_count, b)
        total = pass_count * b
        target = (total - 1) / 2.0
        by_query = np.zeros((b, x.shape[2]), dtype=np.complex128)
        query_valid = np.zeros(b, dtype=bool)
        inverse_orders = [np.argsort(order.astype(np.int64)) for order in orders]
        max_degree = 1 if pass_count == 2 else 3

        for query in range(b):
            observations: list[np.ndarray] = []
            times: list[float] = []
            for p, inverse in enumerate(inverse_orders):
                within = int(inverse[query])
                if mask[p, within]:
                    observations.append(x[p, within])
                    times.append(float(p * b + within))
            if not observations:
                continue
            query_valid[query] = True
            degree = min(max_degree, len(observations) - 1)
            if degree == 0:
                by_query[query] = observations[0]
            else:
                weights = moment_weights(np.asarray(times), degree=degree, target_time=target)
                by_query[query] = np.tensordot(weights, np.asarray(observations), axes=(0, 0))

        # The rest of the engine consumes samples in the primary physical order.
        return by_query[self.plan.local_order.astype(np.int64)], query_valid[self.plan.local_order.astype(np.int64)]

    def _prepare_views(
        self,
        slots: np.ndarray,
        valid: np.ndarray,
        *,
        pass_count: int,
        receiver_calibration: np.ndarray | None = None,
        switch_fir_taps: tuple[float, ...] | np.ndarray | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        expected = self.plan.n_views * self.plan.slots_per_view * pass_count
        if slots.shape[0] != expected:
            raise ValueError(f"expected {expected} physical slots, received {slots.shape[0]}")
        corrected = np.array(slots, dtype=np.complex128, copy=True)
        corrected[~valid] = 0.0
        taps = self.config.switch_fir_taps if switch_fir_taps is None else switch_fir_taps
        corrected = deconvolve_fir_memory(corrected, taps, valid)

        calibration = None if receiver_calibration is None else np.asarray(receiver_calibration)
        if calibration is not None and calibration.ndim == 1:
            if calibration.shape != (corrected.shape[1],):
                raise ValueError("receiver_calibration shape mismatch")
            corrected = corrected / calibration[None, :]
        elif calibration is not None and calibration.ndim != 2:
            raise ValueError("receiver_calibration must have shape (rx,) or (views, rx)")

        b = self.plan.slots_per_view
        block = pass_count * b
        prepared: list[tuple[np.ndarray, np.ndarray]] = []
        for view in range(self.plan.n_views):
            sl = slice(view * block, (view + 1) * block)
            view_samples = corrected[sl]
            if calibration is not None and calibration.ndim == 2:
                if calibration.shape != (self.plan.n_views, corrected.shape[1]):
                    raise ValueError("receiver_calibration shape mismatch")
                view_samples = view_samples / calibration[view][None, :]
            prepared.append(
                self._synthesize_view(view_samples, valid[sl], pass_count=pass_count)
            )
        return prepared

    def process_capture(
        self,
        slots: np.ndarray,
        valid: np.ndarray | None = None,
        *,
        pass_count: int = 1,
        receiver_calibration: np.ndarray | None = None,
        switch_fir_taps: tuple[float, ...] | np.ndarray | None = None,
    ) -> ReconstructionResult:
        if pass_count not in (1, 2, 4):
            raise ValueError("pass_count must be 1, 2, or 4")
        x = np.asarray(slots)
        if x.ndim == 1:
            x = x[:, None]
        if valid is None:
            valid_a = np.ones(x.shape[0], dtype=bool)
        else:
            valid_a = np.asarray(valid, dtype=bool)
            if valid_a.shape != (x.shape[0],):
                raise ValueError("valid mask shape mismatch")

        spectra: list[np.ndarray] = []
        detected: list[np.ndarray] = []
        diagnostics: list[ViewDiagnostics] = []

        for view_index, (view_slots, view_valid) in enumerate(
            self._prepare_views(
                x,
                valid_a,
                pass_count=pass_count,
                receiver_calibration=receiver_calibration,
                switch_fir_taps=switch_fir_taps,
            )
        ):
            initial_spectrum = self._spectrum_from_slots(view_slots)
            initial = self.view_decoder.decode(initial_spectrum)

            if not np.all(view_valid) and initial.buckets.size:
                view_slots = repair_erased_gray_slots(
                    view_slots,
                    view_valid,
                    local_order=self.plan.local_order,
                    candidate_buckets=initial.buckets[: self.config.erasure_repair_candidate_limit],
                    degree=self.config.erasure_repair_degree,
                    ridge=self.config.erasure_repair_ridge,
                )

            spectrum = self._spectrum_from_slots(view_slots)
            final = self.view_decoder.decode(spectrum)
            spectra.append(spectrum)
            detected.append(final.buckets)
            diagnostic_spans = [
                abs(c.phase_span_cycles)
                for c in final.components
                if c.score >= self.config.diagnostic_min_score_ratio * final.detection_threshold
            ]
            phase_span = (
                float(np.quantile(diagnostic_spans, self.config.diagnostic_phase_quantile))
                if diagnostic_spans
                else 0.0
            )
            diagnostics.append(
                ViewDiagnostics(
                    view_index=view_index,
                    detected_buckets=int(final.buckets.size),
                    erasure_fraction=float(1.0 - np.mean(view_valid)),
                    noise_floor=float(final.noise_floor),
                    detection_threshold=float(final.detection_threshold),
                    residual_energy=float(final.residual_energy),
                    estimated_phase_span_cycles=float(phase_span),
                    fiber_tail_energy=float(self.aperture_budget.tail_energy(phase_span)),
                )
            )

        early_groups = [(7, 8, 9), (10, 11, 12)]
        results = [
            recover_global_support(
                detected,
                spectra,
                list(self.plan.views),
                min_view_support=self.config.min_view_support,
                spatial_consistency_threshold=self.config.spatial_consistency_threshold,
                spatial_subspace_rank=self.config.spatial_subspace_rank,
                low_support_spatial_margin=self.config.low_support_spatial_margin,
                max_pre_candidates=self.config.max_pre_candidates,
                early_group=group,
            )
            for group in early_groups
            if max(group) < self.plan.n_views
        ]
        if not results:
            results = [
                recover_global_support(
                    detected,
                    spectra,
                    list(self.plan.views),
                    min_view_support=self.config.min_view_support,
                    spatial_consistency_threshold=self.config.spatial_consistency_threshold,
                    spatial_subspace_rank=self.config.spatial_subspace_rank,
                low_support_spatial_margin=self.config.low_support_spatial_margin,
                    max_pre_candidates=self.config.max_pre_candidates,
                    early_group=tuple(range(7, min(10, self.plan.n_views))),
                )
            ]

        identities = sorted({int(v) for result in results for v in result.identities})
        record_map = {}
        for result in results:
            for record in result.candidates:
                previous = record_map.get(record.identity)
                if previous is None or (record.view_support, record.spatial_consistency) > (
                    previous.view_support,
                    previous.spatial_consistency,
                ):
                    record_map[record.identity] = record

        return ReconstructionResult(
            identities=np.asarray(identities, dtype=np.uint32),
            candidates=tuple(record_map[key] for key in sorted(record_map)),
            views=tuple(diagnostics),
            pass_count=pass_count,
        )
