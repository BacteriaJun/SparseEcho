from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from sparseecho.transforms import gray_order

from .hashes import LinearHashView, default_hash_views


@dataclass(frozen=True)
class QuerySlot:
    physical_index: int
    view_index: int
    within_view_index: int
    local_query: int
    challenge: int


@dataclass(frozen=True)
class QueryPlan:
    views: tuple[LinearHashView, ...]
    local_order: np.ndarray
    slots: tuple[QuerySlot, ...]
    ordering: str = "gray"

    @property
    def physical_slots(self) -> int:
        return len(self.slots)

    @property
    def slots_per_view(self) -> int:
        return int(self.local_order.size)

    @property
    def n_views(self) -> int:
        return len(self.views)

    @property
    def identity_bits(self) -> int:
        return self.views[0].identity_bits

    @property
    def local_bits(self) -> int:
        return self.views[0].local_bits

    def slice_for_view(self, view_index: int) -> slice:
        b = self.slots_per_view
        return slice(view_index * b, (view_index + 1) * b)

    def pass_orders(self, pass_count: int) -> tuple[np.ndarray, ...]:
        if pass_count == 1:
            return (self.local_order,)
        if pass_count == 2:
            return (self.local_order, self.local_order[::-1])
        if pass_count == 4:
            return (self.local_order, self.local_order[::-1], self.local_order, self.local_order[::-1])
        raise ValueError("pass_count must be 1, 2, or 4")

    def expanded_challenges(self, pass_count: int) -> np.ndarray:
        """Physical challenge sequence with repeated passes grouped by local view."""
        orders = self.pass_orders(pass_count)
        values: list[int] = []
        for view in self.views:
            for order in orders:
                values.extend(view.challenge(int(query)) for query in order)
        return np.asarray(values, dtype=np.uint64)

    def fingerprint(self) -> str:
        """Stable SHA-256 identifier for the logical single-pass query plan."""
        payload = {
            "identity_bits": self.identity_bits,
            "local_bits": self.local_bits,
            "n_views": self.n_views,
            "ordering": self.ordering,
            "local_order": self.local_order.astype(int).tolist(),
            "view_names": [view.name for view in self.views],
            "challenges": self.expanded_challenges(1).astype(object).tolist(),
        }
        canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def to_dict(self, *, pass_count: int = 1, aperture_scale: float = 1.0) -> dict:
        challenges = self.expanded_challenges(pass_count)
        return {
            "identity_bits": self.identity_bits,
            "local_bits": self.local_bits,
            "n_views": self.n_views,
            "slots_per_view": self.slots_per_view,
            "pass_count": int(pass_count),
            "physical_slots": int(challenges.size),
            "aperture_scale": float(aperture_scale),
            "plan_fingerprint": self.fingerprint(),
            "view_names": [view.name for view in self.views],
            "ordering": self.ordering,
            "local_order": self.local_order.astype(int).tolist(),
            "challenges": challenges.astype(object).tolist(),
        }


def _compile_order(local_bits: int, ordering: str, seed: int) -> np.ndarray:
    n = 1 << local_bits
    if ordering == "gray":
        return gray_order(local_bits)
    if ordering == "binary":
        return np.arange(n, dtype=np.uint32)
    if ordering == "random":
        rng = np.random.default_rng(seed ^ 0xA5A5_1EAF)
        return rng.permutation(n).astype(np.uint32)
    raise ValueError("ordering must be 'gray', 'binary', or 'random'")


def compile_query_plan(
    *,
    identity_bits: int = 32,
    local_bits: int = 8,
    n_views: int = 16,
    seed: int = 0x5EED,
    ordering: str = "gray",
) -> QueryPlan:
    views = tuple(
        default_hash_views(
            identity_bits=identity_bits,
            local_bits=local_bits,
            n_views=n_views,
            seed=seed,
        )
    )
    order = _compile_order(local_bits, ordering, seed)
    slots: list[QuerySlot] = []
    physical = 0
    for v, view in enumerate(views):
        for within, local_query in enumerate(order):
            slots.append(
                QuerySlot(
                    physical_index=physical,
                    view_index=v,
                    within_view_index=within,
                    local_query=int(local_query),
                    challenge=view.challenge(int(local_query)),
                )
            )
            physical += 1
    return QueryPlan(views=views, local_order=order, slots=tuple(slots), ordering=ordering)
