from __future__ import annotations

from dataclasses import dataclass

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

    def to_dict(self) -> dict:
        return {
            "identity_bits": self.identity_bits,
            "local_bits": self.local_bits,
            "n_views": self.n_views,
            "slots_per_view": self.slots_per_view,
            "physical_slots": self.physical_slots,
            "view_names": [view.name for view in self.views],
            "local_order": self.local_order.astype(int).tolist(),
            "challenges": [slot.challenge for slot in self.slots],
        }


def compile_query_plan(
    *, identity_bits: int = 32, local_bits: int = 8, n_views: int = 14, seed: int = 0x5EED
) -> QueryPlan:
    views = tuple(
        default_hash_views(
            identity_bits=identity_bits,
            local_bits=local_bits,
            n_views=n_views,
            seed=seed,
        )
    )
    order = gray_order(local_bits)
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
    return QueryPlan(views=views, local_order=order, slots=tuple(slots))
