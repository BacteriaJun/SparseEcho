from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np

from sparseecho.planning import LinearHashView


@dataclass(frozen=True)
class SupportCandidate:
    identity: int
    view_support: int
    spatial_consistency: float


@dataclass(frozen=True)
class SupportResult:
    identities: np.ndarray
    candidates: tuple[SupportCandidate, ...]


def _pack_bytes(b0: np.ndarray, b1: np.ndarray, b2: np.ndarray, b3: np.ndarray) -> np.ndarray:
    return (
        b0.astype(np.uint32)
        | (b1.astype(np.uint32) << 8)
        | (b2.astype(np.uint32) << 16)
        | (b3.astype(np.uint32) << 24)
    )


def _membership_mask(values: np.ndarray) -> np.ndarray:
    mask = np.zeros(256, dtype=bool)
    mask[np.asarray(values, dtype=np.int64)] = True
    return mask


def _strict_gate(
    candidates: np.ndarray,
    views: list[LinearHashView],
    membership: list[np.ndarray],
    indices: tuple[int, ...],
) -> np.ndarray:
    out = candidates
    for v in indices:
        if v >= len(views) or out.size == 0:
            continue
        z = views[v].hash_ids(out).astype(np.int64)
        out = out[membership[v][z]]
    return out



def _soft_gate(
    candidates: np.ndarray,
    views: list[LinearHashView],
    membership: list[np.ndarray],
    indices: tuple[int, ...],
    *,
    min_hits: int,
) -> np.ndarray:
    if candidates.size == 0 or not indices:
        return candidates
    hits = np.zeros(candidates.size, dtype=np.int8)
    for v in indices:
        if v >= len(views):
            continue
        z = views[v].hash_ids(candidates).astype(np.int64)
        hits += membership[v][z]
    return candidates[hits >= int(min_hits)]

def _gf2_rank(a: np.ndarray) -> int:
    x = (np.asarray(a, dtype=np.uint8) & 1).copy()
    rank = 0
    for col in range(x.shape[1]):
        pivot = np.flatnonzero(x[rank:, col])
        if pivot.size == 0:
            continue
        p = rank + int(pivot[0])
        x[[rank, p]] = x[[p, rank]]
        for row in range(x.shape[0]):
            if row != rank and x[row, col]:
                x[row] ^= x[rank]
        rank += 1
        if rank == x.shape[0]:
            break
    return rank


def _gf2_inverse(a: np.ndarray) -> np.ndarray:
    x = np.concatenate(
        [(np.asarray(a, dtype=np.uint8) & 1).copy(), np.eye(a.shape[0], dtype=np.uint8)], axis=1
    )
    n = a.shape[0]
    for col in range(n):
        pivot = np.flatnonzero(x[col:, col])
        if pivot.size == 0:
            raise ValueError("matrix is singular over GF(2)")
        p = col + int(pivot[0])
        x[[col, p]] = x[[p, col]]
        for row in range(n):
            if row != col and x[row, col]:
                x[row] ^= x[col]
    return x[:, n:]


# Coefficients of the seven byte-wise structured views in planning.hashes.
_STRUCTURED_COEFFICIENTS = np.asarray(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
    ],
    dtype=np.uint8,
)


def _decoding_quads() -> tuple[tuple[tuple[int, ...], np.ndarray], ...]:
    # Seven decoding bases cover every pattern with up to two missing structured views.  Keeping
    # this set explicit bounds candidate enumeration and makes the erasure tolerance auditable.
    selected = (
        (0, 1, 2, 3),
        (0, 1, 4, 5),
        (0, 2, 5, 6),
        (0, 3, 4, 6),
        (1, 2, 4, 6),
        (1, 3, 5, 6),
        (2, 3, 4, 5),
    )
    return tuple(
        (indices, _gf2_inverse(_STRUCTURED_COEFFICIENTS[list(indices)]))
        for indices in selected
    )


_DECODING_QUADS = _decoding_quads()


def _decode_tuple_to_ids(values: list[np.ndarray], inverse: np.ndarray) -> np.ndarray:
    decoded: list[np.ndarray] = []
    for output_byte in range(4):
        value = np.zeros_like(values[0], dtype=np.uint16)
        for observed_index in range(4):
            if inverse[output_byte, observed_index]:
                value ^= values[observed_index].astype(np.uint16)
        decoded.append(value)
    return _pack_bytes(decoded[0], decoded[1], decoded[2], decoded[3])


def _structured_candidates(
    detected: list[np.ndarray],
    views: list[LinearHashView],
    *,
    early_group: tuple[int, ...],
    max_pre_candidates: int,
) -> np.ndarray:
    """List-recover 32-bit IDs from redundant 8-bit structured views.

    Candidate enumeration uses only small local bucket lists.  Alternate GF(2) bases prevent a
    single structured-view miss from becoming a hard failure.  No operation traverses the 32-bit
    address space.
    """
    if len(detected) < 10:
        return np.empty(0, dtype=np.uint32)
    sets = [np.unique(np.asarray(v, dtype=np.uint16)) for v in detected]
    membership = [_membership_mask(v) for v in sets]
    chunks: list[np.ndarray] = []

    for indices, inverse in _DECODING_QUADS:
        selected = [sets[i] for i in indices]
        if any(values.size == 0 for values in selected):
            continue
        # Chunk on the first list.  The other three lists form at most ~O(10^4-10^5) tuples in
        # normal operation, which is cheap compared with any population-scale scan.
        m1, m2, m3 = np.meshgrid(selected[1], selected[2], selected[3], indexing="ij")
        tails = [m1.ravel(), m2.ravel(), m3.ravel()]
        for first in selected[0]:
            values = [np.full(tails[0].size, first, dtype=np.uint16), *tails]
            candidates = _decode_tuple_to_ids(values, inverse)
            candidates = _soft_gate(
                candidates, views, membership, early_group, min_hits=max(1, len(early_group) - 1)
            )
            if candidates.size:
                chunks.append(candidates)

    if not chunks:
        return np.empty(0, dtype=np.uint32)
    candidates = np.unique(np.concatenate(chunks))

    # Rank by all non-structured views before the slower spatial check.  This is an incidence
    # consistency score, not an estimate of the number of active sources.
    score = np.zeros(candidates.size, dtype=np.int16)
    for v in range(7, len(views)):
        z = views[v].hash_ids(candidates).astype(np.int64)
        score += membership[v][z]
    validation_count = len(views) - 7
    keep = score >= max(2, validation_count - 3)
    candidates = candidates[keep]
    score = score[keep]
    if candidates.size > max_pre_candidates:
        order = np.argsort(score)[::-1][:max_pre_candidates]
        candidates = candidates[order]
    return candidates


def _filter_candidates(
    candidates: np.ndarray,
    detected: list[np.ndarray],
    spectra: list[np.ndarray],
    views: list[LinearHashView],
    *,
    min_view_support: int,
    spatial_consistency_threshold: float,
    spatial_subspace_rank: int = 2,
    low_support_spatial_margin: float = 0.0,
) -> SupportResult:
    if candidates.size == 0:
        return SupportResult(np.empty(0, dtype=np.uint32), tuple())
    membership = [_membership_mask(v) for v in detected]
    support = np.zeros(candidates.size, dtype=np.int16)
    for v, view in enumerate(views):
        z = view.hash_ids(candidates).astype(np.int64)
        support += membership[v][z]
    keep = support >= int(min_view_support)
    candidates = candidates[keep]
    support = support[keep]

    records: list[SupportCandidate] = []
    accepted: list[int] = []
    for identity, view_support in zip(candidates, support, strict=False):
        normalized_rows: list[np.ndarray] = []
        for v, view in enumerate(views):
            bucket = int(view.hash_ids(np.array([identity], dtype=np.uint32))[0])
            vector = np.asarray(spectra[v][bucket])
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                normalized_rows.append(vector / norm)
        if len(normalized_rows) < 2:
            consistency = 0.0
        else:
            singular = np.linalg.svd(np.asarray(normalized_rows), compute_uv=False)
            rank = max(1, min(int(spatial_subspace_rank), singular.size))
            consistency = float(np.sum(singular[:rank] ** 2) / (np.sum(singular**2) + 1e-18))
        records.append(SupportCandidate(int(identity), int(view_support), consistency))
        threshold = float(spatial_consistency_threshold)
        if int(view_support) <= int(min_view_support):
            threshold += float(low_support_spatial_margin)
        if consistency >= threshold:
            accepted.append(int(identity))
    return SupportResult(np.asarray(sorted(set(accepted)), dtype=np.uint32), tuple(records))


def recover_global_support(
    detected: list[np.ndarray],
    spectra: list[np.ndarray],
    views: list[LinearHashView],
    *,
    min_view_support: int | None = None,
    spatial_consistency_threshold: float = 0.65,
    spatial_subspace_rank: int = 2,
    low_support_spatial_margin: float = 0.0,
    max_pre_candidates: int = 20000,
    early_group: tuple[int, ...] = (7, 8, 9),
) -> SupportResult:
    """Recover global 32-bit support from local hash views without population scanning."""
    if len(detected) != len(views) or len(spectra) != len(views):
        raise ValueError("detected/spectra/views length mismatch")
    candidates = _structured_candidates(
        detected,
        views,
        early_group=early_group,
        max_pre_candidates=max_pre_candidates,
    )
    if min_view_support is None:
        min_view_support = max(3, len(views) - 3)
    return _filter_candidates(
        candidates,
        detected,
        spectra,
        views,
        min_view_support=int(min_view_support),
        spatial_consistency_threshold=spatial_consistency_threshold,
        spatial_subspace_rank=spatial_subspace_rank,
        low_support_spatial_margin=low_support_spatial_margin,
    )
