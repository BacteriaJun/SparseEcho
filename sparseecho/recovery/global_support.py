from __future__ import annotations

from dataclasses import dataclass

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


def _structured_candidates(
    detected: list[np.ndarray],
    views: list[LinearHashView],
    *,
    early_group: tuple[int, ...] = (5, 6, 7),
    allow_structured_erasure: bool = False,
    max_pre_candidates: int = 20000,
) -> np.ndarray:
    """List-recover 32-bit IDs from fixed 8-bit structured views.

    Primary recovery enumerates the four systematic bytes and uses the parity view as a check.
    A slower fallback can replace one missing systematic byte from parity. Both modes are bounded
    by local candidate lists and never traverse the 32-bit population.
    """
    if len(detected) < 6:
        return np.empty(0, dtype=np.uint32)
    sets = [np.unique(np.asarray(v, dtype=np.uint16)) for v in detected]
    membership = [_membership_mask(v) for v in sets]
    s0, s1, s2, s3, parity = sets[:5]
    structured = [s0, s1, s2, s3]
    chunks: list[np.ndarray] = []

    if all(s.size for s in structured):
        b1, b2, b3 = np.meshgrid(s1, s2, s3, indexing="ij")
        b1f, b2f, b3f = b1.ravel(), b2.ravel(), b3.ravel()
        for b0 in s0:
            b0f = np.full(b1f.size, b0, dtype=np.uint16)
            cand = _pack_bytes(b0f, b1f, b2f, b3f)
            p = (
                (cand & 255)
                ^ ((cand >> 8) & 255)
                ^ ((cand >> 16) & 255)
                ^ ((cand >> 24) & 255)
            ).astype(np.int64)
            cand = cand[membership[4][p]]
            cand = _strict_gate(cand, views, membership, early_group)
            if cand.size:
                chunks.append(cand)

    if allow_structured_erasure:
        for missing in range(4):
            others = [i for i in range(4) if i != missing]
            if any(structured[i].size == 0 for i in others) or parity.size == 0:
                continue
            a_set, b_set, c_set = [structured[i] for i in others]
            b_mesh, c_mesh, p_mesh = np.meshgrid(b_set, c_set, parity, indexing="ij")
            bf = b_mesh.ravel().astype(np.uint32)
            cf = c_mesh.ravel().astype(np.uint32)
            pf = p_mesh.ravel().astype(np.uint32)
            for a0 in a_set:
                af = np.full(bf.size, a0, dtype=np.uint32)
                vals: list[np.ndarray | None] = [None, None, None, None]
                vals[others[0]] = af
                vals[others[1]] = bf
                vals[others[2]] = cf
                vals[missing] = pf ^ af ^ bf ^ cf
                cand = _pack_bytes(vals[0], vals[1], vals[2], vals[3])  # type: ignore[arg-type]
                cand = _strict_gate(cand, views, membership, early_group)
                if cand.size:
                    chunks.append(cand)

    if not chunks:
        return np.empty(0, dtype=np.uint32)
    candidates = np.unique(np.concatenate(chunks))

    validation_count = len(views) - 5
    score = np.zeros(candidates.size, dtype=np.int16)
    for v in range(5, len(views)):
        z = views[v].hash_ids(candidates).astype(np.int64)
        score += membership[v][z]
    threshold = max(3, validation_count - 2)
    keep = score >= threshold
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
            consistency = float((singular[0] ** 2) / (np.sum(singular**2) + 1e-18))
        records.append(SupportCandidate(int(identity), int(view_support), consistency))
        # A candidate that only reaches the minimum view support is inherently one view away
        # from a hash-list ambiguity. Require a modestly stronger receiver-space consistency
        # certificate in that boundary case; candidates with redundant view support use the
        # nominal threshold.
        boundary_threshold = spatial_consistency_threshold + 0.05
        required_consistency = (
            boundary_threshold if int(view_support) == int(min_view_support) else spatial_consistency_threshold
        )
        if consistency >= required_consistency:
            accepted.append(int(identity))
    return SupportResult(np.asarray(sorted(set(accepted)), dtype=np.uint32), tuple(records))


def recover_global_support(
    detected: list[np.ndarray],
    spectra: list[np.ndarray],
    views: list[LinearHashView],
    *,
    min_view_support: int | None = None,
    spatial_consistency_threshold: float = 0.47,
    max_pre_candidates: int = 20000,
    early_group: tuple[int, ...] = (5, 6, 7),
    allow_structured_erasure: bool = False,
) -> SupportResult:
    """Recover global 32-bit support from local hash views without population scanning."""
    if len(detected) != len(views) or len(spectra) != len(views):
        raise ValueError("detected/spectra/views length mismatch")
    candidates = _structured_candidates(
        detected,
        views,
        early_group=early_group,
        allow_structured_erasure=allow_structured_erasure,
        max_pre_candidates=max_pre_candidates,
    )
    if min_view_support is None:
        min_view_support = max(3, len(views) - 1)
    return _filter_candidates(
        candidates,
        detected,
        spectra,
        views,
        min_view_support=int(min_view_support),
        spatial_consistency_threshold=spatial_consistency_threshold,
    )
