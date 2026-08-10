from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gf2_rank(matrix: np.ndarray) -> int:
    a = np.asarray(matrix, dtype=np.uint8).copy() & 1
    rows, cols = a.shape
    rank = 0
    for col in range(cols):
        pivot = np.flatnonzero(a[rank:, col])
        if pivot.size == 0:
            continue
        p = rank + int(pivot[0])
        a[[rank, p]] = a[[p, rank]]
        for r in range(rows):
            if r != rank and a[r, col]:
                a[r] ^= a[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def _matrix_row_masks(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.uint8) & 1
    r, m = matrix.shape
    if m > 64:
        raise ValueError("row-mask representation supports identity_bits <= 64")
    masks = np.zeros(r, dtype=np.uint64)
    for i in range(r):
        value = 0
        for bit in np.flatnonzero(matrix[i]):
            value |= 1 << int(bit)
        masks[i] = value
    return masks


def _byte_luts(matrix: np.ndarray) -> tuple[np.ndarray, ...]:
    matrix = np.asarray(matrix, dtype=np.uint8) & 1
    r, m = matrix.shape
    nbytes = (m + 7) // 8
    out: list[np.ndarray] = []
    for byte in range(nbytes):
        sub = matrix[:, byte * 8 : min((byte + 1) * 8, m)]
        lut = np.zeros(256, dtype=np.uint16)
        for value in range(256):
            bits = np.array([(value >> i) & 1 for i in range(sub.shape[1])], dtype=np.uint8)
            z = (sub @ bits) & 1
            encoded = 0
            for i, bit in enumerate(z):
                encoded |= int(bit) << i
            lut[value] = encoded
        out.append(lut)
    return tuple(out)


@dataclass(frozen=True)
class LinearHashView:
    """Linear identity projection M q over F_2."""

    matrix: np.ndarray
    name: str
    structured: bool = False

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=np.uint8) & 1
        object.__setattr__(self, "matrix", m)
        object.__setattr__(self, "row_masks", _matrix_row_masks(m))
        object.__setattr__(self, "byte_luts", _byte_luts(m))

    @property
    def local_bits(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def identity_bits(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def bucket_count(self) -> int:
        return 1 << self.local_bits

    def hash_ids(self, identities: np.ndarray | int) -> np.ndarray:
        ids = np.asarray(identities, dtype=np.uint64)
        z = np.zeros(ids.shape, dtype=np.uint16)
        for byte, lut in enumerate(self.byte_luts):
            z ^= lut[((ids >> (8 * byte)) & 0xFF).astype(np.int64)]
        return z

    def challenge(self, local_query: int) -> int:
        """Return global parity mask a=M^T u encoded as an integer."""
        value = np.uint64(0)
        u = int(local_query)
        for bit, row_mask in enumerate(self.row_masks):
            if (u >> bit) & 1:
                value ^= row_mask
        return int(value)

    def challenge_table(self) -> np.ndarray:
        return np.array([self.challenge(u) for u in range(self.bucket_count)], dtype=np.uint64)


def _random_full_rank(local_bits: int, identity_bits: int, rng: np.random.Generator) -> np.ndarray:
    while True:
        matrix = rng.integers(0, 2, size=(local_bits, identity_bits), dtype=np.uint8)
        if gf2_rank(matrix) == local_bits:
            return matrix


def default_hash_views(
    *, identity_bits: int = 32, local_bits: int = 8, n_views: int = 14, seed: int = 0x5EED
) -> list[LinearHashView]:
    """Build the public 32-bit/8-bit view family.

    The first four views are systematic byte projections, the fifth is a parity projection,
    and the remaining views are deterministic random full-rank projections used for list
    consistency. Coding/list-recovery is an implementation tool rather than a novelty claim.
    """
    if identity_bits != 4 * local_bits:
        raise ValueError("default structured family expects identity_bits == 4 * local_bits")
    if n_views < 6:
        raise ValueError("n_views must be at least 6")

    views: list[LinearHashView] = []
    for block in range(4):
        matrix = np.zeros((local_bits, identity_bits), dtype=np.uint8)
        for i in range(local_bits):
            matrix[i, block * local_bits + i] = 1
        views.append(LinearHashView(matrix, f"systematic-{block}", True))

    parity = np.zeros((local_bits, identity_bits), dtype=np.uint8)
    for i in range(local_bits):
        for block in range(4):
            parity[i, block * local_bits + i] = 1
    views.append(LinearHashView(parity, "parity-0", True))

    rng = np.random.default_rng(seed)
    while len(views) < n_views:
        matrix = _random_full_rank(local_bits, identity_bits, rng)
        views.append(LinearHashView(matrix, f"validation-{len(views)-5}", False))
    return views
