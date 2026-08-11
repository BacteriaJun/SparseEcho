from __future__ import annotations

import numpy as np


def fwht(x: np.ndarray, *, axis: int = 0, normalize: bool = False) -> np.ndarray:
    """Fast Walsh-Hadamard transform along *axis*.

    The transform uses Sylvester/natural binary ordering. Input length on the selected axis
    must be a power of two. A copy is returned.
    """
    a = np.array(x, copy=True)
    n = a.shape[axis]
    if n < 1 or n & (n - 1):
        raise ValueError("FWHT axis length must be a power of two")
    a = np.swapaxes(a, 0, axis)
    h = 1
    while h < n:
        for start in range(0, n, 2 * h):
            left = a[start : start + h].copy()
            right = a[start + h : start + 2 * h].copy()
            a[start : start + h] = left + right
            a[start + h : start + 2 * h] = left - right
        h *= 2
    if normalize:
        a = a / n
    return np.swapaxes(a, 0, axis)


def gray_code(index: int | np.ndarray) -> int | np.ndarray:
    """Binary-reflected Gray code."""
    return index ^ (index >> 1)


def gray_inverse(code: int) -> int:
    """Inverse of binary-reflected Gray code for a scalar integer."""
    value = int(code)
    shift = 1
    while (value >> shift) > 0:
        value ^= value >> shift
        shift <<= 1
    # The doubling form above is compact but easy to misread. Verify by ordinary fold.
    out = 0
    g = int(code)
    while g:
        out ^= g
        g >>= 1
    return out


def gray_order(bits: int) -> np.ndarray:
    """Return local query labels in physical Gray execution order."""
    n = 1 << bits
    idx = np.arange(n, dtype=np.uint32)
    return gray_code(idx).astype(np.uint32)


def gray_time_generators(bits: int) -> np.ndarray:
    """Walsh masks that generate centered Gray-rank time.

    For local query u in F_2^r, the centered physical rank under reflected Gray execution is

        tau(u) = -1/2 * sum_i 2^i chi_{s_i}(u)

    where s_i has ones in bit positions i..r-1.
    """
    full = (1 << bits) - 1
    return np.array([full ^ ((1 << i) - 1) for i in range(bits)], dtype=np.uint32)


def centered_gray_rank(query: int | np.ndarray, bits: int) -> np.ndarray:
    """Centered physical rank j-(2^r-1)/2 for Gray-labeled queries."""
    q = np.asarray(query, dtype=np.uint32)
    flat = q.ravel()
    inv = np.fromiter((gray_inverse(int(v)) for v in flat), dtype=np.int64, count=flat.size)
    inv = inv.reshape(q.shape)
    return inv.astype(np.float64) - ((1 << bits) - 1) / 2.0


def parity_u32(values: np.ndarray | int) -> np.ndarray:
    """Parity of unsigned integers as 0/1 array."""
    x = np.asarray(values, dtype=np.uint64)
    y = x.copy()
    y ^= y >> 32
    y ^= y >> 16
    y ^= y >> 8
    y ^= y >> 4
    y &= 0xF
    lut = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0], dtype=np.uint8)
    return lut[y.astype(np.int64)]


def walsh_character(query: np.ndarray | int, index: np.ndarray | int) -> np.ndarray:
    """chi_index(query)=(-1)^(query dot index) over F_2."""
    p = parity_u32(np.bitwise_and(np.asarray(query, dtype=np.uint64), np.asarray(index, dtype=np.uint64)))
    return 1.0 - 2.0 * p.astype(np.float64)


def hadamard_matrix(bits: int, *, dtype=np.float64) -> np.ndarray:
    """Small dense Hadamard matrix, intended for local apertures (e.g. 2^8)."""
    n = 1 << bits
    h = np.array([[1.0]], dtype=dtype)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h
