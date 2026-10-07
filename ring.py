"""Polynomial ring arithmetic for Z_Q[X]/(X^N + 1)."""

from __future__ import annotations
from typing import Sequence


def poly_add(a: list[int], b: list[int], Q: int) -> list[int]:
    return [(x + y) % Q for x, y in zip(a, b)]


def poly_sub(a: list[int], b: list[int], Q: int) -> list[int]:
    return [(x - y) % Q for x, y in zip(a, b)]


def poly_mul(a: list[int], b: list[int], Q: int) -> list[int]:
    """Schoolbook polynomial multiplication in Z_Q[X]/(X^N + 1)."""
    N = len(a)
    result = [0] * N
    for i in range(N):
        for j in range(N):
            idx = (i + j) % N
            sign = -1 if (i + j) >= N else 1
            result[idx] = (result[idx] + sign * a[i] * b[j]) % Q
    return result


def poly_reduce(a: list[int], Q: int) -> list[int]:
    return [x % Q for x in a]


def center(v: int, Q: int) -> int:
    """Map v mod Q to centered representative in (-(Q-1)/2, (Q-1)/2]."""
    v = v % Q
    if v > (Q - 1) // 2:
        v -= Q
    return v


def poly_center(a: list[int], Q: int) -> list[int]:
    return [center(x, Q) for x in a]


def poly_norm_inf(a: list[int], Q: int) -> int:
    return max(abs(center(x, Q)) for x in a)


def chknorm(a: list[int], bound: int, Q: int) -> bool:
    """Returns True if infinity norm >= bound (i.e. REJECT)."""
    return any(abs(center(x, Q)) >= bound for x in a)


def power2round(a: list[int], D: int, Q: int) -> tuple[list[int], list[int]]:
    """Split a into (a1, a0) where a = a1 * 2^D + a0."""
    a1, a0 = [], []
    for v in a:
        v = center(v, Q)
        v1 = (v + (1 << (D - 1)) - 1) >> D
        v0 = v - (v1 << D)
        a1.append(v1)
        a0.append(v0)
    return a1, a0


def decompose(a: list[int], GAMMA2: int, Q: int) -> tuple[list[int], list[int]]:
    """FIPS 204 Decompose: a = a1 * 2*GAMMA2 + a0 with a1 in [0, (Q-1)/(2*GAMMA2) - 1]."""
    a1_out, a0_out = [], []
    ALPHA = 2 * GAMMA2
    for v in a:
        v = v % Q
        a0 = v % ALPHA
        if a0 > ALPHA // 2:
            a0 -= ALPHA
        if v - a0 == Q - 1:
            a1_out.append(0)
            a0_out.append(a0 - 1)
        else:
            a1_out.append((v - a0) // ALPHA)
            a0_out.append(a0)
    return a1_out, a0_out


def make_hint_poly(a0: list[int], a1: list[int], GAMMA2: int) -> tuple[list[int], int]:
    """Compute hint polynomial and return (h, popcount)."""
    h = []
    count = 0
    for v0, v1 in zip(a0, a1):
        bit = 1 if (v0 > GAMMA2 or v0 < -GAMMA2 or (v0 == -GAMMA2 and v1 != 0)) else 0
        h.append(bit)
        count += bit
    return h, count


def matmul_vec(A: list[list[list[int]]], s: list[list[int]], Q: int) -> list[list[int]]:
    """Matrix-vector product A*s over polynomial ring. A is KxL matrix of polys."""
    K = len(A)
    L = len(A[0])
    N = len(A[0][0])
    result = [[0] * N for _ in range(K)]
    for k in range(K):
        for l in range(L):
            prod = poly_mul(A[k][l], s[l], Q)
            result[k] = poly_add(result[k], prod, Q)
    return result


def expand_A(rho: bytes, K: int, L: int, N: int, Q: int) -> list[list[list[int]]]:
    """Generate matrix A from seed rho using SHAKE128."""
    import hashlib
    A = []
    for i in range(K):
        row = []
        for j in range(L):
            seed = rho + bytes([j, i])
            h = hashlib.shake_128(seed).digest(N * 4)
            coeffs = []
            idx = 0
            while len(coeffs) < N:
                b0, b1, b2 = h[idx], h[idx + 1], h[idx + 2]
                idx += 3
                if idx + 3 > len(h):
                    h += hashlib.shake_128(seed + idx.to_bytes(4, 'little')).digest(N * 4)
                t = ((b2 & 0x7F) << 16) | (b1 << 8) | b0
                if t < Q:
                    coeffs.append(t)
            row.append(coeffs[:N])
        A.append(row)
    return A


def sample_secret(seed: bytes, index: int, ETA: int, N: int) -> list[int]:
    """Sample short polynomial with coefficients in [-ETA, ETA]."""
    import hashlib
    h = hashlib.shake_256(seed + bytes([index])).digest(N * 4)
    coeffs = []
    i = 0
    while len(coeffs) < N:
        b = h[i % len(h)]
        i += 1
        if ETA == 1:
            b0 = b & 0xF
            b1 = b >> 4
            for bv in [b0, b1]:
                if bv < 3 and len(coeffs) < N:
                    coeffs.append(1 - bv)
        elif ETA == 2:
            b0 = b & 0xF
            b1 = b >> 4
            for bv in [b0, b1]:
                if bv < 15 and len(coeffs) < N:
                    coeffs.append(2 - (bv % 5))
        else:  # ETA == 4
            b0 = b & 0xF
            b1 = b >> 4
            for bv in [b0, b1]:
                if bv < 9 and len(coeffs) < N:
                    coeffs.append(4 - bv)
    return coeffs[:N]
