"""
Fast ML-DSA signing simulation using numpy.
Core logic identical to reference — optimized for analysis, not constant-time.
"""

from __future__ import annotations
import os
import hashlib
from dataclasses import dataclass

import numpy as np

from params import DilithiumParams


def _shake256(data: bytes, n: int) -> bytes:
    return hashlib.shake_256(data).digest(n)


def _center(v: int, Q: int) -> int:
    v = v % Q
    return v - Q if v > (Q - 1) // 2 else v


def _center_arr(a: np.ndarray, Q: int) -> np.ndarray:
    a = a % Q
    return np.where(a > (Q - 1) // 2, a - Q, a)


def _poly_mul(a: np.ndarray, b: np.ndarray, Q: int) -> np.ndarray:
    """Poly mul in Z_Q[X]/(X^N+1) using numpy."""
    N = len(a)
    conv = np.convolve(a, b)
    result = conv[:N].copy()
    if len(conv) > N:
        result[:len(conv) - N] -= conv[N:]
    return result % Q


def _poly_chknorm(a: np.ndarray, bound: int, Q: int) -> bool:
    """Returns True if ||a||_inf >= bound (REJECT)."""
    return bool(np.any(np.abs(_center_arr(a, Q)) >= bound))


def expand_A(rho: bytes, K: int, L: int, N: int, Q: int) -> np.ndarray:
    """Matrix A from seed rho. Returns (K, L, N) array."""
    A = np.zeros((K, L, N), dtype=np.int64)
    for i in range(K):
        for j in range(L):
            seed = rho + bytes([j, i])
            h = _shake256(seed, N * 6)
            coeffs = []
            pos = 0
            while len(coeffs) < N:
                if pos + 3 > len(h):
                    h += _shake256(seed + pos.to_bytes(4, 'little'), N * 6)
                b0, b1, b2 = h[pos], h[pos+1], h[pos+2]
                pos += 3
                t = ((b2 & 0x7F) << 16) | (b1 << 8) | b0
                if t < Q:
                    coeffs.append(t)
            A[i, j] = coeffs[:N]
    return A


def sample_secret(seed: bytes, index: int, ETA: int, N: int) -> np.ndarray:
    """Short polynomial in [-ETA, ETA]."""
    h = _shake256(seed + index.to_bytes(4, 'little'), N * 6)
    coeffs = []
    i = 0
    while len(coeffs) < N:
        b = h[i % len(h)]
        i += 1
        if ETA == 2:
            for bv in [b & 0xF, b >> 4]:
                if bv < 15 and len(coeffs) < N:
                    coeffs.append(2 - (bv % 5))
        else:
            for bv in [b & 0xF, b >> 4]:
                if bv < 9 and len(coeffs) < N:
                    coeffs.append(4 - bv)
    return np.array(coeffs[:N], dtype=np.int64)


def power2round(a: np.ndarray, D: int, Q: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a into (a1*2^D + a0)."""
    ac = _center_arr(a, Q)
    a1 = (ac + (1 << (D - 1)) - 1) >> D
    a0 = ac - (a1 << D)
    return a1, a0


def decompose_scalar(v: int, GAMMA2: int, Q: int) -> tuple[int, int]:
    v = v % Q
    ALPHA = 2 * GAMMA2
    a1 = (v + ALPHA // 2) // ALPHA
    if a1 == (Q - 1) // ALPHA + 1:
        a1 = 0
        a0 = v - Q
    else:
        a0 = v - a1 * ALPHA
    return a1, a0


def challenge(c_tilde: bytes, p: DilithiumParams) -> np.ndarray:
    """TAU-weight challenge polynomial."""
    signs_bits = int.from_bytes(c_tilde[:8], 'little')
    positions = []
    seen: set[int] = set()
    h = _shake256(c_tilde + b'challenge', p.N * 4)
    idx = 0
    while len(positions) < p.TAU:
        if idx + 4 > len(h):
            h += _shake256(h, p.N * 4)
        pos = int.from_bytes(h[idx:idx+4], 'little') % p.N
        idx += 4
        if pos not in seen:
            positions.append(pos)
            seen.add(pos)
    c = np.zeros(p.N, dtype=np.int64)
    for k, pos in enumerate(positions):
        sign = -1 if (signs_bits >> k) & 1 else 1
        c[pos] = (c[pos] + sign) % p.Q
    return c


@dataclass
class FastKey:
    rho: bytes
    s1: np.ndarray  # (L, N)
    s2: np.ndarray  # (K, N)
    t0: np.ndarray  # (K, N)
    t1: np.ndarray  # (K, N)


def fast_keygen(p: DilithiumParams, seed: bytes | None = None) -> FastKey:
    if seed is None:
        seed = os.urandom(32)
    expanded = _shake256(seed, 96)
    rho = expanded[:32]
    rhoprime = expanded[32:]

    A = expand_A(rho, p.K, p.L, p.N, p.Q)
    s1 = np.stack([sample_secret(rhoprime, i, p.ETA, p.N) for i in range(p.L)])
    s2 = np.stack([sample_secret(rhoprime, p.L + i, p.ETA, p.N) for i in range(p.K)])

    t = np.zeros((p.K, p.N), dtype=np.int64)
    for k in range(p.K):
        for l in range(p.L):
            t[k] = (t[k] + _poly_mul(A[k, l], s1[l], p.Q)) % p.Q
        t[k] = (t[k] + s2[k]) % p.Q

    t1_arr = np.zeros((p.K, p.N), dtype=np.int64)
    t0_arr = np.zeros((p.K, p.N), dtype=np.int64)
    for k in range(p.K):
        t1_arr[k], t0_arr[k] = power2round(t[k], p.D, p.Q)

    return FastKey(rho=rho, s1=s1, s2=s2, t0=t0_arr, t1=t1_arr)


@dataclass
class FastSignResult:
    total_rejections: int
    check1_rejections: int
    check2_rejections: int
    check3_rejections: int
    check4_rejections: int
    hint_count: int
    c_final: np.ndarray  # accepted challenge
    challenges_c3_rejected: list[np.ndarray]


def fast_sign(p: DilithiumParams, key: FastKey, msg: bytes) -> FastSignResult:
    """Instrumented signing using numpy. ~10x faster than pure Python."""
    A = expand_A(key.rho, p.K, p.L, p.N, p.Q)
    mu = _shake256(key.rho + msg, 64)
    rnd = os.urandom(32)
    rhoprime = _shake256(key.rho + rnd + mu, 64)

    total = c1 = c2 = c3 = c4 = 0
    challenges_c3: list[np.ndarray] = []
    nonce = 0

    while True:
        # Sample y in [-GAMMA1+1, GAMMA1]
        y = np.zeros((p.L, p.N), dtype=np.int64)
        for l in range(p.L):
            buf = _shake256(rhoprime + nonce.to_bytes(4, 'little'), p.N * 6)
            coeffs = []
            pos = 0
            while len(coeffs) < p.N:
                if pos + 4 > len(buf):
                    buf += _shake256(buf, p.N * 6)
                v = int.from_bytes(buf[pos:pos+4], 'little') % (2 * p.GAMMA1)
                pos += 4
                yv = v - p.GAMMA1
                if -(p.GAMMA1 - 1) <= yv <= p.GAMMA1:
                    coeffs.append(yv)
            y[l] = coeffs[:p.N]
            nonce += 1

        # w = A*y
        w = np.zeros((p.K, p.N), dtype=np.int64)
        for k in range(p.K):
            for l in range(p.L):
                w[k] = (w[k] + _poly_mul(A[k, l], y[l], p.Q)) % p.Q

        # decompose w → w1, w0
        w1 = np.zeros((p.K, p.N), dtype=np.int64)
        w0 = np.zeros((p.K, p.N), dtype=np.int64)
        for k in range(p.K):
            for i in range(p.N):
                a1, a0 = decompose_scalar(int(w[k, i]), p.GAMMA2, p.Q)
                w1[k, i] = a1
                w0[k, i] = a0

        # Challenge from w1
        w1_packed = w1.tobytes()
        c_tilde = _shake256(mu + w1_packed, 32)
        c = challenge(c_tilde, p)

        # z = y + c*s1; check 1
        z = np.zeros((p.L, p.N), dtype=np.int64)
        reject = False
        for l in range(p.L):
            cs1_l = _poly_mul(c, key.s1[l], p.Q)
            z[l] = (y[l] + cs1_l) % p.Q
            if _poly_chknorm(z[l], p.GAMMA1 - p.BETA, p.Q):
                reject = True
                break
        if reject:
            total += 1; c1 += 1
            continue

        # cs2; check 2
        reject = False
        for k in range(p.K):
            cs2_k = _poly_mul(c, key.s2[k], p.Q)
            w0_cs2 = (w0[k] - cs2_k) % p.Q
            if _poly_chknorm(w0_cs2, p.GAMMA2 - p.BETA, p.Q):
                reject = True
                break
        if reject:
            total += 1; c2 += 1
            continue

        # ct0; check 3 ← SECRET-DEPENDENT
        reject = False
        for k in range(p.K):
            ct0_k = _poly_mul(c, key.t0[k], p.Q)
            if _poly_chknorm(ct0_k, p.GAMMA2, p.Q):
                reject = True
                break
        if reject:
            total += 1; c3 += 1
            challenges_c3.append(c.copy())
            continue

        # Hint count; check 4
        hint_total = 0
        for k in range(p.K):
            cs2_k = _poly_mul(c, key.s2[k], p.Q)
            ct0_k = _poly_mul(c, key.t0[k], p.Q)
            combined = _center_arr((w0[k] - cs2_k + ct0_k) % p.Q, p.Q)
            hint_total += int(np.sum(np.abs(combined) > p.GAMMA2))

        if hint_total > p.OMEGA:
            total += 1; c4 += 1
            continue

        return FastSignResult(
            total_rejections=total,
            check1_rejections=c1,
            check2_rejections=c2,
            check3_rejections=c3,
            check4_rejections=c4,
            hint_count=hint_total,
            c_final=c,
            challenges_c3_rejected=challenges_c3,
        )
