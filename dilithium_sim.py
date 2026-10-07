"""
Pure-Python Dilithium2 simulation (reference, not constant-time).
Used for formal analysis — DO NOT use in production.

Key output: (signature, rejection_count, hint_polynomial) for each signing call.
"""

from __future__ import annotations
import hashlib
import os
from dataclasses import dataclass
from typing import Optional

from params import DilithiumParams, DILITHIUM2, TINY
from ring import (
    poly_add, poly_sub, poly_mul, poly_center, poly_reduce,
    power2round, decompose, make_hint_poly, chknorm,
    expand_A, sample_secret, matmul_vec,
)


@dataclass
class SecretKey:
    rho: bytes
    tr: bytes
    key: bytes
    s1: list[list[int]]   # L polynomials
    s2: list[list[int]]   # K polynomials
    t0: list[list[int]]   # K polynomials (low bits of t)


@dataclass
class PublicKey:
    rho: bytes
    t1: list[list[int]]   # K polynomials (high bits of t)


@dataclass
class SignatureData:
    c_tilde: bytes
    z: list[list[int]]    # L polynomials
    h: list[list[int]]    # K polynomials (hint)
    hint_count: int
    rejection_count: int  # number of goto-rej iterations


def keygen(p: DilithiumParams, seed: Optional[bytes] = None) -> tuple[PublicKey, SecretKey]:
    if seed is None:
        seed = os.urandom(32)

    expanded = hashlib.shake_256(seed + bytes([p.K, p.L])).digest(128)
    rho = expanded[:32]
    rhoprime = expanded[32:96]
    key = expanded[96:128]

    A = expand_A(rho, p.K, p.L, p.N, p.Q)

    s1 = [sample_secret(rhoprime, i, p.ETA, p.N) for i in range(p.L)]
    s2 = [sample_secret(rhoprime, p.L + i, p.ETA, p.N) for i in range(p.K)]

    # t = A*s1 + s2
    t = matmul_vec(A, s1, p.Q)
    for k in range(p.K):
        t[k] = poly_add(t[k], s2[k], p.Q)

    # t1, t0 = power2round(t)
    t1 = []
    t0 = []
    for k in range(p.K):
        hi, lo = power2round(t[k], p.D, p.Q)
        t1.append(hi)
        t0.append(lo)

    pk = PublicKey(rho=rho, t1=t1)
    tr = hashlib.shake_256(rho + _pack_t1(t1, p)).digest(64)
    sk = SecretKey(rho=rho, tr=tr, key=key, s1=s1, s2=s2, t0=t0)
    return pk, sk


def sign(p: DilithiumParams, sk: SecretKey, msg: bytes,
         randomized: bool = True) -> SignatureData:
    """Sign msg. Returns full SignatureData including hint and rejection count."""

    A = expand_A(sk.rho, p.K, p.L, p.N, p.Q)

    # mu = H(tr || msg)
    mu = hashlib.shake_256(sk.tr + msg).digest(64)

    if randomized:
        rnd = os.urandom(32)
    else:
        rnd = b'\x00' * 32

    # rhoprime = H(key || rnd || mu)
    rhoprime = hashlib.shake_256(sk.key + rnd + mu).digest(64)

    # NTT of secrets (simulate — we work in coefficient domain)
    nonce = 0
    rejection_count = 0

    while True:
        # Sample y
        y = [_sample_gamma1(rhoprime, nonce + i, p) for i in range(p.L)]
        nonce += p.L

        # w = A*y
        w = matmul_vec(A, y, p.Q)

        # w1 = highbits(w), w0 = lowbits(w)
        w1 = []
        w0 = []
        for k in range(p.K):
            hi, lo = decompose(w[k], p.GAMMA2, p.Q)
            w1.append(hi)
            w0.append(lo)

        # c = H(mu || w1)
        w1_packed = _pack_w1(w1, p)
        c_tilde = hashlib.shake_256(mu + w1_packed).digest(p.N // 4)
        c = _challenge(c_tilde, p)

        # z = y + c*s1
        z = []
        for l in range(p.L):
            cs1_l = poly_mul(c, sk.s1[l], p.Q)
            z_l = poly_add(y[l], cs1_l, p.Q)
            z.append(z_l)

        # Reject if ||z||_inf >= GAMMA1 - BETA
        reject = any(chknorm(z[l], p.GAMMA1 - p.BETA, p.Q) for l in range(p.L))
        if reject:
            rejection_count += 1
            continue

        # w0 - cs2
        cs2 = [poly_mul(c, sk.s2[k], p.Q) for k in range(p.K)]
        w0_cs2 = [poly_sub(w0[k], cs2[k], p.Q) for k in range(p.K)]

        # Reject if ||w0 - cs2||_inf >= GAMMA2 - BETA
        reject = any(chknorm(w0_cs2[k], p.GAMMA2 - p.BETA, p.Q) for k in range(p.K))
        if reject:
            rejection_count += 1
            continue

        # ct0
        ct0 = [poly_mul(c, sk.t0[k], p.Q) for k in range(p.K)]

        # Reject if ||ct0||_inf >= GAMMA2
        reject = any(chknorm(ct0[k], p.GAMMA2, p.Q) for k in range(p.K))
        if reject:
            rejection_count += 1
            continue

        # Compute hints
        h = []
        total_hints = 0
        for k in range(p.K):
            r = poly_add(w0_cs2[k], ct0[k], p.Q)
            r_centered = poly_center(r, p.Q)
            hk, cnt = make_hint_poly(r_centered, w1[k], p.GAMMA2)
            h.append(hk)
            total_hints += cnt

        # Reject if hint count > OMEGA
        if total_hints > p.OMEGA:
            rejection_count += 1
            continue

        return SignatureData(
            c_tilde=c_tilde,
            z=z,
            h=h,
            hint_count=total_hints,
            rejection_count=rejection_count,
        )


def _sample_gamma1(seed: bytes, idx: int, p: DilithiumParams) -> list[int]:
    """Sample polynomial with coefficients uniform in [-(GAMMA1-1), GAMMA1]."""
    h = hashlib.shake_256(seed + idx.to_bytes(4, 'little')).digest(p.N * 5)
    bits = p.GAMMA1.bit_length()
    coeffs = []
    i = 0
    while len(coeffs) < p.N:
        chunk = int.from_bytes(h[i:i + 4], 'little') & ((1 << (bits + 1)) - 1)
        i += 4
        if i >= len(h):
            h += hashlib.shake_256(seed + idx.to_bytes(4, 'little') + i.to_bytes(4, 'little')).digest(p.N * 5)
        v = chunk - p.GAMMA1
        if -(p.GAMMA1 - 1) <= v <= p.GAMMA1:
            coeffs.append(v % p.Q)
    return coeffs[:p.N]


def _challenge(c_tilde: bytes, p: DilithiumParams) -> list[int]:
    """Generate challenge polynomial with exactly TAU non-zero coefficients."""
    signs_stream = int.from_bytes(c_tilde[:8], 'little')
    h = hashlib.shake_256(c_tilde).digest(p.N)
    c = [0] * p.N
    count = 0
    i = p.N - 1
    sign_pos = 0
    indices = list(range(p.N))

    used = set()
    positions = []
    for byte in h:
        if len(positions) == p.TAU:
            break
        pos = byte % (i + 1)
        if pos not in used:
            positions.append(pos)
            used.add(pos)

    # Simpler: just use hash bytes as positions
    positions = []
    buf = hashlib.shake_256(c_tilde + b'challenge').digest(p.TAU * 4)
    for idx in range(p.TAU):
        pos = int.from_bytes(buf[idx * 4:(idx + 1) * 4], 'little') % p.N
        positions.append(pos)

    for k, pos in enumerate(positions):
        sign = 1 if (signs_stream >> k) & 1 == 0 else -1
        c[pos] = (c[pos] + sign) % p.Q

    return c


def _pack_t1(t1: list[list[int]], p: DilithiumParams) -> bytes:
    result = b''
    for poly in t1:
        for coeff in poly:
            result += (coeff % (1 << 16)).to_bytes(2, 'little')
    return result


def _pack_w1(w1: list[list[int]], p: DilithiumParams) -> bytes:
    result = b''
    for poly in w1:
        for coeff in poly:
            result += (coeff % (1 << 16)).to_bytes(2, 'little')
    return result
