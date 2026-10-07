"""
Idealized oracle at ML-DSA-44 parameters (Section 6 of the paper): the full key
(s1, s2, t0) determined from one exact oracle answer on c*t0 and one on c*s1.

NOT AN ATTACK. Both oracle answers are computed from the secret key; nothing
here is measured on a device or derived from public signatures. Published
side-channel attacks recover such intermediate values partially and with noise;
obtaining the exact values assumed here is not addressed.

STEPS:
  1. t0 from c*t0: solve M(c) t0 = ct0 mod Q (M(c) the negacyclic matrix of c).
  2. s1 from c*s1: same linear system.
  3. s2 = 2^D t1 + t0 - A s1 (key-generation identity; needs t0).
This is linear algebra and touches no lattice problem: exact access to both
products at one signing attempt determines the key.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dilithium_sim import keygen, PublicKey, SecretKey, _sample_gamma1, _challenge as _ch, _pack_w1  # type: ignore[attr-defined]
from params import DilithiumParams, DILITHIUM2
from ring import expand_A, matmul_vec, poly_mul, poly_add, poly_sub, poly_center, decompose, center

BANNER = "=" * 64
SEP    = "─" * 64
Q      = DILITHIUM2.Q


# ── Negacyclic matrix inversion (shared by Phase 1 and Phase 3) ──────────────

def build_negacyclic_matrix(f: list[int], N: int, Q: int) -> np.ndarray:
    M = np.zeros((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            idx  = (i - j) % N
            sign = -1 if i < j else 1
            M[i, j] = (sign * f[idx]) % Q
    return M


def mat_inv_mod_q(M: np.ndarray, Q: int) -> np.ndarray | None:
    n   = M.shape[0]
    aug = np.zeros((n, 2 * n), dtype=object)
    for i in range(n):
        for j in range(n):
            aug[i, j] = int(M[i, j]) % Q
        aug[i, n + i] = 1

    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r, col] % Q != 0), -1)
        if pivot == -1:
            return None
        aug[[col, pivot]] = aug[[pivot, col]]
        inv_p = pow(int(aug[col, col]) % Q, -1, Q)
        for j in range(2 * n):
            aug[col, j] = int(aug[col, j]) * inv_p % Q
        for row in range(n):
            if row == col:
                continue
            factor = int(aug[row, col]) % Q
            if factor == 0:
                continue
            for j in range(2 * n):
                aug[row, j] = (int(aug[row, j]) - factor * int(aug[col, j])) % Q

    return aug[:, n:]


def invert_from_product(c: list[int], cx: list[int], N: int, Q: int) -> list[int] | None:
    """Given c and c·x, recover x via Gaussian elimination over Z_Q."""
    M     = build_negacyclic_matrix(c, N, Q)
    M_inv = mat_inv_mod_q(M, Q)
    if M_inv is None:
        return None
    cx_vec = np.array([v % Q for v in cx], dtype=object)
    x_vec  = np.array([
        sum(int(M_inv[i, j]) * int(cx_vec[j]) for j in range(N)) % Q
        for i in range(N)
    ])
    return [center(int(v), Q) for v in x_vec]


# ── Assumed oracle: exact products computed from the secret key ──────────────

def oracle_ct0(p: DilithiumParams, sk: SecretKey, n_obs: int = 1) -> list[dict]:
    """
    Software oracle: returns exact c·t0[k][i] per coefficient.
    Computed from sk.t0; not a measurement.
    """
    A   = expand_A(sk.rho, p.K, p.L, p.N, p.Q)
    obs = []
    for msg_i in range(n_obs):
        msg      = f"oracle_t0_{msg_i}".encode()
        mu       = hashlib.shake_256(sk.tr + msg).digest(64)
        rnd      = os.urandom(32)
        rhoprime = hashlib.shake_256(sk.key + rnd + mu).digest(64)
        y        = [_sample_gamma1(rhoprime, i, p) for i in range(p.L)]
        w        = matmul_vec(A, y, p.Q)
        w1       = [decompose(w[k], p.GAMMA2, p.Q)[0] for k in range(p.K)]
        c        = _ch(hashlib.shake_256(mu + _pack_w1(w1, p)).digest(p.N // 4), p)
        ct0_vals = [
            [center(v, p.Q) for v in poly_mul(c, sk.t0[k], p.Q)]
            for k in range(p.K)
        ]
        obs.append({"c": c, "vals": ct0_vals})
    return obs


def oracle_cs1(p: DilithiumParams, sk: SecretKey, n_obs: int = 1) -> list[dict]:
    """
    Software oracle: returns exact c·s1[l][i] per coefficient.
    Computed from sk.s1; not a measurement.
    """
    A   = expand_A(sk.rho, p.K, p.L, p.N, p.Q)
    obs = []
    for msg_i in range(n_obs):
        msg      = f"oracle_s1_{msg_i}".encode()
        mu       = hashlib.shake_256(sk.tr + msg).digest(64)
        rnd      = os.urandom(32)
        rhoprime = hashlib.shake_256(sk.key + rnd + mu).digest(64)
        y        = [_sample_gamma1(rhoprime, i, p) for i in range(p.L)]
        w        = matmul_vec(A, y, p.Q)
        w1       = [decompose(w[k], p.GAMMA2, p.Q)[0] for k in range(p.K)]
        c        = _ch(hashlib.shake_256(mu + _pack_w1(w1, p)).digest(p.N // 4), p)
        cs1_vals = [
            [center(v, p.Q) for v in poly_mul(c, sk.s1[l], p.Q)]
            for l in range(p.L)
        ]
        obs.append({"c": c, "vals": cs1_vals})
    return obs


# ── Phase 1: t0 recovery ──────────────────────────────────────────────────────

def phase1_recover_t0(p: DilithiumParams, sk: SecretKey) -> list[list[int]] | None:
    print(f"\n{SEP}")
    print("PHASE 1 — t0 recovery via c·t0 oracle + matrix inversion")
    print(SEP)

    t0_true = [[center(v, p.Q) for v in sk.t0[k]] for k in range(p.K)]
    t0_rec  = [None] * p.K

    for attempt in range(10):
        obs = oracle_ct0(p, sk, n_obs=1)[0]
        c   = obs["c"]
        t0_start = time.time()
        M     = build_negacyclic_matrix(c, p.N, p.Q)
        M_inv = mat_inv_mod_q(M, p.Q)
        ms    = (time.time() - t0_start) * 1000

        if M_inv is None:
            print(f"  Attempt {attempt}: M(c) singular — retry")
            continue

        print(f"  M(c) inverted  ({ms:.0f}ms)")
        for k in range(p.K):
            if t0_rec[k] is not None:
                continue
            cx_vec = np.array([v % p.Q for v in obs["vals"][k]], dtype=object)
            x_vec  = np.array([
                sum(int(M_inv[i, j]) * int(cx_vec[j]) for j in range(p.N)) % p.Q
                for i in range(p.N)
            ])
            t0_k = [center(int(v), p.Q) for v in x_vec]
            # Accepted without looking at the key; the comparison is reported only.
            t0_rec[k] = t0_k
            print(f"  k={k}: {'equal to' if t0_k == t0_true[k] else 'DIFFERS from'} the generated t0")

        if all(r is not None for r in t0_rec):
            break

    return t0_rec if all(r is not None for r in t0_rec) else None


# ── Phase 3: s1 recovery ──────────────────────────────────────────────────────

def phase3_recover_s1(p: DilithiumParams, sk: SecretKey) -> list[list[int]] | None:
    print(f"\n{SEP}")
    print("PHASE 3 — s1 recovery via c·s1 oracle + matrix inversion")
    print(SEP)

    s1_true = [[center(v, p.Q) for v in sk.s1[l]] for l in range(p.L)]
    s1_rec  = [None] * p.L

    for attempt in range(10):
        obs = oracle_cs1(p, sk, n_obs=1)[0]
        c   = obs["c"]
        t0_start = time.time()
        M     = build_negacyclic_matrix(c, p.N, p.Q)
        M_inv = mat_inv_mod_q(M, p.Q)
        ms    = (time.time() - t0_start) * 1000

        if M_inv is None:
            print(f"  Attempt {attempt}: M(c) singular — retry")
            continue

        print(f"  M(c) inverted  ({ms:.0f}ms)")
        for l in range(p.L):
            if s1_rec[l] is not None:
                continue
            cx_vec = np.array([v % p.Q for v in obs["vals"][l]], dtype=object)
            x_vec  = np.array([
                sum(int(M_inv[i, j]) * int(cx_vec[j]) for j in range(p.N)) % p.Q
                for i in range(p.N)
            ])
            s1_l = [center(int(v), p.Q) for v in x_vec]
            # Accepted without looking at the key; the comparison is reported only.
            s1_rec[l] = s1_l
            print(f"  l={l}: {'equal to' if s1_l == s1_true[l] else 'DIFFERS from'} the generated s1")

        if all(r is not None for r in s1_rec):
            break

    return s1_rec if all(r is not None for r in s1_rec) else None


# ── Phase 2: s2 algebraic recovery ───────────────────────────────────────────

def phase2_recover_s2(
    p: DilithiumParams,
    sk: SecretKey,
    pk: PublicKey,
    t0_rec: list[list[int]],
    s1_rec: list[list[int]],
) -> list[list[int]]:
    """s2 = 2^D·t1 + t0 - A·s1  (keygen identity, no oracle needed)"""
    print(f"\n{SEP}")
    print("PHASE 2 — s2 algebraic recovery (keygen identity)")
    print(SEP)

    A  = expand_A(sk.rho, p.K, p.L, p.N, p.Q)
    As1 = matmul_vec(A, [[v % p.Q for v in s1_rec[l]] for l in range(p.L)], p.Q)

    s2_rec = []
    s2_true = [[center(v, p.Q) for v in sk.s2[k]] for k in range(p.K)]

    for k in range(p.K):
        t1k  = [v * (2 ** p.D) % p.Q for v in pk.t1[k]]
        t0k  = [v % p.Q for v in t0_rec[k]]
        tk   = poly_add(t1k, t0k, p.Q)
        s2k  = poly_sub(tk, As1[k], p.Q)
        s2k  = [center(v, p.Q) for v in s2k]
        match = (s2k == s2_true[k])
        print(f"  k={k}: {'EXACT ✓' if match else '✗ mismatch'}")
        s2_rec.append(s2k)

    return s2_rec


# ── Verification ──────────────────────────────────────────────────────────────

def verify_full_key(
    p: DilithiumParams,
    sk: SecretKey,
    t0_rec: list[list[int]],
    s1_rec: list[list[int]],
    s2_rec: list[list[int]],
) -> bool:
    t0_ok = all(
        [center(v, p.Q) for v in sk.t0[k]] == t0_rec[k]
        for k in range(p.K)
    )
    s1_ok = all(
        [center(v, p.Q) for v in sk.s1[l]] == s1_rec[l]
        for l in range(p.L)
    )
    s2_ok = all(
        [center(v, p.Q) for v in sk.s2[k]] == s2_rec[k]
        for k in range(p.K)
    )
    return t0_ok and s1_ok and s2_ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = DILITHIUM2  # Real ML-DSA-44: N=256, K=4, L=4

    print()
    print(BANNER)
    print("Idealized oracle, ML-DSA-44 (N=256): key from exact c*t0 and c*s1 answers")
    print(BANNER)
    print(f"  Params  : {p.name}  N={p.N}  K={p.K}  L={p.L}  Q={p.Q}")
    print(f"  D={p.D}  ETA={p.ETA}  TAU={p.TAU}  GAMMA2={p.GAMMA2}")
    print(f"  Oracle  : assumed, exact, computed from the secret key (not a measurement)")
    print(f"  Target  : full private key (s1, s2, t0) — {(p.K+p.L)*p.N} secret coefficients")
    print()

    seed    = hashlib.sha256(b"ironproof-full-key-real").digest()
    pk, sk  = keygen(p, seed=seed)

    t_total = time.time()

    # Phase 1
    t1_start = time.time()
    t0_rec   = phase1_recover_t0(p, sk)
    t1_ms    = (time.time() - t1_start) * 1000

    if t0_rec is None:
        print("\n[FAIL] Phase 1 failed — t0 not recovered")
        return

    # Phase 3
    t3_start = time.time()
    s1_rec   = phase3_recover_s1(p, sk)
    t3_ms    = (time.time() - t3_start) * 1000

    if s1_rec is None:
        print("\n[FAIL] Phase 3 failed — s1 not recovered")
        return

    # Phase 2
    t2_start = time.time()
    s2_rec   = phase2_recover_s2(p, sk, pk, t0_rec, s1_rec)
    t2_ms    = (time.time() - t2_start) * 1000

    total_ms = (time.time() - t_total) * 1000

    # Verify
    ok = verify_full_key(p, sk, t0_rec, s1_rec, s2_rec)

    print(f"\n{BANNER}")
    print(f"  Key from oracle answers — {'equal to the generated key' if ok else 'MISMATCH'}")
    print(BANNER)
    print(f"  t0 : {p.K}×{p.N} = {p.K*p.N:,} coefficients   Phase 1  {t1_ms:.0f}ms")
    print(f"  s1 : {p.L}×{p.N} = {p.L*p.N:,} coefficients   Phase 3  {t3_ms:.0f}ms")
    print(f"  s2 : {p.K}×{p.N} = {p.K*p.N:,} coefficients   Phase 2  {t2_ms:.0f}ms  (algebraic)")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Total: {(p.K+p.L+p.K)*p.N:,} coefficients determined in {total_ms:.0f}ms")
    print()
    print(f"  SIZES:")
    bits_t0 = p.D * p.K * p.N
    bits_s  = (p.ETA.bit_length() + 1) * (p.L + p.K) * p.N
    print(f"  t0 : {bits_t0:,} bits")
    print(f"  s1 : {p.L*p.N} coefficients ∈ [-{p.ETA},{p.ETA}]")
    print(f"  s2 : {p.K*p.N} coefficients ∈ [-{p.ETA},{p.ETA}]")
    print()
    print(f"  Oracle model: exact and noise-free, computed from the secret key")
    print(f"  Obtaining such an oracle from a device: not addressed")
    print(BANNER)


if __name__ == "__main__":
    main()
