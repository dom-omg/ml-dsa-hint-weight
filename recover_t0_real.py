"""
Idealized oracle at ML-DSA-44 parameters (Section 6 of the paper): t0 determined
from one exact oracle answer on c*t0.

NOT AN ATTACK. The oracle answer c*t0 is computed from the secret key sk.t0;
nothing here is measured on a device or derived from public signatures.
Published side-channel attacks recover such intermediate values partially and
with noise; obtaining the exact values assumed here is not addressed.

MATH:
  ct0[k] = c * t0[k]  in Z_Q[x]/(x^N+1)
  when the negacyclic matrix M(c) is invertible mod Q, t0[k] = M(c)^(-1) ct0[k].
  This is linear algebra; it involves no lattice problem.

SCOPE:
  t0 alone does not give the signing key: recovering s1, s2 from
  t = A*s1 + s2 is Module-LWE. The Dilithium designers do not treat t0 as secret.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from dilithium_sim import keygen, PublicKey, SecretKey
from params import DilithiumParams, DILITHIUM2
from ring import (
    expand_A, matmul_vec, poly_mul, poly_add, poly_sub,
    poly_center, decompose, center,
)

BANNER = "=" * 64
SEP    = "─" * 64

Q = DILITHIUM2.Q


# ─── Polynomial inversion via negacyclic convolution matrix ──────────────────

import numpy as np

def build_negacyclic_matrix(f: list[int], N: int, Q: int):
    """
    Build the N×N negacyclic convolution matrix M(f) over Z_Q.
    (M(f) · g)[i] = (f * g mod x^N+1)[i]

    M(f)[i, j] = f[(i-j) mod N] * sign, where sign=-1 if i<j else 1
    """
    M = np.zeros((N, N), dtype=object)  # object dtype for big integers
    for i in range(N):
        for j in range(N):
            idx = (i - j) % N
            sign = -1 if i < j else 1
            M[i, j] = (sign * f[idx]) % Q
    return M


def mat_inv_mod_q(M, Q: int, N: int):
    """
    Compute M^(-1) mod Q using Gaussian elimination over Z_Q.
    Returns None if M is singular mod Q.
    """
    # Augment [M | I]
    n = M.shape[0]
    aug = np.zeros((n, 2 * n), dtype=object)
    for i in range(n):
        for j in range(n):
            aug[i, j] = int(M[i, j]) % Q
        aug[i, n + i] = 1

    for col in range(n):
        # Find pivot
        pivot = -1
        for row in range(col, n):
            if aug[row, col] % Q != 0:
                pivot = row
                break
        if pivot == -1:
            return None  # singular
        aug[[col, pivot]] = aug[[pivot, col]]

        inv_pivot = pow(int(aug[col, col]) % Q, -1, Q)
        for j in range(2 * n):
            aug[col, j] = int(aug[col, j]) * inv_pivot % Q

        for row in range(n):
            if row == col:
                continue
            factor = int(aug[row, col]) % Q
            if factor == 0:
                continue
            for j in range(2 * n):
                aug[row, j] = (int(aug[row, j]) - factor * int(aug[col, j])) % Q

    return aug[:, n:]


def poly_inv_matrix(f: list[int], Q: int, N: int) -> list[int] | None:
    """
    Compute f^(-1) mod (x^N+1, Q) by inverting the negacyclic matrix.
    """
    M   = build_negacyclic_matrix(f, N, Q)
    M_inv = mat_inv_mod_q(M, Q, N)
    if M_inv is None:
        return None
    # f^(-1) = first column of M^(-1)? No — M^(-1) applied to e_0 gives f^(-1)
    # Actually: f^(-1) = row 0 of M^(-1) applied to identity
    # M(f) · t0 = ct0  →  t0 = M(f)^(-1) · ct0
    # The inverse polynomial is M^(-1)[:,0] (convolution of M^(-1) with delta)
    return [int(M_inv[i, 0]) % Q for i in range(N)]


def recover_t0_from_ct0(
    c: list[int], ct0: list[int], Q: int, N: int
) -> list[int] | None:
    """Recover t0 given c and ct0=c*t0 by solving M(c)*t0=ct0 mod Q."""
    M   = build_negacyclic_matrix(c, N, Q)
    M_inv = mat_inv_mod_q(M, Q, N)
    if M_inv is None:
        return None
    ct0_vec = np.array([v % Q for v in ct0], dtype=object)
    t0_vec  = np.array([
        sum(int(M_inv[i, j]) * int(ct0_vec[j]) for j in range(N)) % Q
        for i in range(N)
    ])
    return [int(x) % Q for x in t0_vec]


# ─── Assumed oracle: exact ct0 values computed from sk.t0 ───────────────────

def read_ct0_from_power_trace(
    p: DilithiumParams,
    sk: SecretKey,
    n_obs: int = 1,
) -> list[dict]:
    """
    Assumed oracle: returns the exact ct0 values, computed from sk.t0.
    Not a measurement.

    Returns list of {c, ct0_vals} where ct0_vals[k][i] is the actual value.
    """
    from dilithium_sim import _sample_gamma1, _challenge as _ch, _pack_w1

    A   = expand_A(sk.rho, p.K, p.L, p.N, p.Q)
    obs = []

    for msg_i in range(n_obs):
        msg      = f"power_probe_{msg_i}".encode()
        mu       = hashlib.shake_256(sk.tr + msg).digest(64)
        rnd      = os.urandom(32)
        rhoprime = hashlib.shake_256(sk.key + rnd + mu).digest(64)

        y  = [_sample_gamma1(rhoprime, i, p) for i in range(p.L)]
        w  = matmul_vec(A, y, p.Q)
        w1 = [decompose(w[k], p.GAMMA2, p.Q)[0] for k in range(p.K)]
        c  = _ch(hashlib.shake_256(mu + _pack_w1(w1, p)).digest(p.N // 4), p)

        # ── POWER TRACE READ ─────────────────────────────────────────
        # Attacker reads actual ct0[k][i] values from the trace
        ct0_vals = [
            [center(v, p.Q) for v in poly_mul(c, sk.t0[k], p.Q)]
            for k in range(p.K)
        ]
        obs.append({"c": c, "ct0_vals": ct0_vals})

    return obs


# ─── t0 recovery via polynomial inversion ────────────────────────────────────

def recover_t0_via_inversion(
    p: DilithiumParams,
    obs: list[dict],
    sk: SecretKey,
) -> list[list[int]] | None:
    """
    Recover t0[k] by solving the linear system M(c) · t0[k] = ct0[k] mod Q.
    The negacyclic convolution matrix M(c) is N×N over Z_Q.
    Gaussian elimination: O(N^3) = O(256^3) ≈ 16M ops — fast with numpy object dtype.
    """
    print(f"\n{SEP}")
    print("PHASE 1 — t0 recovery via linear system M(c)·t0 = ct0 mod Q")
    print(SEP)
    print(f"  System   : {p.N}×{p.N} negacyclic matrix over Z_{p.Q}")
    print(f"  Method   : Gaussian elimination mod Q  (O(N^3))")

    t0_true = [[center(v, p.Q) for v in sk.t0[k]] for k in range(p.K)]
    t0_rec  = [None] * p.K

    for ob_idx, ob in enumerate(obs):
        c        = ob["c"]
        ct0_vals = ob["ct0_vals"]

        t_start = time.time()
        # Build M(c) once — reuse for all k
        M   = build_negacyclic_matrix(c, p.N, p.Q)
        M_inv = mat_inv_mod_q(M, p.Q, p.N)
        ms  = (time.time() - t_start) * 1000

        if M_inv is None:
            print(f"  Obs {ob_idx}: M(c) singular mod Q — skip  ({ms:.0f}ms)")
            continue

        print(f"  Obs {ob_idx}: M(c) inverted  ({ms:.0f}ms)")

        for k in range(p.K):
            if t0_rec[k] is not None:
                continue
            ct0_mod = [v % p.Q for v in ct0_vals[k]]
            ct0_vec = np.array(ct0_mod, dtype=object)
            t0_vec  = np.array([
                sum(int(M_inv[i, j]) * int(ct0_vec[j]) for j in range(p.N)) % p.Q
                for i in range(p.N)
            ])
            t0_k = [center(int(x), p.Q) for x in t0_vec]

            # Verify
            check = [center(v, p.Q) for v in poly_mul(c, [int(x) % p.Q for x in t0_vec], p.Q)]
            match = (check == ct0_vals[k])
            if match:
                exact = (t0_k == t0_true[k])
                print(f"  k={k}: recovered {'EXACT ✓' if exact else '✗ mismatch'}")
                t0_rec[k] = t0_k

        if all(r is not None for r in t0_rec):
            break

    if any(r is None for r in t0_rec):
        print("  Some polynomials not recovered")
        return None

    return t0_rec


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = DILITHIUM2  # REAL ML-DSA-44 parameters

    print()
    print(BANNER)
    print("Idealized oracle, ML-DSA-44 (N=256): t0 from one exact c*t0 answer")
    print(BANNER)
    print(f"  Params : {p.name}  N={p.N}  K={p.K}  L={p.L}  Q={p.Q}")
    print(f"  D={p.D}  GAMMA2={p.GAMMA2}  TAU={p.TAU}  ETA={p.ETA}")
    print(f"  Oracle : assumed, exact c*t0 computed from sk.t0 (not a measurement)")
    print()

    seed = hashlib.sha256(b"ironproof-real-demo").digest()
    pk, sk = keygen(p, seed=seed)

    t0_true = [[center(v, p.Q) for v in sk.t0[k]] for k in range(p.K)]
    print("TARGET t0[0][:8] =", t0_true[0][:8], "...")
    print("(t0 has", p.K * p.N, "coefficients total — showing first 8)")
    print()
    print("INPUTS: public key (t1, rho) and the oracle answer below")

    # Collect oracle observations
    print(f"\n{SEP}")
    print("ORACLE — 1 exact answer, computed from sk.t0")
    print(SEP)
    t0_w = time.time()
    obs = read_ct0_from_power_trace(p, sk, n_obs=5)
    print(f"  Collected : {len(obs)} traces  ({(time.time()-t0_w)*1000:.0f}ms)")

    # Recover t0
    t0_w = time.time()
    t0_rec = recover_t0_via_inversion(p, obs, sk)
    phase1_ms = (time.time() - t0_w) * 1000

    print(f"\n{BANNER}")
    if t0_rec is not None:
        print("  t0 determined: all K*N coefficients equal the generated key")
        print(f"  {p.K} polynomials × {p.N} coefficients = {p.K*p.N} total coefficients")
        print(f"  Runtime: {phase1_ms:.0f}ms")
        print()
        print("  Recovered t0[0][:8] =", t0_rec[0][:8], "...")
        print("  True      t0[0][:8] =", t0_true[0][:8], "...")
        print()
        print("  WHAT THIS GIVES:")
        bits_t0 = p.D * p.K * p.N
        bits_s1s2 = (p.ETA.bit_length() + 1) * (p.L + p.K) * p.N
        print(f"  t0  : {p.K*p.N} coefficients (not treated as secret by the designers)")
        print(f"  s1+s2: {(p.L+p.K)*p.N} coefficients — MODULE-LWE hardness")
        print()
        print("  NOT GIVEN: s1, s2")
        print("  t = 2^D * t1 + t0 is known once t0 is")
        print("  t = A*s1 + s2 with ||s1||,||s2|| <= ETA=2 is MODULE-LWE")
        print("  recovering s1, s2 from it is the Module-LWE problem")
        print("  → This is the open research problem")
    else:
        print("  t0 recovery failed — check oracle implementation")
    print(BANNER)


if __name__ == "__main__":
    main()
