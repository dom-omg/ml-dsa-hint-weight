"""
Section 6.1 of the paper: passive estimate of t0 from public signature data.

Procedure (Python model of the signing algorithm, dilithium_sim.py):
  1. Generate key A and key B from independent seeds.
  2. Sign M messages with key A.
  3. Correlate the low bits of w' = Az - c*t1*2^d with the challenge,
     using public data only, to estimate t0 up to a scale factor.
  4. Report the Pearson correlation of the estimate with t0 of key A
     (the signing key) and with t0 of key B (an unrelated key).

Signing is randomized, so the third decimal of the correlations varies between runs.
"""

import sys, os, hashlib
import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
os.chdir(BASE)

# Monkey-patch poly_mul with numpy FFT before any imports
import ring as _ring
def _poly_mul_np(a, b, Q):
    N = len(a)
    an, bn = np.array(a, np.float64), np.array(b, np.float64)
    fa, fb = np.fft.rfft(an, 2*N), np.fft.rfft(bn, 2*N)
    c_full = np.fft.irfft(fa * fb, 2*N)
    pos = np.round(c_full[:N]).astype(np.int64)
    neg = np.round(c_full[N:]).astype(np.int64)
    return ((pos - neg) % Q).tolist()
_ring.poly_mul = _poly_mul_np

from dilithium_sim import keygen, sign, _challenge
from params import DILITHIUM2
from ring import expand_A

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--sigs", type=int, default=3000)
args = ap.parse_args()

p = DILITHIUM2
Q, N, K, L, D, GAMMA2, TAU = p.Q, p.N, p.K, p.L, p.D, p.GAMMA2, p.TAU

print("=" * 60)
print("  L2d DIFFERENTIAL KEY TEST")
print("=" * 60)

# Two completely independent keys
pk_A, sk_A = keygen(p, seed=hashlib.sha256(b'differential-key-A').digest())
pk_B, sk_B = keygen(p, seed=hashlib.sha256(b'differential-key-B').digest())
print(f"  Key A seed: 'differential-key-A'  ||t0||_inf={max(abs(c) for poly in sk_A.t0 for c in poly)}")
print(f"  Key B seed: 'differential-key-B'  ||t0||_inf={max(abs(c) for poly in sk_B.t0 for c in poly)}")
print(f"  Signatures from key A: M={args.sigs}")
print()

# Build matrices from key A public key only
A_np  = np.array(expand_A(pk_A.rho, K, L, N, Q), dtype=np.int64)
t1_np = np.array(pk_A.t1, dtype=np.int64)
t1_2d = (t1_np * (1 << D)) % Q

def poly_mul_np(a, b, Q):
    N = len(a)
    fa = np.fft.rfft(a.astype(np.float64), 2*N)
    fb = np.fft.rfft(b.astype(np.float64), 2*N)
    c_full = np.fft.irfft(fa * fb, 2*N)
    return (np.round(c_full[:N]).astype(np.int64) - np.round(c_full[N:]).astype(np.int64)) % Q

def lowbits_np(poly, GAMMA2, Q):
    ALPHA = 2 * GAMMA2
    p = poly % Q
    r0 = p % ALPHA
    r0 = np.where(r0 > ALPHA // 2, r0 - ALPHA, r0)
    return np.where(p - r0 == Q - 1, r0 - 1, r0).astype(np.int64)

# Accumulation
acc = np.zeros((K, N), dtype=np.float64)
hints_total = 0
zeros_N = np.zeros(N, dtype=np.float64)

print("  Accumulating signatures from KEY A only...")
for i in range(args.sigs):
    msg = f"l2d-difftest-{i}".encode()
    sig = sign(p, sk_A, msg)

    c    = _challenge(sig.c_tilde, p)
    c_np = np.array(c, dtype=np.int64)
    z_np = np.array(sig.z, dtype=np.int64)
    h_np = np.array(sig.h, dtype=np.float64)

    # Passive: A·z - c·t1·2^D   (public key + public signature only)
    Az = np.zeros((K, N), dtype=np.int64)
    for k in range(K):
        for l in range(L):
            Az[k] = (Az[k] + poly_mul_np(A_np[k, l], z_np[l], Q)) % Q
    ct1_2d = np.zeros((K, N), dtype=np.int64)
    for k in range(K):
        ct1_2d[k] = poly_mul_np(c_np, t1_2d[k], Q)
    wprime = (Az - ct1_2d) % Q

    r_np = np.zeros((K, N), dtype=np.float64)
    for k in range(K):
        r_np[k] = lowbits_np(wprime[k], GAMMA2, Q)

    hints_total += int(h_np.sum())

    # Negacyclic cross-correlation (2N FFT)
    c_centered = np.where(c_np > Q // 2, c_np - Q, c_np).astype(np.float64)
    c_neg_ext  = np.concatenate([c_centered, -c_centered])
    C_2N_fft   = np.fft.fft(c_neg_ext)
    for k in range(K):
        s_k = (1.0 - h_np[k]) * r_np[k]
        s_2N = np.concatenate([s_k, zeros_N])
        corr_2N = np.real(np.fft.ifft(np.fft.fft(s_2N) * np.conj(C_2N_fft)))
        acc[k] += corr_2N[:N]

    if (i + 1) % 1000 == 0:
        print(f"    [{i+1}/{args.sigs}]")

hint_rate = hints_total / (args.sigs * K * N)
scale     = (1.0 - hint_rate) * TAU
t0_hat    = acc / (args.sigs * scale)

print()
print(f"  hint_rate={hint_rate:.4f}  scale={scale:.2f}")
print()
print("  RESULT — Pearson r of t0_hat vs true t0:")
print(f"  {'k':>3}  {'r vs t0_A (correct)':>22}  {'r vs t0_B (wrong key)':>22}")
print("  " + "─" * 55)

verdict = True
for k in range(K):
    hat_k = t0_hat[k]
    r_A, _ = stats.pearsonr(hat_k, np.array(sk_A.t0[k], dtype=np.float64))
    r_B, _ = stats.pearsonr(hat_k, np.array(sk_B.t0[k], dtype=np.float64))
    key_specific = abs(r_A) > 0.5 and abs(r_B) < 0.15
    if not key_specific:
        verdict = False
    print(f"  {k:>3}  {r_A:>22.4f}  {r_B:>22.4f}  {'OK' if key_specific else 'FAIL'}")

print()
if verdict:
    print("  RESULT: the estimate correlates with the correct key, not with the unrelated key.")
else:
    print("  RESULT: the estimate does not distinguish the two keys.")
print("=" * 60)
