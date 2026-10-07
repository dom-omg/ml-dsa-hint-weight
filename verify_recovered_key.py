"""
Functional check for Section 6 of the paper: does the key determined from the
idealized oracle sign?

The key is determined by full_key_recovery_real.py (oracle answers computed from
the secret key; see that file). A verifier that sees only the public key then
checks signatures made with it. The verifier is first tested both ways: it must
accept the honest signer and reject a wrong message and a key changed in one
coefficient.

The model in dilithium_sim.py does not wrap the top high-bits value of Decompose
to 0, as FIPS 204 does, so most signatures it produces fail a conformant
verifier. This script substitutes the FIPS 204 Decompose for the check.
"""
import contextlib
import hashlib
import io
import os

import dilithium_sim
import full_key_recovery_real as fk
import ring
from dilithium_sim import SecretKey, _challenge, _pack_t1, _pack_w1, keygen, sign
from params import DILITHIUM2 as p
from ring import chknorm, expand_A, matmul_vec, poly_mul, poly_sub

N_MSG = 20


def decompose_fips(a, GAMMA2, Q):
    """FIPS 204 Decompose: high bits in [0, (Q-1)/(2*GAMMA2) - 1]."""
    ALPHA = 2 * GAMMA2
    hi, lo = [], []
    for v in a:
        v %= Q
        r0 = v % ALPHA
        if r0 > ALPHA // 2:
            r0 -= ALPHA
        if v - r0 == Q - 1:
            hi.append(0)
            lo.append(r0 - 1)
        else:
            hi.append((v - r0) // ALPHA)
            lo.append(r0)
    return hi, lo


ring.decompose = decompose_fips
dilithium_sim.decompose = decompose_fips


def use_hint(h, r):
    r1, r0 = decompose_fips(r, p.GAMMA2, p.Q)
    m = (p.Q - 1) // (2 * p.GAMMA2)
    return [((a1 + 1) % m if a0 > 0 else (a1 - 1) % m) if hb else a1
            for hb, a1, a0 in zip(h, r1, r0)]


def verify(pk, msg, sig):
    """Uses the public key only."""
    if any(chknorm(sig.z[l], p.GAMMA1 - p.BETA, p.Q) for l in range(p.L)):
        return False
    if sum(sum(hk) for hk in sig.h) > p.OMEGA:
        return False
    A = expand_A(pk.rho, p.K, p.L, p.N, p.Q)
    tr = hashlib.shake_256(pk.rho + _pack_t1(pk.t1, p)).digest(64)
    mu = hashlib.shake_256(tr + msg).digest(64)
    c = _challenge(sig.c_tilde, p)
    Az = matmul_vec(A, sig.z, p.Q)
    w1 = []
    for k in range(p.K):
        ct1 = poly_mul(c, [(v << p.D) % p.Q for v in pk.t1[k]], p.Q)
        w1.append(use_hint(sig.h[k], poly_sub(Az[k], ct1, p.Q)))
    return hashlib.shake_256(mu + _pack_w1(w1, p)).digest(p.N // 4) == sig.c_tilde


def main():
    pk, sk = keygen(p, seed=hashlib.sha256(b"forge-test").digest())
    msgs = [f"message {i}".encode() for i in range(N_MSG)]

    honest = sum(verify(pk, m, sign(p, sk, m)) for m in msgs)
    wrong_msg = sum(verify(pk, m + b"!", sign(p, sk, m)) for m in msgs)
    off = SecretKey(rho=sk.rho, tr=sk.tr, key=sk.key,
                    s1=[list(x) for x in sk.s1], s2=sk.s2, t0=sk.t0)
    off.s1[0][0] += 1
    off_key = sum(verify(pk, m, sign(p, off, m)) for m in msgs)

    with contextlib.redirect_stdout(io.StringIO()):
        t0 = fk.phase1_recover_t0(p, sk)
        s1 = fk.phase3_recover_s1(p, sk)
        s2 = fk.phase2_recover_s2(p, sk, pk, t0, s1)
    tr = hashlib.shake_256(pk.rho + _pack_t1(pk.t1, p)).digest(64)
    determined = SecretKey(rho=pk.rho, tr=tr, key=os.urandom(32), s1=s1, s2=s2, t0=t0)
    det = sum(verify(pk, m, sign(p, determined, m)) for m in msgs)

    print(f"honest signer accepted           : {honest}/{N_MSG}")
    print(f"wrong message accepted           : {wrong_msg}/{N_MSG}")
    print(f"key off by one coefficient       : {off_key}/{N_MSG}")
    print(f"key determined from the oracle   : {det}/{N_MSG}")


if __name__ == "__main__":
    main()
