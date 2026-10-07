"""ML-DSA (Dilithium) parameter sets — FIPS 204 / NIST PQC Standard."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DilithiumParams:
    name: str
    N: int    # ring dimension
    Q: int    # modulus
    D: int    # dropped bits
    K: int    # rows in A
    L: int    # cols in A
    ETA: int  # secret bound
    TAU: int  # challenge weight
    BETA: int # = TAU * ETA
    GAMMA1: int
    GAMMA2: int
    OMEGA: int


DILITHIUM2 = DilithiumParams(
    name="Dilithium2",
    N=256, Q=8380417, D=13,
    K=4, L=4, ETA=2, TAU=39, BETA=78,
    GAMMA1=(1 << 17), GAMMA2=(8380417 - 1) // 88, OMEGA=80,
)

DILITHIUM3 = DilithiumParams(
    name="Dilithium3",
    N=256, Q=8380417, D=13,
    K=6, L=5, ETA=4, TAU=49, BETA=196,
    GAMMA1=(1 << 19), GAMMA2=(8380417 - 1) // 32, OMEGA=55,
)

DILITHIUM5 = DilithiumParams(
    name="Dilithium5",
    N=256, Q=8380417, D=13,
    K=8, L=7, ETA=2, TAU=60, BETA=120,
    GAMMA1=(1 << 19), GAMMA2=(8380417 - 1) // 32, OMEGA=75,
)

# Reduced toy parameters for Z3 formal verification (tractable)
# Calibrated to mirror Dilithium2 geometry with small N for Z3:
#   D=6 → t0 in [-32,31]; TAU=3 → max||ct0||_inf=96 > GAMMA2=88 → check-3 fires
#   ETA=1, BETA=3 → GAMMA2-BETA=85 ≫ BETA → check-2 rarely fires
#   OMEGA=8=N*K → hint check never fires
#   Expected iterations per sig ≈ 2 (comparable to real Dilithium2)
TINY = DilithiumParams(
    name="Tiny-N4",
    N=4, Q=8191, D=6,
    K=2, L=2, ETA=1, TAU=3, BETA=3,
    GAMMA1=512, GAMMA2=88, OMEGA=8,
)

# Full-recovery demo parameters — calibrated for Z3 uniqueness in all 3 phases:
#   D=2  → t0 in [-2, 1]^(K*N),  4^8 = 65536 total → Z3-tractable for uniqueness
#   GAMMA2=3 → TAU*max_t0=2*2=4 > 3 → check-3 fires ~30-70% of challenges
#   ETA=1 → s1,s2 in [-1,1]^(L*N or K*N), 3^8 = 6561 → tractable
#   GAMMA1=32, BETA=2 → |cs1_i| <= TAU*ETA=2, check-1 rarely fires
#   UseHint constraint: width 2*GAMMA2=6 per coeff → very tight for s1 recovery
TINY_DEMO = DilithiumParams(
    name="Tiny-Demo",
    N=4, Q=65537, D=2,
    K=2, L=2, ETA=1, TAU=2, BETA=2,
    GAMMA1=32, GAMMA2=3, OMEGA=8,
)
