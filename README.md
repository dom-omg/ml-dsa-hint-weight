# The Hint Weight of ML-DSA Signatures Is Key-Dependent — data and code

Data, measurement harness and analysis scripts for the paper
*The Hint Weight of ML-DSA Signatures Is Key-Dependent: An Empirical Study
across the Three FIPS 204 Parameter Sets* (Dominik Blain, Ironproof, 2026).
The LaTeX source is in `paper/`.

## What is here

| Path | Content |
|---|---|
| `harness/` | C harness that signs with the unmodified pq-crystals reference implementation and records the per-polynomial hint weights (`hint_collect.c`), its deterministic SHAKE256 random source (`det_randombytes.c`), `Makefile`, and `run_all.sh` (12 parallel shards). |
| `results/c_ref/mldsa{44,65,87}.npz` | Raw data of the paper: 200 keys × 2,000 signatures per parameter set, ordinary and bounded-weight signing, 2,400,000 signatures in all, with the norm of each polynomial of t0. |
| `scripts/paper_numbers.py` | Recomputes every figure of Sections 3, 4 and 7.1 of the paper from the raw files. |
| `scripts/csv_to_npz.py` | Converts the harness output to the `.npz` files above. |
| `IRONPROOF_50KEY_2000SIG_ANOVA.json` | Earlier independent ML-DSA-44 run (50 keys × 2,000 signatures, GCC 13.3.0, Linux aarch64), cited in Section 4.1. |
| `full_statistical_analysis.py`, `fast_sign.py`, `results/statistical_report.json` | Fixed-iteration signing figures of Section 5. They use a Python model of the signing algorithm that is **not** byte-compatible with FIPS 204. The script calls the variant PRISM-DSA. |
| `dilithium_sim.py`, `params.py`, `ring.py`, `recover_t0_real.py`, `full_key_recovery_real.py`, `recover_key.py`, `l2d_differential_test.py` | Section 6. Same Python model. The oracle is simulated, exact and noise-free; no device was measured. |

## Reproduce the figures from the raw data

Python 3 with NumPy and SciPy.

```bash
python3 scripts/paper_numbers.py
```

About 90 s on a laptop. The output is deterministic and is the source of every
number in Sections 3, 4 and 7.1.

## Re-collect the raw data

```bash
cd harness
git clone https://github.com/pq-crystals/dilithium dilithium-ref
git -C dilithium-ref checkout 6e00625
make DILITHIUM_REF=dilithium-ref/ref
bash run_all.sh ../out
python3 ../scripts/csv_to_npz.py ...   # see the header of csv_to_npz.py
```

The random source is a SHAKE256 stream keyed by a fixed seed and the run
arguments, so a run is repeatable bit for bit. The paper's run took about
10 minutes on an Apple M4 (Apple clang 17.0.0, `-O2`).

## Scope

The hint weight is a weak statistical fingerprint of the signing key. This
repository does not contain, and the paper does not claim, any key recovery from
public signatures. See the Limitations section of the paper.

## License

Code: MIT (see `LICENSE`). Data and paper: CC BY 4.0.
