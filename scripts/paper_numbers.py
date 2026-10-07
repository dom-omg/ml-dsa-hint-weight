"""
Re-derive every measured number quoted in paper/ironproof_pqc.tex from the raw data.

The paper must never quote a figure this script does not print.

Data: results/c_ref/mldsa{44,65,87}.npz, produced by harness/hint_collect.c on
the pq-crystals reference implementation (see scripts/csv_to_npz.py):
    norms   (keys, K)        Euclidean norm of each polynomial of t0
    std     (keys, n, K)     per-polynomial hint weights, ordinary signing
    bw      (keys, n, K)     same, bounded-weight signing
    runs    (keys,)          signing runs used to obtain the bw signatures
"""

import json
import math
import os

import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "results", "c_ref")

# (name, file stem, tau, gamma2)
VARIANTS = [
    ("ML-DSA-44", "mldsa44", 39, 95232),
    ("ML-DSA-65", "mldsa65", 49, 261888),
    ("ML-DSA-87", "mldsa87", 60, 261888),
]
N = 256
Z_001 = float(stats.norm.ppf(0.999))  # one-sided alpha = 0.001
N_PERM = 1000
N_PAIRS = 100


def anova(groups: np.ndarray):
    """One-way ANOVA on a (keys, n) array. Returns F, p, eta squared."""
    keys, n = groups.shape
    grand = groups.mean()
    ss_b = n * ((groups.mean(axis=1) - grand) ** 2).sum()
    ss_t = ((groups - grand) ** 2).sum()
    ss_w = ss_t - ss_b
    f_stat = (ss_b / (keys - 1)) / (ss_w / (keys * (n - 1)))
    p = float(stats.f.sf(f_stat, keys - 1, keys * (n - 1)))
    return float(f_stat), p, float(ss_b / ss_t)


def permutation_p(groups: np.ndarray, rng: np.random.Generator) -> str:
    """Permutation test on the key labels; returns the p-value bound as text."""
    observed = anova(groups)[0]
    flat = groups.ravel().copy()
    hits = 0
    for _ in range(N_PERM):
        rng.shuffle(flat)
        if anova(flat.reshape(groups.shape))[0] >= observed:
            hits += 1
    return f"{hits}/{N_PERM} permuted F >= observed (p <= {(hits + 1) / (N_PERM + 1):.4f})"


def omnibus_n(groups: np.ndarray) -> int:
    """Smallest n per key such that the ANOVA on the first n signatures has p < 0.001."""
    lo, hi, best = 2, groups.shape[1], groups.shape[1]
    while lo <= hi:
        mid = (lo + hi) // 2
        if anova(groups[:, :mid])[1] < 0.001:
            best, hi = mid, mid - 1
        else:
            lo = mid + 1
    return best


def pair_n(delta: float, sigma: float) -> float:
    """Signatures per key for a one-sided two-sample z-test at alpha = 0.001."""
    return 2.0 * (Z_001 * sigma / delta) ** 2


def scalar_lift(total: np.ndarray) -> float:
    """Single-signature nearest-mean classification, first half train, second half test."""
    keys, n = total.shape
    half = n // 2
    cent = total[:, :half].mean(axis=1)
    hit = 0
    for k in range(keys):
        hit += int((np.abs(total[k, half:, None] - cent[None, :]).argmin(axis=1) == k).sum())
    return hit / (keys * (n - half)) * keys


def pooled_cov(train: np.ndarray) -> np.ndarray:
    keys, n, dim = train.shape
    d = (train - train.mean(axis=1, keepdims=True)).reshape(-1, dim)
    return d.T @ d / (keys * (n - 1))


def vector_lift(vec: np.ndarray) -> float:
    """Single-signature nearest-centroid classification in Mahalanobis distance."""
    keys, n, _ = vec.shape
    half = n // 2
    train, test = vec[:, :half], vec[:, half:]
    chol = np.linalg.cholesky(np.linalg.inv(pooled_cov(train)))
    cw = train.mean(axis=1) @ chol
    c2 = (cw ** 2).sum(axis=1)
    hit = 0
    for k in range(keys):
        xw = test[k] @ chol
        dist = (xw ** 2).sum(axis=1)[:, None] - 2.0 * xw @ cw.T + c2[None, :]
        hit += int((dist.argmin(axis=1) == k).sum())
    return hit / (keys * (n - half)) * keys


def t2_fraction(vec: np.ndarray, rng: np.random.Generator) -> float:
    """Share of random key pairs separated by a two-sample Hotelling T2 test at alpha = 0.05."""
    keys, n, dim = vec.shape
    sig = 0
    for _ in range(N_PAIRS):
        i, j = rng.choice(keys, 2, replace=False)
        diff = vec[i].mean(axis=0) - vec[j].mean(axis=0)
        s_pool = (np.cov(vec[i].T) + np.cov(vec[j].T)) / 2.0
        t2 = (n / 2.0) * float(diff @ np.linalg.solve(s_pool, diff))
        f_stat = t2 * (2 * n - dim - 1) / (dim * (2 * n - 2))
        if stats.f.sf(f_stat, dim, 2 * n - dim - 1) < 0.05:
            sig += 1
    return sig / N_PAIRS


def report_channel(label: str, vec: np.ndarray, rng: np.random.Generator) -> None:
    keys, n, dim = vec.shape
    total = vec.sum(axis=2)
    means = total.mean(axis=1)
    f_stat, p, eta2 = anova(total)
    sigma_w = float(np.sqrt(np.mean(total.var(axis=1, ddof=1))))
    sd_means = float(means.std(ddof=1))
    spread = float(means.max() - means.min())
    print(f"-- {label}: total weight")
    print(
        f"   mean={total.mean():.3f} sigma_within={sigma_w:.3f} sd_of_key_means={sd_means:.3f} "
        f"spread={spread:.3f}"
    )
    print(f"   ANOVA F={f_stat:.2f} df=({keys - 1},{keys * (n - 1)}) p={p:.1e} eta^2={eta2:.4f}")
    print(f"   lift over 1/{keys}: total weight={scalar_lift(total):.3f} weight vector={vector_lift(vec):.3f}")
    coord = [anova(vec[:, :, k]) for k in range(dim)]
    cohen = []
    for k in range(dim):
        m = vec[:, :, k].mean(axis=1)
        sd = math.sqrt(float(np.mean(vec[:, :, k].var(axis=1, ddof=1))))
        cohen.append((m.max() - m.min()) / sd)
    print("   per-coordinate F: " + " ".join(f"{c[0]:.2f}" for c in coord))
    print("   per-coordinate p: " + " ".join(f"{c[1]:.1e}" for c in coord))
    print("   per-coordinate Cohen d (extreme pair): " + " ".join(f"{c:.3f}" for c in cohen))
    if label != "standard":
        return
    hi, lo = int(means.argmax()), int(means.argmin())
    t, pt = stats.ttest_ind(total[hi], total[lo], equal_var=False)
    print(f"   Welch extreme pair (keys {hi},{lo}): t={t:.2f} p={pt:.1e}")
    print(f"   permutation test: {permutation_p(total, rng)}")
    print(
        f"   n/key at 0.001: omnibus ANOVA={omnibus_n(total)} "
        f"extreme pair={pair_n(spread, sigma_w):.0f} pair 1 SD apart={pair_n(sd_means, sigma_w):.0f}"
    )
    print(f"   Hotelling T2, {N_PAIRS} random pairs, alpha=0.05: {t2_fraction(vec, rng) * 100:.0f}% separated")
    print("   pooled within-key covariance:\n" + np.array2string(pooled_cov(vec), precision=2))


def report_model(norms: np.ndarray, std: np.ndarray, tau: int, gamma2: int) -> None:
    """First-order model: E[wt(h_k)] = sqrt(2 tau N / pi) * ||t0_k||_2 / (2 gamma2)."""
    keys, n, dim = std.shape
    scale = math.sqrt(2 * tau * N / math.pi) / (2 * gamma2)
    pred = scale * norms
    obs = std.mean(axis=1)
    r_poly = float(np.corrcoef(pred.ravel(), obs.ravel())[0, 1])
    r_tot = float(np.corrcoef(pred.sum(axis=1), obs.sum(axis=1))[0, 1])
    print("-- first-order model")
    print(
        f"   Pearson r: per polynomial={r_poly:.3f} (n={obs.size}), per key total={r_tot:.3f} (n={keys}); "
        f"predicted mean total={pred.sum(axis=1).mean():.2f} observed={obs.sum(axis=1).mean():.2f}"
    )
    # Residual analysis: what is left once the norm is regressed out, against sampling noise.
    slope, intercept = np.polyfit(pred.ravel(), obs.ravel(), 1)
    resid = obs - (slope * pred + intercept)
    noise_var = std.var(axis=1, ddof=1) / n
    signal = obs.var(ddof=1) - noise_var.mean()
    left = resid.var(ddof=2) - noise_var.mean()
    chi2 = float((resid ** 2 / noise_var).sum())
    dof = obs.size - 2
    print(
        f"   fit obs = {slope:.3f} * pred + {intercept:.3f}; variance of key-polynomial means: "
        f"total={obs.var(ddof=1):.5f} residual={resid.var(ddof=2):.5f} sampling noise={noise_var.mean():.5f}"
    )
    print(f"   share of the key-dependent variance (sampling noise removed) explained by the norm: {1 - left / signal:.3f}")
    print(f"   residual against sampling noise: chi2={chi2:.1f} dof={dof} p={stats.chi2.sf(chi2, dof):.2e}")
    uniform_norm = math.sqrt(N) * 8192 / math.sqrt(12)
    print(f"   uniform-t0 prediction of the total: {dim * scale * uniform_norm:.2f}")


def report_variant(name: str, stem: str, tau: int, gamma2: int) -> None:
    path = os.path.join(DATA, stem + ".npz")
    if not os.path.exists(path):
        print(f"{name}: MISSING {path}")
        return
    z = np.load(path)
    norms = z["norms"].astype(np.float64)
    std = z["std"].astype(np.float64)
    bw = z["bw"].astype(np.float64)
    runs = z["runs"].astype(np.float64)
    keys, n, dim = std.shape
    rng = np.random.default_rng(2026)
    print(f"\n==================== {name}: {keys} keys x {n} signatures, K={dim} ====================")
    report_channel("standard", std, rng)
    report_channel("BW-Sign", bw, rng)
    print(f"   cost={runs.sum() / (keys * n):.2f} signing runs per output signature")
    report_model(norms, std, tau, gamma2)


def legacy_check() -> None:
    """Earlier independent ML-DSA-44 run (same C code, GCC 13.3.0, Linux 6.8.0 aarch64, 50 keys)."""
    path = os.path.join(BASE, "legacy_mldsa44_linux_50x2000.json")
    if not os.path.exists(path):
        print("legacy run: MISSING")
        return
    with open(path) as fh:
        raw = json.load(fh)
    g = np.array([r["hint_counts"] for r in raw.get("results", [])], dtype=np.float64)
    if g.size == 0:
        print("legacy run: EMPTY")
        return
    f_stat, p, eta2 = anova(g)
    print(
        f"\n== legacy ML-DSA-44 run (Linux, 50 keys x 2000): mean={g.mean():.3f} "
        f"sigma_within={math.sqrt(float(np.mean(g.var(axis=1, ddof=1)))):.3f} "
        f"sd_of_key_means={g.mean(axis=1).std(ddof=1):.3f} F={f_stat:.2f} p={p:.1e} eta^2={eta2:.4f}"
    )


if __name__ == "__main__":
    for variant in VARIANTS:
        report_variant(*variant)
    legacy_check()
