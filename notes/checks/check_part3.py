"""Part 3: finite-sample ERM from teacher samples (Theorem 2 of the note).

Same exchangeable model as part 2. For each T, draw n teacher samples, fit
  - clipped-ratio ERM:  argmin_mu (1/n) sum_i (1 - min(1, q_mu(x_i)/p(x_i))^beta)
  - log-loss ERM:       argmax_mu (1/n) sum_i log q_mu(x_i)   (= sample mean)
over a grid of mu, and report TV(p, q_hat)/OPT (median over repetitions).
The sample size n is held fixed while T grows.
"""
import numpy as np
from scipy.special import gammaln

MU0, MU1, EPS = 0.5, 0.7, 0.01
rng = np.random.default_rng(1)


def log_binom(T):
    k = np.arange(T + 1)
    return gammaln(T + 1) - gammaln(k + 1) - gammaln(T - k + 1)


def lseq(T, mu):
    k = np.arange(T + 1)
    return k * np.log(mu) + (T - k) * np.log1p(-mu)


def lteach(T):
    return np.logaddexp(np.log1p(-EPS) + lseq(T, MU0), np.log(EPS) + lseq(T, MU1))


def tv(T, mu, lb, lp):
    return 0.5 * np.abs(np.exp(lb + lp) - np.exp(lb + lseq(T, mu))).sum()


n, reps = 50_000, 15
print(f"n = {n} teacher samples (fixed), median over {reps} repetitions")
print(f"{'T':>7} {'OPT':>8} | {'D_1-ERM':>8} {'D_1/2-ERM':>9} {'logloss-ERM':>11}")
for T in (100, 1000, 10000):
    lb, lp = log_binom(T), lteach(T)
    # grid concentrated where it matters, plus a coarse global part
    grid = np.unique(np.concatenate([np.linspace(0.01, 0.99, 400),
                                     MU0 + np.linspace(-0.1, 0.1, 1601)]))
    LQ = np.stack([lseq(T, m) for m in grid])          # (grid, T+1)
    tvs = np.array([tv(T, m, lb, lp) for m in grid])
    opt = tvs.min()
    ratios = {"D1": [], "Dh": [], "LL": []}
    for _ in range(reps):
        comp = rng.random(n) < EPS
        k = np.where(comp, rng.binomial(T, MU1, n), rng.binomial(T, MU0, n))
        counts = np.bincount(k, minlength=T + 1) / n
        u = np.exp(np.minimum(LQ - lp[None, :], 0.0))  # min(q/p, 1) per count
        for key, beta in (("D1", 1.0), ("Dh", 0.5)):
            Lhat = (counts[None, :] * (1 - u ** beta)).sum(axis=1)
            ratios[key].append(tvs[np.argmin(Lhat)] / opt)
        mu_ll = k.mean() / T
        ratios["LL"].append(tv(T, mu_ll, lb, lp) / opt)
    print(f"{T:>7} {opt:8.5f} | {np.median(ratios['D1']):8.2f} {np.median(ratios['Dh']):9.2f} "
          f"{np.median(ratios['LL']):11.2f}")
