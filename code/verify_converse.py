"""Fixed-length coding of Gaussian teachers: converse vs achievability.

For a rate of R bits (2^R codewords) we compare
  * the one-shot converse, valid for EVERY code:
        P[success] <= P[j(F,eps) < R + gamma] + 2^-gamma      (optimised over gamma)
  * the exact success probability of a random codebook drawn from P_{G*}:
        P[success] = E[1 - (1 - p_F)^(2^R)],  p_F = P_{G*}(B_eps(F)) = 2^-(j+Delta)
  * an entropy-coded uniform scalar quantiser of the KL coordinates
    (the analogue of a 'quantised student'), rate = empirical entropy.
"""
import numpy as np
from fdl import (water_level, rate_gaussian, dispersion_gaussian_nats2,
                 tilted_info, achievability_gap, n_above_water, LN2)

rng = np.random.default_rng(1)
K = 40
lam = 1.0 / np.arange(1, K + 1) ** 1.5
eps = 0.02
R0 = rate_gaussian(lam, eps)
V = dispersion_gaussian_nats2(lam, eps) / LN2 ** 2
ka = n_above_water(lam, eps)
print(f"K={K} eps={eps}: R(eps)={R0:.2f} bits, sqrt(V)={np.sqrt(V):.2f} bits, k_a={ka}")

n_f = 400
Z = rng.standard_normal((n_f, K)) * np.sqrt(lam)
J = tilted_info(Z, lam, eps)
Dl = np.array([achievability_gap(z, lam, eps, n_mc=40_000, rng=rng) for z in Z])
logp = -(J + Dl)                                    # log2 P_{G*}(B_eps(F))
print(f"mean Delta = {Dl.mean():.2f} bits  (1/2 log2(pi k_a) = {0.5*np.log2(np.pi*ka):.2f})")

Jbig = tilted_info(rng.standard_normal((400_000, K)) * np.sqrt(lam), lam, eps)
print("\n  R    converse-UB  random-code  (success probabilities)")
rows = []
for R in np.arange(np.floor(R0 - 3 * np.sqrt(V)), np.ceil(R0 + 3 * np.sqrt(V)) + 8, 2.0):
    conv = min(1.0, min(np.mean(Jbig < R + g) + 2.0 ** -g for g in np.linspace(0, 12, 121)))
    # 1-(1-p)^M computed stably
    M = 2.0 ** R
    succ = np.mean(-np.expm1(M * np.log1p(-np.minimum(2.0 ** logp, 1 - 1e-16))))
    rows.append((R, conv, succ))
    print(f"{R:5.0f}   {conv:8.4f}    {succ:8.4f}")

# entropy-coded scalar quantiser of the above-water coordinates at the same eps
th = water_level(lam, eps)
a = lam > th
eps_a = eps - lam[~a].sum()                        # budget for above-water comps
step = np.sqrt(12 * eps_a / a.sum())               # high-resolution equal steps
q = np.round(Z[:, a] / step)
bits_ecsq = 0.0
for i in range(a.sum()):
    # ideal entropy of the quantised Gaussian N(0, lam_i) with this step
    grid = np.arange(-60, 61) * step
    from scipy.stats import norm
    pm = norm.cdf((grid + step / 2) / np.sqrt(lam[a][i])) - norm.cdf((grid - step / 2) / np.sqrt(lam[a][i]))
    pm = pm[pm > 0]
    bits_ecsq += -np.sum(pm * np.log2(pm))
dist = np.mean(np.sum((Z[:, a] - q * step) ** 2, 1) + np.sum(Z[:, ~a] ** 2, 1))
print(f"\nECSQ student: rate={bits_ecsq:.2f} bits at distortion {dist:.4f} (target {eps});"
      f" R(eps)={R0:.2f}; excess={bits_ecsq - R0:.2f} bits"
      f" = {(bits_ecsq - R0)/a.sum():.3f} bits/dim (theory 0.254)")
np.save("../results/converse_rows.npy", np.array(rows))
