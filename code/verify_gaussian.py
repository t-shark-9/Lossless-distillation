"""Numerical checks of the Gaussian-prior FDL theory (Part III of the notes)."""
import numpy as np
from fdl import (water_level, rate_gaussian, dispersion_gaussian_nats2,
                 tilted_info, achievability_gap, ball_logmass_direct,
                 n_above_water, LN2)

rng = np.random.default_rng(0)

print("== 1. E[j] = R(eps), Var[j] = V(eps) ==")
lam = 1.0 / np.arange(1, 201) ** 2.0          # power-law spectrum, alpha = 2
for eps in [0.3, 0.05, 0.005]:
    Z = rng.standard_normal((200_000, lam.size)) * np.sqrt(lam)
    J = tilted_info(Z, lam, eps)
    R = rate_gaussian(lam, eps)
    V = dispersion_gaussian_nats2(lam, eps) / LN2 ** 2
    print(f"eps={eps:6.3f} k_a={n_above_water(lam, eps):3d}  R={R:8.3f}  "
          f"mean j={J.mean():8.3f}   V={V:8.3f}  var j={J.var():8.3f}")

print("\n== 2. Csiszar condition  sup_g E_F exp(j + lam* eps - lam* d(F,g)) <= 1 ==")
lam_s = np.array([2.0, 1.0, 0.5, 0.2, 0.05])
eps = 0.6
th = water_level(lam_s, eps)
Z = rng.standard_normal((400_000, lam_s.size)) * np.sqrt(lam_s)
jn = tilted_info(Z, lam_s, eps) * LN2
for trial in range(6):
    g = rng.standard_normal(lam_s.size) * (0 if trial == 0 else 0.7)
    val = np.mean(np.exp(jn + eps / (2 * th) - np.sum((Z - g) ** 2, 1) / (2 * th)))
    print(f"  g={np.round(g, 2)}  E[...] = {val:.4f}")

print("\n== 3. -log2 P_{G*}(B_eps(f)) : direct MC vs j + Delta (tilted IS) ==")
for k, eps in [(3, 0.4), (5, 0.8), (8, 1.5)]:
    lam_k = 1.0 / np.arange(1, k + 1) ** 1.0
    for _ in range(2):
        z = rng.standard_normal(k) * np.sqrt(lam_k)
        direct, hits = ball_logmass_direct(z, lam_k, eps, n_mc=4_000_000, rng=rng)
        j = tilted_info(z, lam_k, eps)
        D = achievability_gap(z, lam_k, eps, rng=rng)
        print(f"k={k} eps={eps}: direct={direct:7.3f} (hits={hits:6d})  "
              f"j={j:7.3f}  j+Delta={j + D:7.3f}  Delta={D:5.3f}")

print("\n== 4. Gap Delta_f for typical teachers vs 1/2 log2(pi k_a) ==")
for K in [10, 30, 100, 300, 1000]:
    lam_K = np.ones(K)
    eps = 0.01 * K                               # all components above water
    gaps = [achievability_gap(rng.standard_normal(K), lam_K, eps,
                              n_mc=100_000, rng=rng) for _ in range(20)]
    print(f"k_a={K:5d}: mean Delta={np.mean(gaps):6.3f} bits  "
          f"(sd {np.std(gaps):5.3f});  1/2 log2(pi k_a)={0.5*np.log2(np.pi*K):6.3f}")
