"""Cross-fitted Gaussian-model estimator of R_pi(eps) and of the single-teacher
FDL, and a direct test of its predictions against an actual prior-matched
random codebook (nearest retrained teacher among M others)."""
import numpy as np
import json
from fdl import (rate_diag_gaussian, tilted_info, ball_logmass_gauss,
                 n_above_water, water_level)

rng = np.random.default_rng(11)
out = {}
for name in ["mlp_seed", "mlp_seed_data"]:
    F = np.load(f"../results/{name}.npz")["F"]
    N, n = F.shape
    perm = rng.permutation(N)
    A, B = F[perm[: N // 2]], F[perm[N // 2:]]
    mA = A.mean(0)
    U, s, Vt = np.linalg.svd((A - mA) / np.sqrt(n), full_matrices=False)
    r = int(np.sum(s > s[0] * 1e-8))
    Phi = Vt[:r]                                          # basis from half A
    ZB = (B - mA) / np.sqrt(n) @ Phi.T                    # coords of half B
    resB = np.sum(((B - mA) / np.sqrt(n)) ** 2, 1) - np.sum(ZB ** 2, 1)
    sig2 = np.mean(ZB ** 2, 0)                            # 2nd moments (held out)
    e_res = resB.mean()
    # bootstrap band for the cross-fitted upper bound
    eps_grid = np.logspace(-7, -2.5, 19)
    Rcf = np.array([rate_diag_gaussian(sig2, e - e_res) for e in eps_grid])
    boots = []
    for _ in range(200):
        bi = rng.integers(0, len(B), len(B))
        s2b = np.mean(ZB[bi] ** 2, 0)
        boots.append([rate_diag_gaussian(s2b, e - np.mean(resB[bi])) for e in eps_grid])
    boots = np.array(boots)
    # single-teacher FDL for held-out teachers (leave-one-out inside B)
    Jt = []
    for j in range(len(B)):
        s2_j = (len(B) * sig2 - ZB[j] ** 2) / (len(B) - 1)
        order = np.argsort(-s2_j)
        Jt.append([tilted_info(ZB[j][order], s2_j[order], e) for e in eps_grid])
    Jt = np.array(Jt)
    print(f"\n=== {name}: basis rank {r}, held-out residual energy {e_res:.2e}")
    print("   eps     k_a   R_crossfit [boot 5%,95%]    mean j (sd)")
    for i, e in enumerate(eps_grid):
        lo, hi = np.percentile(boots[:, i], [5, 95])
        print(f"{e:9.2e} {n_above_water(np.sort(sig2)[::-1], e - e_res):4d} "
              f"{Rcf[i]:8.1f} [{lo:7.1f},{hi:7.1f}]   {Jt[:, i].mean():8.1f} ({Jt[:, i].std():5.1f})")

    # --- prior-matched random codebook test -------------------------------
    # codebook = M teachers from half A; distortion of nearest codeword for
    # each teacher in B, versus the Gaussian-model prediction
    #   log2 M  ~  -log2 pi_G(B_eps(f)),   pi_G = N(mA, diag sig2) in basis Phi
    DAB = (np.sum(B ** 2, 1)[:, None] + np.sum(A ** 2, 1)[None] - 2 * B @ A.T) / n
    rows = []
    for M in [2, 4, 8, 16, 32, 64, 128, 200]:
        dmin = np.array([DAB[j, rng.choice(len(A), M, replace=False)].min() for j in range(len(B))])
        eps_med = float(np.median(dmin))
        pred = []
        for j in rng.choice(len(B), 25, replace=False):
            e = eps_med - resB[j]
            pred.append(ball_logmass_gauss(ZB[j], sig2, e, n_mc=40_000, rng=rng))
        pred = np.array(pred)
        rows.append((M, eps_med, float(np.median(pred))))
        print(f"  codebook M={M:4d} (log2 M={np.log2(M):4.1f}): median NN sq.dist {eps_med:.3e};"
              f"  Gaussian-model median -log2 pi(B_eps(f)) = {np.median(pred):5.2f} bits")
    # per-teacher: does j(f, eps) predict who is hard to code?
    M = 64
    dmin = np.array([DAB[j, rng.choice(len(A), M, replace=False)].min() for j in range(len(B))])
    e_ref = float(np.median(dmin))
    jref = []
    for j in range(len(B)):
        s2_j = (len(B) * sig2 - ZB[j] ** 2) / (len(B) - 1)
        order = np.argsort(-s2_j)
        jref.append(tilted_info(ZB[j][order], s2_j[order], e_ref))
    rho = np.corrcoef(np.log(dmin), np.array(jref))[0, 1]
    print(f"  per-teacher: corr(log NN-distortion at M={M}, j(f, eps_ref)) = {rho:.2f}")
    out[name] = {"eps_grid": eps_grid.tolist(), "R_crossfit": Rcf.tolist(),
                 "R_boot5": np.percentile(boots, 5, 0).tolist(),
                 "R_boot95": np.percentile(boots, 95, 0).tolist(),
                 "j_mean": Jt.mean(0).tolist(), "j_sd": Jt.std(0).tolist(),
                 "codebook_rows": rows, "corr_j_vs_nn": rho, "sig2": np.sort(sig2)[::-1].tolist(),
                 "e_res": e_res}
json.dump(out, open("../results/mlp_crossfit.json", "w"))
