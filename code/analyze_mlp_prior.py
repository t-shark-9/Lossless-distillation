"""Computable bounds on R_pi(eps) and on the single-teacher FDL j_pi(f, eps)
for the MLP training-procedure priors produced by mlp_prior.py.

Distortion: squared L2(D) error of the teacher output, estimated on the
shared query sample Xq ~ D.  All rates in bits.
"""
import numpy as np
from scipy.stats import beta, kurtosis
from fdl import rate_gaussian, tilted_info, n_above_water, water_level
import json

rng = np.random.default_rng(7)
res = {}


def spectrum(F):
    """Eigen-decomposition of the empirical covariance operator on L2(D_n)."""
    N, n = F.shape
    m = F.mean(0)
    Fc = (F - m) / np.sqrt(n)                    # L2(D_n) scaling
    U, s, Vt = np.linalg.svd(Fc, full_matrices=False)
    lam = s ** 2 / (N - 1)
    keep = lam > lam[0] * 1e-12
    return m, lam[keep], Vt[keep]                # eigenfunctions as rows (unit in l2 / sqrt(n))


def coords(f, m, V, n):
    return (f - m) / np.sqrt(n) @ V.T            # <f - m, phi_i>_{L2(D_n)}


def cp_upper(k, n, delta):
    """Clopper-Pearson upper confidence bound for a binomial proportion."""
    return 1.0 if k >= n else beta.ppf(1 - delta, k + 1, n - k)


def jac_rank(npz, i, Xq):
    """Numerical rank of the Jacobian of theta -> (f_theta(x))_{x in Xq}."""
    W1, b1, W2, b2, W3 = (npz[k][i] for k in ["W1", "b1", "W2", "b2", "W3"])
    a1 = np.tanh(Xq @ W1 + b1)
    a2 = np.tanh(a1 @ W2 + b2)
    d2 = (1 - a2 ** 2) * W3[:, 0]                # d out / d pre2   (n,H)
    d1 = (d2 @ W2.T) * (1 - a1 ** 2)             # d out / d pre1   (n,H)
    J = np.concatenate([
        (Xq[:, :, None] * d1[:, None, :]).reshape(len(Xq), -1),   # W1
        d1,                                                         # b1
        (a1[:, :, None] * d2[:, None, :]).reshape(len(Xq), -1),    # W2
        d2,                                                         # b2
        a2,                                                         # W3
        np.ones((len(Xq), 1))], axis=1) / np.sqrt(len(Xq))          # b3
    s = np.linalg.svd(J, compute_uv=False)
    return s, J.shape[1]


def quantised_teacher_curve(npz, i, Xq, f_ref, steps):
    """Direct (prior-free) student: the teacher's own weights, uniformly
    quantised with step `st` and ideally entropy-coded (per-tensor empirical
    entropy + 32-bit header per tensor).  Returns (bits, distortion)."""
    out = []
    names = ["W1", "b1", "W2", "b2", "W3", "b3"]
    for st in steps:
        P = {k: npz[k][i] for k in names}
        bits = 0.0
        Q = {}
        for k in names:
            q = np.round(P[k] / st)
            vals, cnt = np.unique(q, return_counts=True)
            pr = cnt / cnt.sum()
            bits += -np.sum(cnt * np.log2(pr)) + 32 + len(vals) * 8   # data + header + table
            Q[k] = q * st
        a1 = np.tanh(Xq @ Q["W1"] + Q["b1"])
        a2 = np.tanh(a1 @ Q["W2"] + Q["b2"])
        fq = (a2 @ Q["W3"] + Q["b3"])[:, 0]
        out.append((bits, np.mean((fq - f_ref) ** 2)))
    return out


eps_grid = np.logspace(-8, -2, 25)
for name in ["mlp_seed", "mlp_seed_data"]:
    npz = np.load(f"../results/{name}.npz")
    F, Xq = npz["F"], npz["Xq"]
    N, n = F.shape
    m, lam, V = spectrum(F)
    trC = lam.sum()
    r = {"N": N, "n_query": n, "P": int(npz["P"]), "trace_C": trC,
         "mean_pairwise_sqdist": 2 * trC, "top_eigs": lam[:10].tolist()}
    # power-law fit on the middle of the observed spectrum
    idx = np.arange(5, min(200, len(lam)))
    slope = np.polyfit(np.log(idx + 1), np.log(lam[idx]), 1)[0]
    r["powerlaw_alpha"] = -slope
    r["R_gauss"] = [rate_gaussian(lam, e) for e in eps_grid]
    r["k_above"] = [n_above_water(lam, e) for e in eps_grid]

    # leave-one-out single-teacher FDL j(f, eps) for 40 held-out teachers
    J = []
    for i in rng.choice(N, 40, replace=False):
        mask = np.arange(N) != i
        m_i, lam_i, V_i = spectrum(F[mask])
        z = coords(F[i], m_i, V_i, n)
        resid = np.mean((F[i] - m_i) ** 2) - np.sum(z ** 2)   # energy outside observed span
        J.append([tilted_info(z, lam_i, e) for e in eps_grid] + [resid])
    J = np.array(J)
    r["j_loo_mean"] = J[:, :-1].mean(0).tolist()
    r["j_loo_sd"] = J[:, :-1].std(0).tolist()
    r["loo_out_of_span_energy_mean"] = float(J[:, -1].mean())

    # Gaussianity of the top KL coordinates
    Z = coords(F, m, V, n)
    r["kurtosis_top5"] = kurtosis(Z[:, :5], axis=0, fisher=True).tolist()

    # model-free certified lower bound: collision entropy at radius 2 sqrt(eps)
    D2 = np.sum((F[:, None, :] - F[None, :, :]) ** 2, -1) / n
    iu = np.triu_indices(N, 1)
    pair_d = D2[iu]
    perm = rng.permutation(N)
    disj = D2[perm[0::2], perm[1::2]]           # N/2 independent pairs
    lb = []
    for e in eps_grid:
        k_hit = int(np.sum(disj <= 4 * e))       # ||F-F'|| <= 2 sqrt(e)
        pc_up = cp_upper(k_hit, len(disj), 0.05)
        # success prob >= 1/2 requires R >= 1/2 log2(1/pc) - 1
        lb.append(max(0.0, 0.5 * np.log2(1 / pc_up) - 1))
    r["collision_LB_bits"] = lb
    r["min_pairwise_sqdist"] = float(pair_d.min())
    # prior-aware constructive code: index of nearest other teacher (log2(N-1) bits)
    nn = np.array([np.min(np.delete(D2[i], i)) for i in range(N)])
    r["nn_code_bits"] = float(np.log2(N - 1))
    r["nn_code_dist_median"] = float(np.median(nn))

    # functional dimension of one teacher
    s, P = jac_rank(npz, 0, Xq)
    r["jac_rank_1e-6"] = int(np.sum(s > s[0] * 1e-6))
    r["jac_rank_1e-9"] = int(np.sum(s > s[0] * 1e-9))
    r["jac_sv"] = s.tolist()
    # direct (prior-free) quantised-teacher code
    qc = quantised_teacher_curve(npz, 0, Xq, F[0], np.logspace(-5, -0.5, 30))
    r["quant_curve"] = qc
    r["spectrum"] = lam.tolist()
    res[name] = r
    print(f"\n=== {name}: N={N}, n_query={n}, P={r['P']}")
    print(f"tr C = {trC:.3e}  (mean pairwise sq. dist {2*trC:.3e}); power-law alpha ~ {r['powerlaw_alpha']:.2f}")
    print(f"top eigs: {np.array2string(lam[:6], precision=3)}")
    print(f"kurtosis of top-5 KL coords: {np.round(r['kurtosis_top5'], 2)}")
    print(f"Jacobian numerical rank: {r['jac_rank_1e-6']} (tol 1e-6), {r['jac_rank_1e-9']} (1e-9) of P={P}")
    print(f"nearest-other-teacher code: {r['nn_code_bits']:.2f} bits at median sq. dist {r['nn_code_dist_median']:.2e}")
    print(f"LOO energy outside the observed span: {r['loo_out_of_span_energy_mean']:.2e}")
    print("   eps      k_a   R_gauss   j_loo(mean+-sd)   collisionLB")
    for e, R, ka, jm, js, c in zip(eps_grid, r["R_gauss"], r["k_above"], r["j_loo_mean"], r["j_loo_sd"], lb):
        print(f"{e:9.2e} {ka:5d} {R:9.1f} {jm:9.1f} +- {js:5.1f}   {c:6.2f}")
    print("direct quantised-teacher code (bits, sq. dist):")
    for b, d in qc[::4]:
        print(f"   {b:9.0f} bits   {d:.3e}")

json.dump({"eps_grid": eps_grid.tolist(), **res}, open("../results/mlp_analysis.json", "w"))
