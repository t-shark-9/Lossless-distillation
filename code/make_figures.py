"""Figures for notes/functional_information.md (static PNG, light surface)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fdl import achievability_gap

S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"      # categorical slots 1-4
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2, "lines.solid_capstyle": "round",
})

cf = json.load(open("../results/mlp_crossfit.json"))
an = json.load(open("../results/mlp_analysis.json"))

# ---------------- Figure 1: rates for the MLP training priors -----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.9))
a = ax[0]
for key, col, lab in [("mlp_seed", S1, "seed-only prior"), ("mlp_seed_data", S2, "seed + data prior")]:
    e = np.array(cf[key]["eps_grid"]); R = np.array(cf[key]["R_crossfit"])
    lo, hi = np.array(cf[key]["R_boot5"]), np.array(cf[key]["R_boot95"])
    m = R > 0.5
    a.plot(e[m], R[m], color=col, label=f"$R_\\pi(\\varepsilon)$ estimate, {lab}")
    a.fill_between(e[m], lo[m], hi[m], color=col, alpha=0.10, lw=0)
q = np.array(an["mlp_seed"]["quant_curve"])
o = np.argsort(q[:, 1]); q = q[o]
m = (q[:, 1] > 1e-7) & (q[:, 1] < 3e-3)
a.plot(q[m, 1], q[m, 0], color=S3, label="teacher's own weights, quantised + entropy-coded")
eg = np.array(an["eps_grid"]); lb = np.array(an["mlp_seed"]["collision_LB_bits"])
m = (lb > 0) & (eg >= 1e-7) & (eg <= 3e-3)
a.plot(eg[m], lb[m], color=S4, label="model-free certified lower bound (N = 400)")
a.set_xscale("log"); a.set_yscale("log")
a.set_xlim(1e-7, 3e-3); a.set_ylim(0.3, 1e5)
a.set_xlabel("tolerance $\\varepsilon$  (squared $L^2(D)$ error)")
a.set_ylabel("bits")
a.set_title("Functional description length of MLP teachers", loc="left", fontsize=11, color=INK)
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8, ncol=2)

b = ax[1]
for key, col, lab in [("mlp_seed", S1, "seed-only prior"), ("mlp_seed_data", S2, "seed + data prior")]:
    rows = np.array(cf[key]["codebook_rows"])
    b.plot(np.log2(rows[:, 0]), rows[:, 2], "o", color=col, ms=7, mec=SURF, mew=1.5, label=lab)
xs = np.linspace(0, 8, 50)
b.plot(xs, xs + np.log2(1 / np.log(2)), color=MUTED, lw=1, label="ideal: $\\log_2 M + \\log_2(1/\\ln 2)$")
b.set_xlabel("actual codebook size $\\log_2 M$ (M other retrained teachers)")
b.set_ylabel("Gaussian-model $-\\log_2 \\pi(B_\\varepsilon(f))$ at achieved $\\varepsilon$")
b.set_title("Second-order model predicts real random-coding", loc="left", fontsize=11, color=INK)
b.set_xlim(0, 8); b.set_ylim(0, 10)
b.legend(loc="upper left", fontsize=8.5)
fig.tight_layout()
fig.savefig("../results/fig_mlp_rates.png", dpi=150)

# ---------------- Figure 2: Gaussian-prior checks ------------------------------
rng = np.random.default_rng(3)
Ks = np.array([3, 10, 30, 100, 300, 1000])
gaps = []
for K in Ks:
    lam = np.ones(K)
    g = [achievability_gap(rng.standard_normal(K), lam, 0.01 * K, n_mc=60_000, rng=rng) for _ in range(10)]
    gaps.append(np.mean(g))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
a = ax[0]
kk = np.logspace(0.4, 3.1, 100)
a.plot(kk, 0.5 * np.log2(np.pi * kk), color=MUTED, lw=1, label="$\\frac{1}{2}\\log_2(\\pi k_a)$")
a.plot(Ks, gaps, "o", color=S1, ms=7, mec=SURF, mew=1.5, label="measured gap $\\Delta_f(\\varepsilon)$")
a.set_xscale("log"); a.set_xlabel("dimensions above water $k_a$"); a.set_ylabel("bits")
a.set_title("Upper and lower bounds on a single teacher's FDL differ by $\\Delta_f$", loc="left", fontsize=10.5, color=INK)
a.legend(loc="upper left", fontsize=9)
b = ax[1]
rows = np.load("../results/converse_rows.npy")
b.plot(rows[:, 0], rows[:, 1], color=S1, label="converse: no $R$-bit code can do better")
b.plot(rows[:, 0], rows[:, 2], color=S2, label="achieved by a random code from $P_{G^*}$")
b.axvline(99.95, color=AXIS, lw=1)
b.text(100.6, 0.02, "$R_\\pi(\\varepsilon)$", color=INK2, fontsize=9)
b.set_xlabel("student size $R$ (bits)"); b.set_ylabel("P[student is $\\varepsilon$-lossless]")
b.set_title("Gaussian prior, 40 dims: success vs. size", loc="left", fontsize=10.5, color=INK)
b.legend(loc="upper left", fontsize=8.5)
fig.tight_layout()
fig.savefig("../results/fig_gaussian_checks.png", dpi=150)

# ---------------- Figure 3: spectra -------------------------------------------
fig, a = plt.subplots(figsize=(5.6, 4.0))
for key, col, lab in [("mlp_seed", S1, "seed-only prior"), ("mlp_seed_data", S2, "seed + data prior")]:
    lam = np.array(an[key]["spectrum"])
    lam = lam[lam > lam[0] * 1e-9]
    a.plot(np.arange(1, lam.size + 1), lam, color=col, label=lab)
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel("index $i$"); a.set_ylabel("eigenvalue $\\lambda_i$ of the teacher covariance on $L^2(D)$")
a.set_title("Function-space spectrum of trained teachers", loc="left", fontsize=11, color=INK)
a.legend(loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig("../results/fig_mlp_spectrum.png", dpi=150)
print("gaps", dict(zip(Ks.tolist(), np.round(gaps, 3).tolist())))
