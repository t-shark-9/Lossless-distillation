"""Checks for Section 3 (capacity lower bounds) and Section 2 (softmax-bottleneck floor).

Run:  python3 verification/check_capacity.py
"""
import numpy as np
from scipy.optimize import brentq

rng = np.random.default_rng(2)


def report(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return ok


results = []

# prop:powerlaw: reverse water-filling for a power-law spectrum gives D(R) ~ R^{-(alpha-1)}
def distortion_rate(lams, R):
    """Gaussian source, eigenvalues lams, rate R nats: D(R) = sum min(w, lam_i) (per-coordinate MSE)."""
    rate = lambda w: 0.5 * np.sum(np.log(np.maximum(lams / w, 1.0))) - R
    w = brentq(rate, 1e-300, lams.max())
    return float(np.sum(np.minimum(w, lams)))


for alpha in [1.5, 2.0, 3.0]:
    lams = np.arange(1, 2_000_001, dtype=float) ** (-alpha)
    Rs = np.array([1e2, 1e3, 1e4])
    Ds = np.array([distortion_rate(lams, R) for R in Rs])
    slope = np.polyfit(np.log(Rs), np.log(Ds), 1)[0]
    results.append(report(f"alpha={alpha}: fitted exponent {slope:.3f} vs predicted {-(alpha-1):.3f}",
                          abs(slope + (alpha - 1)) < 0.05))

# prop:phase: truncated spectrum (teacher with k_T components): power law, then exponential decay
lams = np.arange(1, 1001, dtype=float) ** (-2.0)
for R in [50, 500, 2000, 4000]:
    print(f"      k_T=1000, R={R:5d} nats: D(R) = {distortion_rate(lams, R):.3e}")

# thm:bottleneck: softmax-bottleneck floor, checked against a width-d student fitted by L-BFGS.
# Near-lossless instance: teacher logits = rank-d part + small rank-3 part, so M is small.
from scipy.optimize import minimize

N, Vs, d = 300, 10, 3
Zt = rng.normal(size=(N, d)) @ rng.normal(size=(d, Vs)) * 0.5 \
    + rng.normal(size=(N, 3)) @ rng.normal(size=(3, Vs)) * 0.15
Pt = np.exp(Zt - Zt.max(1, keepdims=True)); Pt /= Pt.sum(1, keepdims=True)
logPt = np.log(Pt)
Cc = np.eye(Vs) - np.ones((Vs, Vs)) / Vs
sv = np.linalg.svd(logPt @ Cc, compute_uv=False)
tail = float(np.sum(sv[d + 1:] ** 2))   # centred student log-prob matrix has rank <= d+1
pmin = Pt.min()


def unpack(x):
    H = x[:N * d].reshape(N, d); W = x[N * d:N * d + d * Vs].reshape(d, Vs); b = x[N * d + d * Vs:]
    return H, W, b


def obj(x):
    H, W, b = unpack(x)
    Zs = H @ W + b
    logQ = Zs - np.logaddexp.reduce(Zs, axis=1, keepdims=True)
    f = float(np.mean(np.sum(Pt * (logPt - logQ), axis=1)))
    G = (np.exp(logQ) - Pt) / N
    return f, np.concatenate([(G @ W.T).ravel(), (H.T @ G).ravel(), G.sum(0)])


x0 = rng.normal(size=N * d + d * Vs + Vs) * 0.1
res = minimize(obj, x0, jac=True, method="L-BFGS-B", options={"maxiter": 20000, "gtol": 1e-10})
H, W, b = unpack(res.x)
Zs = H @ W + b
logQ = Zs - np.logaddexp.reduce(Zs, axis=1, keepdims=True)
klbar = float(np.mean(np.sum(Pt * (logPt - logQ), axis=1)))
M = float(max((logPt - logQ).max(), 0.0))
bound = np.exp(-M) * pmin / (2 * N) * tail
# row-weighted version: Pi = diag(min_y p_i(y)) preserves rank under left multiplication
Pi_half = np.sqrt(Pt.min(axis=1))[:, None]
sv_w = np.linalg.svd(Pi_half * (logPt @ Cc), compute_uv=False)
bound_w = np.exp(-M) / (2 * N) * float(np.sum(sv_w[d + 1:] ** 2))
print(f"      fitted student: mean KL = {klbar:.5f}; floor (global p_min) = {bound:.2e}; "
      f"floor (row-weighted) = {bound_w:.2e}  (M = {M:.3f})")
results.append(report("softmax-bottleneck floors <= achieved KL of a width-d student (non-vacuous)",
                      0 < bound <= bound_w <= klbar))

# cor:facts: the log-loss floor ln V - R ln 2 / K is attained by
# a student that memorises R / log2(V) facts and is uniform on the rest.
K, Vf = 1000, 16
for Rbits in [0, 1000, 2000, 4000]:
    known = min(K, int(Rbits / np.log2(Vf)))
    achieved = (K - known) * np.log(Vf) / K
    floor = max(0.0, np.log(Vf) - Rbits * np.log(2) / K)
    results.append(report(f"K-facts R={Rbits}: achieved {achieved:.3f} >= floor {floor:.3f} (tight)",
                          achieved >= floor - 1e-12 and achieved - floor < np.log(Vf) / K + 1e-12))

print("\nALL PASS" if all(results) else "\nSOME CHECKS FAILED")
