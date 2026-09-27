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

# thm:capacity: randomized exact test  E_pi min_theta (1/N) sum_i KL >= (I(T;Y) - R ln2)/(mN)
import itertools


def kl_rows(P, Q):
    return np.sum(P * (np.log(P) - np.log(Q)), axis=-1)


worst = np.inf
for trial in range(300):
    Vc, X, Mt = 3, 2, int(rng.integers(2, 9))                 # vocabulary, inputs, number of teachers
    teachers = rng.dirichlet(np.ones(Vc) * 10 ** rng.uniform(-1, 0.7), size=(Mt, X))
    prior = rng.dirichlet(np.ones(Mt))
    Rbits = int(rng.integers(0, 3))
    k = 2 ** Rbits
    n_copy = int(rng.integers(0, min(k, Mt) + 1))                 # seed the class with some teachers
    students = np.concatenate([teachers[rng.choice(Mt, n_copy, replace=False)],
                               rng.dirichlet(np.ones(Vc), size=(k - n_copy, X))])[:k]
    for N in (1, 2):
        xs = rng.integers(0, X, size=N)
        lhs = sum(prior[j] * min(np.mean(kl_rows(teachers[j, xs], st[xs])) for st in students) for j in range(Mt))
        for m in (1, 2):
            # exact I(T;Y), Y = (Y_ij) with Y_ij ~ p_T(.|x_i)
            HY_T = sum(prior[j] * m * np.sum(-teachers[j, xs] * np.log(teachers[j, xs])) for j in range(Mt))
            HY = 0.0
            for y in itertools.product(range(Vc), repeat=m * N):
                yy = np.array(y).reshape(N, m)
                py = sum(prior[j] * np.prod(teachers[j, xs[:, None], yy]) for j in range(Mt))
                HY -= py * np.log(py)
            rhs = (HY - HY_T - Rbits * np.log(2)) / (m * N)
            worst = min(worst, lhs - rhs)
results.append(report(f"capacity floor holds on 300 random instances (min slack {worst:.2e})", worst >= -1e-10))

# cor:facts-tv: at R = 0 the implicit bound forces e >= 1 - 1/V, attained by the uniform student
for Vf in (2, 3, 16):
    g = lambda e: -e * np.log(e) - (1 - e) * np.log(1 - e) + e * np.log(Vf - 1)
    e_star = 1 - 1 / Vf
    results.append(report(f"facts-TV bound at R=0, V={Vf}: g(1-1/V) = ln V", abs(g(e_star) - np.log(Vf)) < 1e-12))

print("\nALL PASS" if all(results) else "\nSOME CHECKS FAILED")
