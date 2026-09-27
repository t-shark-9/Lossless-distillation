"""Checks for Section 7 (a solvable distillation scaling law; open problem P4).

Run:  python3 verification/check_scaling_law.py      (about two minutes)

Claims are identified by their LaTeX labels.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import logsumexp

rng = np.random.default_rng(7)


def report(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return ok


results = []

# ---------------------------------------------------------------------------
# lem:quad  e^{-osc} <= KL(p || softmax(log p + delta)) / (Var_p(delta)/2) <= e^{osc}
# ---------------------------------------------------------------------------
lo, hi = np.inf, -np.inf
for _ in range(20000):
    V = int(rng.integers(2, 12))
    z = rng.normal(size=V) * rng.uniform(0, 4)
    d = rng.normal(size=V) * rng.uniform(0, 3)
    p = np.exp(z - z.max()); p /= p.sum()
    lq = z + d - np.logaddexp.reduce(z + d)
    kl = float(np.sum(p * (np.log(p) - lq)))
    var = float(np.sum(p * d ** 2) - np.sum(p * d) ** 2)
    o = d.max() - d.min()
    if var > 1e-10:
        r = kl / (0.5 * var)
        lo, hi = min(lo, r * np.exp(o)), max(hi, r * np.exp(-o))
results.append(report(f"lem:quad sandwich on 20000 random instances (min {lo:.4f} >= 1, max {hi:.4f} <= 1)",
                      lo >= 1 - 1e-9 and hi <= 1 + 1e-9))

# ---------------------------------------------------------------------------
# lem:deeplin  products of L matrices with a width-d layer are exactly the rank-<=d maps
# ---------------------------------------------------------------------------
dims = [7, 3, 5, 6]                                     # d_0=7 (features), widths 3, 5, output 6
Ws = [rng.normal(size=(dims[i + 1], dims[i])) for i in range(3)]
prod = Ws[2] @ Ws[1] @ Ws[0]
M = rng.normal(size=(6, 3)) @ rng.normal(size=(3, 7))  # an arbitrary rank-3 target
U, s, Vt = np.linalg.svd(M)
W1 = np.zeros((3, 7)); W1[:3] = (np.sqrt(s[:3])[:, None] * Vt[:3])
W2 = np.zeros((5, 3)); W2[:3, :3] = np.eye(3)
W3 = np.zeros((6, 5)); W3[:, :3] = U[:, :3] * np.sqrt(s[:3])
results.append(report("lem:deeplin rank(product) <= min width, and an explicit factorisation realises a rank-3 map",
                      np.linalg.matrix_rank(prod) <= 3 and np.allclose(W3 @ W2 @ W1, M)))

# ---------------------------------------------------------------------------
# thm:curve / cor:nested  exact learning curve of the well-optimised (OLS) student
#   E eps_G = 1/2 [ Lambda(min(k,kT)) + (Lambda(k,kT) + sigma^2 V') k/(n-k-1) ]
# ---------------------------------------------------------------------------
Pdim, Vp, alpha = 60, 8, 1.7
lam = np.arange(1, Pdim + 1.0) ** (-alpha)
G = rng.normal(size=(Vp, Pdim)); G *= np.sqrt(lam / (G ** 2).sum(0))   # ||g_i||^2 = lam_i, not aligned
Lam = lambda a, b=Pdim: float(lam[a:b].sum())


def mc_curve(k, kT, n, s2, reps=3000):
    T = G.copy(); T[:, kT:] = 0
    errs = np.empty(reps)
    for r in range(reps):
        X = rng.normal(size=(Pdim, n))
        Y = T @ X + np.sqrt(s2) * rng.normal(size=(Vp, n))
        Xk = X[:k]
        W = np.linalg.solve(Xk @ Xk.T, Xk @ Y.T).T
        D = G.copy(); D[:, :k] -= W
        errs[r] = 0.5 * (D ** 2).sum()
    return errs.mean(), errs.std() / np.sqrt(reps)


def exact_curve(k, kT, n, s2):
    return 0.5 * (Lam(min(k, kT)) + (Lam(k, kT) if kT > k else 0.0) * k / (n - k - 1) + s2 * Vp * k / (n - k - 1))


zs = []
for (k, kT, n, s2) in [(5, 60, 20, 0.0), (5, 12, 20, 0.0), (10, 60, 40, 0.0), (6, 60, 30, 0.5), (4, 20, 12, 0.2)]:
    m, se = mc_curve(k, kT, n, s2)
    zs.append((m - exact_curve(k, kT, n, s2)) / se)
m, _ = mc_curve(5, 3, 20, 0.0, reps=20)                 # k >= kT, soft labels: deterministic, exact recovery
results.append(report(f"thm:curve Monte Carlo vs exact formula, 5 configurations (max |z| = {max(map(abs, zs)):.2f} < 4)",
                      max(map(abs, zs)) < 4))
results.append(report("thm:curve soft labels with k >= k_T: error is exactly Lambda(k_T)/2 (exact recovery of the teacher)",
                      abs(m - 0.5 * Lam(3)) < 1e-10))

# multinomial hard labels, whitened: the formula needs only second moments (linearised model)
Vv, k, n = 5, 3, 25
Gam_p = np.full(Vv, 1.0 / Vv); Gam = np.diag(Gam_p) - np.outer(Gam_p, Gam_p)
ev, evec = np.linalg.eigh(Gam); Q = evec[:, 1:]                     # orthonormal basis of 1-perp
iwh = Q @ np.diag(ev[1:] ** -0.5) @ Q.T                               # Gamma^{+1/2}
Gs = rng.normal(size=(Vv - 1, k)) * 0.3
errs = []
for _ in range(6000):
    X = rng.normal(size=(k, n))
    y = rng.integers(0, Vv, size=n)                                   # uniform teacher: p = 1/V
    xi = Q.T @ iwh @ (np.eye(Vv)[y].T - Gam_p[:, None])              # whitened one-hot noise, Cov = I_{V-1}
    Y = Gs @ X + xi
    W = np.linalg.solve(X @ X.T, X @ Y.T).T
    errs.append(0.5 * ((W - Gs) ** 2).sum())
pred = 0.5 * (Vv - 1) * k / (n - k - 1)
z = (np.mean(errs) - pred) / (np.std(errs) / np.sqrt(len(errs)))
results.append(report(f"thm:curve with actual multinomial labels (whitened): MC {np.mean(errs):.4f} vs (V-1)k/(2(n-k-1)) = {pred:.4f}, z={z:+.2f}",
                      abs(z) < 4))

# cor:nested capacity gap: optimal teacher size is k_T = k for every n and sigma^2; penalty <= (n-1)/(n-k-1)
ok = True
for (k, n, s2) in [(5, 12, 0.0), (5, 40, 0.0), (10, 200, 0.0), (5, 40, 0.3)]:
    vals = [exact_curve(k, kT, n, s2) for kT in range(0, Pdim + 1)]
    ok &= int(np.argmin(vals)) == k and max(vals[k:]) / vals[k] <= (n - 1) / (n - k - 1) + 1e-12
results.append(report("cor:nested optimal teacher size equals student capacity; too-strong-teacher penalty <= (n-1)/(n-k-1)", ok))

# ---------------------------------------------------------------------------
# thm:law  envelope exponents (soft alpha-1, hard (alpha-1)/alpha) and the unified two-sided law
#   eps*(k,n) := min_{k'<=min(k,n-2)} E eps  ~  k^{1-alpha} + n^{1-alpha} + (nu/n)^{(alpha-1)/alpha},  nu = sigma^2 V'
# ---------------------------------------------------------------------------
alpha = 2.0
lamL = np.arange(1, 2_000_001, dtype=float) ** (-alpha)
tailL = np.concatenate([[lamL.sum()], lamL.sum() - np.cumsum(lamL)])          # tailL[k] = Lambda(k)


def env(k, n, nu):
    kk = np.arange(0, min(k, n - 2) + 1)
    return float(np.min(0.5 * (tailL[kk] * (n - 1) / (n - kk - 1) + nu * kk / (n - kk - 1))))


ns = np.array([1e3, 3e3, 1e4, 3e4, 1e5]).astype(int)
s_soft = np.polyfit(np.log(ns), np.log([env(10 ** 7, n, 0.0) for n in ns]), 1)[0]
s_hard = np.polyfit(np.log(ns), np.log([env(10 ** 7, n, 30.0) for n in ns]), 1)[0]
results.append(report(f"thm:law soft envelope slope {s_soft:.3f} vs -(alpha-1) = {-(alpha-1):.3f}", abs(s_soft + alpha - 1) < 0.05))
results.append(report(f"thm:law hard envelope slope {s_hard:.3f} vs -(alpha-1)/alpha = {-(alpha-1)/alpha:.3f}",
                      abs(s_hard + (alpha - 1) / alpha) < 0.05))
ratios = []
for k in [1, 10, 100, 1000, 10 ** 6]:
    for n in [10, 100, 1000, 10 ** 4, 10 ** 5]:
        for nu in [0.0, 1e-6, 1e-3, 1.0, 10.0]:
            if n >= max(4, nu):
                ratios.append(env(k, n, nu) / (k ** (1 - alpha) + n ** (1 - alpha) + (nu / n) ** ((alpha - 1) / alpha)))
results.append(report(f"thm:law two-sided law on a (k,n,nu) grid incl. the soft/hard crossover: ratio in "
                      f"[{min(ratios):.3f}, {max(ratios):.3f}]", max(ratios) / min(ratios) < 25))

# ---------------------------------------------------------------------------
# lem:trunc  ||[B+N]_d - B||_F^2 <= 2 sum_{j>d} s_j(B)^2 + 31 d ||N||_op^2
# ---------------------------------------------------------------------------
worst = 0.0
for _ in range(20000):
    m_, k_ = rng.integers(2, 9, size=2); d = int(rng.integers(1, min(m_, k_) + 1))
    B = rng.normal(size=(m_, k_)) * rng.uniform(0, 3, size=k_)
    N = rng.normal(size=(m_, k_)) * 10 ** rng.uniform(-2, 1)
    u, s, vt = np.linalg.svd(B + N)
    sB = np.linalg.svd(B, compute_uv=False)
    worst = max(worst, (((u[:, :d] * s[:d]) @ vt[:d] - B) ** 2).sum() / (2 * (sB[d:] ** 2).sum() + 31 * d * np.linalg.norm(N, 2) ** 2))
results.append(report(f"lem:trunc on 20000 random instances (max lhs/rhs = {worst:.3f} <= 1)", worst <= 1))

# ---------------------------------------------------------------------------
# thm:rrr  the width-d global minimiser (reduced-rank regression) obeys the deterministic bound
#   ||M_rrr - W*||_F^2 <= [ 2 lmax(S) sum_{j>d} s_j(W*)^2 + 31 d ||N S^{1/2}||_op^2 ] / lmin(S)
# ---------------------------------------------------------------------------
ok = True
for _ in range(200):
    Vq, k, d, n = 12, 10, 3, int(rng.integers(15, 80))
    Wst = rng.normal(size=(Vq, k)) * np.arange(1, k + 1.0) ** -1.0
    X = rng.normal(size=(k, n)); Y = Wst @ X + rng.normal(size=(Vq, n)) * rng.uniform(0, 1)
    S = X @ X.T; ev, evec = np.linalg.eigh(S); Sh = evec @ np.diag(ev ** 0.5) @ evec.T; Sih = evec @ np.diag(ev ** -0.5) @ evec.T
    Wols = np.linalg.solve(S, X @ Y.T).T
    u, s, vt = np.linalg.svd(Wols @ Sh); Af, Bf = u[:, :d] * s[:d], vt[:d] @ Sih; Wrrr = Af @ Bf
    # Wrrr minimises the empirical loss over rank-d maps: compare with nearby rank-d competitors A'B'
    loss = lambda W: ((Y - W @ X) ** 2).sum()
    ok &= all(loss(Wrrr) <= loss((Af + 1e-3 * rng.normal(size=Af.shape)) @ (Bf + 1e-3 * rng.normal(size=Bf.shape))) + 1e-9
              for _ in range(3))
    N = Wols - Wst
    sW = np.linalg.svd(Wst, compute_uv=False)
    rhs = (2 * ev.max() * (sW[d:] ** 2).sum() + 31 * d * np.linalg.norm(N @ Sh, 2) ** 2) / ev.min()
    ok &= ((Wrrr - Wst) ** 2).sum() <= rhs
results.append(report("thm:rrr deterministic bound for the width-d global minimiser (200 random instances)", ok))

# ---------------------------------------------------------------------------
# prop:flow  depth-L gradient flow, spectral init eps: s(t)/shat -> 1{shat*tau > 1} at t = tau*c_L(eps)
# ---------------------------------------------------------------------------
def flow(L, shat, eps, tau):
    cL = np.log(1 / eps) if L == 2 else eps ** (-(L - 2)) / (L - 2)
    f = lambda t, s: L * np.maximum(s, 0) ** (2 - 2 / L) * (shat - s)
    return solve_ivp(f, [0, tau * cL], [eps ** L], rtol=1e-10, atol=1e-300, method="LSODA").y[0, -1] / shat


vals = {(L, tau): flow(L, 1.0, 1e-3, tau) for L in (2, 3, 4) for tau in (0.8, 1.25)}
results.append(report("prop:flow hard-threshold limit, eps=1e-3: " +
                      ", ".join(f"L={L}: {vals[(L, 0.8)]:.3f}/{vals[(L, 1.25)]:.3f}" for L in (2, 3, 4)) +
                      "  (below/above threshold)",
                      all(vals[(L, 0.8)] < 0.1 and vals[(L, 1.25)] > 0.95 for L in (2, 3, 4))))

# ---------------------------------------------------------------------------
# thm:depth  deep (thresholding path) vs shallow (uniform shrinkage) early stopping, whitened sequence model
# ---------------------------------------------------------------------------
Vq = kq = 300; alpha = 2.0; lamq = np.arange(1, kq + 1.0) ** (-alpha)
Uq = np.linalg.qr(rng.normal(size=(Vq, Vq)))[0]; Wq = np.linalg.qr(rng.normal(size=(kq, kq)))[0]
Bq = (Uq[:, :kq] * np.sqrt(lamq)) @ Wq.T
ok, msg = True, []
for n in (1e4, 1e5):
    deep, shal = [], []
    for _ in range(3):
        Shat = Bq + rng.normal(size=(Vq, kq)) / np.sqrt(n)
        u, s, vt = np.linalg.svd(Shat)
        deep.append(min((((u[:, :r] * s[:r]) @ vt[:r] - Bq) ** 2).sum() for r in range(kq + 1)))
        g = (Bq * Shat).sum() / (Shat * Shat).sum(); shal.append(((g * Shat - Bq) ** 2).sum())
    envq = min(lamq[r:].sum() + r * (Vq + kq) / n for r in range(kq + 1))
    B2, N2 = lamq.sum(), Vq * kq / n
    msg.append(f"n={n:.0e}: deep {np.mean(deep):.3f}, envelope {envq:.3f}, shallow {np.mean(shal):.3f} "
               f"(harmonic mean {B2 * N2 / (B2 + N2):.3f})")
    ok &= np.mean(deep) <= 1.5 * envq and np.mean(shal) >= 2 * np.mean(deep) and abs(np.mean(shal) / (B2 * N2 / (B2 + N2)) - 1) < 0.1
for line in msg:
    print("      " + line)
results.append(report("thm:depth deep early stopping tracks the envelope; shallow is >= 2x worse and matches ||B||^2 E||N||^2/(||B||^2+E||N||^2)", ok))

# ---------------------------------------------------------------------------
# thm:expo  exponential phase of D(rho) (rho in nats) for the Gaussian source of prop:powerlaw
# ---------------------------------------------------------------------------
def DRlog(ll, rho):
    f = lambda lw: 0.5 * np.sum(np.maximum(ll - lw, 0.0)) - rho
    lw = brentq(f, ll.min() - 2 * rho - 10, ll.max())
    act = ll > lw
    parts = ([np.log(act.sum()) + lw] if act.any() else []) + ([logsumexp(ll[~act])] if (~act).any() else [])
    return np.log(0.5) + logsumexp(parts), lw


ll = np.log(np.arange(1, 200001.0) ** -1.5)
ok = True
for rho in (50.0, 500.0):
    l1, lw = DRlog(ll, rho); l2, _ = DRlog(ll, rho + 1e-4)
    w = np.exp(lw); lamx = np.exp(ll)
    Keff = (lamx > w).sum() + lamx[lamx <= w].sum() / w
    ok &= abs(-(l2 - l1) / 1e-4 * Keff / 2 - 1) < 1e-3
results.append(report("thm:expo (i) local-rate identity -dlnD/drho = 2/K_eff(w)", ok))
lamf = np.arange(1, 1001.0) ** -2.0; llf = np.log(lamf); kT = 1000
rho0 = 0.5 * np.sum(np.log(lamf / lamf[-1]))
exact = lambda rho: np.log(kT / 2) + np.mean(llf) - 2 * rho / kT
results.append(report(f"thm:expo (ii) finite rank: D = (k_T/2) GM e^(-2 rho/k_T) beyond rho_kT = {rho0:.0f} nats",
                      all(abs(DRlog(llf, r)[0] - exact(r)) < 1e-8 for r in (rho0 + 1, 2 * rho0, 3 * rho0))))
beta = 0.5; lle = -beta * np.arange(1, 6001.0)
se = -DRlog(lle, 1e4)[0] / np.sqrt(1e4)
results.append(report(f"thm:expo (iv) exponential spectrum: -lnD/sqrt(rho) = {se:.3f} -> 2 sqrt(beta) = {2*np.sqrt(beta):.3f}",
                      abs(se / (2 * np.sqrt(beta)) - 1) < 0.05))
ok, msg = True, []
kT, a = 200, 2.0
for g in (1.0, 10.0, 1e3):
    lamg = np.arange(1, 100001.0) ** -a; lamg[kT:] /= g; llg = np.log(lamg)
    rhos = np.linspace(50, 3000, 2951); inwin = []
    for r in rhos:
        l1, lw = DRlog(llg, r); l2, _ = DRlog(llg, r + 1e-3)
        inwin.append(lw <= llg[kT - 1] and -(l2 - l1) / 1e-3 >= 2 / (1.25 * kT))
    length = np.sum(inwin) * (rhos[1] - rhos[0])
    wlow = max(lamg[kT], lamg[kT:].sum() / (0.25 * kT))
    pred = max(0.5 * kT * np.log(lamg[kT - 1] / wlow), 0.0)
    msg.append(f"g={g:g}: {length:.0f} vs {pred:.0f}")
    ok &= abs(length - pred) <= 3
results.append(report("thm:expo (v) exponential window length (k_T/2) ln(lam_kT / w_eta), eta=1/4: " + "; ".join(msg), ok))

# ---------------------------------------------------------------------------
# prop:rf  random-feature approximation error ~ k^{-min(alpha-1, a)}
# ---------------------------------------------------------------------------
P_, a = 6000, 1.5
mu = np.arange(1, P_ + 1.0) ** -a
ok, msg = True, []
for alpha in (2.0, 4.0):
    bvec = np.sqrt(np.arange(1, P_ + 1.0) ** -alpha / mu) * rng.choice([-1, 1], size=P_)
    ks = np.array([25, 50, 100, 200, 400]); A = []
    for k in ks:
        v = []
        for _ in range(3):
            F = rng.normal(size=(k, P_)); FS = F * mu; c = FS @ bvec
            v.append(np.sum(mu * bvec ** 2) - c @ np.linalg.solve(FS @ F.T, c))
        A.append(np.mean(v))
    sl = np.polyfit(np.log(ks), np.log(A), 1)[0]
    msg.append(f"alpha={alpha}: {sl:.3f} vs {-min(alpha - 1, a):.3f}")
    ok &= abs(sl + min(alpha - 1, a)) < 0.12
results.append(report("prop:rf slopes (a=1.5): " + "; ".join(msg), ok))

print("\nALL PASS" if all(results) else "\nSOME CHECKS FAILED")
