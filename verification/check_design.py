"""Checks for Section 7 (optimal input design).

Every claim is identified by its LaTeX label. Expectations are computed exactly (by enumeration or
closed form) where possible; the asymptotic constants of thm:local-design(a) and prop:iw-design(a)
are checked by Monte Carlo with fixed seeds and a tolerance of four standard errors.
Run:  python3 verification/check_design.py
"""
import itertools
from math import comb, lgamma, log

import numpy as np
from scipy.optimize import linprog, minimize, minimize_scalar
from scipy.special import zeta

rng = np.random.default_rng(7)
TOL = 1e-9


def report(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return ok


results = []
sig = lambda z: 1 / (1 + np.exp(-z))
dsig = lambda z: sig(z) * (1 - sig(z))

# ---------------------------------------------------------------------------------------------
# prop:design-game -- class-restricted transfer coefficient and the design LP
# ---------------------------------------------------------------------------------------------


def design_game(Gh):
    """Gh[g, x] = g(x) / E_mu g. Returns (v, D, v_dual): v = max_D min_g E_D Gh[g] (primal LP)."""
    nG, nX = Gh.shape
    res = linprog(np.r_[np.zeros(nX), -1.0], A_ub=np.c_[-Gh, np.ones(nG)], b_ub=np.zeros(nG),
                  A_eq=np.r_[np.ones(nX), 0.0][None], b_eq=[1.0], bounds=[(0, None)] * nX + [(None, None)])
    dual = linprog(np.r_[np.zeros(nG), 1.0], A_ub=np.c_[Gh.T, -np.ones(nX)], b_ub=np.zeros(nX),
                   A_eq=np.r_[np.ones(nG), 0.0][None], b_eq=[1.0], bounds=[(0, None)] * nG + [(None, None)])
    return -res.fun, res.x[:nX], dual.fun


ok = True
for _ in range(200):
    nX, nG = rng.integers(2, 7), rng.integers(1, 9)
    mu = rng.dirichlet(np.ones(nX))
    G = rng.random((nG, nX)) ** 3 * (rng.random((nG, nX)) < 0.7)
    G = G[(G @ mu) > 0]
    if len(G) == 0:
        continue
    v, D, v_dual = design_game(G / (G @ mu)[:, None])
    kappa = np.max((G @ mu) / np.maximum(G @ D, 1e-300))
    rho = np.max(mu / np.maximum(D, 1e-300))
    ok &= abs(v - v_dual) < 1e-7 and v >= 1 - 1e-9 and abs(kappa - 1 / v) < 1e-6 * kappa and kappa <= rho * (1 + 1e-9)
results.append(report("prop:design-game (iii): LP duality, v >= 1, min_D kappa = 1/v, kappa <= rho (200 games)", ok))
mu = rng.dirichlet(np.ones(5))
v, D, _ = design_game(np.eye(5) / mu[:, None])
results.append(report("prop:design-game (i): all singleton profiles -> kappa = rho, optimum D = mu",
                      abs(v - 1) < 1e-9 and np.allclose(D, mu, atol=1e-7)))
eps = 0.01
v, D, _ = design_game(np.array([[0.0, 0.3]]) / (np.array([[0.0, 0.3]]) @ np.array([1 - eps, eps]))[:, None])
results.append(report("prop:design-game (iv): needle, min kappa = eps at the point mass on x_1",
                      abs(1 / v - eps) < 1e-9 and np.allclose(D, [0, 1])))

# ---------------------------------------------------------------------------------------------
# thm:separable -- ingredients of the proof, the identity inf_D S_n = L_n, and the upper bound
# ---------------------------------------------------------------------------------------------


def laplace_expected_kl(p, N):
    V = len(p)
    tot = 0.0
    for c in itertools.product(range(N + 1), repeat=V - 1):
        if sum(c) > N:
            continue
        cnt = np.array(list(c) + [N - sum(c)])
        logpr = lgamma(N + 1) - sum(lgamma(ci + 1) for ci in cnt) + float(np.sum(cnt * np.log(p)))
        ph = (cnt + 1) / (N + V)
        tot += np.exp(logpr) * float(np.sum(p * np.log(p / ph)))
    return tot


ok = True
for _ in range(40):
    V, N = int(rng.integers(2, 5)), int(rng.integers(0, 11))
    p = rng.dirichlet(np.ones(V) * 0.7)
    ok &= laplace_expected_kl(p, N) <= log(1 + (V - 1) / (N + 1)) + TOL
results.append(report("thm:separable: Laplace estimator E KL <= log(1+(V-1)/(N+1)) (exact enumeration)", ok))

ok = True
for _ in range(200):
    n, pi, a = int(rng.integers(1, 60)), float(rng.random()), float(rng.uniform(0.5, 5))
    pmf = np.array([comb(n, j) * pi ** j * (1 - pi) ** (n - j) for j in range(n + 1)])
    lhs = float(pmf @ np.array([1.0] + [min(1.0, a / j) for j in range(1, n + 1)]))
    ok &= lhs <= min(1.0, 2 * a / ((n + 1) * pi)) + TOL
results.append(report("thm:separable: E min{1, a/Bin(n,pi)} <= min{1, 2a/((n+1)pi)} for a >= 1/2", ok))

ok = True
for _ in range(200):
    Vp, T = 2 * int(rng.integers(1, 5)), int(rng.integers(1, 40))
    d = float(rng.uniform(0.01, min(0.5, 0.5 / np.sqrt(T))))
    s1 = rng.choice([-1, 1], Vp // 2); s2 = s1.copy(); r = int(rng.integers(1, Vp // 2 + 1))
    s2[:r] *= -1
    mk = lambda s: np.ravel(np.column_stack([(1 + s * d), (1 - s * d)])) / Vp
    p1, p2 = mk(s1), mk(s2)
    h2 = 0.5 * np.sum((np.sqrt(p1) - np.sqrt(p2)) ** 2)
    g = 2 / Vp * (1 - np.sqrt(1 - d * d))
    h2T = 1 - (1 - h2) ** T
    kl1 = float(np.sum(p1 * np.log(p1 / p2)))
    ok &= (abs(h2 - r * g) < 1e-12 and h2T >= (1 - np.exp(-1)) * r * T * d * d / Vp - 1e-12
           and (r > 1 or kl1 <= 16 * d * d / (3 * Vp) + 1e-15))
results.append(report("thm:separable: hypercube separation Hel^2 = r g, sequence bound, and token KL <= 16 delta^2/(3V')", ok))


def L_formula(mu, k, n):
    m = np.sort(mu)[::-1]
    cs = np.concatenate([[0.0], np.cumsum(np.sqrt(m))])
    tail = np.concatenate([np.cumsum(m[::-1])[::-1], [0.0]])
    G = k * cs ** 2 / n + tail
    j = int(np.argmin(G))
    return float(G[j]), j


def S_n(mu, D, k, n):
    with np.errstate(divide="ignore"):
        return float(np.sum(mu * np.minimum(1.0, np.where(D > 0, k / (n * np.where(D > 0, D, 1)), np.inf))))


ok = True
for _ in range(8):
    K, k, n = 4, int(rng.integers(1, 4)), int(rng.integers(2, 30))
    mu = rng.dirichlet(np.ones(K) * 0.5)
    best = 1.0
    for S in range(1, 2 ** K):
        idx = [i for i in range(K) if S >> i & 1]

        def f(z):
            D = np.zeros(K); D[idx] = np.abs(z) / np.abs(z).sum()
            return S_n(mu, D, k, n)
        for _ in range(3):
            best = min(best, minimize(f, rng.random(len(idx)) + 0.1, method="Nelder-Mead",
                                      options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000}).fun)
    L, m = L_formula(mu, k, n)
    Dst = np.zeros(K); top = np.argsort(-mu)[:m]; Dst[top] = np.sqrt(mu[top]); Dst /= max(Dst.sum(), 1e-300)
    ok &= abs(best - L) < 1e-6 and (m == 0 or abs(S_n(mu, Dst, k, n) - L) < 1e-12)
results.append(report("thm:separable (ii): inf_D S_n(D) = L_n (direct search), attained by sqrt(mu) on the top m*", ok))

# Upper bound (i): Monte Carlo risk of the product Laplace estimator never exceeds S_n(D)
K, V, T, k = 3000, 3, 6, 2
mu = np.arange(1, K + 1, dtype=float) ** -1.6; mu /= mu.sum()
P = rng.dirichlet(np.ones(V), size=K)
ok = True
for n in [300, 3000]:
    for D in [mu, np.sqrt(mu) / np.sqrt(mu).sum()]:
        risks = []
        for _ in range(20):
            nx = rng.multinomial(n, D)
            cnt = np.array([rng.multinomial(nx[x] * T, P[x]) for x in range(K)])
            ph = (cnt + 1) / (nx[:, None] * T + V)
            h1 = 0.5 * np.sum((np.sqrt(P) - np.sqrt(ph)) ** 2, axis=1)
            risks.append(float(mu @ -np.expm1(T * np.log1p(-h1))))
        ok &= np.mean(risks) <= S_n(mu, D, k, n)
results.append(report("thm:separable (i): Monte Carlo risk of the Laplace estimator <= S_n(D) (D = mu and sqrt(mu))", ok))

ok = True
for _ in range(300):
    Vv, Tt, n, pi = int(rng.integers(2, 6)), int(rng.integers(1, 200)), int(rng.integers(1, 80)), float(rng.uniform(0.001, 1))
    A, kk_ = Tt * log(Vv), Vv - 1
    pmf = np.array([comb(n, j) * pi ** j * (1 - pi) ** (n - j) for j in range(n + 1)])
    lhs = pmf[0] * A + sum(pmf[j] * min(A, kk_ / j) for j in range(1, n + 1))
    ok &= lhs <= min(A, 2 * kk_ / ((n + 1) * pi)) + A * (1 - pi) ** n + 1e-9
results.append(report("rem:sep-general: E min{T log V, k/N} <= min{T log V, 2k/((n+1)pi)} + T log V (1-pi)^n", ok))

# ---------------------------------------------------------------------------------------------
# cor:sep-rates and cor:sep-soft -- rate exponents (Zipf laws on all of N: explicit head + Hurwitz-zeta tail)
# ---------------------------------------------------------------------------------------------
k = 1
Kz = 4_000_000
ns = np.array([1e3, 1e4, 1e5, 1e6])
slope = lambda ys: float(np.polyfit(np.log(ns), np.log(ys), 1)[0])


def zipf_quantities(a, n):
    """S_n(mu), L_n, top-n tail and missing mass for mu_i = i^-a / zeta(a), i = 1, 2, ..."""
    i = np.arange(1, Kz + 1, dtype=float)
    mu = i ** -a / zeta(a)
    tail_after = lambda m: zeta(a, m + 1) / zeta(a)          # sum_{i > m} mu_i, exact
    i0 = int(np.sum(mu >= k / n))                            # mu is decreasing
    S = i0 * k / n + tail_after(i0)
    cs = np.concatenate([[0.0], np.cumsum(np.sqrt(mu))])
    ms = np.unique(np.r_[np.arange(0, 2000), np.geomspace(2000, Kz, 4000).astype(int)])
    L = float(min(k * cs[m] ** 2 / n + tail_after(m) for m in ms))
    top = tail_after(int(n))
    miss = float(np.sum(mu * np.exp(n * np.log1p(-mu)))) + tail_after(Kz)   # (1-mu)^n ~ 1 beyond Kz
    return S, L, top, miss


for a, pred_mu, pred_opt in [(1.5, -1 / 3, -0.5), (3.0, -2 / 3, -1.0)]:
    Q = np.array([zipf_quantities(a, n) for n in ns])
    results.append(report(f"cor:sep-rates (b), a={a}: slopes of S_n(mu) {slope(Q[:, 0]):.3f} (pred {pred_mu:.3f}), "
                          f"L_n {slope(Q[:, 1]):.3f} (pred {pred_opt:.3f})",
                          abs(slope(Q[:, 0]) - pred_mu) < 0.05 and abs(slope(Q[:, 1]) - pred_opt) < 0.08))
    if a == 1.5:
        results.append(report(f"cor:sep-soft: Zipf a=1.5 slopes, top-n tail {slope(Q[:, 2]):.3f} (pred -0.5), "
                              f"missing mass {slope(Q[:, 3]):.3f} (pred -0.333)",
                              abs(slope(Q[:, 2]) + 0.5) < 0.03 and abs(slope(Q[:, 3]) + 1 / 3) < 0.03))
r = 0.7
mu = (1 - r) * r ** np.arange(400)
Sg = np.array([float(np.sum(np.minimum(mu, k / n))) for n in ns]) * ns / np.log(ns)
Lg = np.array([L_formula(mu, k, n)[0] for n in ns]) * ns
results.append(report("cor:sep-rates (c): geometric mu, n S_n(mu)/log n and n L_n stay bounded",
                      Sg.max() / Sg.min() < 2 and Lg.max() / Lg.min() < 1.5))
mu = (1 - 0.5) * 0.5 ** np.arange(200)
nm = np.array([n * float(np.sum(mu * np.exp(n * np.log1p(-mu)))) for n in [1e2, 1e3, 1e4, 1e5]])
results.append(report(f"cor:sep-soft: geometric mu, n * missing mass in [{nm.min():.3f}, {nm.max():.3f}] (order 1/n)",
                      nm.max() / nm.min() < 1.5))



def best_iid_missing(mu, n):
    """min_D sum_x mu_x (1 - D_x)^n: convex; KKT gives D_x = 1 - (lam/(n mu_x))^(1/(n-1)) where positive."""
    def D_of(lam):
        return np.clip(1 - (lam / (n * mu)) ** (1 / (n - 1)), 0, 1)
    lo, hi = 1e-300, n * mu.max()
    for _ in range(400):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if D_of(mid).sum() > 1 else (lo, mid)
    D = D_of(hi); D /= D.sum()
    return float(np.sum(mu * (1 - D) ** n))


mu = 0.5 ** np.arange(1, 400); mu /= mu.sum()
vals = {n: best_iid_missing(mu, n) for n in [40, 160, 640, 2560]}
rat = [-np.log(v) / np.sqrt(n) for n, v in vals.items()]
results.append(report(f"cor:sep-soft: geometric mu, best i.i.d. design -log(risk)/sqrt(n) in [{min(rat):.2f}, {max(rat):.2f}] "
                      f"(e^(-Theta(sqrt n))), top-n tail at n=160: {mu[160:].sum():.1e}",
                      max(rat) / min(rat) < 1.6 and mu[160:].sum() < 1e-40 < vals[160]))

# ---------------------------------------------------------------------------------------------
# thm:local-design -- autoregressive logistic model, V = 2, T = 3, k = 3 shared parameters
# ---------------------------------------------------------------------------------------------
rng = np.random.default_rng(83)   # dedicated seed: the example quoted in the text of thm:local-design
T, kk, nX = 3, 3, 6
A = rng.normal(size=(nX, kk, 3))
scale = np.exp(rng.normal(size=nX) * 0.9)
seqs = list(itertools.product([0, 1], repeat=T))
feat = {(x, y[:t]): scale[x] * (A[x] @ np.array([1.0, (y[t - 1] if t else 0.0), sum(y[:t]) - t / 2]))
        for x in range(nX) for y in seqs for t in range(T)}


def seq_logp(th, x, y):
    return sum(y[t] * (th @ feat[(x, y[:t])]) - np.logaddexp(0, th @ feat[(x, y[:t])]) for t in range(T))


def fisher_chain(th, x):
    I = np.zeros((kk, kk))
    for y in seqs:
        w = np.exp(seq_logp(th, x, y))
        for t in range(T):
            f = feat[(x, y[:t])]; q = sig(th @ f)
            I += w * q * (1 - q) * np.outer(f, f)
    return I


def fisher_score(th, x):
    out = np.zeros((kk, kk))
    for y in seqs:
        s = sum((y[t] - sig(th @ feat[(x, y[:t])])) * feat[(x, y[:t])] for t in range(T))
        out += np.exp(seq_logp(th, x, y)) * np.outer(s, s)
    return out


th0 = np.array([0.6, -0.4, 0.3])
Ix = np.array([fisher_chain(th0, x) for x in range(nX)])
results.append(report("thm:local-design: sequence Fisher information = sum_t E I(s_t) (chain rule)",
                      all(np.allclose(Ix[x], fisher_score(th0, x)) for x in range(nX))))
mu = rng.dirichlet(np.ones(nX) * 0.6)
Imu = np.tensordot(mu, Ix, 1)
Phi = lambda D, I=Ix: float(np.trace(np.linalg.solve(np.tensordot(D, I, 1), np.tensordot(mu, I, 1))))
lev = np.array([np.trace(np.linalg.solve(Imu, Ix[x])) for x in range(nX)])
D = np.ones(nX) / nX
for _ in range(20000):   # multiplicative algorithm for the I-optimal design
    Mi = np.linalg.inv(np.tensordot(D, Ix, 1))
    psi = np.array([np.trace(Mi @ Imu @ Mi @ Ix[x]) for x in range(nX)])
    D = D * np.sqrt(psi); D /= D.sum()
Mi = np.linalg.inv(np.tensordot(D, Ix, 1))
psi = np.array([np.trace(Mi @ Imu @ Mi @ Ix[x]) for x in range(nX)])
Dstar, Phistar = D.copy(), Phi(D)
rand_best = min(Phi(rng.dirichlet(np.ones(nX) * 0.3)) for _ in range(5000))
results += [
    report(f"thm:local-design (a,e): Phi(mu) = k = {Phi(mu):.6f}", abs(Phi(mu) - kk) < 1e-9),
    report(f"thm:local-design (d): equivalence theorem at D* (max psi/Phi = {psi.max()/Phistar:.6f}, = 1 on support)",
           psi.max() <= Phistar * (1 + 1e-6) and np.allclose(psi[Dstar > 1e-4], Phistar, rtol=1e-5)
           and rand_best >= Phistar - 1e-9),
    report(f"thm:local-design (e): k^2/max lev = {kk**2/lev.max():.3f} <= Phi* = {Phistar:.3f} <= "
           f"(E sqrt lev)^2 = {(mu @ np.sqrt(lev))**2:.3f} <= k, attained by D ~ mu sqrt(lev)",
           kk ** 2 / lev.max() <= Phistar + 1e-9 and Phistar <= (mu @ np.sqrt(lev)) ** 2 + 1e-9
           and (mu @ np.sqrt(lev)) ** 2 <= kk + 1e-9
           and Phi(mu * np.sqrt(lev) / (mu @ np.sqrt(lev))) <= (mu @ np.sqrt(lev)) ** 2 + 1e-9),
    report("thm:local-design (e): mu is optimal iff max lev <= k (here max lev > k and Phi* < k)",
           lev.max() > kk and Phistar < kk - 1e-6),
    report(f"thm:local-design: D* drops {int(np.sum(Dstar < 1e-6))} of 6 prompts, incl. the most frequent "
           f"(mu = {mu.max():.3f}, lev = {lev[np.argmax(mu)]:.3f} < k)",
           Dstar[np.argmax(mu)] < 1e-6 and int(np.sum(Dstar < 1e-6)) == 3 and lev[np.argmax(mu)] < kk),
    report("thm:local-design (f): Phi and D* invariant under a common rescaling of the information",
           abs(Phi(Dstar, 7.3 * Ix) - Phistar) < 1e-9 and abs(Phi(mu, 7.3 * Ix) - kk) < 1e-9),
]
# block (separable) case: Phi* = (sum sqrt(mu_x k_x))^2 at D ~ sqrt(mu k)
kx = np.array([1, 2, 3]); mub = np.array([0.7, 0.2, 0.1]); off = np.r_[0, np.cumsum(kx)]
Ib = []
for x in range(3):
    B = np.zeros((kx.sum(), kx.sum())); R = rng.normal(size=(kx[x], kx[x]))
    B[off[x]:off[x + 1], off[x]:off[x + 1]] = R @ R.T + np.eye(kx[x]); Ib.append(B)
Ib = np.array(Ib)
PhiB = lambda D: float(np.trace(np.linalg.solve(np.tensordot(D, Ib, 1), np.tensordot(mub, Ib, 1))))
Dsq = np.sqrt(mub * kx) / np.sqrt(mub * kx).sum()
results.append(report("thm:local-design (e): block case Phi* = (sum_x sqrt(mu_x k_x))^2 at D ~ sqrt(mu k)",
                      abs(PhiB(Dsq) - np.sqrt(mub * kx).sum() ** 2) < 1e-9
                      and min(PhiB(rng.dirichlet(np.ones(3))) for _ in range(3000)) >= PhiB(Dsq) - 1e-9))

# (a) Monte Carlo: n E_mu KL(P || Q_mle) ~ Phi(D)/2 for D = mu and D = D*
P0 = np.array([[np.exp(seq_logp(th0, x, y)) for y in seqs] for x in range(nX)])
Fx = [np.array([[feat[(x, y[:t])] for t in range(T)] for y in seqs]) for x in range(nX)]   # [x][seq, t, k]
Yx = np.array(seqs, dtype=float)


def kl_mu(th):
    return float(sum(mu[x] * np.sum(P0[x] * (np.log(P0[x]) - np.array([seq_logp(th, x, y) for y in seqs])))
                     for x in range(nX)))


def mle_counts(C):
    th = th0.copy()
    for _ in range(40):
        g = np.zeros(kk); H = np.zeros((kk, kk))
        for x in range(nX):
            z = Fx[x] @ th; q = sig(z); w = C[x][:, None]
            g += np.einsum("st,stk->k", w * (Yx - q), Fx[x])
            H += np.einsum("st,stk,stl->kl", w * q * (1 - q), Fx[x], Fx[x])
        step = np.linalg.solve(H, g); th += step
        if np.abs(step).max() < 1e-12:
            break
    return th


rmc = np.random.default_rng(11)
n, R = 2000, 300
for name, Dd in [("D = mu", mu), ("D = D*", Dstar)]:
    vals = np.array([n * kl_mu(mle_counts([rmc.multinomial(c, P0[x]) for x, c in enumerate(rmc.multinomial(n, Dd))]))
                     for _ in range(R)])
    se = vals.std() / np.sqrt(R)
    results.append(report(f"thm:local-design (a): {name}: n E_mu KL = {vals.mean():.3f} +- {se:.3f} vs Phi/2 = {Phi(Dd)/2:.3f}",
                          abs(vals.mean() - Phi(Dd) / 2) < 4 * se))

# ---------------------------------------------------------------------------------------------
# prop:adaptivity-gap -- every fixed design loses a factor close to K at some theta
# ---------------------------------------------------------------------------------------------
u_star = minimize_scalar(lambda u: -(u ** 2) * dsig(u), bounds=(0.1, 10), method="bounded").x
for K, rr in [(3, 30.0), (5, 60.0)]:
    thetas = rr ** np.arange(K); c = u_star / thetas
    info = lambda th: c ** 2 * dsig(th * c)
    worst = lambda z: max(info(th).max() / (np.exp(z) / np.exp(z).sum() @ info(th)) for th in thetas)
    best = minimize(worst, np.zeros(K), method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-12}).fun
    results.append(report(f"prop:adaptivity-gap: K={K}: best fixed design has worst-case Phi/Phi* = {best:.3f} >= 0.99 K",
                          best >= 0.99 * K))

ok = True
for _ in range(100):
    nXr, kr = int(rng.integers(2, 7)), int(rng.integers(1, 4))
    Ir = []
    for _ in range(nXr):
        R_ = rng.normal(size=(kr, int(rng.integers(1, kr + 1)))); Ir.append(R_ @ R_.T * np.exp(rng.normal() * 2))
    Ir = np.array(Ir); mur = rng.dirichlet(np.ones(nXr))
    if np.linalg.eigvalsh(np.tensordot(np.ones(nXr), Ir, 1)).min() < 1e-8:
        continue
    Ph = lambda D: float(np.trace(np.linalg.solve(np.tensordot(D, Ir, 1) + 1e-300 * np.eye(kr), np.tensordot(mur, Ir, 1))))
    best = min(Ph(rng.dirichlet(np.ones(nXr) * 0.5)) for _ in range(2000))
    ok &= Ph(np.ones(nXr) / nXr) <= nXr * best * (1 + 1e-9)
results.append(report("prop:adaptivity-gap: converse, Phi(unif) <= |X| Phi(D) for every D (100 random models)", ok))

# ---------------------------------------------------------------------------------------------
# prop:bias-design -- the factor rho(mu||D) rho(D||mu) is sharp
# ---------------------------------------------------------------------------------------------
ok = True
for _ in range(100):
    mu2, D2 = rng.dirichlet([1, 1]), rng.dirichlet([1, 1])
    if mu2[0] / D2[0] < 1:
        mu2, D2 = mu2[::-1], D2[::-1]
    eta = 1e-3; s = (1 - eta) * D2[1] / D2[0]
    e1, e2 = np.array([0.0, 1.0]), np.array([s, 0.0])
    win_D = e2 if D2 @ e2 < D2 @ e1 else e1
    best_mu = min(mu2 @ e1, mu2 @ e2)
    factor = max(mu2 / D2) * max(D2 / mu2)
    ok &= (mu2 @ win_D) <= factor * best_mu * (1 + 1e-12) and (mu2 @ win_D) >= (1 - eta) * factor * best_mu * (1 - 1e-9)
results.append(report("prop:bias-design: E_mu e(theta*_D) <= rho(mu||D) rho(D||mu) E_mu e(theta*_mu), sharp", ok))

# ---------------------------------------------------------------------------------------------
# prop:iw-design -- misspecified logistic student, importance-weighted MLE, D* ~ mu sqrt(c)
# ---------------------------------------------------------------------------------------------
nX = 8
phi = np.stack([np.ones(nX), np.linspace(-2, 2, nX)], 1)
pt = np.clip(0.5 + 0.45 * np.sin(np.linspace(0, 5, nX)), 0.05, 0.95)   # teacher not logistic in phi
mu = np.array([0.30, 0.02, 0.25, 0.01, 0.02, 0.2, 0.05, 0.15])


def risk(th):
    q = sig(phi @ th)
    return float(mu @ (pt * np.log(pt / q) + (1 - pt) * np.log((1 - pt) / (1 - q))))


ths = minimize(risk, np.zeros(2), method="BFGS", options={"gtol": 1e-12}).x
q = sig(phi @ ths)
H = (phi * (mu * q * (1 - q))[:, None]).T @ phi
cx = np.array([np.trace(np.linalg.solve(H, (pt[x] * (1 - q[x]) ** 2 + (1 - pt[x]) * q[x] ** 2) * np.outer(phi[x], phi[x])))
               for x in range(nX)])
Dst = mu * np.sqrt(cx); Dst /= Dst.sum()
pred = lambda D: 0.5 * float(np.sum(mu ** 2 * cx / D))
grid_best = min(pred(rng.dirichlet(np.ones(nX))) for _ in range(20000))
results.append(report(f"prop:iw-design (b): optimum at mu sqrt(c): {pred(Dst):.4f} = (E sqrt c)^2/2 = "
                      f"{0.5*(mu @ np.sqrt(cx))**2:.4f} <= E c/2 = {pred(mu):.4f}",
                      abs(pred(Dst) - 0.5 * (mu @ np.sqrt(cx)) ** 2) < 1e-12 and grid_best >= pred(Dst) - 1e-12
                      and pred(Dst) <= pred(mu)))
chi2 = float(np.sum(mu ** 2 / Dst) - 1)
results.append(report("prop:iw-design (b): constant c gives c (1 + chi^2(mu||D))",
                      abs(float(np.sum(mu ** 2 * 3.0 / Dst)) - 3.0 * (1 + chi2)) < 1e-9))


def wfit(nx, ones, w):
    th = ths.copy()
    for _ in range(60):
        pr = sig(phi @ th)
        g = phi.T @ (w * (ones - nx * pr)); Hh = (phi * (w * nx * pr * (1 - pr))[:, None]).T @ phi
        st = np.linalg.solve(Hh, g); th += st
        if np.abs(st).max() < 1e-12:
            break
    return th


rmc = np.random.default_rng(12)
n, R = 3000, 1500
for name, Dd in [("D = mu", mu), ("D = mu sqrt(c)", Dst)]:
    vals = []
    for _ in range(R):
        nx = rmc.multinomial(n, Dd); ones = rmc.binomial(nx, pt)
        vals.append(n * (risk(wfit(nx, ones, mu / Dd)) - risk(ths)))
    vals = np.array(vals); se = vals.std() / np.sqrt(R)
    results.append(report(f"prop:iw-design (a): {name}: n excess = {vals.mean():.3f} +- {se:.3f} vs {pred(Dd):.3f}",
                          abs(vals.mean() - pred(Dd)) < 4 * se))

# ---------------------------------------------------------------------------------------------
# prop:path-shift -- state-level shift compounds with depth
# ---------------------------------------------------------------------------------------------
V, T = 3, 6
prefixes = [p for t in range(T) for p in itertools.product(range(V), repeat=t)]
index = {p: i for i, p in enumerate(prefixes)}


def softmax_rows(Z):
    Z = Z - Z.max(axis=1, keepdims=True); E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


Pt = softmax_rows(rng.normal(size=(len(prefixes), V)) * 1.2)
Rt = softmax_rows(np.log(Pt) + rng.normal(size=Pt.shape) * 0.6)
muX, DX = np.array([0.6, 0.4]), np.array([0.3, 0.7])   # two prompts sharing the same tree, for simplicity
chi0 = min(float(np.sum(Pt[i] ** 2 / Rt[i]) - 1) for i in range(len(prefixes)))
gam = min(float((Pt[i] / Rt[i]).max()) for i in range(len(prefixes)))
chiX, rhoX = float(np.sum(muX ** 2 / DX) - 1), float(max(muX / DX))
ok_i, ok_ii = True, True
for t in range(1, T + 1):
    m2, sup = 0.0, 0.0
    for y in itertools.product(range(V), repeat=t - 1):
        pp = np.prod([Pt[index[y[:j]], y[j]] for j in range(t - 1)]); rr = np.prod([Rt[index[y[:j]], y[j]] for j in range(t - 1)])
        m2 += (muX ** 2 / DX).sum() * pp ** 2 / rr; sup = max(sup, rhoX * pp / rr)
    ok_i &= m2 >= (1 + chiX) * (1 + chi0) ** (t - 1) - 1e-12 and sup >= rhoX * gam ** (t - 1) - 1e-12
results.append(report("prop:path-shift (i): E[w_t^2] >= (1+chi^2(mu||D))(1+chi_0)^(t-1) and ||w_t|| >= rho gamma^(t-1)", ok_i))
for t in range(1, T + 1):   # (ii): teacher-generated prefixes keep the prompt-level constants
    m2 = sum((muX ** 2 / DX).sum() * np.prod([Pt[index[y[:j]], y[j]] for j in range(t - 1)])
             for y in itertools.product(range(V), repeat=t - 1))
    ok_ii &= abs(m2 - (1 + chiX)) < 1e-12
results.append(report("prop:path-shift (ii): with r = p, E[w_t^2] = 1 + chi^2(mu||D) at every depth", ok_ii))
# (iii): deterministic teacher prefix a^(t-1); proposal r0 hits it w.p. r0(a)^(t-1) = e^{-(t-1)K0}
ok = True
for _ in range(200):
    r0a, t, N = float(rng.uniform(0.05, 0.95)), int(rng.integers(2, 40)), int(rng.integers(1, 10 ** 6))
    K0 = -log(r0a)
    miss = float(np.exp(N * np.log1p(-r0a ** (t - 1))))                 # P(s* never drawn in N proposal states)
    ok &= abs(r0a ** (t - 1) - np.exp(-(t - 1) * K0)) < 1e-12 and miss >= (1 - N * np.exp(-(t - 1) * K0)) - 1e-12
results.append(report("prop:path-shift (iii): P(miss s*) = (1 - e^{-(t-1)K0})^N >= 1 - N e^{-(t-1)K0}; teacher prefixes hit s* surely", ok))

print("\nALL PASS" if all(results) else "\nSOME CHECKS FAILED")
