"""Checks for Sections 5.5-5.6 (representation matching beyond the linear-Gaussian model).

Run:  python3 verification/check_repr_nonlinear.py
Gaussian expectations use tensor Gauss-Hermite quadrature, so most checks are exact up to quadrature error.
"""
import itertools

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import minimize

rng = np.random.default_rng(7)
results = []


def report(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    results.append(bool(ok))
    return ok


def gh(m):
    """Nodes and weights for E over N(0,1)."""
    x, w = hermegauss(m)
    return x, w / w.sum()


def grid(x, w, d):
    """Tensor quadrature over N(0, I_d): nodes (d x M) and weights (M,)."""
    idx = np.array(list(itertools.product(range(len(x)), repeat=d)))
    return x[idx].T, np.prod(w[idx], axis=1)


def softmax(Z, axis=-1):
    Z = Z - Z.max(axis=axis, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=axis, keepdims=True)


def kl_rows(P, Q):
    return np.sum(P * (np.log(P) - np.log(Q)), axis=-1)


# ---------------------------------------------------------------------------
# lem:penalty on random finite parameter sets
# ---------------------------------------------------------------------------
ok_mono = ok_bounds = ok_limit = ok_compat = True
for _ in range(200):
    m = 60
    L, R = rng.exponential(size=m), rng.exponential(size=m)
    if rng.random() < 0.3:                      # make argmin L and argmin R intersect
        j = rng.integers(m); L[j], R[j] = 0.0, 0.0
    Ls, R0 = L.min(), R.min()
    feas = np.isclose(R, R0)
    Dinf = L[feas].min() - Ls
    gammas = np.logspace(-3, 4, 60)
    sel = [int(np.argmin(L + g * R)) for g in gammas]
    Lg, Rg = L[sel], R[sel]
    ok_mono &= np.all(np.diff(Lg) >= -1e-12) and np.all(np.diff(Rg) <= 1e-12)
    ok_bounds &= np.all(Lg - Ls <= Dinf + 1e-12) and np.all(Rg - R0 <= Dinf / gammas + 1e-12)
    ok_limit &= np.isclose(Lg[-1] - Ls, Dinf) if Dinf < 1e3 else True
    if np.any(np.isclose(L, Ls) & feas):
        ok_compat &= all(np.isclose(L[s], Ls) and np.isclose(R[s], R0) for s in sel)
report("lem:penalty (ii) monotone path, 200 random instances", ok_mono)
report("lem:penalty (iii) Delta(gamma) <= Delta_inf and R - R0 <= Delta_inf/gamma", ok_bounds)
report("lem:penalty (iv) Delta(gamma) -> Delta_inf", ok_limit)
report("lem:penalty (i) compatible constraint: penalised minimisers are output-optimal", ok_compat)

# ---------------------------------------------------------------------------
# thm:nl-overconstraint
# ---------------------------------------------------------------------------
V = 5
# (a) min_kappa E KL(pi || kappa(Bh)) = I(Y; h | Bh) on a discrete grid; B = first coordinate
vals = np.arange(4)
H = np.array(list(itertools.product(vals, vals)), dtype=float)          # 16 points
ph = rng.dirichlet(np.ones(len(H)))
Pi = softmax(rng.normal(size=(len(H), V)) * 2)                           # nonlinear head, arbitrary
groups = H[:, 0]
kappa = np.zeros_like(Pi)
for gv in vals:
    msk = groups == gv
    kappa[msk] = (ph[msk] @ Pi[msk]) / ph[msk].sum()
val_opt = ph @ kl_rows(Pi, kappa)
pY_given_u = {gv: kappa[groups == gv][0] for gv in vals}
cmi = sum(ph[i] * np.sum(Pi[i] * np.log(Pi[i] / pY_given_u[groups[i]])) for i in range(len(H)))
report(f"thm:nl-overconstraint (a) min_kappa loss {val_opt:.6f} = I(Y;h|Bh) {cmi:.6f}", abs(val_opt - cmi) < 1e-12)
def perturbed_head():
    # a perturbation that depends on h only through Bh = h[0], so kappa stays Bh-measurable
    noise = {gv: 0.3 * rng.normal(size=V) for gv in vals}
    return softmax(np.log(kappa) + np.array([noise[gv] for gv in groups]))


worse = all(ph @ kl_rows(Pi, perturbed_head()) >= val_opt - 1e-12 for _ in range(50))
report("thm:nl-overconstraint (a) conditional-mean head beats perturbed heads", worse)

# (b) R_LS minimisers are PCA for a non-Gaussian law (only second moments enter)
dT, dS, n = 5, 2, 20000
Z = rng.standard_t(df=5, size=(n, dT)) @ rng.normal(size=(dT, dT))
Z -= Z.mean(0)
S = Z.T @ Z / n
lam, U = np.linalg.eigh(S); lam, U = lam[::-1], U[:, ::-1]


def R_LS(B):
    Hs = Z @ B.T
    A = np.linalg.lstsq(Hs, Z, rcond=None)[0]
    return np.mean(np.sum((Hs @ A - Z) ** 2, 1))


pca = R_LS(U[:, :dS].T)
rand_min = min(R_LS(rng.normal(size=(dS, dT))) for _ in range(300))
report(f"thm:nl-overconstraint (b) R_LS(PCA) {pca:.4f} = tail {lam[dS:].sum():.4f} < best random {rand_min:.4f}",
       abs(pca - lam[dS:].sum()) < 1e-8 and pca < rand_min)

# (c) the example: h1 continuous (high variance), h2 uniform on V values, teacher reads only h2
eta = 1e-3
Pi2 = (1 - eta) * np.eye(V) + eta / V                                    # pi(h2 = j) = row j
pY = Pi2.mean(0)
I_h2 = np.mean(kl_rows(Pi2, pY[None, :]))                                 # = I(Y; h2) = I(Y; h_T)
var_h2 = np.var(np.arange(V) - (V - 1) / 2)
report(f"thm:nl-overconstraint (c) PCA keeps h1 (Var 10 > {var_h2}); cost = I(Y;h_T) = {I_h2:.4f} "
       f"-> log V = {np.log(V):.4f}", 10 > var_h2 and np.log(V) - I_h2 < 0.02)

# ---------------------------------------------------------------------------
# thm:fisher-central: a multi-index teacher whose index space is orthogonal to the principal subspace
# ---------------------------------------------------------------------------
dT, V, Cc = 4, 5, np.eye(5) - 1 / 5
lam_c = np.array([9.0, 4.0, 1.0, 0.5])
Rot = np.linalg.qr(rng.normal(size=(dT, dT)))[0]
Sig = Rot @ np.diag(lam_c) @ Rot.T
Sh = Rot @ np.diag(np.sqrt(lam_c)) @ Rot.T
P = rng.normal(size=(2, 2)) @ Rot[:, 2:].T                    # index space = span of the two smallest-variance directions
W1, b1, W2 = rng.normal(size=(6, 2)), rng.normal(size=(6, 1)), 2 * rng.normal(size=(V, 6))
g = lambda Hm: (Cc @ W2 @ np.tanh(W1 @ (P @ Hm) + b1)).T      # centred logits, (M, V)


def jac(Hm):                                                   # (M, V, dT)
    A = 1 - np.tanh(W1 @ (P @ Hm) + b1) ** 2
    return np.einsum('vk,km,kd->mvd', Cc @ W2, A, W1 @ P)


xq, wq = gh(14)
XI, WT = grid(xq, wq, dT)
Hq = Sh @ XI
Zq, Jq = g(Hq), jac(Hq)
Pq = softmax(Zq)
Fq = np.einsum('mv,vu->mvu', Pq, np.eye(V)) - np.einsum('mv,mu->mvu', Pq, Pq)
K = np.einsum('m,mvd,mvu,mue->de', WT, Jq, Fq, Jq)
mu, UK = np.linalg.eigh(K); mu, UK = mu[::-1], UK[:, ::-1]
UF = UK[:, :2]
same_space = np.linalg.norm(UF @ UF.T - Rot[:, 2:] @ Rot[:, 2:].T) < 1e-8
report(f"thm:fisher-central (i) rank K = 2 (eigenvalues {np.round(mu, 10)}), range K = index space",
       mu[2] < 1e-10 * mu[0] and same_space)
phi = lambda Um: g(UF @ Um)                                     # reduced head
kl_student = WT @ kl_rows(Pq, softmax(phi(UF.T @ Hq)))
report(f"thm:fisher-central (ii) student h_S = U_F^T h_T with head phi is lossless (KL {kl_student:.1e})",
       kl_student < 1e-12)
Fh = np.array([np.real(np.linalg.cholesky(Fm + 1e-14 * np.eye(V))).T for Fm in Fq])   # F = Fh^T Fh
resid = -(np.eye(dT) - UF @ UF.T) @ Hq                          # A h_S - h_T with A = U_F
RJ = WT @ np.sum(np.einsum('mvu,mud,dm->mv', Fh, Jq, resid) ** 2, axis=1)
report(f"thm:fisher-central (iii) Jacobian constraint R_J = {RJ:.1e} at that student", RJ < 1e-20)
Uw = np.linalg.eigh(Sh @ K @ Sh)[1][:, ::-1][:, :2]
N = np.linalg.lstsq(UF, np.linalg.solve(Sh, Uw), rcond=None)[0]
report("thm:fisher-central (ii) whitened target of def:fisher-rep is an invertible image of t_F",
       np.allclose(UF @ N, np.linalg.solve(Sh, Uw), atol=1e-8) and abs(np.linalg.det(N)) > 1e-6)
pbar = WT @ Pq                                                  # PCA keeps the independent high-variance directions
I_Yh = WT @ kl_rows(Pq, pbar[None, :])
report(f"thm:fisher-central vs PCA matching: PCA student loses I(Y;h_T) = {I_Yh:.4f} > 0", I_Yh > 1e-2)

# ---------------------------------------------------------------------------
# thm:stein: nonlinear teacher, linear read-out, constant F
# ---------------------------------------------------------------------------
dT, dS, V = 3, 1, 4
Cc = np.eye(V) - 1 / V
p0 = rng.dirichlet(np.ones(V)); F = np.diag(p0) - np.outer(p0, p0)
Fh = np.real(np.linalg.cholesky(F + 1e-15 * np.eye(V))).T
X = rng.normal(size=(dT, dT)); Sig = X @ X.T / dT + 0.3 * np.eye(dT)
ws, Us = np.linalg.eigh(Sig); Sh = Us @ np.diag(np.sqrt(ws)) @ Us.T
lam, Ul = ws[::-1], Us[:, ::-1]
A1, a1, A2 = rng.normal(size=(5, dT)), rng.normal(size=(5, 1)), rng.normal(size=(V, 5))
# an entire nonlinearity, so that Gauss-Hermite quadrature is accurate to machine precision
g = lambda Hm: (Cc @ A2 @ np.sin(A1 @ Hm + a1)).T
jac = lambda Hm: np.einsum('vk,km,kd->mvd', Cc @ A2, np.cos(A1 @ Hm + a1), A1)
xq, wq = gh(40)
XI, WT = grid(xq, wq, dT)
Hq = Sh @ XI; Zq = g(Hq); Jq = jac(Hq)
Ez = WT @ Zq; Jbar = np.einsum('m,mvd->vd', WT, Jq)
Cm = (Zq - Ez).T @ (WT[:, None] * Hq.T)
report(f"thm:stein (c) Stein: |Cov(g,h) - Jbar Sigma| = {np.abs(Cm - Jbar @ Sig).max():.1e}",
       np.abs(Cm - Jbar @ Sig).max() < 1e-8)
Gs = Fh @ Cm @ np.linalg.inv(Sh)
sig = np.linalg.svd(Gs, compute_uv=False); Vt = np.linalg.svd(Gs)[2]
lin = Cm @ np.linalg.inv(Sig)
Nres = 0.5 * WT @ np.sum(((Zq - Ez) - (lin @ Hq).T) @ Fh.T * (((Zq - Ez) - (lin @ Hq).T) @ Fh.T), 1)


def loss_lin(L):                                                  # optimal bias included
    Rm = (Zq - Ez) - (L @ Hq).T
    return 0.5 * WT @ np.sum((Rm @ Fh.T) ** 2, 1)


def best_given_B(B):
    L = Cm @ B.T @ np.linalg.inv(B @ Sig @ B.T) @ B              # weighted LS read-out for fixed B
    return loss_lin(L)


opt = min(minimize(lambda b: best_given_B(b.reshape(dS, dT)), rng.normal(size=dS * dT),
                   method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 20000}).fun
          for _ in range(4))
ey = Nres + 0.5 * np.sum(sig[dS:] ** 2)
report(f"thm:stein (a) output optimum {opt:.8f} = N + EY tail {ey:.8f}", abs(opt - ey) < 1e-7)
t_sharp_B = Vt[:dS] @ np.linalg.inv(Sh)
report("thm:stein (b) target t_sharp attains the optimum", abs(best_given_B(t_sharp_B) - ey) < 1e-10)
pca_val = best_given_B(Ul[:, :dS].T)
pca_formula = Nres + 0.5 * sum(np.linalg.norm(Fh @ Cm @ Ul[:, i]) ** 2 / lam[i] for i in range(dS, dT))
report(f"thm:stein (a) PCA value {pca_val:.6f} = formula {pca_formula:.6f}", abs(pca_val - pca_formula) < 1e-10)
Kbar = Jbar.T @ F @ Jbar
Ub = np.linalg.eigh(Sh @ Kbar @ Sh)[1][:, ::-1][:, :dS]
report("thm:stein (c) the Kbar-aligned target equals t_sharp (same subspace)",
       abs(best_given_B(Ub.T @ np.linalg.inv(Sh)) - ey) < 1e-10)
# (d) the counterexample, by quadrature in d = 2
xq2, wq2 = gh(40)
XI2, WT2 = grid(xq2, wq2, 2)
w1, w2 = Cc @ rng.normal(size=V), Cc @ rng.normal(size=V)
w1 *= np.sqrt(0.3 * (w2 @ F @ w2) / (w1 @ F @ w1))              # 4a > b with a/b = 0.3
a, b = w1 @ F @ w1, w2 @ F @ w2
Zc = np.outer(XI2[0] ** 2 - 1, w1) + np.outer(XI2[1], w2)
Jc = np.stack([np.outer(2 * XI2[0], w1), np.tile(w2, (XI2.shape[1], 1))], axis=2)
Kc = np.einsum('m,mvd,vu,mue->de', WT2, Jc, F, Jc)
Jbc = np.einsum('m,mvd->vd', WT2, Jc)


def keep(i):                                                      # best linear student keeping coordinate i
    c = Zc.T @ (WT2 * XI2[i])
    Rm = Zc - np.outer(XI2[i], c)
    return 0.5 * WT2 @ np.sum((Rm @ Fh.T) ** 2, 1)


okd = (np.allclose(Kc, np.diag([4 * a, b]), atol=1e-10) and np.argmax(np.diag(Kc)) == 0
       and np.allclose(Jbc[:, 0], 0, atol=1e-12) and abs(keep(0) - (a + b / 2)) < 1e-10 and abs(keep(1) - a) < 1e-10)
report(f"thm:stein (d) K keeps h1: loss {keep(0):.5f} = a+b/2; Kbar keeps h2: loss {keep(1):.5f} = a; "
       f"cost b/2 = {b / 2:.5f}", okd)

# ---------------------------------------------------------------------------
# prop:no-moment-rule
# ---------------------------------------------------------------------------
x1, w1q = gh(120)
E1 = lambda f: np.sum(w1q * f(x1))
c = np.sqrt(4 / (9 * (1 - np.exp(-18))))
psis = [(lambda t: (t ** 2 - 1) / np.sqrt(2), lambda t: np.sqrt(2) * t),
        (lambda t: c * (np.cos(3 * t) - np.exp(-4.5)), lambda t: -3 * c * np.sin(3 * t))]
mom_ok = all(abs(E1(p)) < 1e-12 and abs(E1(dp)) < 1e-12 and abs(E1(lambda t: dp(t) ** 2) - 2) < 1e-12
             for p, dp in psis)
report("prop:no-moment-rule: both teachers have E psi = E psi' = 0 and E psi'^2 = 2 (same Sigma, Jbar, K)", mom_ok)


def loss_dir(p, rho2, a=1.0, b=0.5):
    rho, s = np.sqrt(rho2), np.sqrt(1 - rho2)
    cond = np.array([np.sum(w1q * p(rho * t - s * x1)) for t in x1])
    return 0.5 * (b + a * E1(lambda t: p(t) ** 2) - b * (1 - rho2) - a * np.sum(w1q * cond ** 2))


rg = np.linspace(0, 1, 201)
L1 = np.array([loss_dir(psis[0][0], r) for r in rg]); L2 = np.array([loss_dir(psis[1][0], r) for r in rg])
f2 = (1 / 9) * (1 - np.exp(-9)) / (1 + np.exp(-9))
ok1 = np.argmin(L1) == len(rg) - 1 and abs(L1[-1] - 0.25) < 1e-12 and abs(L1[0] - 0.5) < 1e-12 and np.all(L1[:-1] > 0.25)
ok2 = np.argmin(L2) == 0 and abs(L2[0] - f2) < 1e-12 and abs(L2[-1] - 0.25) < 1e-10 and np.all(L2[1:] > f2)
report(f"prop:no-moment-rule teacher 1: unique optimum e1 (0.25 vs 0.5)", ok1)
report(f"prop:no-moment-rule teacher 2: unique optimum e2 ({L2[0]:.6f} = formula {f2:.6f} vs 0.25); "
       f"K-rule loses {0.25 - f2:.4f}", ok2)

# ---------------------------------------------------------------------------
# thm:poincare: the sandwich for universal heads, by nested quadrature
# ---------------------------------------------------------------------------
dT, V = 3, 4
Cc = np.eye(V) - 1 / V
p0 = rng.dirichlet(np.ones(V)); F = np.diag(p0) - np.outer(p0, p0)
Fh = np.real(np.linalg.cholesky(F + 1e-15 * np.eye(V))).T
X = rng.normal(size=(dT, dT)); Sig = X @ X.T / dT + 0.3 * np.eye(dT)
ws, Us = np.linalg.eigh(Sig); Sh = Us @ np.diag(np.sqrt(ws)) @ Us.T
A1, a1, A2 = 0.8 * rng.normal(size=(5, dT)), rng.normal(size=(5, 1)), rng.normal(size=(V, 5))
fz = lambda XIm: (Cc @ A2 @ np.sin(A1 @ (Sh @ XIm) + a1)).T @ Fh.T       # f(xi) = F^1/2 g(Sigma^1/2 xi)
Df = lambda XIm: np.einsum('uv,vk,km,kd->mud', Fh, Cc @ A2, np.cos(A1 @ (Sh @ XIm) + a1), A1 @ Sh)
xq, wq = gh(24)
XI, WT = grid(xq, wq, dT)
Dq = Df(XI)
Kw = np.einsum('m,mud,mue->de', WT, Dq, Dq)
Dbar = np.einsum('m,mud->ud', WT, Dq)
Kbw = Dbar.T @ Dbar


def Lstar(Upar):
    k = Upar.shape[1]
    Q = np.linalg.qr(np.hstack([Upar, rng.normal(size=(dT, dT - k))]))[0]
    Uperp = Q[:, k:]
    Xp, Wp = grid(xq, wq, k)
    Xc, Wc = grid(xq, wq, dT - k)
    tot = 0.0
    for j in range(Xp.shape[1]):
        vals_ = fz(Upar @ Xp[:, [j]] + Uperp @ Xc)
        m = Wc @ vals_
        tot += Wp[j] * (Wc @ np.sum((vals_ - m) ** 2, 1))
    return 0.5 * tot


ok_sw, worst = True, 0.0
for k in (1, 2):
    for _ in range(4):
        Up = np.linalg.qr(rng.normal(size=(dT, k)))[0]
        Pp = np.eye(dT) - Up @ Up.T
        lo, Ls_ = 0.5 * np.trace(Pp @ Kbw @ Pp), Lstar(Up)
        hi = lo + 0.5 * np.trace(Pp @ (Kw - Kbw) @ Pp)
        ok_sw &= lo - 1e-10 <= Ls_ <= hi + 1e-10
        worst = max(worst, (Ls_ - lo) / max(hi - lo, 1e-15))
report(f"thm:poincare sandwich on 8 random subspaces (nonlinear part uses at most {worst:.2f} of the Poincare room)",
       ok_sw)
muK, UKw = np.linalg.eigh(Kw); muK, UKw = muK[::-1], UKw[:, ::-1]
muB = np.sort(np.linalg.eigvalsh(Kbw))[::-1]
ok_fa = True
for dS in (1, 2):
    upper_at_SK = Lstar(UKw[:, :dS])
    lower_all = 0.5 * muB[dS:].sum()
    scan = min(Lstar(np.linalg.qr(rng.normal(size=(dT, dS)))[0]) for _ in range(12))
    ok_fa &= upper_at_SK <= 0.5 * muK[dS:].sum() + 1e-10 and scan >= lower_all - 1e-10
report("thm:poincare: L*(S_K) <= tail of K_w and every L*(S) >= tail of Kbar_w (so cost <= the gap)", ok_fa)

# ---------------------------------------------------------------------------
# ex:retrieval: exact Bayes failure probability of end-to-end learning
# ---------------------------------------------------------------------------
ok_ret = True
for N in (8, 16, 32, 64, 128):
    M = N - 1
    for n in range(1, N // 2 + 1):
        dist = np.zeros(n + 1); dist[0] = 1.0                  # law of #distinct values among draws from M values
        for _ in range(n):
            new = np.zeros(n + 1)
            new[:] += dist * np.arange(n + 1) / M
            new[1:] += dist[:-1] * (M - np.arange(n)) / M
            dist = new
        succ_given_miss = np.sum(dist / (N - np.arange(n + 1)))
        fail = (1 - 1 / N) ** n * (1 - succ_given_miss)
        bound = (1 - 1 / N) ** n * (1 - 1 / (N - n))
        ok_ret &= fail >= bound - 1e-15 and bound >= (1 - 1 / N) ** (N / 2) * (1 - 2 / N) - 1e-15
    ok_ret &= (1 - 1 / N) ** (N / 2) * (1 - 2 / N) >= 0.25
report("ex:retrieval: E2E failure >= (1-1/N)^n (1-1/(N-n)) >= 1/4 for n <= N/2, N in {8,...,128}", ok_ret)

# ---------------------------------------------------------------------------
# prop:attention-convex
# ---------------------------------------------------------------------------
d, Ntok, dp = 4, 3, 2
Q0, V0 = rng.normal(size=(d, d)), rng.normal(size=(dp, d))
n_needed = d * int(np.ceil(d / min(Ntok - 1, d)))


def contexts(n):
    return rng.normal(size=(n, d, Ntok)), rng.normal(size=(n, d))


def attn(Q, Xs, xs):
    return softmax(np.einsum('cd,de,cej->cj', xs, Q, Xs))


def att_obj(qflat, Xs, xs, A0):
    return np.sum(kl_rows(A0, attn(qflat.reshape(d, d), Xs, xs)))


def feat_rank(Xs, xs):
    rows = [np.outer(xs[c], Xs[c][:, j] - Xs[c][:, 0]).ravel() for c in range(len(xs)) for j in range(1, Ntok)]
    return np.linalg.matrix_rank(np.array(rows))


Xs, xs = contexts(n_needed); A0 = attn(Q0, Xs, xs)
Phi = np.einsum('cd,cej->cjde', xs, Xs).reshape(len(xs), Ntok, d * d)   # attention logits = Phi @ vec(Q)


def att_grad(qflat, *_):
    return np.einsum('cj,cjk->k', softmax(Phi @ qflat) - A0, Phi)


def att_hess(qflat, *_):
    A = softmax(Phi @ qflat)
    Fm = np.einsum('cj,jk->cjk', A, np.eye(Ntok)) - np.einsum('cj,ck->cjk', A, A)
    return np.einsum('cjk,cjm,ckn->mn', Fm, Phi, Phi)


# the objective is convex but flat where attention saturates, so use Newton steps with the exact Hessian
res = minimize(att_obj, rng.normal(size=d * d), args=(Xs, xs, A0), jac=att_grad, hess=att_hess,
               method="trust-exact", options={"gtol": 1e-14})
Qh = res.x.reshape(d, d)
Xt, xt = contexts(200)
report(f"prop:attention-convex (ii) n = {n_needed} contexts: feature rank {feat_rank(Xs, xs)} = d^2, Q recovered "
       f"(err {np.abs(Qh - Q0).max():.1e}), attention exact on fresh contexts",
       feat_rank(Xs, xs) == d * d and np.abs(Qh - Q0).max() < 1e-5
       and np.abs(attn(Qh, Xt, xt) - attn(Q0, Xt, xt)).max() < 1e-5)
Xf, xf = contexts(n_needed - 1)
report(f"prop:attention-convex (ii) n = {n_needed - 1} contexts: feature rank {feat_rank(Xf, xf)} < d^2 (not identified)",
       feat_rank(Xf, xf) < d * d)
mid_ok = True
for _ in range(200):
    q1, q2 = 3 * rng.normal(size=d * d), 3 * rng.normal(size=d * d)
    mid_ok &= att_obj((q1 + q2) / 2, Xs, xs, A0) <= (att_obj(q1, Xs, xs, A0) + att_obj(q2, Xs, xs, A0)) / 2 + 1e-12
report("prop:attention-convex (i) midpoint convexity of the attention-map objective (200 pairs)", mid_ok)
Xa, xa = contexts(d); Aa = attn(Q0, Xa, xa)
Mv = np.einsum('cdj,cj->cd', Xa, Aa)                           # rows X_c alpha_c
O = Mv @ V0.T
Vh = np.linalg.lstsq(Mv, O, rcond=None)[0].T
report("prop:attention-convex (ii) V recovered by least squares from d contexts", np.abs(Vh - V0).max() < 1e-10)
mfun = lambda Q: 2 - 1 / (1 + np.exp(Q))                      # m(Q) = 2 - sigma(-Q)
o_ = 1.5
e2e = lambda Q, Vv: (Vv * mfun(Q) - o_) ** 2
qa, qb = -2.0, 3.0
za, zb = (qa, o_ / mfun(qa)), (qb, o_ / mfun(qb))
mid = e2e((qa + qb) / 2, (za[1] + zb[1]) / 2)
report(f"prop:attention-convex (iii) E2E: two zeros, midpoint loss {mid:.2e} > 0 (not convex)",
       e2e(*za) < 1e-20 and e2e(*zb) < 1e-20 and mid > 1e-6)

print("\nALL PASS" if all(results) else "\nSOME CHECKS FAILED")
