"""Exact checks of the recoverability results (Section 2.6).

Tiny autoregressive teachers and students are enumerated exactly, so every
expectation below is computed without Monte Carlo. Latent-chain results are
checked by exact dynamic programming on the latent state.
Run:  python3 verification/check_recoverability.py
"""
import itertools
import numpy as np

rng = np.random.default_rng(3)
TOL = 1e-9
results = []


def report(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    results.append(bool(ok))
    return ok


# ---------------------------------------------------------------------------
# Prefix-tree models: states s = (x, y_1, ..., y_{t-1}); tables indexed by prefix
# ---------------------------------------------------------------------------
class Tree:
    def __init__(self, X, V, T):
        self.X, self.V, self.T = X, V, T
        self.prefixes = [(x,) + y for x in X for t in range(T) for y in itertools.product(range(V), repeat=t)]
        self.index = {p: i for i, p in enumerate(self.prefixes)}
        self.by_time = [[p for p in self.prefixes if len(p) == t + 1] for t in range(T)]

    def laws(self, P, Dx):
        """d_t for t = 0..T-1 (0-based): dict prefix -> probability."""
        out = [{(x,): Dx[i] for i, x in enumerate(self.X)}]
        for t in range(1, self.T):
            d = {}
            for s, w in out[-1].items():
                for a in range(self.V):
                    d[s + (a,)] = w * P[self.index[s], a]
            out.append(d)
        return out

    def Qfun(self, P, C):
        """Teacher cost-to-go Q_t(s,a) = c_t(s,a) + E_P[sum_{k>t} c_k | s, a]."""
        Q = np.zeros((len(self.prefixes), self.V))
        for s in sorted(self.prefixes, key=len, reverse=True):
            i = self.index[s]
            for a in range(self.V):
                nxt = s + (a,)
                Q[i, a] = C[i, a] + (P[self.index[nxt]] @ Q[self.index[nxt]] if len(nxt) <= self.T else 0.0)
        return Q

    def J(self, P, C, Dx):
        return sum(w * P[self.index[s]] @ C[self.index[s]] for d in self.laws(P, Dx) for s, w in d.items())

    def seq_table(self, P, C, x):
        """(prob, total cost) of every full sequence given x."""
        rows = []
        for y in itertools.product(range(self.V), repeat=self.T):
            pr, c = 1.0, 0.0
            for t in range(self.T):
                i = self.index[(x,) + y[:t]]
                pr *= P[i, y[t]]
                c += C[i, y[t]]
            rows.append((pr, c))
        return np.array(rows)


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def chi2(q, p):
    return float(np.sum(q ** 2 / p) - 1.0)


def dirichlet_rows(n, V, conc):
    return rng.dirichlet(conc * np.ones(V), size=n)


def perturb(P, scale):
    Z = np.log(P) + rng.normal(scale=scale, size=P.shape)
    Z -= Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def local_quantities(tree, Pt, Ps, C, Dx):
    """Per-(t,s) student-prefix weights, TV_t, u_t and the PDL summand Delta_t."""
    Q = tree.Qfun(Pt, C)
    rows = []  # (t, s, weight under d^theta_t, weight under d^p_t, TV, u, Delta)
    dS, dP = tree.laws(Ps, Dx), tree.laws(Pt, Dx)
    for t in range(tree.T):
        for s in tree.by_time[t]:
            i = tree.index[s]
            rows.append((t, s, dS[t].get(s, 0.0), dP[t].get(s, 0.0), tv(Pt[i], Ps[i]),
                         Q[i].max() - Q[i].min(), float((Ps[i] - Pt[i]) @ Q[i])))
    return Q, rows


# ---------------------------------------------------------------------------
# thm:local-u  (i) localisation, (ii) moments, (iii) commitment split, (iv) sharpness
# ---------------------------------------------------------------------------
V, T, X, Dx = 3, 4, (0, 1), np.array([0.35, 0.65])
tree = Tree(X, V, T)
n = len(tree.prefixes)
ok_i = ok_id = ok_ii = ok_iii = True
for trial in range(60):
    conc = [0.3, 1.0, 5.0][trial % 3]
    Pt = dirichlet_rows(n, V, conc)
    Ps = perturb(Pt, [0.05, 0.5, 2.0][(trial // 3) % 3])
    C = rng.random((n, V)) if trial % 2 else (rng.random((n, V)) < 0.3).astype(float)
    Q, rows = local_quantities(tree, Pt, Ps, C, Dx)
    dJ = tree.J(Ps, C, Dx) - tree.J(Pt, C, Dx)
    w = np.array([r[2] for r in rows]) / T                     # nu^theta
    TVv, uv, gv = (np.array([r[k] for r in rows]) for k in (4, 5, 6))
    tvec = np.array([r[0] for r in rows])
    Teon = T * float(w @ TVv)
    u = uv.max()
    ok_id &= abs(dJ - T * float(w @ gv)) < TOL                 # Delta J = a_on T eps_on (PDL)
    loc = T * float(w @ (uv * TVv))
    ok_i &= abs(dJ) <= loc + TOL and loc <= u * Teon + TOL
    for r in [1.0, 1.5, 2.0, 3.0, np.inf]:
        if r == np.inf:
            un, tvn = float(uv[w > 0].max()), float(w @ TVv)
            b2 = T * un * float(w @ TVv)
            b1 = b2
        else:
            rp = np.inf if r == 1.0 else r / (r - 1.0)
            un = float(w @ uv ** r) ** (1 / r)
            tvn = float(TVv[w > 0].max()) if rp == np.inf else float(w @ TVv ** rp) ** (1 / rp)
            b1 = T * un * tvn
            b2 = T * un * float(w @ TVv) ** (1 - 1 / r)
        ok_ii &= abs(dJ) <= b1 + TOL and b1 <= b2 + TOL
    for U in np.quantile(uv, [0.0, 0.3, 0.6, 0.9, 1.0]):
        B = uv > U
        bound = U * Teon + T * float(w @ ((T - tvec) * TVv * B))
        zeta = float(w @ (TVv * B)) / float(w @ TVv)
        ok_iii &= abs(dJ) <= bound + TOL and bound <= (U + zeta * T) * Teon + TOL
report("thm:local-u(i): Delta J = T E_nu[<q-p,Q>] (PDL identity), random instances", ok_id)
report("thm:local-u(i): |dJ| <= sum_t E_{d^theta}[u_t TV_t] <= u T eps_on", ok_i)
report("thm:local-u(ii): Hoelder bounds for r in {1, 1.5, 2, 3, inf}", ok_ii)
report("thm:local-u(iii): |dJ| <= U T eps_on + sum_t (T-t+1) E[TV_t 1{u_t>U}] <= (U + zeta T) T eps_on", ok_iii)

# (iv) sharpness: a single-state deviation attains Delta J = u_t(s) T eps_on exactly
Pt = dirichlet_rows(n, V, 1.0)
C = rng.random((n, V))
Q = tree.Qfun(Pt, C)
J0 = tree.J(Pt, C, Dx)
dP = tree.laws(Pt, Dx)
ratios, ok_iv = [], True
for s in tree.prefixes:
    i, t = tree.index[s], len(s) - 1
    ap, am = int(Q[i].argmax()), int(Q[i].argmin())
    if ap == am:
        continue
    for frac in [1.0, 1e-3]:
        eta = frac * Pt[i, am]
        Ps = Pt.copy()
        Ps[i, ap] += eta
        Ps[i, am] -= eta
        dJ = tree.J(Ps, C, Dx) - J0
        dS = tree.laws(Ps, Dx)
        Teon = sum(w_ * tv(Pt[tree.index[s_]], Ps[tree.index[s_]]) for d in dS for s_, w_ in d.items())
        ok_iv &= abs(dJ - (Q[i].max() - Q[i].min()) * Teon) < 1e-9 * max(1.0, Teon)
        ratios.append(dJ / Teon)
u_all = max(Q[i].max() - Q[i].min() for i in range(n))
report("thm:local-u(iv): single-state deviation gives Delta J = u_t(s) T eps_on exactly", ok_iv)
report("thm:local-u(iv): sup over students of |dJ|/(T eps_on) equals u (full-support teacher)",
       abs(max(ratios) - u_all) < 1e-9)

# rem:measure-u: |Delta_t| <= TV_t * (range of Q over k tokens) + (T-t+1) * (|q-p| mass outside them)
ok_res = True
for trial in range(2000):
    Vv, H = int(rng.integers(2, 12)), float(rng.integers(1, 20))
    p_, q_ = rng.dirichlet(0.5 * np.ones(Vv)), rng.dirichlet(0.5 * np.ones(Vv))
    Qa = H * rng.random(Vv)
    k = int(rng.integers(1, Vv + 1))
    top = np.argsort(-np.abs(q_ - p_))[:k]
    rest = np.setdiff1d(np.arange(Vv), top)
    Delta = float((q_ - p_) @ Qa)
    bound = tv(p_, q_) * (Qa[top].max() - Qa[top].min()) + H * float(np.abs(q_ - p_)[rest].sum())
    trunc = float((q_ - p_)[top] @ Qa[top])
    ok_res &= abs(Delta) <= bound + TOL and abs(Delta - trunc) <= H * float(np.abs(q_ - p_)[rest].sum()) + TOL
report("rem:measure-u: top-k range plus residual bounds |Delta_t|; truncated Delta_t within the residual", ok_res)

# rem:measure-u: the TV-proportional position draw gives unbiased estimates (exact expectation)
Pt = dirichlet_rows(n, V, 1.0)
Ps = perturb(Pt, 0.7)
C = rng.random((n, V))
Q = tree.Qfun(Pt, C)
num_u = est_u = est_d = 0.0
for k_, x in enumerate(X):
    for y in itertools.product(range(V), repeat=T):
        pr, path = Dx[k_], []
        for t in range(T):
            i = tree.index[(x,) + y[:t]]
            path.append((tv(Pt[i], Ps[i]), Q[i].max() - Q[i].min(), float((Ps[i] - Pt[i]) @ Q[i])))
            pr *= Ps[i, y[t]]
        S = sum(a for a, _, _ in path)
        est_u += pr * sum(a / S * S * b for a, b, _ in path if a > 0)
        est_d += pr * sum(a / S * S * d / a for a, _, d in path if a > 0)
_, rows = local_quantities(tree, Pt, Ps, C, Dx)
w = np.array([r[2] for r in rows])
TVv, uv = np.array([r[4] for r in rows]), np.array([r[5] for r in rows])
dJ = tree.J(Ps, C, Dx) - tree.J(Pt, C, Dx)
report("rem:measure-u: S u_t and S Delta_t/TV_t are unbiased for u_on T eps_on and Delta J",
       abs(est_u - float(w @ (uv * TVv))) < 1e-9 and abs(est_d - dJ) < 1e-9)

# ---------------------------------------------------------------------------
# prop:valmart  (i) value martingale, (ii) terminal rewards, (iii) chi^2 transfer
# ---------------------------------------------------------------------------
ok_mart = ok_pop = ok_cs = ok_chain = ok_near = True
for trial in range(40):
    Pt = dirichlet_rows(n, V, [0.5, 2.0][trial % 2])
    Ps = perturb(Pt, [0.02, 0.3, 1.5][trial % 3])
    C = rng.random((n, V))
    Q = tree.Qfun(Pt, C)
    dP = tree.laws(Pt, Dx)
    sig2 = {s: float(Pt[tree.index[s]] @ Q[tree.index[s]] ** 2 - (Pt[tree.index[s]] @ Q[tree.index[s]]) ** 2)
            for s in tree.prefixes}
    lhs = sum(w_ * sig2[s] for d in dP for s, w_ in d.items())
    Sigma2, chi_x = 0.0, []
    for k, x in enumerate(X):
        tp, tq = tree.seq_table(Pt, C, x), tree.seq_table(Ps, C, x)
        m = tp[:, 0] @ tp[:, 1]
        Sigma2 += Dx[k] * float(tp[:, 0] @ (tp[:, 1] - m) ** 2)
        chi_x.append(float(np.sum(tq[:, 0] ** 2 / tp[:, 0]) - 1.0))
    ok_mart &= abs(lhs - Sigma2) < 1e-9
    ok_pop &= all(sig2[s] <= (Q[tree.index[s]].max() - Q[tree.index[s]].min()) ** 2 / 4 + TOL for s in tree.prefixes)
    dJ = tree.J(Ps, C, Dx) - tree.J(Pt, C, Dx)
    Echi = float(Dx @ np.array(chi_x))
    ok_cs &= abs(dJ) <= np.sqrt(Echi * Sigma2) + TOL
    kappa = max(chi2(Ps[i], Pt[i]) for i in range(n))
    ok_chain &= Echi <= (1 + kappa) ** T - 1 + TOL
    if T * kappa <= 1:
        Enu_sig2 = Sigma2 / T
        ok_near &= abs(dJ) <= T * np.sqrt(2 * kappa * Enu_sig2) + TOL
report("prop:valmart(i): sum_t E_{d^p_t} Var_{p_t} Q_t = E_x Var_P(C)", ok_mart)
report("prop:valmart: sigma_t(s) <= u_t(s)/2 (Popoviciu)", ok_pop)
report("prop:valmart(iii): |dJ| <= sqrt(E chi2(Q||P) * E_x Var_P(C))", ok_cs)
report("prop:valmart(iii): E chi2(Q^x||P^x) <= (1+chi2_inf)^T - 1", ok_chain)

# near-lossless students (T kappa <= 1) for the last bound
ok_near_cnt = 0
for trial in range(40):
    Pt = dirichlet_rows(n, V, 3.0)
    Ps = perturb(Pt, 0.03)
    C = rng.random((n, V))
    kappa = max(chi2(Ps[i], Pt[i]) for i in range(n))
    if T * kappa > 1:
        continue
    Q = tree.Qfun(Pt, C)
    Sigma2 = 0.0
    for k, x in enumerate(X):
        tp = tree.seq_table(Pt, C, x)
        m = tp[:, 0] @ tp[:, 1]
        Sigma2 += Dx[k] * float(tp[:, 0] @ (tp[:, 1] - m) ** 2)
    dJ = tree.J(Ps, C, Dx) - tree.J(Pt, C, Dx)
    ok_near &= abs(dJ) <= T * np.sqrt(2 * kappa * Sigma2 / T) + TOL
    ok_near_cnt += 1
report(f"prop:valmart(iii): |dJ| <= T sqrt(2 chi2_inf E_nu^p sigma^2) when T chi2_inf <= 1 ({ok_near_cnt} cases)",
       ok_near and ok_near_cnt > 10)

# (ii) terminal rewards: u_t(s) = one-token swing of the success probability W, u <= 1, Sigma^2 <= E W(1-W)
Pt = dirichlet_rows(n, V, 1.0)
Ps = perturb(Pt, 0.8)
r = {(x,) + y: rng.random() for x in X for y in itertools.product(range(V), repeat=T)}
C = np.zeros((n, V))
for s in tree.by_time[T - 1]:
    for a in range(V):
        C[tree.index[s], a] = 1.0 - r[s + (a,)]
Q = tree.Qfun(Pt, C)


def W(s):
    """Teacher success probability from prefix s (len(s)-1 tokens generated)."""
    if len(s) == T + 1:
        return r[s]
    return float(Pt[tree.index[s]] @ np.array([W(s + (a,)) for a in range(V)]))


ok_term = all(abs((Q[tree.index[s]].max() - Q[tree.index[s]].min())
                  - (max(W(s + (a,)) for a in range(V)) - min(W(s + (a,)) for a in range(V)))) < TOL
              and Q[tree.index[s]].max() - Q[tree.index[s]].min() <= 1 + TOL for s in tree.prefixes)
Sigma2 = 0.0
for k, x in enumerate(X):
    tp = tree.seq_table(Pt, C, x)
    m = tp[:, 0] @ tp[:, 1]
    Sigma2 += Dx[k] * float(tp[:, 0] @ (tp[:, 1] - m) ** 2)
EW = sum(Dx[k] * W((x,)) * (1 - W((x,))) for k, x in enumerate(X))
dS = tree.laws(Ps, Dx)
refined = sum(w_ * tv(Pt[tree.index[s]], Ps[tree.index[s]]) *
              (max(W(s + (a,)) for a in range(V)) - min(W(s + (a,)) for a in range(V)))
              for d in dS for s, w_ in d.items())
dJr = tree.J(Ps, C, Dx) - tree.J(Pt, C, Dx)
Teon = sum(w_ * tv(Pt[tree.index[s]], Ps[tree.index[s]]) for d in dS for s, w_ in d.items())
report("prop:valmart(ii): terminal reward, u_t(s) = one-token swing of E_p[r | sa] <= 1", ok_term)
report("prop:valmart(ii): E_x Var_P(C) <= E_x W(1-W) <= 1/4", Sigma2 <= EW + TOL and EW <= 0.25)
report("prop:valmart(ii): |dJ_r| <= sum_t E_{d^theta}[TV_t u_t] <= T eps_on",
       abs(dJr) <= refined + TOL and refined <= Teon + TOL)

# ---------------------------------------------------------------------------
# thm:forgetting (i) influence decomposition, (ii) TV of cost statistics and couplings
# ---------------------------------------------------------------------------


def future_profile(tree, P, C, s_next):
    """E_P[c_k | s_{t+1} = s_next] for every later step k (list, in time order)."""
    prof, layer = [], {s_next: 1.0}
    while layer and len(next(iter(layer))) <= tree.T:
        prof.append(sum(w_ * P[tree.index[s]] @ C[tree.index[s]] for s, w_ in layer.items()))
        nxt = {}
        for s, w_ in layer.items():
            for a in range(tree.V):
                nxt[s + (a,)] = w_ * P[tree.index[s], a]
        layer = nxt
    return prof


Pt = dirichlet_rows(n, V, 1.0)
C = rng.random((n, V))
Q = tree.Qfun(Pt, C)
ok_inf = ok_cons = True
for s in tree.prefixes:
    i = tree.index[s]
    profs = [future_profile(tree, Pt, C, s + (a,)) if len(s) < T else [] for a in range(V)]
    ok_cons &= all(abs(Q[i, a] - C[i, a] - sum(profs[a])) < TOL for a in range(V))
    iota = [max(pr[k] for pr in profs) - min(pr[k] for pr in profs) for k in range(len(profs[0]))]
    ok_inf &= Q[i].max() - Q[i].min() <= C[i].max() - C[i].min() + sum(iota) + TOL
report("thm:forgetting(i): Q_t(s,a) = c_t(s,a) + sum_k V^(k)_{t+1}(sa)", ok_cons)
report("thm:forgetting(i): u_t(s) <= osc_a c_t(s,.) + sum_{k>t} iota_{t,k}(s)", ok_inf)

# (ii) window-local costs c_k = g_k(y_{k-1}, y_k): beta_{t,k} <= TV of the window law; any coupling bound
g = rng.random((T, V + 1, V))  # g[k, prev token (V = none), token]
Cw = np.zeros((n, V))
for s in tree.prefixes:
    k = len(s) - 1
    prev = s[-1] if k >= 1 else V
    Cw[tree.index[s]] = g[k, prev]
Qw = tree.Qfun(Pt, Cw)


def window_law(tree, P, s_next, k):
    """Law of (y_{k-1}, y_k) (0-based step k) under P from prefix s_next."""
    layer = {s_next: 1.0}
    while len(next(iter(layer))) < k + 2:
        nxt = {}
        for s, w_ in layer.items():
            for a in range(tree.V):
                nxt[s + (a,)] = w_ * P[tree.index[s], a]
        layer = nxt
    out = {}
    for s, w_ in layer.items():
        key = s[-2:] if k >= 1 else (s[-1],)
        out[key] = out.get(key, 0.0) + w_
    return out


def quantile_coupling(p, q):
    """Joint law of (F_p^{-1}(U), F_q^{-1}(U)), U uniform: shared-noise sampling."""
    Fp, Fq = np.concatenate([[0], np.cumsum(p)]), np.concatenate([[0], np.cumsum(q)])
    return np.array([[max(0.0, min(Fp[a + 1], Fq[b + 1]) - max(Fp[a], Fq[b])) for b in range(len(q))]
                     for a in range(len(p))])


ok_win = ok_coup = True
for s in [p_ for p_ in tree.prefixes if len(p_) <= T - 1]:
    i, t = tree.index[s], len(s) - 1
    for a, a2 in itertools.combinations(range(V), 2):
        sa, sa2 = s + (a,), s + (a2,)
        # (ii) TV of the window statistic bounds the swing of each future expected cost
        for k in range(t + 1, T):
            la, lb = window_law(tree, Pt, sa, k), window_law(tree, Pt, sa2, k)
            keys = set(la) | set(lb)
            tvk = 0.5 * sum(abs(la.get(kk, 0) - lb.get(kk, 0)) for kk in keys)
            Ea = sum(pr * g[k, kk[0], kk[1]] for kk, pr in la.items())
            Eb = sum(pr * g[k, kk[0], kk[1]] for kk, pr in lb.items())
            ok_win &= abs(Ea - Eb) <= tvk + TOL
        # coupling bound with the shared-noise (quantile) coupling of the two continuations
        EN, layer = 0.0, {(sa, sa2): 1.0}
        while len(next(iter(layer))[0]) <= T:
            nxt = {}
            for (u1, u2), w_ in layer.items():
                J2 = quantile_coupling(Pt[tree.index[u1]], Pt[tree.index[u2]])
                for b1 in range(V):
                    for b2 in range(V):
                        if J2[b1, b2] > 0:
                            EN += w_ * J2[b1, b2] * (Cw[tree.index[u1], b1] != Cw[tree.index[u2], b2])
                            nxt[(u1 + (b1,), u2 + (b2,))] = nxt.get((u1 + (b1,), u2 + (b2,)), 0) + w_ * J2[b1, b2]
            layer = nxt
        ok_coup &= abs(Qw[i, a] - Qw[i, a2]) <= abs(Cw[i, a] - Cw[i, a2]) + EN + TOL
report("thm:forgetting(ii): swing of E[g_k(window)] <= TV of the window law", ok_win)
report("thm:forgetting(ii): |Q(s,a)-Q(s,a')| <= |c diff| + E N under shared-noise coupling", ok_coup)

# ---------------------------------------------------------------------------
# thm:forgetting (iii) latent chains: u <= 1 + m/(1-delta_m); post-step costs: u <= m/(1-delta_m)
# ---------------------------------------------------------------------------


def dob(K):
    return max(0.5 * np.abs(K[z] - K[w]).sum() for z in range(K.shape[0]) for w in range(K.shape[0]))


def latent_u(Pz, f, cz, T):
    """Exact latent DP; Pz[t]: (nz, V), f: (nz, V) -> next latent, cz[t]: (nz, V). Returns u and Q tables."""
    nz = f.shape[0]
    Vn = np.zeros(nz)
    Qs = [None] * T
    for t in reversed(range(T)):
        Qs[t] = cz[t] + Vn[f]
        Vn = (Pz[t] * Qs[t]).sum(axis=1)
    return max((Qs[t].max(axis=1) - Qs[t].min(axis=1)).max() for t in range(T)), Qs


def kernels(Pz, f):
    nz, Vv = f.shape
    Ks = []
    for P_ in Pz:
        K = np.zeros((nz, nz))
        for z in range(nz):
            for a in range(Vv):
                K[z, f[z, a]] += P_[z, a]
        Ks.append(K)
    return Ks


def delta_m(Ks, m):
    vals = []
    for t in range(len(Ks) - m + 1):
        K = np.eye(Ks[0].shape[0])
        for j in range(t, t + m):
            K = K @ Ks[j]
        vals.append(dob(K))
    return max(vals) if vals else 0.0


ok_gen = ok_post = ok_tau = ok_doe = ok_mix = ok_tmix = True
checked = n_tmix = 0
for trial in range(300):
    nz, Vv, TT = int(rng.integers(2, 6)), int(rng.integers(2, 5)), int(rng.integers(3, 25))
    f = rng.integers(0, nz, size=(nz, Vv))
    homog = trial % 2 == 0
    base = rng.dirichlet(rng.uniform(0.3, 3) * np.ones(Vv), size=nz)
    Pz = [base if homog else rng.dirichlet(np.ones(Vv), size=nz) for _ in range(TT)]
    Ks = kernels(Pz, f)
    cz = [rng.random((nz, Vv)) for _ in range(TT)]
    gpost = [rng.random(nz) for _ in range(TT + 1)]
    cpost = [gpost[t + 1][f] for t in range(TT)]
    u_gen, _ = latent_u(Pz, f, cz, TT)
    u_post, _ = latent_u(Pz, f, cpost, TT)
    for m in range(1, min(4, TT) + 1):
        dm = delta_m(Ks, m)
        if dm < 1 - 1e-12:
            checked += 1
            ok_gen &= u_gen <= 1 + m / (1 - dm) + TOL
            ok_post &= u_post <= m / (1 - dm) + TOL
            # Doeblin: alpha_m = min over t of sum_z' min_z K(z,z') gives delta_m <= 1 - alpha_m
            alphas = []
            for t in range(len(Ks) - m + 1):
                K = np.eye(nz)
                for j in range(t, t + m):
                    K = K @ Ks[j]
                alphas.append(K.min(axis=0).sum())
            if alphas:
                am = min(alphas)
                ok_doe &= dm <= 1 - am + TOL and (am <= 0 or u_gen <= 1 + m / am + TOL)
    taus = [m for m in range(1, TT + 1) if delta_m(Ks, m) <= 0.5]   # m_half, if it exists
    if taus:
        ok_tau &= u_gen <= 1 + 2 * taus[0] + TOL
    if homog:
        K = Ks[0]
        w_, vecs = np.linalg.eig(K.T)
        pi = np.real(vecs[:, np.argmin(np.abs(w_ - 1))])
        pi = pi / pi.sum()
        tmix = None
        for m in range(1, 400):
            Km = np.linalg.matrix_power(K, m)
            dmix = max(0.5 * np.abs(Km[z] - pi).sum() for z in range(nz))
            ok_mix &= dmix - TOL <= dob(Km) <= 2 * dmix + TOL
            if tmix is None and dmix <= 0.25:
                tmix = m
        if tmix is not None:
            n_tmix += 1
            ok_tmix &= u_gen <= 1 + 2 * tmix + TOL
report(f"thm:forgetting(iii): u <= 1 + m/(1-delta_m) on random latent chains ({checked} cases)", ok_gen)
report("thm:forgetting(iii): post-step costs, u <= m/(1-delta_m)", ok_post)
report("thm:forgetting(iii): u <= 1 + 2 m_half, m_half = min{m <= T : delta_m <= 1/2}", ok_tau)
report(f"thm:forgetting(iii): homogeneous chains, u <= 1 + 2 t_mix(1/4) ({n_tmix} cases)", ok_tmix and n_tmix > 50)
report("thm:forgetting(iii): Doeblin, delta_m <= 1 - alpha and u <= 1 + m/alpha", ok_doe)
report("thm:forgetting(iii): d(m) <= delta(K^m) <= 2 d(m) (homogeneous chains)", ok_mix)

# prefix-level u of a latent-chain teacher never exceeds the latent-level u
nz, Vv, TT = 3, 2, 5
f = rng.integers(0, nz, size=(nz, Vv))
Pz = [rng.dirichlet(np.ones(Vv), size=nz) for _ in range(TT)]
cz = [rng.random((nz, Vv)) for _ in range(TT)]
u_lat, _ = latent_u(Pz, f, cz, TT)
tr2 = Tree((0,), Vv, TT)
z_of = {}
for s in sorted(tr2.prefixes, key=len):
    z_of[s] = 0 if len(s) == 1 else int(f[z_of[s[:-1]], s[-1]])
Pp = np.array([Pz[len(s) - 1][z_of[s]] for s in tr2.prefixes])
Cp = np.array([cz[len(s) - 1][z_of[s]] for s in tr2.prefixes])
Qp = tr2.Qfun(Pp, Cp)
u_pref = max(Qp[i].max() - Qp[i].min() for i in range(len(tr2.prefixes)))
report("thm:forgetting(iii): prefix-level u <= latent-level u", u_pref <= u_lat + TOL)

# (iv) the raw prefix chain has Dobrushin coefficient 1
Kp = np.zeros((len(tr2.by_time[2]), len(tr2.by_time[3])))
idx3 = {s: j for j, s in enumerate(tr2.by_time[3])}
for j, s in enumerate(tr2.by_time[2]):
    for a in range(Vv):
        Kp[j, idx3[s + (a,)]] = Pp[tr2.index[s], a]
report("thm:forgetting(iv): the raw prefix chain has delta = 1", abs(dob(Kp) - 1.0) < TOL)

# ---------------------------------------------------------------------------
# ex:selfcorrect: latent states ok (0) and err (1); error rate eta_err, repair rate eta_rep;
# each step that ends in err costs 1
# ---------------------------------------------------------------------------
f2 = np.array([[0, 1], [0, 1]])  # token 0 -> ok, token 1 -> err, from either state
ok_sc = ok_lim = ok_cliff = ok_doe2 = True
for eta_rep, eta_err in [(0.3, 0.05), (0.1, 0.0), (0.5, 0.5), (0.02, 0.01), (0.0, 0.2)]:
    lam = 1 - eta_rep - eta_err
    K1 = np.array([[1 - eta_err, eta_err], [eta_rep, 1 - eta_rep]])
    ok_doe2 &= abs(K1.min(axis=0).sum() - (eta_err + eta_rep)) < TOL   # best Doeblin constant, m = 1
    for TT in [1, 2, 5, 50, 400]:
        Pz = [K1] * TT
        cpost = [np.array([[0.0, 1.0], [0.0, 1.0]])] * TT
        u_ex, _ = latent_u(Pz, f2, cpost, TT)
        closed = TT if lam == 1 else (1 - lam ** TT) / (1 - lam)
        ok_sc &= abs(u_ex - closed) < 1e-9 * max(1, closed)
        ok_sc &= abs(dob(kernels(Pz, f2)[0]) - abs(lam)) < TOL
        if lam < 1:
            ok_sc &= u_ex <= 1 / (1 - lam) + TOL
    if lam < 1:  # gap to the limit is exactly lambda_2^T / (1 - lambda_2) -> 0
        ok_lim &= abs(1 / (eta_err + eta_rep) - u_ex - lam ** TT / (1 - lam)) < 1e-9 * (1 / (eta_err + eta_rep))
for TT in [1, 7, 100]:
    Pz = [np.array([[1.0, 0.0], [0.0, 1.0]])] * TT
    u_ex, _ = latent_u(Pz, f2, [np.array([[0.0, 1.0], [0.0, 1.0]])] * TT, TT)
    ok_cliff &= abs(u_ex - TT) < TOL and abs(delta_m(kernels(Pz, f2), 3 if TT >= 3 else 1) - 1) < TOL
report("ex:selfcorrect: u = (1-lambda_2^T)/(1-lambda_2) <= 1/(1-delta_1), delta_1 = lambda_2", ok_sc)
report("ex:selfcorrect: Doeblin constant (m=1) is eta_err + eta_rep", ok_doe2)
report("ex:selfcorrect: u -> 1/(eta_err+eta_rep) as T grows", ok_lim)
report("ex:selfcorrect: no repair and no errors (err absorbing): delta_m = 1 and u = T", ok_cliff)

# rem:where-avg: an absorbing error entered with probability 1/T per step gives Var_p(C) of order T^2
ratios_v = []
for TT in [10, 100, 1000, 10000]:
    eta = 1.0 / TT
    t = np.arange(1, TT + 1)
    pt = (1 - eta) ** (t - 1) * eta          # P(first deviation at step t)
    Cv = TT - t + 1.0                        # the deviating step and every later one cost 1
    m1, m2 = float(pt @ Cv), float(pt @ Cv ** 2)
    ratios_v.append((m2 - m1 ** 2) / TT ** 2)
report(f"rem:where-avg: absorbing error at rate 1/T gives Var_p(C)/T^2 -> {ratios_v[-1]:.4f} > 0",
       min(ratios_v) > 0.05 and abs(ratios_v[-1] - ratios_v[-2]) < 1e-3)

# ---------------------------------------------------------------------------
# prop:trap: teacher-side averages of u do not control on-policy compounding
# latent states ok, drift, sink
# ---------------------------------------------------------------------------
STEP = {("ok", 0): "ok", ("ok", 1): "drift", ("drift", 0): "ok", ("drift", 1): "sink",
        ("sink", 0): "sink", ("sink", 1): "sink"}


def trap_tree(TT):
    tr = Tree((0,), 2, TT)
    z = {}
    for s in sorted(tr.prefixes, key=len):
        z[s] = "ok" if len(s) == 1 else STEP[(z[s[:-1]], s[-1])]
    C = np.array([[float(STEP[(z[s], a)] != "ok") for a in range(2)] for s in tr.prefixes])
    return tr, z, C


def trap_models(TT, eps, eta=0.0):
    tr, z, C = trap_tree(TT)
    Pt = np.array([[1 - eta, eta] for s in tr.prefixes])
    Ps = np.array([{"ok": [1 - eps, eps], "drift": [0.0, 1.0], "sink": [1.0, 0.0]}[z[s]] for s in tr.prefixes])
    return tr, z, Pt, Ps, C


def ratio_terms(tr, Pt, Ps, C):
    Dx1 = np.array([1.0])
    dJ = tr.J(Ps, C, Dx1) - tr.J(Pt, C, Dx1)
    Teon = sum(w_ * tv(Pt[tr.index[s]], Ps[tr.index[s]]) for d in tr.laws(Ps, Dx1) for s, w_ in d.items())
    return dJ, Teon


ok_trap = ok_teach = ok_D = ok_lb = ok_off = True
for TT in [3, 6, 10]:
    for eps in [0.3 / TT, 1.0 / TT, 0.05]:
        tr, z, Pt, Ps, C = trap_models(TT, eps)
        Dx1 = np.array([1.0])
        Q = tr.Qfun(Pt, C)
        dJ, Teon = ratio_terms(tr, Pt, Ps, C)
        dS, dP = tr.laws(Ps, Dx1), tr.laws(Pt, Dx1)
        ok_trap &= abs(dJ - sum(1 - (1 - eps) ** t for t in range(1, TT + 1))) < 1e-9
        ok_trap &= abs(Teon - (2 - (1 - eps) ** TT - (1 - eps) ** (TT - 1))) < 1e-9
        ok_teach &= all(abs(sum(w_ * (Q[tr.index[s]].max() - Q[tr.index[s]].min()) for s, w_ in d.items()) - 1) < TOL
                        for d in dP)
        ok_D &= all(abs(Q[tr.index[s]].max() - Q[tr.index[s]].min() - (TT - (len(s) - 1))) < TOL
                    for s in tr.prefixes if z[s] == "drift")
        eoff = sum(w_ * tv(Pt[tr.index[s]], Ps[tr.index[s]]) for d in dP for s, w_ in d.items()) / TT
        u_off = sum(w_ * tv(Pt[tr.index[s]], Ps[tr.index[s]]) * (Q[tr.index[s]].max() - Q[tr.index[s]].min())
                    for d in dP for s, w_ in d.items()) / (TT * eoff)
        ok_off &= abs(eoff - eps) < TOL and abs(u_off - 1) < TOL
        if TT * eps <= 1:
            u_on = sum(w_ * tv(Pt[tr.index[s]], Ps[tr.index[s]]) * (Q[tr.index[s]].max() - Q[tr.index[s]].min())
                       for d in dS for s, w_ in d.items()) / Teon
            ok_lb &= dJ >= (TT + 1) / 8 * Teon - TOL and u_on >= (TT + 1) / 8 - TOL
report("prop:trap: exact Delta J and T eps_on (enumeration)", ok_trap)
report("prop:trap: E_{d^p_t} u_t = 1 for every t", ok_teach)
report("prop:trap: u_t = T-t+1 at the drift state", ok_D)
report("prop:trap: eps_off = eps and E_nu^p[u TV]/E_nu^p[TV] = 1", ok_off)
report("prop:trap: Delta J >= (T+1)/8 * T eps_on and u_on >= (T+1)/8 when T eps <= 1", ok_lb)

# full-support teacher: the averaged conclusions hold up to factors close to 1
tr, z, Pt, Ps, C = trap_models(10, 0.1, eta=1e-9)
Q = tr.Qfun(Pt, C)
dJ, Teon = ratio_terms(tr, Pt, Ps, C)
dP = tr.laws(Pt, np.array([1.0]))
Eu = [sum(w_ * (Q[tr.index[s]].max() - Q[tr.index[s]].min()) for s, w_ in d.items()) for d in dP]
num = sum(w_ * tv(Pt[tr.index[s]], Ps[tr.index[s]]) * (Q[tr.index[s]].max() - Q[tr.index[s]].min())
          for d in dP for s, w_ in d.items())
den = sum(w_ * tv(Pt[tr.index[s]], Ps[tr.index[s]]) for d in dP for s, w_ in d.items())
report("prop:trap: full-support teacher (eta=1e-9): teacher-side averages < 1.001, ratio >= (T+1)/8 * 0.999",
       max(Eu) < 1.001 and num / den < 1.001 and dJ >= 0.999 * (10 + 1) / 8 * Teon)

# the supremum for the deterministic trap teacher is T/2 (while u_D = T-1): random students never exceed it,
# and deviating only at t=1, then always emitting 1 at drift, attains it
ok_sup = ok_att = True
for TT in [2, 3, 4, 6]:
    tr, z, C = trap_tree(TT)
    Pt = np.array([[1.0, 0.0] for s in tr.prefixes])
    Q = tr.Qfun(Pt, C)
    uD = max(Q[tr.index[s]].max() - Q[tr.index[s]].min() for s in tr.prefixes)
    for trial in range(300):
        Ps = rng.dirichlet(rng.choice([0.2, 1.0, 5.0]) * np.ones(2), size=len(tr.prefixes))
        dJ, Teon = ratio_terms(tr, Pt, Ps, C)
        ok_sup &= dJ <= TT / 2 * Teon + TOL
    Ps = np.array([[1.0, 0.0] if z[s] != "drift" else [0.0, 1.0] for s in tr.prefixes])
    Ps[tr.index[(0,)]] = [0.9, 0.1]
    dJ, Teon = ratio_terms(tr, Pt, Ps, C)
    ok_att &= abs(dJ / Teon - TT / 2) < 1e-9 and abs(uD - (TT - 1)) < TOL
report("app:recoverability: deterministic trap teacher, sup |dJ|/(T eps_on) <= T/2 (random students)", ok_sup)
report("app:recoverability: T/2 attained by one deviation at t=1; u_D = T-1", ok_att)

# large T via the closed forms
for TT, eps in [(100, 1e-2), (1000, 1e-3)]:
    dJ = sum(1 - (1 - eps) ** t for t in range(1, TT + 1))
    Teon = 2 - (1 - eps) ** TT - (1 - eps) ** (TT - 1)
    print(f"      trap T={TT:5d} eps={eps:g}: dJ={dJ:8.3f}  T*eps_on={Teon:6.4f}  ratio={dJ / Teon:7.2f}"
          f"  (T+1)/8={(TT + 1) / 8:7.2f}")

print("\nALL PASS" if all(results) else "\nSOME CHECKS FAILED")
