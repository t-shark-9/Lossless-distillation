"""Checks for Section 7 (decoding-level losslessness).

Flip thresholds are checked against direct constrained optimisation over the simplex; sequence-level
statements by exact enumeration of small autoregressive teacher/student pairs; the knapsack
characterisation by brute force over subsets.
Run:  python3 verification/check_decoding.py
"""
import itertools
import warnings
import numpy as np
from scipy.optimize import linprog, minimize, minimize_scalar

warnings.filterwarnings("ignore", category=RuntimeWarning)  # inf - inf inside the 1-D searches is expected

rng = np.random.default_rng(7)
TOL = 1e-9


def report(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return bool(ok)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def xlogy(x, y):
    """x log(x/y) with 0 log 0 = 0 and x log(x/0) = inf."""
    if x <= 0:
        return 0.0
    if y <= 0:
        return np.inf
    return x * np.log(x / y)


# ---------------------------------------------------------------------------
# f-divergences D_f(P||Q) = sum_y Q f(P/Q), written through their perspectives g(p, q) = q f(p/q)
# ---------------------------------------------------------------------------
BETA = 0.3
PERSP = {
    "TV": lambda p, q: 0.5 * abs(p - q),
    "KL": lambda p, q: xlogy(p, q),                                  # KL(P||Q)
    "rKL": lambda p, q: xlogy(q, p),                                 # KL(Q||P)
    "H2": lambda p, q: 0.5 * (np.sqrt(p) - np.sqrt(q)) ** 2,
    "chi2": lambda p, q: (0.0 if p == q else np.inf) if q <= 0 else (p - q) ** 2 / q,
    "JSD": lambda p, q: BETA * xlogy(p, BETA * p + (1 - BETA) * q) + (1 - BETA) * xlogy(q, BETA * p + (1 - BETA) * q),
}
FPP1 = {"KL": 1.0, "rKL": 1.0, "H2": 0.25, "chi2": 2.0, "JSD": BETA * (1 - BETA)}  # f''(1)


def fdiv(name, P, Q):
    g = PERSP[name]
    return float(sum(g(float(a), float(b)) for a, b in zip(P, Q)))


def merge_cost(name, P1, P2, R):
    """k_f(P1,P2,R) = min_{0<=c<=1/2} g(P1,c)+g(P2,c)+g(R,1-2c)  (numerically, by convex 1-D search)."""
    g = PERSP[name]
    R = max(R, 0.0)
    phi = lambda c: g(P1, c) + g(P2, c) + g(R, max(1 - 2 * c, 0.0))
    res = minimize_scalar(phi, bounds=(0.0, 0.5), method="bounded", options={"xatol": 1e-13})
    cands = [res.x, 0.0, 0.5, (P1 + P2) / 2]
    vals = [phi(c) for c in cands]
    i = int(np.argmin(vals))
    return vals[i], cands[i]


def closed_form(name, P1, P2, R):
    R = max(R, 0.0)
    s = (np.sqrt(P1) - np.sqrt(P2)) ** 2
    if name == "TV":
        return (P1 - P2) / 2
    if name == "KL":
        return xlogy(P1, (P1 + P2) / 2) + xlogy(P2, (P1 + P2) / 2)
    if name == "rKL":
        return -np.log(1 - s)
    if name == "H2":
        return 1 - np.sqrt(1 - s / 2)
    if name == "chi2":
        return (np.sqrt(2 * (P1 ** 2 + P2 ** 2)) + R) ** 2 - 1
    raise ValueError(name)


def top2(p):
    o = np.argsort(-p, kind="stable")
    return o[0], o[1], p[o[0]], p[o[1]]


def direct_min(name, p, b):
    """min D(p||q) over the simplex subject to q_b >= q_{a*}, by direct optimisation (upper bound on the min)."""
    V = len(p)
    a = int(np.argmax(p))
    if name == "TV":  # LP in (q, t): min sum t/2, t >= |q-p|, q_b >= q_a, sum q = 1, q >= 0
        c = np.r_[np.zeros(V), 0.5 * np.ones(V)]
        A, bb = [], []
        for i in range(V):
            r = np.zeros(2 * V); r[i] = 1; r[V + i] = -1; A.append(r); bb.append(p[i])
            r = np.zeros(2 * V); r[i] = -1; r[V + i] = -1; A.append(r); bb.append(-p[i])
        r = np.zeros(2 * V); r[a] = 1; r[b] = -1; A.append(r); bb.append(0.0)
        Aeq = [np.r_[np.ones(V), np.zeros(V)]]
        res = linprog(c, A_ub=np.array(A), b_ub=bb, A_eq=np.array(Aeq), b_eq=[1.0], bounds=[(0, None)] * (2 * V))
        return res.fun
    best = np.inf
    obj = lambda z: fdiv(name, p, softmax(z))
    cons = [{"type": "ineq", "fun": lambda z: z[b] - z[a]}]
    for s in range(6):
        z0 = np.log(p) + rng.normal(scale=0.3 if s else 0.0, size=V)
        z0[a], z0[b] = z0[b], z0[a]  # start from a feasible point
        res = minimize(obj, z0, constraints=cons, method="SLSQP", options={"ftol": 1e-14, "maxiter": 2000})
        if res.success or res.status == 8:
            best = min(best, obj(res.x) if res.x[b] >= res.x[a] - 1e-10 else np.inf)
    return best


results = []

# ---------------------------------------------------------------------------
# prop:flip  (flip thresholds)
# ---------------------------------------------------------------------------
ok_formula = ok_direct = ok_runner = ok_lb = ok_rand = True
worst_gap = 0.0
for trial in range(120):
    V = int(rng.integers(2, 7))
    p = rng.dirichlet(10 ** rng.uniform(-0.5, 1.0) * np.ones(V))
    p = np.clip(p, 1e-6, None); p /= p.sum()
    a, b2, p1, p2 = top2(p)
    R = max(1 - p1 - p2, 0.0)
    gam, m = p1 - p2, p1 + p2
    for name in ["TV", "KL", "rKL", "H2", "chi2", "JSD"]:
        kf, c = merge_cost(name, p1, p2, R)
        if name != "JSD":
            cf = closed_form(name, p1, p2, R)
            ok_formula &= abs(kf - cf) <= 1e-7 * max(1.0, cf)
        if trial < 40:
            # the runner-up is the cheapest competitor, and the reduction to k_f is exact
            costs = [direct_min(name, p, b) for b in range(V) if b != a]
            dm = min(costs)
            worst_gap = max(worst_gap, abs(dm - kf) / max(kf, 1e-12))
            ok_direct &= dm >= kf - 1e-7 and abs(dm - kf) <= 1e-5 * max(kf, 1e-3)
            ok_runner &= abs(direct_min(name, p, b2) - dm) <= 1e-5 * max(kf, 1e-3)
        # random students that flip never beat the threshold
        for _ in range(200):
            q = softmax(np.log(p) + rng.normal(scale=rng.uniform(0.05, 2.0), size=V))
            if np.argmax(q) != a:
                ok_rand &= fdiv(name, p, q) >= kf - 1e-10
    ok_lb &= closed_form("KL", p1, p2, R) >= gam ** 2 / (2 * m) - TOL
    ok_lb &= closed_form("rKL", p1, p2, R) >= gam ** 2 / (2 * m) - TOL
    ok_lb &= merge_cost("JSD", p1, p2, R)[0] >= BETA * (1 - BETA) * gam ** 2 / 2 - TOL
results += [report("k_f equals the closed forms for TV, KL, rKL, H^2, chi^2", ok_formula),
            report(f"k_f equals the direct constrained minimum over the simplex (max rel. gap {worst_gap:.1e})", ok_direct),
            report("the runner-up is the cheapest competitor", ok_runner),
            report("random flipping students never beat k_f", ok_rand),
            report("KL, rKL >= gamma^2/(2m);  JSD_b >= b(1-b) gamma^2/2", ok_lb)]

# local expansion: k_f ~ f''(1) gamma^2 / (2m) as gamma -> 0
ok = True
for name, fpp in FPP1.items():
    for m, R in [(1.0, 0.0), (0.6, 0.4), (0.2, 0.8)]:
        gam = 1e-3 * m
        k = merge_cost(name, (m + gam) / 2, (m - gam) / 2, R)[0]
        ok &= abs(k / (fpp * gam ** 2 / (2 * m)) - 1) < 1e-2
results.append(report("k_f = f''(1) gamma^2/(2m) (1+o(1)) as gamma/m -> 0", ok))

# logit thresholds: flip => osc(z - v) >= Gamma and ||C(z-v)||_2 >= Gamma/sqrt(2), both attained
ok = True
for _ in range(20000):
    V = int(rng.integers(2, 9))
    v = rng.normal(scale=2.0, size=V)
    p = softmax(v)
    a, b2, p1, p2 = top2(p)
    G = np.log(p1 / p2)
    d = rng.normal(scale=rng.uniform(0.1, 3.0), size=V)
    if np.argmax(v + d) != a:
        ok &= d.max() - d.min() >= G - TOL and np.linalg.norm(d - d.mean()) >= G / np.sqrt(2) - TOL
d = np.zeros(V); d[a] = -G / 2; d[b2] = G / 2
z = v + d
ok &= abs(z[a] - z[b2]) < 1e-12 and abs(np.linalg.norm(d - d.mean()) - G / np.sqrt(2)) < 1e-12
results.append(report("logit flips need osc(z-v) >= Gamma and ||C(z-v)|| >= Gamma/sqrt2 (tight)", ok))

# ---------------------------------------------------------------------------
# Small autoregressive models, enumerated exactly
# ---------------------------------------------------------------------------


class Tree:
    def __init__(self, V, T, cond):
        self.V, self.T = V, T
        self.prefixes = [p for t in range(T) for p in itertools.product(range(V), repeat=t)]
        self.index = {p: i for i, p in enumerate(self.prefixes)}
        self.cond = cond  # array [num_prefixes, V]
        self.seqs = list(itertools.product(range(V), repeat=T))

    def c(self, s):
        return self.cond[self.index[tuple(s)]]

    def prob(self, y):
        return float(np.prod([self.c(y[:t])[y[t]] for t in range(len(y))]))

    def seq_law(self):
        return np.array([self.prob(y) for y in self.seqs])

    def prefix_law(self, t):  # law of y_{1..t}
        return {y: self.prob(y) for y in itertools.product(range(self.V), repeat=t)}

    def greedy(self):
        s = ()
        for t in range(self.T):
            s = s + (int(np.argmax(self.c(s))),)  # ties: lowest index
        return s


def random_tree(V, T, scale=1.5):
    n = sum(V ** t for t in range(T))
    return Tree(V, T, np.array([softmax(rng.normal(scale=scale, size=V)) for _ in range(n)]))


def tree_from_law(template, Q):
    """Autoregressive model with sequence law Q (dict or array over template.seqs)."""
    law = dict(zip(template.seqs, Q))
    pre = {}
    for y, w in law.items():
        for t in range(template.T + 1):
            pre[y[:t]] = pre.get(y[:t], 0.0) + w
    cond = np.zeros_like(template.cond)
    for s in template.prefixes:
        tot = pre[s]
        cond[template.index[s]] = [pre[s + (a,)] / tot for a in range(template.V)] if tot > 0 else template.c(s)
    return Tree(template.V, template.T, cond)


def greedy_path(P):
    """Greedy states, greedy tokens, runner-up tokens, top-two probs and greedy-prefix probabilities pi_t."""
    s, pi, out = (), 1.0, []
    for t in range(P.T):
        a, b, p1, p2 = top2(P.c(s))
        out.append(dict(s=s, a=int(a), b=int(b), p1=p1, p2=p2, pi=pi))
        pi *= p1
        s = s + (int(a),)
    return out


def omega(name, path):
    return min(merge_cost(name, st["pi"] * st["p1"], st["pi"] * st["p2"], 1 - st["pi"] * (st["p1"] + st["p2"]))[0]
               for st in path)


def first_flip(P, Qm):
    for t, st in enumerate(greedy_path(P)):
        if int(np.argmax(Qm.c(st["s"]))) != st["a"]:
            return t
    return None


# thm:greedy-seq (i): disagreement => D_f(P||Q) >= omega_f, for every f-divergence
NAMES = ["TV", "KL", "rKL", "H2", "chi2", "JSD"]
ok = True
n_dis = 0
for _ in range(30):
    P = random_tree(3, 4, scale=rng.uniform(0.5, 2.5))
    Pl = P.seq_law()
    path = greedy_path(P)
    om = {nm: omega(nm, path) for nm in NAMES}
    for _ in range(60):
        Qm = Tree(3, 4, np.array([softmax(np.log(r) + rng.normal(scale=rng.uniform(0.02, 1.5), size=3)) for r in P.cond]))
        if Qm.greedy() != P.greedy():
            n_dis += 1
            Ql = Qm.seq_law()
            for nm in NAMES:
                ok &= fdiv(nm, Pl, Ql) >= om[nm] - 1e-10
results.append(report(f"greedy disagreement => D_f(P||Q) >= omega_f for TV, KL, rKL, H2, chi2, JSD ({n_dis} cases)", ok))

# thm:greedy-seq (ii): omega_f is attained (cell reweighting at the argmin step), for every f
ok = True
delta = 1e-7
for _ in range(15):
    P = random_tree(3, 4, scale=rng.uniform(0.5, 2.5))
    Pl = P.seq_law()
    path = greedy_path(P)
    for nm in NAMES:
        vals = [merge_cost(nm, st["pi"] * st["p1"], st["pi"] * st["p2"], 1 - st["pi"] * (st["p1"] + st["p2"])) for st in path]
        t = int(np.argmin([v[0] for v in vals]))
        k, c = vals[t]
        st = path[t]
        P1, P2 = st["pi"] * st["p1"], st["pi"] * st["p2"]
        R = 1 - P1 - P2
        c1, c2 = c - delta, c + delta
        cell1, cell2 = st["s"] + (st["a"],), st["s"] + (st["b"],)
        Q = np.array([w * (c1 / P1 if y[:t + 1] == cell1 else c2 / P2 if y[:t + 1] == cell2 else (1 - c1 - c2) / R)
                      for y, w in zip(P.seqs, Pl)])
        Qm = tree_from_law(P, Q)
        ok &= Qm.greedy() != P.greedy() and abs(fdiv(nm, Pl, Q) - k) < 1e-4 * max(k, 1e-3)
results.append(report("omega_f is attained: a student at divergence omega_f (+o(1)) changes the greedy output", ok))

# TV and KL: omega = min_t pi_t * kappa_t, attained by changing one conditional only; hybrid sums coincide
ok = True
for _ in range(20):
    P = random_tree(3, 4, scale=rng.uniform(0.5, 2.5))
    Pl = P.seq_law()
    path = greedy_path(P)
    om_tv = min(st["pi"] * (st["p1"] - st["p2"]) / 2 for st in path)
    om_kl = min(st["pi"] * closed_form("KL", st["p1"], st["p2"], 0) for st in path)
    ok &= abs(om_tv - omega("TV", path)) < 1e-9 and abs(om_kl - omega("KL", path)) < 1e-7
    t = int(np.argmin([st["pi"] * (st["p1"] - st["p2"]) for st in path]))
    st = path[t]
    cond = P.cond.copy()
    row = cond[P.index[st["s"]]].copy()
    mv = (st["p1"] - st["p2"]) / 2 + 1e-9
    row[st["a"]] -= mv; row[st["b"]] += mv
    cond[P.index[st["s"]]] = row
    Qm = Tree(3, 4, cond)
    Ql = Qm.seq_law()
    e_off = sum(w * fdiv("TV", P.c(s), Qm.c(s)) for tt in range(4) for s, w in P.prefix_law(tt).items())
    e_on = sum(w * fdiv("TV", P.c(s), Qm.c(s)) for tt in range(4) for s, w in Qm.prefix_law(tt).items())
    ok &= Qm.greedy() != P.greedy() and abs(fdiv("TV", Pl, Ql) - om_tv) < 1e-8
    ok &= abs(e_off - om_tv) < 1e-8 and abs(e_on - om_tv) < 1e-8
results.append(report("TV/KL: omega = min_t pi_t kappa_t; single-state student attains it for TV_seq, T*e_off^TV, T*e_on^TV", ok))

# rKL closed form along the path: -log(1 - pi_t (sqrt p1 - sqrt p2)^2) <= pi_t kappa_rKL
ok = True
for _ in range(10):
    path = greedy_path(random_tree(3, 4))
    for st in path:
        lhs = merge_cost("rKL", st["pi"] * st["p1"], st["pi"] * st["p2"], 1 - st["pi"] * (st["p1"] + st["p2"]))[0]
        cf = -np.log(1 - st["pi"] * (np.sqrt(st["p1"]) - np.sqrt(st["p2"])) ** 2)
        ok &= abs(lhs - cf) < 1e-7 and cf <= st["pi"] * closed_form("rKL", st["p1"], st["p2"], 0) + TOL
results.append(report("rKL path threshold = -log(1 - pi_t (sqrt p1 - sqrt p2)^2) <= pi_t kappa_rKL", ok))

# prop:hard: disagreement => min_t q(a*_t|s*_t) <= 1/2 => Q(g_p(x)) <= 1/2
ok = True
for _ in range(3000):
    P = random_tree(3, 4, scale=2.0)
    Qm = Tree(3, 4, np.array([softmax(np.log(r) + rng.normal(scale=1.0, size=3)) for r in P.cond]))
    if Qm.greedy() != P.greedy():
        path = greedy_path(P)
        t = first_flip(P, Qm)
        ok &= Qm.c(path[t]["s"])[path[t]["a"]] <= 0.5 + TOL and Qm.prob(P.greedy()) <= 0.5 + TOL
results.append(report("greedy disagreement => q(a*_t|s*_t) <= 1/2 at the first flip => Q(g_p(x)) <= 1/2", ok))

# ---------------------------------------------------------------------------
# thm:knapsack  (exact worst case given a per-input threshold distribution)
# ---------------------------------------------------------------------------


def Phi(mu, om, eps):
    best = 0.0
    n = len(mu)
    for r in range(n + 1):
        for A in itertools.combinations(range(n), r):
            A = list(A)
            if np.sum(mu[A] * om[A]) <= eps + 1e-15:
                best = max(best, float(np.sum(mu[A])))
    return best


def Phibar_greedy(mu, om, eps):
    o = np.argsort(om, kind="stable")
    tot, spent = 0.0, 0.0
    for i in o:
        cost = mu[i] * om[i]
        if spent + cost <= eps:
            spent += cost; tot += mu[i]
        else:
            tot += mu[i] * (eps - spent) / cost if cost > 0 else mu[i]
            break
    return tot


def Phibar_dual(mu, om, eps):
    taus = [t for t in np.unique(om) if t > 0]
    vals = [eps / t + float(np.sum(mu * np.clip(1 - om / t, 0, None))) for t in taus] + [1.0]
    return min(vals)


ok_primal = ok_order = True
for _ in range(300):
    n = int(rng.integers(3, 10))
    mu = rng.dirichlet(np.ones(n))
    om = rng.uniform(0, 1, n) * (rng.random(n) > 0.1)
    eps = rng.uniform(0, 0.5) * float(np.sum(mu * om))
    ok_primal &= abs(Phibar_greedy(mu, om, eps) - Phibar_dual(mu, om, eps)) < 1e-9
    ok_primal &= Phibar_dual(mu, om, eps) <= min(eps / t + float(np.sum(mu * (om < t))) for t in (0.1, 0.3, 0.7)) + 1e-12
    ph, pb = Phi(mu, om, eps), Phibar_greedy(mu, om, eps)
    ok_order &= pb - mu.max() - 1e-12 <= ph <= pb + 1e-12
results += [report("fractional knapsack = dual formula inf_tau {eps/tau + E(1-omega/tau)_+} <= eps/tau + mu[omega<tau]", ok_primal),
            report("Phibar - max_x mu(x) <= Phi <= Phibar (0-1 knapsack by brute force)", ok_order)]

# end-to-end: many inputs, each with its own teacher; per-token errors on greedy paths; bound and attainment
ok_bound = ok_step = ok_attain = ok_unif = True
for name in ["TV", "KL"]:
    kap = lambda st: merge_cost(name, st["p1"], st["p2"], 1 - st["p1"] - st["p2"])[0]
    for _ in range(12):
        n = 7
        mu = rng.dirichlet(np.ones(n))
        teachers = [random_tree(3, 3, scale=rng.uniform(0.3, 2.0)) for _ in range(n)]
        paths = [greedy_path(P) for P in teachers]
        kt = np.array([[kap(st) for st in pth] for pth in paths])  # [input, step]
        kbar = kt.min(axis=1)
        for _ in range(25):
            dis, cost, costmax, per_t = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros((n, 3))
            for i, P in enumerate(teachers):
                Qm = Tree(3, 3, np.array([softmax(np.log(r) + rng.normal(scale=rng.uniform(0.0, 1.0), size=3)) for r in P.cond]))
                dis[i] = Qm.greedy() != P.greedy()
                per_t[i] = [fdiv(name, P.c(st["s"]), Qm.c(st["s"])) for st in paths[i]]
            cost, costmax = per_t.sum(axis=1), per_t.max(axis=1)
            pdis = float(np.sum(mu * dis))
            ok_bound &= pdis <= Phi(mu, kbar, float(np.sum(mu * costmax))) + 1e-12
            ok_bound &= Phi(mu, kbar, float(np.sum(mu * costmax))) <= Phibar_dual(mu, kbar, float(np.sum(mu * cost))) + 1e-12
            ok_step &= pdis <= sum(Phibar_dual(mu, kt[:, t], float(np.sum(mu * per_t[:, t]))) for t in range(3)) + 1e-12
        # attainment: flip exactly the optimal knapsack set, one state per input
        eps = 0.4 * float(np.sum(mu * kbar))
        best, bestA = -1, None
        for r in range(n + 1):
            for A in itertools.combinations(range(n), r):
                if np.sum(mu[list(A)] * kbar[list(A)]) <= eps and np.sum(mu[list(A)]) > best:
                    best, bestA = float(np.sum(mu[list(A)])), A
        eta = float(np.median(kbar))
        for target, budget in [(bestA, None), (tuple(i for i in range(n) if kbar[i] < eta), eta)]:
            dis, cost, costmax = np.zeros(n), np.zeros(n), np.zeros(n)
            for i, P in enumerate(teachers):
                cond = P.cond.copy()
                if i in target:
                    pth = paths[i]
                    t = int(np.argmin(kt[i]))
                    st = pth[t]
                    k, c = merge_cost(name, st["p1"], st["p2"], 1 - st["p1"] - st["p2"])
                    R = 1 - st["p1"] - st["p2"]
                    row = cond[P.index[st["s"]]].copy()
                    rest = [a for a in range(3) if a not in (st["a"], st["b"])]
                    row[rest] *= (1 - 2 * c) / R if R > 0 else 1.0
                    row[st["a"]], row[st["b"]] = c - 1e-10, c + 1e-10
                    cond[P.index[st["s"]]] = row
                Qm = Tree(3, 3, cond)
                dis[i] = Qm.greedy() != P.greedy()
                e = [fdiv(name, P.c(st["s"]), Qm.c(st["s"])) for st in paths[i]]
                cost[i], costmax[i] = sum(e), max(e)
            if budget is None:
                ok_attain &= abs(float(np.sum(mu * dis)) - best) < 1e-12 and float(np.sum(mu * cost)) <= eps + 1e-7
            else:
                ok_unif &= abs(float(np.sum(mu * dis)) - float(np.sum(mu * (kbar < eta)))) < 1e-12
                ok_unif &= costmax.max() <= eta + 1e-7
results += [report("L_dec <= Phi(E max_t e*_t) <= Phibar(E sum_t e*_t), TV and KL (random students)", ok_bound),
            report("L_dec <= sum_t Phibar_{kappa*_t}(E e*_t)  (per-step form)", ok_step),
            report("the knapsack value is attained by students that change one greedy state per input", ok_attain),
            report("uniform errors: students with max e* <= eta reach mu[kbar < eta]", ok_unif)]

# ---------------------------------------------------------------------------
# cor:rates  (margin-condition rates; continuous margin laws discretised finely)
# ---------------------------------------------------------------------------


def worst_case(thr, eps):
    """Phibar for the empirical law of thr (equal weights) via the fractional greedy."""
    s = np.sort(thr)
    cs = np.cumsum(s) / len(s)
    j = np.searchsorted(cs, eps, side="right")
    if j >= len(s):
        return 1.0
    prev = cs[j - 1] if j > 0 else 0.0
    return (j + (eps - prev) / (s[j] / len(s))) / len(s)


ok = True
ratios = []
for alpha in [0.5, 1.0, 2.0]:
    for M, name in [(400000, "TV"), (20000, "KL")]:
        gam = ((np.arange(M) + 1) / M) ** (1 / alpha)  # mu[gamma <= u] = floor(M u^alpha)/M <= u^alpha, i.e. C = 1
        thr = gam / 2 if name == "TV" else np.array([closed_form("KL", (1 + g) / 2, (1 - g) / 2, 0) for g in gam])
        for eps in [1e-4, 1e-3]:
            if name == "TV":
                pred = (2 * (1 + alpha) * eps / alpha) ** (alpha / (1 + alpha))
            else:
                pred = (2 * (2 + alpha) * eps / alpha) ** (alpha / (2 + alpha))
            wc = worst_case(thr, eps)  # Phibar of the discrete law; the 0-1 value is >= wc - 1/M
            ratios.append(wc / pred)
            ok &= wc <= pred * (1 + 1e-9) and wc - 1 / M >= pred * (1 - (1e-2 if name == "TV" else 6e-2))
results.append(report(f"margin-condition rates hold and are nearly attained (worst/bound in [{min(ratios):.3f}, {max(ratios):.3f}])", ok))

# independent (margin-blind) errors are linear for alpha = 1: P[flip] <= P[gamma <= 2 e]
gam = rng.random(200000)
e = rng.exponential(scale=1e-3, size=gam.size)
results.append(report("independent errors: P[flip] <= E F(2e) = 2 E e for uniform margins",
                      np.mean(gam <= 2 * e) <= 2 * np.mean(e) * 1.05))

# exponential separation: TV_seq = gamma/2 ((1+gamma)/2)^{T-1}, yet greedy outputs differ
g0, TT = 0.2, 12
P = Tree(2, TT, np.array([[(1 + g0) / 2, (1 - g0) / 2]] * sum(2 ** t for t in range(TT))))
cond = P.cond.copy()
cond[P.index[tuple([0] * (TT - 1))]] = [0.5 - 1e-12, 0.5 + 1e-12]
Qm = Tree(2, TT, cond)
tvs = fdiv("TV", P.seq_law(), Qm.seq_law())
results.append(report(f"exponential separation: TV = {tvs:.3e} = (gamma/2) pi_T while greedy outputs differ",
                      Qm.greedy() != P.greedy() and abs(tvs - g0 / 2 * ((1 + g0) / 2) ** (TT - 1)) < 1e-9))
print(f"      T=100, gamma=0.2: TV = {g0 / 2 * 0.6 ** 99:.2e}, KL = {closed_form('KL', 0.6, 0.4, 0) * 0.6 ** 99:.2e}, "
      f"converse student TV = 1 - 0.6^100 = {1 - 0.6 ** 100:.6f}")

# ---------------------------------------------------------------------------
# thm:beam  (beam search)
# ---------------------------------------------------------------------------


def beam_search(M, k):
    """Beams B_1..B_T and output; scores are prefix probabilities; ties broken lexicographically."""
    beams, traj, cand_hist = [()], [], []
    for t in range(M.T):
        cands = [b + (a,) for b in beams for a in range(M.V)]
        cands.sort(key=lambda c: (-M.prob(c), c))
        cand_hist.append(cands)
        beams = cands[:k]
        traj.append(frozenset(beams))
    out = sorted(beams, key=lambda c: (-M.prob(c), c))[0]
    return traj, out, cand_hist


def beam_gaps(P, k):
    traj, out, hist = beam_search(P, k)
    pairs = []
    for t in range(P.T):  # levels with at most k candidates cannot change: threshold +inf
        sc = sorted((P.prob(c) for c in hist[t]), reverse=True)
        pairs.append((sc[k - 1], sc[k]) if len(sc) > k else (np.inf, 0.0))
    fin = sorted((P.prob(c) for c in traj[-1]), reverse=True)
    pairs.append((fin[0], fin[1]) if len(fin) > 1 else (np.inf, 0.0))
    return pairs, traj, out


def first_deviation(P, Qm, k):
    tp, op, _ = beam_search(P, k)
    tq, oq, _ = beam_search(Qm, k)
    for t in range(P.T):
        if tp[t] != tq[t]:
            return t
    return P.T if op != oq else None


def prefix_div(name, P, Qm, t):
    lp, lq = P.prefix_law(t), Qm.prefix_law(t)
    return fdiv(name, [lp[y] for y in lp], [lq[y] for y in lp])


ok_f = ok_local = ok_out = ok_k1 = ok_hard = True
n_dev = 0
for _ in range(25):
    P = random_tree(3, 4, scale=rng.uniform(0.5, 2.0))
    k = int(rng.integers(1, 4))
    pairs, traj, out = beam_gaps(P, k)
    for _ in range(40):
        Qm = Tree(3, 4, np.array([softmax(np.log(r) + rng.normal(scale=rng.uniform(0.02, 1.0), size=3)) for r in P.cond]))
        t = first_deviation(P, Qm, k)
        _, oq, _ = beam_search(Qm, k)
        ok_out &= (oq == out) or (t is not None)
        ok_hard &= (oq == out) or Qm.prob(out) <= 0.5 + TOL
        if k == 1:
            ok_k1 &= (oq == P.greedy()) == (Qm.greedy() == P.greedy())
        if t is None or (k == 1 and t == P.T):
            continue
        n_dev += 1
        P1, P2 = pairs[t]
        ok_f &= np.isfinite(P1)
        lvl = min(t + 1, P.T)  # prefix level; level T+1 is the full sequence law
        for nm in NAMES:
            ok_f &= prefix_div(nm, P, Qm, lvl) >= merge_cost(nm, P1, P2, 1 - P1 - P2)[0] - 1e-10
        # beam-local form: sum_{j<=lvl} sum_{u in B_{j-1}} P(u) D(p(.|u) || q(.|u)) for D in {TV, KL}
        for nm in ["TV", "KL"]:
            loc = 0.0
            for j in range(lvl):
                states = [()] if j == 0 else list(traj[j - 1])
                loc += sum(P.prob(u) * fdiv(nm, P.c(u), Qm.c(u)) for u in states)
            ok_local &= loc >= merge_cost(nm, P1, P2, 1 - P1 - P2)[0] - 1e-10
results += [report(f"beam: output change => trajectory change; k=1 beam = greedy ({n_dev} deviations)", ok_out and ok_k1),
            report("beam: first deviation at level t => D_f(P_<=t || Q_<=t) >= k_f(P(c_k), P(c_k+1)) for all six f", ok_f),
            report("beam: ... and the beam-local sums of per-token TV and KL exceed the same thresholds", ok_local),
            report("beam (any width): outputs differ => Q(teacher's output) <= 1/2", ok_hard)]

# attainment for trajectories: reweight the cells c_(k), c_(k+1), rest at the argmin level, for every f
ok = True
for _ in range(12):
    P = random_tree(3, 4, scale=rng.uniform(0.5, 2.0))
    Pl = P.seq_law()
    k = int(rng.integers(1, 4))
    pairs, traj, out = beam_gaps(P, k)
    _, _, hist = beam_search(P, k)
    for nm in NAMES:
        vals = [merge_cost(nm, a, b, 1 - a - b) if np.isfinite(a) else (np.inf, None) for a, b in pairs]
        t = int(np.argmin([v[0] for v in vals]))
        kf, c = vals[t]
        P1, P2 = pairs[t]
        if t < P.T:
            c1, c2 = hist[t][k - 1], hist[t][k]
        else:
            fin = sorted(traj[-1], key=lambda cc: (-P.prob(cc), cc))
            c1, c2 = fin[0], fin[1]
        L, R = len(c1), 1 - P1 - P2
        Q = np.array([w * ((c - 1e-11) / P1 if y[:L] == c1 else (c + 1e-11) / P2 if y[:L] == c2 else (1 - 2 * c) / R)
                      for y, w in zip(P.seqs, Pl)])
        Qm = tree_from_law(P, Q)
        ok &= first_deviation(P, Qm, k) is not None and abs(fdiv(nm, Pl, Q) - kf) < 1e-9 + 1e-6 * kf
results.append(report("beam: the trajectory threshold min_t k_f(P+_t, P-_t) is attained, for all six f", ok))

# output-level threshold <= final merge cost (TV, KL): moving mass between the two best final hypotheses
ok = True
for _ in range(40):
    P = random_tree(3, 4, scale=rng.uniform(0.5, 2.0))
    k = int(rng.integers(2, 4))
    traj, out, _ = beam_search(P, k)
    fin = sorted(traj[-1], key=lambda cc: (-P.prob(cc), cc))
    y1, y2 = fin[0], fin[1]
    P1, P2 = P.prob(y1), P.prob(y2)
    for nm in ["TV", "KL"]:
        mv = (P1 - P2) / 2 + 1e-10
        Q = np.array([P.prob(y) - mv if y == y1 else P.prob(y) + mv if y == y2 else P.prob(y) for y in P.seqs])
        _, oq, _ = beam_search(tree_from_law(P, Q), k)
        ok &= oq != out and abs(fdiv(nm, P.seq_law(), Q) - merge_cost(nm, P1, P2, 1 - P1 - P2)[0]) < 1e-8
results.append(report("beam: equalising the two best final hypotheses always changes the output (TV, KL)", ok))

# the cut gap, not the output's own gaps, is what matters: a cheap level-1 swap changes the output
cond = np.zeros((1 + 3 + 9, 3))
ex = Tree(3, 2, cond)
ex.cond[ex.index[()]] = [0.36, 0.33, 0.31]
ex.cond[ex.index[(0,)]] = np.array([0.30, 0.059, 0.001]) / 0.36
ex.cond[ex.index[(1,)]] = np.array([0.20, 0.10, 0.03]) / 0.33
ex.cond[ex.index[(2,)]] = np.array([0.309, 0.0005, 0.0005]) / 0.31
pairs, traj, out = beam_gaps(ex, 2)
cut = min(a - b for a, b in pairs) / 2
own = min(ex.prob(out[:1]) - 0.31, ex.prob(out) - 0.10, ex.prob(out) - 0.20) / 2
mv = 0.01 + 1e-6
Q = np.array([ex.prob(y) * (1 - mv / 0.33 if y[0] == 1 else 1 + mv / 0.31 if y[0] == 2 else 1.0) for y in ex.seqs])
Qm = tree_from_law(ex, Q)
_, oq, _ = beam_search(Qm, 2)
results.append(report(f"beam example: output {out} -> {oq} at TV {mv:.4f} = cut threshold {cut:.4f} < own-gap threshold {own:.4f}",
                      oq != out and abs(cut - 0.01) < 1e-12 and abs(own - 0.025) < 1e-12
                      and abs(fdiv("TV", ex.seq_law(), Q) - mv) < 1e-12))

# ---------------------------------------------------------------------------
# zero-probability tokens: where the strict clause of lem:merge needs P^- > 0 or f(0) < inf
# ---------------------------------------------------------------------------
F0_FINITE = ["TV", "KL", "H2", "chi2", "JSD"]  # f(0) < inf; reverse KL has f(0) = inf

# lem:merge: P = (1/2, 0, 1/2) with S+ = {0}, S- = {1}. For reverse KL, k_f = log 2 is reached by a tie at
# zero mass, but every strict reversal has Q(1) > 0 = P(1) and infinite cost. For f(0) < inf, strict
# reversals cost k_f + o(1).
Pz = np.array([0.5, 0.0, 0.5])
k_r, c_r = merge_cost("rKL", 0.5, 0.0, 0.5)
ok = abs(k_r - np.log(2)) < 1e-9 and fdiv("rKL", Pz, np.array([0.5 - 1e-6, 1e-6, 0.5])) == np.inf
for nm in F0_FINITE:
    k, c = merge_cost(nm, 0.5, 0.0, 0.5)
    Qs = np.array([c - 1e-9, c + 1e-9, 1 - 2 * c]) if c > 1e-9 else np.array([0.0, 1e-9, 1 - 1e-9])
    ok &= Qs[1] > Qs[0] and fdiv(nm, Pz, Qs) <= k + 1e-6
results.append(report("merge lemma with P^- = 0: strict order costs k_f + o(1) iff f(0) < inf (fails for reverse KL)", ok))


def sparse_tree(V, T, zero_frac=0.3, det_frac=0.25):
    """Random teacher with some zero probabilities and some deterministic states."""
    n = sum(V ** t for t in range(T))
    rows = []
    for _ in range(n):
        z = rng.normal(scale=1.5, size=V)
        if rng.random() < det_frac:
            z = np.full(V, -np.inf); z[rng.integers(V)] = 0.0
        else:
            z[rng.random(V) < zero_frac] = -np.inf
            if np.isinf(z).all() or np.isfinite(z).sum() < 2:
                z[:2] = rng.normal(size=2)
        rows.append(np.exp(z - z[np.isfinite(z)].max()) / np.exp(z - z[np.isfinite(z)].max()).sum())
    return Tree(V, T, np.array(rows))


def support_student(P, scale):
    """Student with the teacher's support at every state (finite reverse KL)."""
    rows = []
    for r in P.cond:
        z = np.where(r > 0, np.log(np.where(r > 0, r, 1.0)) + rng.normal(scale=scale, size=len(r)), -np.inf)
        rows.append(np.exp(z - z.max()) / np.exp(z - z.max()).sum())
    return Tree(P.V, P.T, np.array(rows))


# thm:greedy-seq with deterministic steps and null tokens: lower bound for all six f; for reverse KL the
# minimum over non-deterministic steps equals omega, and a student reaches it there
ok_lb = ok_att = True
n_dis = 0
for _ in range(25):
    P = sparse_tree(3, 4)
    Pl = P.seq_law()
    path = greedy_path(P)
    vals = [merge_cost("rKL", st["pi"] * st["p1"], st["pi"] * st["p2"], 1 - st["pi"] * (st["p1"] + st["p2"])) for st in path]
    om = min(v[0] for v in vals)
    nondet = [t for t, st in enumerate(path) if st["p2"] > 0]
    if not nondet:
        ok_att &= om == np.inf
        continue
    ok_att &= abs(min(vals[t][0] for t in nondet) - om) < 1e-12 or om == np.inf
    for _ in range(40):
        Qm = support_student(P, rng.uniform(0.05, 1.5))
        if Qm.greedy() != P.greedy():
            n_dis += 1
            Ql = Qm.seq_law()
            for nm in NAMES:
                ok_lb &= fdiv(nm, Pl, Ql) >= omega(nm, path) - 1e-10
    t = min(nondet, key=lambda tt: vals[tt][0])
    st = path[t]
    k, c = vals[t]
    P1, P2 = st["pi"] * st["p1"], st["pi"] * st["p2"]
    R = 1 - P1 - P2
    cell1, cell2 = st["s"] + (st["a"],), st["s"] + (st["b"],)
    Q = np.array([w * ((c - 1e-11) / P1 if y[:t + 1] == cell1 else (c + 1e-11) / P2 if y[:t + 1] == cell2
                       else ((1 - 2 * c) / R if R > 0 else 0.0)) for y, w in zip(P.seqs, Pl)])
    Qm = tree_from_law(P, Q)
    ok_att &= Qm.greedy() != P.greedy() and abs(fdiv("rKL", Pl, Q) - k) < 1e-9 + 1e-6 * k
results += [report(f"deterministic steps and null tokens: D_f >= omega_f for all six f ({n_dis} disagreements)", ok_lb),
            report("reverse KL: a non-deterministic step attains omega_rKL, as in the proof of thm:greedy-seq", ok_att)]

# thm:beam with null candidates: lower bounds always hold; attainment for f(0) < inf; the reverse-KL
# counterexample (T=1, p=(0.9,0.1,0,0), k=3) where the cut lies between null candidates
ok_lb = ok_att = True
for _ in range(20):
    P = sparse_tree(3, 3, det_frac=0.1)
    Pl = P.seq_law()
    k = int(rng.integers(2, 4))
    pairs, traj, out = beam_gaps(P, k)
    _, _, hist = beam_search(P, k)
    for _ in range(30):
        Qm = support_student(P, rng.uniform(0.05, 1.5))
        t = first_deviation(P, Qm, k)
        if t is None:
            continue
        P1, P2 = pairs[t]
        lvl = min(t + 1, P.T)
        for nm in NAMES:
            ok_lb &= prefix_div(nm, P, Qm, lvl) >= merge_cost(nm, P1, P2, 1 - P1 - P2)[0] - 1e-10
    for nm in F0_FINITE:
        vals = [merge_cost(nm, a, b, 1 - a - b) if np.isfinite(a) else (np.inf, None) for a, b in pairs]
        t = int(np.argmin([v[0] for v in vals]))
        kf, c = vals[t]
        if not np.isfinite(kf):
            continue
        P1, P2 = pairs[t]
        if t < P.T:
            c1, c2 = hist[t][k - 1], hist[t][k]
        else:
            fin = sorted(traj[-1], key=lambda cc: (-P.prob(cc), cc))
            c1, c2 = fin[0], fin[1]
        L, R = len(c1), 1 - P1 - P2
        cont = lambda y: float(np.prod([P.c(y[:j])[y[j]] for j in range(L, P.T)]))  # teacher continuation law
        Q = np.array([(c - 1e-11) * cont(y) if y[:L] == c1 else (c + 1e-11) * cont(y) if y[:L] == c2
                      else (P.prob(y) * (1 - 2 * c) / R if R > 0 else 0.0) for y in P.seqs])
        Qm = tree_from_law(P, Q)
        ok_att &= first_deviation(P, Qm, k) is not None and abs(fdiv(nm, Pl, Q) - kf) < 1e-9 + 1e-6 * kf
results += [report("beam with null candidates: first deviation => D_f(P_<=t || Q_<=t) >= k_f for all six f", ok_lb),
            report("beam with null candidates: omega^(k)_f is attained when f(0) < inf", ok_att)]

ex = Tree(4, 1, np.array([[0.9, 0.1, 0.0, 0.0]]))
pairs, traj, out = beam_gaps(ex, 3)
om_r = min(merge_cost("rKL", a, b, 1 - a - b)[0] if np.isfinite(a) else np.inf for a, b in pairs)
best = np.inf
for q in np.linspace(0.0, 1.0, 20001):  # finite reverse KL forces the support {0, 1}
    Qm = Tree(4, 1, np.array([[q, 1 - q, 0.0, 0.0]]))
    if first_deviation(ex, Qm, 3) is not None:
        best = min(best, fdiv("rKL", [0.9, 0.1, 0.0, 0.0], [q, 1 - q, 0.0, 0.0]))
results.append(report(f"beam, reverse KL: omega^(3) = {om_r:.3f} but changing the trajectory costs {best:.3f} >= log(1/0.9)",
                      om_r == 0.0 and best >= np.log(1 / 0.9) - 1e-9))

# ---------------------------------------------------------------------------
# prop:spec  (greedy drafting versus speculative sampling)
# ---------------------------------------------------------------------------
ok = True
for _ in range(20000):
    V = int(rng.integers(2, 8))
    p = softmax(rng.normal(scale=2.0, size=V))
    q = softmax(np.log(p) + rng.normal(scale=rng.uniform(0.05, 2.0), size=V))
    a, b, p1, p2 = top2(p)
    flip = float(np.argmax(q) != a)
    ok &= flip <= min(1.0, 2 * fdiv("TV", p, q) / (p1 - p2)) + TOL
    ok &= abs(np.sum(np.minimum(p, q)) - (1 - fdiv("TV", p, q))) < 1e-12
results.append(report("greedy rejection <= min(1, 2 TV/gamma) = (2/gamma) x speculative rejection; acceptance = 1 - TV", ok))

print("\nALL PASS" if all(results) else "\nSOME CHECKS FAILED")
