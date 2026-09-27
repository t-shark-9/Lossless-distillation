"""Numerical sanity checks for the single-distribution results of Sections 1, 2 and 4.

Each check prints PASS/FAIL. Nothing here is a proof; it guards against
sign errors, wrong constants and wrong limits in the stated results.
Run:  python3 verification/check_divergences.py
"""
import numpy as np

rng = np.random.default_rng(0)
TOL = 1e-9


def kl(p, q):
    m = p > 0
    return float(np.sum(p[m] * (np.log(p[m]) - np.log(q[m]))))


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def jsd(p, q, b):
    m = b * p + (1 - b) * q
    return b * kl(p, m) + (1 - b) * kl(q, m)


def h(b):
    return -b * np.log(b) - (1 - b) * np.log(1 - b)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def rand_simplex(V, conc=1.0):
    return rng.dirichlet(conc * np.ones(V))


def report(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return ok


results = []

# prop:pinsker (Pinsker, Bretagnolle-Huber) and prop:jsd (JSD_beta sandwich, Jeffreys bound)
ok_p = ok_bh = ok_j1 = ok_j2 = ok_j3 = True
for _ in range(20000):
    V = rng.integers(2, 12)
    conc = 10 ** rng.uniform(-1.5, 1.0)
    p, q = rand_simplex(V, conc), rand_simplex(V, conc)
    p = np.clip(p, 1e-300, None); p /= p.sum()
    q = np.clip(q, 1e-300, None); q /= q.sum()
    b = rng.uniform(0.01, 0.99)
    T, K, R, J = tv(p, q), kl(p, q), kl(q, p), jsd(p, q, b)
    ok_p &= T <= np.sqrt(max(K, 0.0) / 2) + TOL
    ok_bh &= T <= np.sqrt(max(1 - np.exp(-K), 0.0)) + TOL
    ok_j1 &= J <= h(b) * T + TOL
    ok_j2 &= J >= 2 * b * (1 - b) * T ** 2 - TOL
    ok_j3 &= J <= b * (1 - b) * (K + R) + TOL
results += [report("Pinsker TV <= sqrt(KL/2)", ok_p),
            report("Bretagnolle-Huber TV <= sqrt(1-exp(-KL))", ok_bh),
            report("JSD_b <= h(b) TV", ok_j1),
            report("JSD_b >= 2 b(1-b) TV^2", ok_j2),
            report("JSD_b <= b(1-b)(KL+rKL)", ok_j3)]

# prop:jsd(iii): limits of JSD_beta
p, q = rand_simplex(6), rand_simplex(6)
b = 1e-6
lim0 = jsd(p, q, b) / b
lim1 = jsd(p, q, 1 - b) / b
results += [report("JSD_b / b -> KL(p||q) as b->0", abs(lim0 - kl(p, q)) < 1e-4),
            report("JSD_b / (1-b) -> KL(q||p) as b->1", abs(lim1 - kl(q, p)) < 1e-4)]

# prop:local: chi^2 sandwich under bounded likelihood ratio
ok = True
for _ in range(5000):
    V = rng.integers(2, 10)
    q = rand_simplex(V, 5.0)
    p = q * np.exp(rng.uniform(-0.7, 0.7, V)); p /= p.sum()
    u = p / q
    a, bb = u.min(), u.max()
    chi2 = float(np.sum(q * (u - 1) ** 2))
    K, R = kl(p, q), kl(q, p)
    ok &= chi2 / (2 * bb) - TOL <= K <= chi2 / (2 * a) + TOL
    ok &= chi2 / (2 * bb ** 2) - TOL <= R <= chi2 / (2 * a ** 2) + TOL
    ok &= K <= (bb ** 2 / a) * R + TOL
    ok &= R <= (bb / a ** 2) * K + TOL
    beta = rng.uniform(0.05, 0.95)
    Jb = jsd(p, q, beta)
    ok &= beta * (1 - beta) * chi2 / (2 * bb * (beta * bb + 1 - beta)) - TOL <= Jb
    ok &= Jb <= beta * (1 - beta) * chi2 / (2 * a * (beta * a + 1 - beta)) + TOL
results.append(report("chi^2 sandwich for KL, rKL, JSD_b; KL <= (b^2/a) rKL; rKL <= (b/a^2) KL", ok))

# prop:events: Q(A) <= (KL(Q||P)+log 2)/log(1/P(A)); Renyi probability preservation
ok = True
for _ in range(5000):
    V = rng.integers(3, 12)
    p, q = rand_simplex(V, 0.3), rand_simplex(V, 0.3)
    p = np.clip(p, 1e-12, None); p /= p.sum()
    q = np.clip(q, 1e-12, None); q /= q.sum()
    A = rng.random(V) < 0.4
    if not A.any() or A.all():
        continue
    PA, QA = p[A].sum(), q[A].sum()
    ok &= QA <= (kl(q, p) + np.log(2)) / np.log(1 / PA) + TOL
    ok &= PA <= (kl(p, q) + np.log(2)) / np.log(1 / QA) + TOL
    alpha = rng.uniform(1.1, 5.0)
    Dalpha = np.log(np.sum(q ** alpha * p ** (1 - alpha))) / (alpha - 1)
    ok &= QA <= (np.exp(Dalpha) * PA) ** ((alpha - 1) / alpha) + 1e-9
results.append(report("event bounds (coverage, precision, Renyi preservation)", ok))

# prop:grad: gradients wrt student logits
ok = True
for _ in range(200):
    V = rng.integers(2, 9)
    p = rand_simplex(V)
    z = rng.normal(size=V)
    b = rng.uniform(0.05, 0.95)
    q = softmax(z)
    g_fwd = q - p
    g_rev = q * (np.log(q / p) - kl(q, p))
    m = b * p + (1 - b) * q
    g_jsd = (1 - b) * q * (np.log(q / m) - np.sum(q * np.log(q / m)))
    for f, g in [(lambda zz: kl(p, softmax(zz)), g_fwd),
                 (lambda zz: kl(softmax(zz), p), g_rev),
                 (lambda zz: jsd(p, softmax(zz), b), g_jsd)]:
        num = np.array([(f(z + 1e-6 * e) - f(z - 1e-6 * e)) / 2e-6 for e in np.eye(V)])
        ok &= np.allclose(num, g, atol=1e-6)
results.append(report("logit gradients of fKL, rKL, JSD_b (finite differences)", ok))

# prop:convexity: reverse KL is non-convex in logits (V=2 example); forward-KL Hessian <= 1/2
pz = 0.3
f = lambda z: kl(softmax(np.array([z, 0.0])), np.array([pz, 1 - pz]))
zp = np.log(pz / (1 - pz))
z0 = zp + 3.0
d2 = (f(z0 + 1e-4) - 2 * f(z0) + f(z0 - 1e-4)) / 1e-8
results.append(report("reverse KL has negative curvature at z = z_p + 3", d2 < 0))
ok = True
for V in [3, 5, 10]:
    pv = rand_simplex(V)
    ray = lambda t: kl(softmax(np.r_[t, np.zeros(V - 1)]), pv)
    ok &= all((ray(t + 1e-3) - 2 * ray(t) + ray(t - 1e-3)) / 1e-6 < 0 for t in (6.0, 10.0))
    bj = 0.999
    rayj = lambda t: jsd(pv, softmax(np.r_[t, np.zeros(V - 1)]), bj) / (1 - bj)
    ok &= (rayj(10 + 1e-3) - 2 * rayj(10) + rayj(10 - 1e-3)) / 1e-6 < 0
results.append(report("reverse KL (V>2 ray) and JSD_0.999 have negative curvature", ok))
sg = lambda u: 1 / (1 + np.exp(-u))
k2 = lambda x, y: x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))
th = np.linspace(-6, 14, 200001)
fr = k2(sg(5.2 - th / 2), 0.79) + k2(sg(-1.5 - th / 3), 0.26)
ff = k2(0.79, sg(5.2 - th / 2)) + k2(0.26, sg(-1.5 - th / 3))
dr, df = np.diff(fr), np.diff(ff)
mins_r = np.where((dr[:-1] < 0) & (dr[1:] >= 0))[0] + 1
mins_f = np.where((df[:-1] < 0) & (df[1:] >= 0))[0] + 1
results.append(report(f"spurious minima: reverse KL minima at {np.round(th[mins_r], 2)}, forward KL at {np.round(th[mins_f], 2)}",
                      len(mins_r) == 2 and len(mins_f) == 1 and abs(fr[mins_r[0]] - 0.2191) < 1e-3
                      and abs(fr[mins_r[1]] - 0.2300) < 1e-3))
ok = True
for _ in range(2000):
    q = rand_simplex(rng.integers(2, 20), 10 ** rng.uniform(-1, 1))
    ok &= np.linalg.eigvalsh(np.diag(q) - np.outer(q, q)).max() <= 0.5 + TOL
results.append(report("Boehning bound lambda_max(diag(q)-qq^T) <= 1/2", ok))

# prop:temperature: high- and low-temperature limits
V = 7
v, z = rng.normal(size=V), rng.normal(size=V)
cen = lambda a: a - a.mean()
tau = 1e3
hi = tau ** 2 * kl(softmax(v / tau), softmax(z / tau))
mse = np.sum((cen(z) - cen(v)) ** 2) / (2 * V)
def kl_logits(a, c):  # KL(softmax(a) || softmax(c)) computed in log space
    la, lc = a - np.logaddexp.reduce(a), c - np.logaddexp.reduce(c)
    return float(np.sum(np.exp(la) * (la - lc)))


tau = 1e-3
lo = tau * kl_logits(v / tau, z / tau)
hinge = z.max() - z[np.argmax(v)]
results += [report("tau^2 KL -> ||z~ - v~||^2/(2V) as tau->inf", abs(hi - mse) / mse < 1e-2),
            report("tau KL -> max_j z_j - z_{argmax v} as tau->0", abs(lo - hinge) < 1e-2)]

# lem:logratio: KL(p||q) >= exp(-M)/2 Var_p(log p/q)
ok = True
for _ in range(5000):
    V = rng.integers(2, 10)
    p, q = rand_simplex(V, 2.0), rand_simplex(V, 2.0)
    d = np.log(p / q)
    M = max(d.max(), 0.0)
    var = float(np.sum(p * d ** 2) - np.sum(p * d) ** 2)
    ok &= kl(p, q) >= np.exp(-M) / 2 * var - TOL
results.append(report("KL >= e^{-M}/2 Var_p(log p/q)", ok))

# prop:sj: skew-Jeffreys picks the uniform student although inf TV = eta
eta, eps, lam = 0.01, 1e-120, 0.5
P2 = np.array([1 - eta, eta]); Q1 = np.array([1 - eps, eps]); U = np.array([0.5, 0.5])
J = lambda Q: (1 - lam) * kl(P2, Q) + lam * kl(Q, P2)
results.append(report(f"skew-Jeffreys example: J(Q1)={J(Q1):.3f} > J(U)={J(U):.3f}, TV(Q1)={tv(P2, Q1):.3f}, "
                      f"TV(U)={tv(P2, U):.3f}", J(Q1) > J(U) and tv(P2, U) > 0.45 and tv(P2, Q1) <= eta + 1e-12))

# prop:invisible: exact cost of perturbing logits on a set A, and the bound 1/2 sum p_i d_i^2 e^{(d_i)_+}
ok = True
for _ in range(5000):
    V = rng.integers(3, 12)
    p = rand_simplex(V, 0.5)
    v = np.log(p)
    A = rng.random(V) < 0.5
    d = np.where(A, rng.normal(scale=3.0, size=V), 0.0)
    exact = kl(p, softmax(v + d))
    formula = np.log(1 + np.sum(p[A] * (np.exp(d[A]) - 1))) - np.sum(p[A] * d[A])
    bound = 0.5 * np.sum(p[A] * d[A] ** 2 * np.exp(np.clip(d[A], 0, None)))
    ok &= abs(exact - formula) < 1e-8 and exact <= bound + 1e-12
results.append(report("logit-perturbation cost: exact formula and e^{(d)_+} bound", ok))

# prop:missing (iii): exact JSD pull formula and upper bound
ok = True
for _ in range(5000):
    V = rng.integers(2, 9)
    p = rand_simplex(V); z = rng.normal(scale=2.0, size=V); q = softmax(z)
    b = rng.uniform(0.05, 0.95); m = b * p + (1 - b) * q
    g = (1 - b) * q * (np.log(q / m) - np.sum(q * np.log(q / m)))
    c = kl(q, m)
    pull = (1 - b) * q * (np.log(1 - b + b * p / q) + c)
    ok &= np.allclose(-g, pull, atol=1e-12)
    miss = p > q
    ok &= np.all(pull[miss] <= b * (1 - b) * (p[miss] - q[miss]) + (1 - b) * q[miss] * np.log(1 / (1 - b)) + 1e-12)
    ok &= 0 <= c <= np.log(1 / (1 - b)) + 1e-12
results.append(report("JSD missing-mode pull: exact formula and upper bound", ok))

print("\nALL PASS" if all(results) else "\nSOME CHECKS FAILED")
