"""Part 1: f-divergence competitiveness constants (Theorem 1 of the note).

Checks, on random distributions:
  kappa_f * TV <= D_f <= (f(0) + f'(inf)) * TV     for kinked f with linear growth
and the two extremal constructions:
  - kinked f: ratio TV(q_hat)/OPT -> C*_f = (f(0)+f'(inf)) / kappa_f
  - JSD (smooth): TV(q_hat) -> sqrt(2 ln2 * OPT)
"""
import numpy as np

rng = np.random.default_rng(0)
LN2 = np.log(2.0)


def Df(p, q, f, fprime_inf):
    """D_f(p||q) = sum_x q f(p/q), with the convention q=0<p -> p * f'(inf)."""
    out = 0.0
    for pi, qi in zip(p, q):
        if qi > 0:
            out += qi * f(pi / qi)
        elif pi > 0:
            out += pi * fprime_inf
    return out


def tv(p, q):
    return 0.5 * np.abs(p - q).sum()


def f_tv(t):
    return 0.5 * abs(t - 1.0)


def make_f_beta(beta):
    # D_beta(p||q) = E_p[(1 - (q/p)^beta)_+]  <->  f(t) = (t - t^(1-beta)) 1{t>=1}
    return lambda t: (t - t ** (1.0 - beta)) if t >= 1.0 else 0.0


def f_js(t):
    if t == 0.0:
        return 0.5 * LN2
    return 0.5 * (t * np.log(2 * t / (1 + t)) + np.log(2 / (1 + t)))


def make_f_js_plus_tv(lam):
    return lambda t: f_js(t) + lam * f_tv(t)


# (name, f, f(0), f'(inf), kink kappa)
families = [("TV", f_tv, 0.5, 0.5, 1.0)]
for beta in (1.0, 0.5, 0.25):
    families.append((f"D_beta(beta={beta})", make_f_beta(beta), 0.0, 1.0, beta))
for lam in (1.0, 0.25):
    families.append((f"JSD+{lam}*TV", make_f_js_plus_tv(lam),
                     0.5 * LN2 + lam / 2, 0.5 * LN2 + lam / 2, lam))

print("== sandwich kappa*TV <= D_f <= (f(0)+f'(inf))*TV on random pairs ==")
for name, f, f0, finf, kappa in families:
    lo, hi = np.inf, 0.0
    for _ in range(3000):
        n = rng.integers(2, 12)
        p = rng.dirichlet(np.ones(n) * rng.uniform(0.1, 2))
        q = rng.dirichlet(np.ones(n) * rng.uniform(0.1, 2))
        if rng.random() < 0.3:  # force some support mismatch
            q[rng.integers(n)] = 0.0
            q /= q.sum()
        if rng.random() < 0.1:  # nearly equal pairs (probe the kink)
            q = p * np.exp(1e-3 * rng.standard_normal(n))
            q /= q.sum()
        t = tv(p, q)
        if t < 1e-12:
            continue
        r = Df(p, q, f, finf) / t
        lo, hi = min(lo, r), max(hi, r)
    print(f"{name:22s} min D/TV = {lo:.4f} (kappa = {kappa:.4f})   "
          f"max D/TV = {hi:.4f} (f0+f'inf = {f0 + finf:.4f})   "
          f"C* = {(f0 + finf) / kappa:.4f}")

print("\n== tightness of C*_f: argmin over {q_smooth, q_disjoint} ==")
# p on 4 atoms {a, b1, b2, c}; q' moves mass delta from a to fresh atom c
# (TV = OPT = delta); q perturbs b1/b2 by +-eta (TV = eta). argmin D_f
# picks q while D_f(q) <= D_f(q'); report the largest eta it would pick.
for name, f, f0, finf, kappa in families:
    delta = 1e-4
    p = np.array([delta, (1 - delta) / 2, (1 - delta) / 2, 0.0])
    qd = np.array([0.0, (1 - delta) / 2, (1 - delta) / 2, delta])
    Dd = Df(p, qd, f, finf)
    lo, hi = delta, 0.49
    for _ in range(200):  # bisection on eta for D(q_eta) = D(q')
        eta = 0.5 * (lo + hi)
        qs = np.array([delta, (1 - delta) / 2 + eta, (1 - delta) / 2 - eta, 0.0])
        if Df(p, qs, f, finf) <= Dd:
            lo = eta
        else:
            hi = eta
    print(f"{name:22s} TV(q_hat)/OPT = {lo / delta:.4f}   predicted C* = {(f0 + finf) / kappa:.4f}")

print("\n== smooth f (JSD): square-root law is exact ==")
for delta in (1e-2, 1e-3, 1e-4, 1e-5):
    p = np.array([delta, (1 - delta) / 2, (1 - delta) / 2, 0.0])
    qd = np.array([0.0, (1 - delta) / 2, (1 - delta) / 2, delta])
    Dd = Df(p, qd, f_js, 0.5 * LN2)
    lo, hi = 0.0, 0.49
    for _ in range(200):
        eta = 0.5 * (lo + hi)
        qs = np.array([delta, (1 - delta) / 2 + eta, (1 - delta) / 2 - eta, 0.0])
        if Df(p, qs, f_js, 0.5 * LN2) <= Dd:
            lo = eta
        else:
            hi = eta
    print(f"OPT = {delta:.0e}: TV(q_hat) = {lo:.5f}, sqrt(2 ln2 OPT) = {np.sqrt(2 * LN2 * delta):.5f}, "
          f"TV(q_hat)/OPT = {lo / delta:9.1f}")
