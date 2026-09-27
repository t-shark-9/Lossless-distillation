"""Numerical checks for notes/optimal-agnostic-exponent.md.

Every claim in the note that can be checked on finite examples is checked here:

  1. psi_f(v) ~ 2 f''(1) v^2 (lower TV-profile of an f-divergence).
  2. The sandwich (1-t) psi^{-1}(K t) <= phi_f(t) <= psi^{-1}(K t): the explicit
     lower-bound instance really makes the D_f-minimiser the far model, and the
     upper bound survives a randomised search.
  3. Local version: full-support perturbations of the uniform law (all likelihood
     ratios in [1-delta, 1+delta]) already force TV(argmin D) ~ sqrt(t), for
     f-divergences (both KL directions included), squared l2, etc.
  4. Kinked generators (TV) are linear; C^{1,a} generators give t^{1/(1+a)}.
  5. Exact per-token chain rule holds for KL(P||Q) (teacher prefixes) and
     KL(Q||P) (student prefixes) and fails for JS / Hellinger / triangular.
  6. The sequence-level TV gradient estimator
         g(x) = -1{Q(x)<P(x)} Q(x)/P(x) * sum_t grad log q(x_t | x_<t),  x ~ P
     is unbiased.
  7. Horizon amplification for token-level surrogates: phi >~ sqrt(T K t / 2f'').
  8. Convexifying the model class (mixture segment) does not remove the sqrt.

Run:  python3 notes/verify_agnostic_exponent.py
"""

import itertools
import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar

rng = np.random.default_rng(0)
EPS = 1e-300


# ----------------------------------------------------------------------------
# f-divergences: D_f(P||Q) = sum_x Q(x) f(P(x)/Q(x)),  P = teacher, Q = student
# ----------------------------------------------------------------------------
class FDiv:
    def __init__(self, name, f, f0, fprime_inf, f2):
        self.name = name
        self.f = f              # generator on (0, inf), f(1) = 0
        self.f0 = f0            # f(0+)
        self.fpi = fprime_inf   # f'(inf) = lim f(u)/u
        self.f2 = f2            # f''(1)  (may be inf / None for kinked f)
        self.K = f0 + fprime_inf

    def __call__(self, P, Q):
        P = np.asarray(P, float)
        Q = np.asarray(Q, float)
        tot = 0.0
        for p, q in zip(P, Q):
            if q > 0 and p > 0:
                tot += q * self.f(p / q)
            elif q > 0:          # p = 0 < q
                tot += q * self.f0
            elif p > 0:          # q = 0 < p
                tot += p * self.fpi
        return float(tot)


def _xlogx(u):
    return u * math.log(u) if u > 0 else 0.0


def skew_kl(a):
    return FDiv(f"skewKL(a={a})", lambda u: _xlogx(u) - u * math.log(a * u + 1 - a),
                0.0, math.log(1 / a), (1 - a) ** 2)


def alpha_div(a):  # Amari alpha-divergence, a in (0,1): f'' (1) = 1
    return FDiv(f"alpha(a={a})", lambda u: (u ** a - 1 - a * (u - 1)) / (a * (a - 1)),
                1 / a, 1 / (1 - a), 1.0)


HELL = FDiv("Hellinger^2 (1-BC)", lambda u: 0.5 * (math.sqrt(u) - 1) ** 2, 0.5, 0.5, 0.25)
TRI = FDiv("triangular", lambda u: (u - 1) ** 2 / (u + 1), 1.0, 1.0, 1.0)
JS = FDiv("Jensen-Shannon", lambda u: 0.5 * (_xlogx(u) - (1 + u) * math.log((1 + u) / 2)),
          0.5 * math.log(2), 0.5 * math.log(2), 0.25)
TV = FDiv("TV", lambda u: 0.5 * abs(u - 1), 0.5, 0.5, None)
KL = FDiv("KL(P||Q)", lambda u: _xlogx(u) - u + 1, 1.0, math.inf, 1.0)
RKL = FDiv("KL(Q||P)", lambda u: -math.log(u) + u - 1, math.inf, 1.0, 1.0)
CHI2 = FDiv("chi^2", lambda u: (u - 1) ** 2, 1.0, math.inf, 2.0)

BOUNDED_SMOOTH = [HELL, TRI, JS, skew_kl(0.5), skew_kl(0.1), alpha_div(0.3)]


def tv(P, Q):
    return 0.5 * float(np.abs(np.asarray(P, float) - np.asarray(Q, float)).sum())


# ----------------------------------------------------------------------------
# psi_f(v) = inf{ D_f(P||Q) : TV(P,Q) = v }.  By Harremoes-Vajda the joint range of
# (TV, D_f) is the convex hull of its binary part, so psi_f is the lower convex
# envelope of the binary profile beta(v) = min_b D_f((b+v,1-b-v) || (b,1-b)).
# The map b -> D_f is convex (joint convexity), so a bounded scalar search is exact.
# ----------------------------------------------------------------------------
def binary_profile(D, v):
    if v <= 0:
        return 0.0, 0.5
    obj = lambda b: D([b + v, 1 - b - v], [b, 1 - b])
    r = minimize_scalar(obj, bounds=(0.0, 1.0 - v), method="bounded",
                        options={"xatol": 1e-13})
    cands = [(r.fun, r.x), (obj(0.0), 0.0), (obj(1.0 - v), 1.0 - v)]
    return min(cands)


class Psi:
    """Tabulated lower convex envelope of the binary profile, with inverse."""

    def __init__(self, D, n=4000):
        self.D = D
        vs = np.concatenate([np.geomspace(1e-7, 1e-2, n // 2), np.linspace(1e-2, 1.0, n // 2)])
        vs = np.unique(np.concatenate([[0.0], vs]))
        bs = np.array([binary_profile(D, v)[0] for v in vs])
        hull = [0]
        for i in range(1, len(vs)):  # lower convex hull (monotone chain)
            while len(hull) >= 2:
                i0, i1 = hull[-2], hull[-1]
                cross = (vs[i1] - vs[i0]) * (bs[i] - bs[i0]) - (bs[i1] - bs[i0]) * (vs[i] - vs[i0])
                if cross <= 0:
                    hull.pop()
                else:
                    break
            hull.append(i)
        self.v, self.val = vs[hull], bs[hull]

    def __call__(self, v):
        return float(np.interp(v, self.v, self.val))

    def inv(self, y):
        """sup{v : psi(v) <= y}."""
        if y >= self.val[-1]:
            return 1.0
        return float(np.interp(y, self.val, self.v))


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        raise SystemExit(1)


# ----------------------------------------------------------------------------
# 1. psi_f(v) ~ 2 f''(1) v^2
# ----------------------------------------------------------------------------
def part1():
    print("\n[1] lower TV-profile psi_f(v) / (2 f''(1) v^2) -> 1")
    for D in BOUNDED_SMOOTH:
        ratios = [binary_profile(D, v)[0] / (2 * D.f2 * v * v) for v in (1e-1, 1e-2, 1e-3)]
        print(f"     {D.name:22s} " + "  ".join(f"{r:.5f}" for r in ratios))
        check(abs(ratios[-1] - 1) < 1e-3, f"{D.name}: psi(v) ~ 2 f''(1) v^2")
    # closed forms
    for v in (0.05, 0.3, 0.7):
        check(abs(Psi_cache[HELL.name](v) - (1 - math.sqrt(1 - v * v))) < 1e-6,
              f"Hellinger psi({v}) = 1 - sqrt(1-v^2)")
        check(abs(Psi_cache[TRI.name](v) - 2 * v * v) < 1e-6, f"triangular psi({v}) = 2 v^2")


# ----------------------------------------------------------------------------
# 2. Sandwich bounds + the explicit lower-bound instance + random search
# ----------------------------------------------------------------------------
def lower_instance(D, t, v):
    """P=(1-t)P0 + t d_a, Q*=(1-t)P0 + t d_b, Qhat=(1-t)Q0 + t d_a with (P0,Q0) the
    binary psi-minimiser at TV v.  Coordinates: [x0, x1, a, b]."""
    _, b = binary_profile(D, v)
    P0, Q0 = np.array([b + v, 1 - b - v]), np.array([b, 1 - b])
    P = np.concatenate([(1 - t) * P0, [t, 0.0]])
    Qs = np.concatenate([(1 - t) * P0, [0.0, t]])
    Qh = np.concatenate([(1 - t) * Q0, [t, 0.0]])
    return P, Qs, Qh


def part2():
    print("\n[2] sandwich (1-t) psi^-1(K t/(1-t)) <= phi_f(t) <= psi^-1(K t)")
    for D in BOUNDED_SMOOTH:
        psi = Psi_cache[D.name]
        sharp = math.sqrt(D.K / (2 * D.f2))
        print(f"   {D.name}:  K_f = {D.K:.4f}, f''(1) = {D.f2:.4f}, "
              f"sqrt(K/2f'') = {sharp:.4f}")
        for t in (1e-2, 1e-3, 1e-4, 1e-5):
            lo_target = psi.inv(D.K * t / (1 - t))
            # realise the lower bound with a concrete pair slightly inside the threshold
            v = 0.999 * lo_target
            P, Qs, Qh = lower_instance(D, t, v)
            dS, dH = D(P, Qs), D(P, Qh)
            ok = dH < dS and abs(tv(P, Qs) - t) < 1e-12 and tv(P, Qh) > t
            lo, up = tv(P, Qh), psi.inv(D.K * t)
            print(f"     t={t:.0e}: TV*={tv(P, Qs):.1e}  D(Q*)={dS:.3e} > D(Qhat)={dH:.3e}"
                  f"  TV(Qhat)/sqrt t = {lo / math.sqrt(t):.4f}   upper/sqrt t = "
                  f"{up / math.sqrt(t):.4f}")
            check(ok and lo <= up + 1e-9, f"instance valid, lower <= upper (t={t:.0e})")
        check(abs(lo / math.sqrt(t) - sharp) < 5e-3 * sharp and abs(up / math.sqrt(t) - sharp) < 5e-3 * sharp,
              f"{D.name}: phi(t)/sqrt(t) -> sqrt(K/2f'') = {sharp:.4f}")

    print("   exact Hellinger sandwich sqrt(2t-3t^2) <= phi <= sqrt(2t-t^2):")
    for t in (0.2, 0.05, 1e-3):
        v = 0.9999 * math.sqrt(t * (2 - 3 * t)) / (1 - t)
        P, Qs, Qh = lower_instance(HELL, t, v)
        check(HELL(P, Qh) < HELL(P, Qs) and tv(P, Qh) > 0.999 * math.sqrt(2 * t - 3 * t * t),
              f"t={t}: Hellinger witness reaches sqrt(2t-3t^2) = {math.sqrt(2*t-3*t*t):.4f}")

    print("   randomised search for a counterexample to the upper bound:")
    worst = {D.name: 0.0 for D in BOUNDED_SMOOTH}
    for trial in range(3000):
        m = rng.integers(2, 7)
        P = rng.dirichlet(rng.choice([0.1, 0.5, 2.0]) * np.ones(m))
        cls = []
        for _ in range(rng.integers(2, 6)):
            kind = rng.integers(3)
            if kind == 0:
                Q = rng.dirichlet(np.ones(m))
            elif kind == 1:  # local perturbation of P
                Q = np.abs(P + 0.1 * rng.normal(size=m) * P)
                Q /= Q.sum()
            else:            # move a small chunk of mass (singular error)
                Q = P.copy()
                i, j = rng.choice(m, 2, replace=False)
                s = rng.uniform(0, P[i])
                Q[i] -= s
                Q[j] += s
            cls.append(Q)
        tstar = min(tv(P, Q) for Q in cls)
        for D in BOUNDED_SMOOTH:
            Qhat = min(cls, key=lambda Q: D(P, Q))
            bound = Psi_cache[D.name].inv(D.K * tstar)
            worst[D.name] = max(worst[D.name], tv(P, Qhat) - bound)
    for D in BOUNDED_SMOOTH:
        check(worst[D.name] < 1e-6, f"{D.name}: max(TV(Qhat) - psi^-1(K TV*)) = {worst[D.name]:.2e}")

    print("   unbounded generators (K = inf): no agnostic guarantee at all")
    for D in (KL, RKL, CHI2):
        t, eta = 1e-4, 1e-3
        # coordinates [x0, x1, a, b]
        P = np.array([1 - t - 1e-3, 1e-3, t, 0.0])
        Qs = np.array([1 - t - 1e-3, 1e-3, 0.0, t])                     # TV = t, D = inf
        Qh = (1 - eta) * np.array([0.0, 1.0, 0.0, 0.0]) + eta * P        # far away but D < inf
        check(D(P, Qs) == math.inf and D(P, Qh) < math.inf and tv(P, Qh) > 0.99,
              f"{D.name}: TV* = {t}, argmin has TV = {tv(P, Qh):.4f}")


# ----------------------------------------------------------------------------
# 3. Local, full-support version (bounded likelihood ratios)
# ----------------------------------------------------------------------------
def part3():
    print("\n[3] local construction: uniform P on n atoms, all ratios in [1-d, 1+d], d = 0.5")
    L2 = ("squared l2", lambda P, Q: float(((np.asarray(P) - np.asarray(Q)) ** 2).sum()))
    divs = [(D.name, D) for D in (KL, RKL, CHI2, HELL, JS, TRI)] + [L2]
    delta = 0.5
    for name, D in divs:
        row = []
        for n in (16, 256, 4096):
            P = np.full(n, 1.0 / n)
            Qs = P.copy()
            Qs[0] += delta / n
            Qs[1] -= delta / n
            h = np.array([1.0, -1.0] * (n // 2))
            dS = D(P, Qs)
            s = brentq(lambda s: D(P, P + s * (delta / n) * h) - dS, 1e-9, 1.0) \
                if D(P, P + (delta / n) * h) > dS else 1.0
            s *= 0.999
            Qh = P + s * (delta / n) * h
            t = tv(P, Qs)
            assert D(P, Qh) < dS
            row.append((n, t, tv(P, Qh)))
        print(f"     {name:20s} " + "  ".join(
            f"n={n}: TV*={t:.1e}, TV(argmin)={x:.3e} (x{x / t:5.1f})" for n, t, x in row))
        # sqrt law: TV(argmin)/TV* grows like sqrt(n/2), i.e. TV(argmin) ~ sqrt(delta * TV* / 2)
        n, t, x = row[-1]
        check(x / t > 0.5 * math.sqrt(n / 2), f"{name}: TV(argmin)/TV* grows like sqrt(n)")


# ----------------------------------------------------------------------------
# 4. Kinked / C^{1,a} generators
# ----------------------------------------------------------------------------
def part4():
    print("\n[4] exponent = 1/(order of contact at 1)")
    psi = Psi(TV, n=400)
    for t in (1e-2, 1e-4):
        check(abs(psi.inv(TV.K * t) - t) < 1e-9, f"TV: psi^-1(K t) = t at t={t}")

    def huber_pow(g, w=0.5):  # |u-1|^g near 1, extended linearly -> bounded (K < inf)
        def f(u):
            s = abs(u - 1)
            return s ** g if s <= w else w ** g + g * w ** (g - 1) * (s - w)
        slope = g * w ** (g - 1)
        return FDiv(f"|u-1|^{g}", f, f(0.0), slope, None)

    for g in (1.25, 1.5, 2.0, 3.0):
        D = huber_pow(g)
        psi = Psi(D, n=1200)
        ts = np.array([1e-5, 1e-6, 1e-7])
        ph = np.array([(1 - t) * psi.inv(D.K * t) for t in ts])
        slope = np.polyfit(np.log(ts), np.log(ph), 1)[0]
        print(f"     f(1+s) = |s|^{g}:  fitted exponent {slope:.3f}   (1/gamma = {1 / g:.3f})")
        check(abs(slope - 1 / g) < 0.02, f"gamma={g}: phi(t) ~ t^(1/gamma)")


# ----------------------------------------------------------------------------
# 5. Exact chain rule characterisation
# ----------------------------------------------------------------------------
def part5():
    print("\n[5] exact per-token chain rule D(P12||Q12) = D(P1||Q1) + E_R D(P2|x||Q2|x)")
    V = 3
    res = {}
    for D in (KL, RKL, JS, HELL, TRI):
        errP = errQ = 0.0
        for _ in range(200):
            P1, Q1 = rng.dirichlet(np.ones(V)), rng.dirichlet(np.ones(V))
            P2, Q2 = rng.dirichlet(np.ones(V), size=V), rng.dirichlet(np.ones(V), size=V)
            joint = D((P1[:, None] * P2).ravel(), (Q1[:, None] * Q2).ravel())
            step = [D(P2[x], Q2[x]) for x in range(V)]
            errP = max(errP, abs(joint - D(P1, Q1) - float(P1 @ step)))
            errQ = max(errQ, abs(joint - D(P1, Q1) - float(Q1 @ step)))
        res[D.name] = (errP, errQ)
        print(f"     {D.name:20s} teacher-prefix error {errP:.2e}   student-prefix error {errQ:.2e}")
    check(res[KL.name][0] < 1e-12 and res[RKL.name][1] < 1e-12, "KL: teacher prefixes; reverse KL: student prefixes")
    check(all(min(res[D.name]) > 1e-3 for D in (JS, HELL, TRI)), "bounded divergences have no exact chain rule")


# ----------------------------------------------------------------------------
# 6. Unbiased per-token gradient for sequence-level TV
# ----------------------------------------------------------------------------
def part6():
    print("\n[6] sequence-level TV has an unbiased, bounded-weight per-token gradient")
    V, T = 3, 3
    prefixes = [p for k in range(T) for p in itertools.product(range(V), repeat=k)]
    idx = {p: i for i, p in enumerate(prefixes)}
    tlogits = rng.normal(size=(len(prefixes), V)) * 1.5
    theta = rng.normal(size=(len(prefixes), V))

    def sm(z):
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    seqs = list(itertools.product(range(V), repeat=T))

    def seqprob(logits, x):
        return math.prod(sm(logits[idx[x[:k]]])[x[k]] for k in range(T))

    def TVfun(th):
        return 0.5 * sum(abs(seqprob(tlogits, x) - seqprob(th, x)) for x in seqs)

    def score(th, x):  # sum_t grad_theta log q(x_t | x_<t) for a tabular softmax student
        g = np.zeros_like(th)
        for k in range(T):
            i = idx[x[:k]]
            g[i] -= sm(th[i])
            g[i, x[k]] += 1.0
        return g

    Pw = np.array([seqprob(tlogits, x) for x in seqs])
    Qw = np.array([seqprob(theta, x) for x in seqs])
    w = np.where(Qw < Pw, Qw / Pw, 0.0)          # weights in [0, 1)
    exact_expect = sum(Pw[k] * (-w[k]) * score(theta, x) for k, x in enumerate(seqs))
    fd = np.zeros_like(theta)
    h = 1e-6
    for i in range(theta.shape[0]):
        for j in range(V):
            e = np.zeros_like(theta)
            e[i, j] = h
            fd[i, j] = (TVfun(theta + e) - TVfun(theta - e)) / (2 * h)
    err = np.abs(exact_expect - fd).max()
    # Monte Carlo from teacher samples
    N = 40000
    draws = rng.choice(len(seqs), size=N, p=Pw)
    mc = sum(np.bincount(draws, minlength=len(seqs))[k] * (-w[k]) * score(theta, x)
             for k, x in enumerate(seqs)) / N
    print(f"     max|E_P[g] - finite-diff grad TV| = {err:.2e};  max|MC(N={N}) - grad| = "
          f"{np.abs(mc - fd).max():.2e};  weights in [{w.min():.2f}, {w.max():.2f}]")
    check(err < 1e-7 and w.max() < 1, "estimator is unbiased and weights are bounded by 1")


# ----------------------------------------------------------------------------
# 7. Horizon amplification for token-level surrogates
# ----------------------------------------------------------------------------
def part7():
    print("\n[7] token-level surrogates  L(Q) = E_{x~P} sum_s D_f(P(.|x<s) || Q(.|x<s))")
    for D in (HELL, TRI):
        for T in (1, 4, 16, 64):
            t = 1e-5
            # first token over [x0, x1, a, b]; afterwards P continues with 'c' always.
            # Q* never emits a but, after a, emits 'd' != 'c' (unvisited by Q*, visited by P).
            # Qhat = diffuse error on the first token only.
            thr = D.K * T * t                     # L(Q*) = K t (step 1) + (T-1) K t
            psi = Psi_cache[D.name]
            v = 0.999 * psi.inv(thr / (1 - t))
            P, Qs, Qh = lower_instance(D, t, v)
            L_star = D(P, Qs) + (T - 1) * t * D([1.0, 0.0], [0.0, 1.0])
            L_hat = D(P, Qh)
            seq_tv_star, seq_tv_hat = tv(P, Qs), tv(P, Qh)   # later tokens agree
            pred = math.sqrt(T * D.K * t / (2 * D.f2))
            print(f"     {D.name:20s} T={T:3d}: TV*={seq_tv_star:.0e}  L(Q*)={L_star:.2e} > "
                  f"L(Qhat)={L_hat:.2e}   TV(Qhat)={seq_tv_hat:.4f}  sqrt(TKt/2f'')={pred:.4f}")
            check(L_hat < L_star and seq_tv_hat > 0.97 * pred, f"T={T}: horizon enters inside the root")


# ----------------------------------------------------------------------------
# 8. Mixture-closed model class
# ----------------------------------------------------------------------------
def part8():
    print("\n[8] convex class {(1-l) Q* + l Qhat}: minimiser still at TV ~ sqrt(t)")
    for D in (HELL, JS, TRI):
        A_half = 0.5 * D.f(2.0) + 0.5 * D.f0      # A(1/2) in the note
        for t in (1e-3, 1e-5):
            v = 0.999 * Psi_cache[D.name].inv(t * A_half / (1 - t))
            P, Qs, Qh = lower_instance(D, t, v)
            lams = np.linspace(0, 1, 20001)
            vals = [D(P, (1 - l) * Qs + l * Qh) for l in lams]
            l_hat = lams[int(np.argmin(vals))]
            Qm = (1 - l_hat) * Qs + l_hat * Qh
            tstar = min(tv(P, (1 - l) * Qs + l * Qh) for l in lams[::100])
            print(f"     {D.name:20s} t={t:.0e}: TV*={tstar:.1e}  argmin lambda={l_hat:.3f}  "
                  f"TV(argmin)/sqrt(t)={tv(P, Qm) / math.sqrt(t):.3f}")
            check(l_hat >= 0.5 and abs(tstar - t) < 1e-12, "minimiser has lambda >= 1/2")


if __name__ == "__main__":
    Psi_cache = {D.name: Psi(D) for D in BOUNDED_SMOOTH}
    part1()
    part2()
    part3()
    part4()
    part5()
    part6()
    part7()
    part8()
    print("\nall checks passed")
