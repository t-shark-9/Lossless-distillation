"""Checks for Section 4.8 (fixed points of stop-gradient on-policy distillation).

Tiny autoregressive teacher/student pairs are enumerated exactly, so every expectation over prefixes is
computed without Monte Carlo. Stochastic updates (thm:fejer) are simulated with sampled roll-outs.
Run:  python3 verification/check_fixed_points.py
"""
import itertools
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize, minimize_scalar

np.seterr(all="ignore")
TOL = 1e-8


def report(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return bool(ok)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def kl(p, q):
    return float(np.sum(p * (np.log(p) - np.log(q))))


class Tree:
    """All prefixes of length 0..T-1 over V tokens for a single prompt, ordered by length."""

    def __init__(self, V, T):
        self.V, self.T = V, T
        self.prefixes = [p for t in range(T) for p in itertools.product(range(V), repeat=t)]
        self.index = {p: i for i, p in enumerate(self.prefixes)}
        self.N = len(self.prefixes)
        self.depth = np.array([len(p) for p in self.prefixes])
        self.child = {(self.index[p], a): self.index.get(p + (a,)) for p in self.prefixes for a in range(V)}

    def prefix_law(self, Q):
        d = np.zeros(self.N)
        d[0] = 1.0
        for i in range(self.N):
            for a in range(self.V):
                c = self.child[(i, a)]
                if c is not None:
                    d[c] += d[i] * Q[i, a]
        return d

    def next_values(self, Q, P):
        """Vn[i, a] = V_{t+1}(s a): reverse KL between continuation laws after extending prefix i by a."""
        Vs = np.zeros(self.N)
        for i in reversed(range(self.N)):
            Vs[i] = sum(Q[i, a] * (np.log(Q[i, a] / P[i, a]) + (Vs[c] if (c := self.child[(i, a)]) is not None else 0.0))
                        for a in range(self.V))
        return np.array([[Vs[c] if (c := self.child[(i, a)]) is not None else 0.0 for a in range(self.V)]
                         for i in range(self.N)])

    def seq_rkl(self, Q, P):
        d = self.prefix_law(Q)
        return float(sum(d[i] * kl(Q[i], P[i]) for i in range(self.N)))


def g_rkl(q, p):
    r = np.log(q / p)
    return q * (r - q @ r)


def g_fkl(q, p):
    return q - p


def make_g_jsd(b):
    def g(q, p):
        lm = np.log(q / (b * p + (1 - b) * q))
        return (1 - b) * q * (lm - q @ lm)
    return g


def loss_val(name, q, p, b=0.5):
    if name == "rkl":
        return kl(q, p)
    if name == "fkl":
        return kl(p, q)
    m = b * p + (1 - b) * q
    return b * kl(p, m) + (1 - b) * kl(q, m)


LOSSES = {"rkl": (g_rkl, 2.0), "fkl": (g_fkl, 2.0), "jsd": (make_g_jsd(0.5), 2.0 / (1 - 0.5))}
rng = np.random.default_rng(2026)
results = []

# ---------------------------------------------------------------------------------------------
# prop:tilt -- value-tilted teacher, covariance form of the bias, soft performance difference
# ---------------------------------------------------------------------------------------------
tree = Tree(3, 4)
N, V = tree.N, tree.V
Zp = rng.normal(scale=1.5, size=(N, V)); P = softmax(Zp)
Z = Zp + rng.normal(scale=0.8, size=(N, V)); Q = softmax(Z)
gF = np.zeros_like(Z)
for idx in np.ndindex(Z.shape):
    E = np.zeros_like(Z); E[idx] = 1e-6
    gF[idx] = (tree.seq_rkl(softmax(Z + E), P) - tree.seq_rkl(softmax(Z - E), P)) / 2e-6
d = tree.prefix_law(Q); Vn = tree.next_values(Q, P)
Pt = P * np.exp(-Vn); Pt /= Pt.sum(axis=1, keepdims=True)
g_tilt = np.array([d[i] * g_rkl(Q[i], Pt[i]) for i in range(N)])
v_sg = np.array([d[i] * g_rkl(Q[i], P[i]) for i in range(N)])
cov = np.array([d[i] * Q[i] * (Vn[i] - Q[i] @ Vn[i]) for i in range(N)])
spd = []
for _ in range(5):
    Q1, Q2 = softmax(Zp + rng.normal(size=(N, V))), softmax(Zp + rng.normal(size=(N, V)))
    Vn1 = tree.next_values(Q1, P); Pt1 = P * np.exp(-Vn1); Pt1 /= Pt1.sum(axis=1, keepdims=True)
    d2 = tree.prefix_law(Q2)
    rhs = sum(d2[i] * (kl(Q2[i], Pt1[i]) - kl(Q1[i], Pt1[i])) for i in range(N))
    spd.append(abs(tree.seq_rkl(Q2, P) - tree.seq_rkl(Q1, P) - rhs))
results += [report("prop:tilt(i): grad of seq. reverse KL = stop-gradient field with the value-tilted teacher",
                   np.abs(gF - g_tilt).max() < 1e-6),
            report("prop:tilt(ii): grad F - v = sum_t E Cov_q(grad log q, V_{t+1}); bias is nonzero here",
                   np.abs(gF - v_sg - cov).max() < 1e-6 and np.abs(cov).max() > 1e-3),
            report("prop:tilt(iii): soft performance-difference identity", max(spd) < 1e-12)]

# ---------------------------------------------------------------------------------------------
# prop:stable -- (a) bound, (b) suboptimality bound, (c) first-order identity
# ---------------------------------------------------------------------------------------------
tree = Tree(2, 3); N, V, T = tree.N, tree.V, tree.T
okA = okB = True
for trial in range(40):
    P = softmax(rng.normal(scale=1.5, size=(N, V)))
    lam = rng.choice([0.0, 0.5, 1.0]); name = rng.choice(["rkl", "fkl", "jsd"])
    cls = [softmax(np.log(P) + rng.normal(scale=1.0, size=(N, V))) for _ in range(8)]
    ell = [np.array([loss_val(name, Qc[i], P[i]) for i in range(N)]) for Qc in cls]
    nu = [lam * tree.prefix_law(Qc) + (1 - lam) * tree.prefix_law(P) for Qc in cls]
    dq = [tree.prefix_law(Qc) for Qc in cls]
    L = np.array([[nu[j] @ ell[i] for j in range(8)] for i in range(8)])     # L[i, j] = L(theta_i; theta_j)
    F = np.diag(L)
    eps_inf = min(e.max() for e in ell)
    for j in range(8):
        if L[j, j] <= L[:, j].min() + 1e-15:                                  # theta_j is stable in the class
            okA &= F[j] <= T * eps_inf + 1e-12
            for i in range(8):
                bound = F[i] + lam * sum((ell[i][tree.depth == t].max() - ell[i][tree.depth == t].min())
                                         * 0.5 * np.abs(dq[j][tree.depth == t] - dq[i][tree.depth == t]).sum()
                                         for t in range(T))
                okB &= F[j] <= bound + 1e-12
results += [report("prop:stable(a): F(stable) <= T eps_inf in random finite classes", okA),
            report("prop:stable(b): F(stable) <= F(theta) + lam sum_t osc_t TV(d_t) in random finite classes", okB)]
# (c) identity grad F - v = lam sum_s ell(s) grad d(s), forward KL, lam = 0.6, tabular student
lam = 0.6
P = softmax(rng.normal(size=(N, V))); Z = rng.normal(size=(N, V))


def Fmix(Zf):
    Qf = softmax(Zf)
    nu = lam * tree.prefix_law(Qf) + (1 - lam) * tree.prefix_law(P)
    return sum(nu[i] * kl(P[i], Qf[i]) for i in range(N))


Q = softmax(Z); nu = lam * tree.prefix_law(Q) + (1 - lam) * tree.prefix_law(P)
vv = np.array([nu[i] * (Q[i] - P[i]) for i in range(N)])
gFm = np.zeros_like(Z); gd = np.zeros((N,) + Z.shape)
for idx in np.ndindex(Z.shape):
    E = np.zeros_like(Z); E[idx] = 1e-6
    gFm[idx] = (Fmix(Z + E) - Fmix(Z - E)) / 2e-6
    gd[(slice(None),) + idx] = (tree.prefix_law(softmax(Z + E)) - tree.prefix_law(softmax(Z - E))) / 2e-6
ellv = np.array([kl(P[i], Q[i]) for i in range(N)])
results.append(report("prop:stable(c): grad F - v = lam sum_s ell(s) grad d_t(s) (forward KL, lam=0.6)",
                      np.abs(gFm - vv - lam * np.einsum("s,sij->ij", ellv, gd)).max() < 1e-6))

# ---------------------------------------------------------------------------------------------
# thm:fp-real -- realisable finite classes: stable points are exactly the lossless members
# ---------------------------------------------------------------------------------------------
ok = True
for trial in range(30):
    P = softmax(rng.normal(scale=1.5, size=(N, V)))
    lam = rng.choice([0.0, 0.3, 1.0]); name = rng.choice(["rkl", "fkl", "jsd"])
    cls = [P.copy()] + [softmax(np.log(P) + rng.normal(scale=rng.uniform(0.2, 2), size=(N, V))) for _ in range(7)]
    ell = [np.array([loss_val(name, Qc[i], P[i]) for i in range(N)]) for Qc in cls]
    nu = [lam * tree.prefix_law(Qc) + (1 - lam) * tree.prefix_law(P) for Qc in cls]
    L = np.array([[nu[j] @ ell[i] for j in range(8)] for i in range(8)])
    stable = [j for j in range(8) if L[j, j] <= L[:, j].min() + 1e-15]
    ok &= stable == [0]
    ok &= all(int(np.argmin(L[:, j])) == 0 for j in range(8))          # exact DAgger: one step to the teacher
results.append(report("thm:fp-real(i,ii): realisable classes: stable = lossless; exact DAgger reaches it in one step", ok))

# ---------------------------------------------------------------------------------------------
# lem:directed -- teacher-directed logit gradients
# ---------------------------------------------------------------------------------------------
ok_id, ratios = True, {"rkl": np.inf, "fkl": np.inf, "jsd": np.inf}
for _ in range(20000):
    Vv = int(rng.integers(2, 7)); b = rng.uniform(0.01, 0.99)
    vt = rng.normal(scale=rng.uniform(0.1, 4), size=Vv); z = rng.normal(scale=rng.uniform(0.1, 6), size=Vv)
    p, q = softmax(vt), softmax(z); r = np.log(q / p)
    gr, gf, gj = g_rkl(q, p), g_fkl(q, p), make_g_jsd(b)(q, p)
    lm = np.log(q / (b * p + (1 - b) * q))
    ok_id &= abs((z - vt) @ gr - (q @ r**2 - (q @ r) ** 2)) < 1e-8 * max(1, q @ r**2)
    ok_id &= abs((z - vt) @ gf - (kl(p, q) + kl(q, p))) < 1e-8 * max(1, kl(p, q))
    ok_id &= abs((z - vt) @ gj - (1 - b) * (q @ (r * lm) - (q @ r) * (q @ lm))) < 1e-8
    for k, g, c in [("rkl", gr, 2.0), ("fkl", gf, 2.0), ("jsd", gj, 2.0 / (1 - b))]:
        if g @ g > 1e-14:
            ratios[k] = min(ratios[k], ((z - vt) @ g) / (c * (g @ g)))
results += [report("lem:directed: <z-v,g> = Var_q r, Jeffreys, (1-beta) Cov_q(r, log q/m)", ok_id),
            report("lem:directed: <z-v,g> >= c_l |g|^2 with c_l = 2, 2, 2/(1-beta)  (min ratio %.4f)" % min(ratios.values()),
                   min(ratios.values()) >= 1 - 1e-9)]

# ---------------------------------------------------------------------------------------------
# thm:fejer -- logit-affine realisable students: pathwise Fejer monotonicity, no violations
# ---------------------------------------------------------------------------------------------
tree = Tree(3, 3); N, V, T = tree.N, tree.V, tree.T
C = np.eye(V) - np.ones((V, V)) / V
Pd = 40                                                  # (V-1) N = 26 <= 40: overparameterised
Psi = rng.normal(size=(N, V, Pd)) / np.sqrt(Pd); b0 = rng.normal(size=(N, V))
th_o = rng.normal(size=Pd) * 1.5
Pm = softmax(np.einsum("svp,p->sv", Psi, th_o) + b0)
J = np.concatenate([C @ Psi[s] for s in range(N)]); Jp = np.linalg.pinv(J)
ok_rank = np.linalg.matrix_rank(J) == (V - 1) * N
Lam = T * max(np.linalg.norm(C @ Psi[s], 2) ** 2 for s in range(N))
viol, steps = 0, 0
for lname, (gfun, c) in LOSSES.items():
    for lam in [1.0, 0.5, 0.0]:
        th = rng.normal(size=Pd) * 2
        others = [th_o] + [th_o + (np.eye(Pd) - Jp @ J) @ rng.normal(size=Pd) * 3 for _ in range(2)]
        for k in range(300):
            Q = softmax(np.einsum("svp,p->sv", Psi, th) + b0)
            vhat = np.zeros(Pd)
            for _ in range(2):                                   # batch of B = 2 trajectories
                Pol = Q if rng.random() < lam else Pm
                pref = ()
                for t in range(T):
                    i = tree.index[pref]
                    vhat += Psi[i].T @ gfun(Q[i], Pm[i]) / 2
                    pref = pref + (int(rng.choice(V, p=Pol[i])),)
            th_new = th - (c / Lam) * vhat
            viol += sum(np.linalg.norm(th_new - o) > np.linalg.norm(th - o) + 1e-12 for o in others)
            steps += 1
            th = th_new
results.append(report(f"thm:fejer(i): pathwise Fejer monotonicity, stochastic updates, 3 losses x 3 mixtures "
                      f"({steps} steps, {viol} violations; J full row rank: {ok_rank})", viol == 0 and ok_rank))

# thm:fejer(iii): exact updates converge to proj_M(theta_0) for every loss and mixture weight
tree = Tree(2, 2); N, V, T = tree.N, tree.V, tree.T
C = np.eye(V) - np.ones((V, V)) / V
Pd = 5                                                   # (V-1) N = 3 <= 5
Psi = rng.normal(size=(N, V, Pd)) / np.sqrt(Pd); b0 = rng.normal(size=(N, V)) * 0.5; th_o = rng.normal(size=Pd)
Pm = softmax(np.einsum("svp,p->sv", Psi, th_o) + b0)
J = np.concatenate([C @ Psi[s] for s in range(N)]); Jp = np.linalg.pinv(J)
th0 = rng.normal(size=Pd) * 2; proj = th0 - Jp @ (J @ (th0 - th_o))
Lam = T * max(np.linalg.norm(C @ Psi[s], 2) ** 2 for s in range(N))
errs = []
for lname, (gfun, c) in LOSSES.items():
    for lam in [1.0, 0.5, 0.0]:
        th = th0.copy()
        for k in range(20000):
            Q = softmax(np.einsum("svp,p->sv", Psi, th) + b0)
            nu = lam * tree.prefix_law(Q) + (1 - lam) * tree.prefix_law(Pm)
            th = th - (c / Lam) * sum(nu[i] * Psi[i].T @ gfun(Q[i], Pm[i]) for i in range(N))
        errs.append(np.linalg.norm(th - proj))
results.append(report("thm:fejer(iii): exact stop-gradient descent converges to proj_M(theta_0) for all losses and "
                      "mixtures (max error %.1e)" % max(errs), max(errs) < 1e-6))

# ---------------------------------------------------------------------------------------------
# rem:not-lyapunov -- F increases along the stop-gradient flow for a tabular realisable student
# ---------------------------------------------------------------------------------------------
tree = Tree(2, 2); N = tree.N
P = np.full((N, 2), 0.5)
Q = np.array([[0.6, 0.4], [0.5, 0.5], [1 - 1e-4, 1e-4]])
d = tree.prefix_law(Q); Vn = tree.next_values(Q, P)
Pt = P * np.exp(-Vn); Pt /= Pt.sum(axis=1, keepdims=True)
dFdt = -sum(d[i] * g_rkl(Q[i], Pt[i]) @ (d[i] * g_rkl(Q[i], P[i])) for i in range(N))
results.append(report("rem:not-lyapunov: dF/dt = %.4f > 0 along the stop-gradient flow; tilted p~(a|x) = %.4f"
                      % (dFdt, Pt[0, 0]), dFdt > 0.01 and abs(Pt[0, 0] - 2 / 3) < 1e-3))

# ---------------------------------------------------------------------------------------------
# rem:spurious -- strict non-lossless local minimum of the sequence-level reverse KL (logit-affine)
# ---------------------------------------------------------------------------------------------
psi = np.array([[2.0, 3.0], [-3.0, -2.5], [3.0, -3.0]]); beta = np.array([0.0, 1.5, -2.0])
logsig = lambda z: -np.logaddexp(0, -z)


def F_and_fields(th):
    z = psi @ th + beta
    q = 1 / (1 + np.exp(-z))
    k = q * (logsig(z) - logsig(beta)) + (1 - q) * (logsig(-z) - logsig(-beta))
    dk = q * (1 - q) * (z - beta)
    Fv = k[0] + (1 - q[0]) * k[1] + q[0] * k[2]
    v = psi[0] * dk[0] + (1 - q[0]) * psi[1] * dk[1] + q[0] * psi[2] * dk[2]
    gradF = v + psi[0] * q[0] * (1 - q[0]) * (k[2] - k[1])
    return Fv, gradF, v, q


res = minimize(lambda th: F_and_fields(th)[:2], np.array([1.2, -1.5]), jac=True, method="BFGS", options={"gtol": 1e-13})
ts = res.x; F0, g0, v0, q0 = F_and_fields(ts)
circle = min(F_and_fields(ts + rad * np.array([np.cos(a), np.sin(a)]))[0]
             for rad in [1e-2, 5e-2, 0.2] for a in np.linspace(0, 2 * np.pi, 360, endpoint=False))
th = ts + 1e-2 * rng.normal(size=2)
for _ in range(20000):
    th -= 0.05 * F_and_fields(th)[1]
stays = np.linalg.norm(th - ts) < 1e-6
th = ts.copy()
for _ in range(20000):
    th -= 0.05 * F_and_fields(th)[2]
results.append(report("rem:spurious: strict local min of F at (%.3f, %.3f), F=%.3f, |v|=%.2f, q(1|x)=%.2f; GD on F "
                      "stays, stop-gradient reaches F=%.1e" % (ts[0], ts[1], F0, np.linalg.norm(v0), q0[0], F_and_fields(th)[0]),
                      np.linalg.norm(g0) < 1e-7 and circle > F0 and abs(F0 - 0.579) < 1e-3 and stays
                      and F_and_fields(th)[0] < 1e-10))

# ---------------------------------------------------------------------------------------------
# prop:local-conv -- nonlinear overparameterised student: Dv = f''(1) H = Hess F at a lossless point
# ---------------------------------------------------------------------------------------------
tree = Tree(2, 3); N, V = tree.N, tree.V
C = np.eye(V) - np.ones((V, V)) / V
rng_nl = np.random.default_rng(7)                      # a well-conditioned instance (cond(H) ~ 2e2)
Hd, Fd = 6, 4
phi = rng_nl.normal(size=(N, Fd))
Pdim = Hd * Fd + V * Hd


def logits(th):
    W1 = th[:Hd * Fd].reshape(Hd, Fd); W2 = th[Hd * Fd:].reshape(V, Hd)
    return np.tanh(phi @ W1.T) @ W2.T


def jac(th):
    """Analytic Jacobian d logits(s)_k / d theta, shape (N, V, Pdim)."""
    W1 = th[:Hd * Fd].reshape(Hd, Fd); W2 = th[Hd * Fd:].reshape(V, Hd)
    A = np.tanh(phi @ W1.T)                                   # (N, Hd)
    Jf = np.zeros((N, V, Pdim))
    Jf[:, :, :Hd * Fd] = np.einsum("kh,sh,sf->skhf", W2, 1 - A**2, phi).reshape(N, V, Hd * Fd)
    for k in range(V):
        Jf[:, k, Hd * Fd + k * Hd: Hd * Fd + (k + 1) * Hd] = A
    return Jf


th_o = 0.5 * rng_nl.normal(size=Pdim)
Pm = softmax(logits(th_o))


def v_nl(th, lam, gfun):
    Q = softmax(logits(th)); nu = lam * tree.prefix_law(Q) + (1 - lam) * tree.prefix_law(Pm)
    Jf = jac(th)
    return sum(nu[i] * Jf[i].T @ gfun(Q[i], Pm[i]) for i in range(N))


Jo = jac(th_o)
Jfd = np.stack([(logits(th_o + 1e-6 * e) - logits(th_o - 1e-6 * e)) / 2e-6 for e in np.eye(Pdim)], axis=-1)
rank_ok = (np.linalg.matrix_rank(np.concatenate([C @ Jo[i] for i in range(N)]), tol=1e-6) == (V - 1) * N
           and np.abs(Jo - Jfd).max() < 1e-8)
dp = tree.prefix_law(Pm)
H = sum(dp[i] * Jo[i].T @ (np.diag(Pm[i]) - np.outer(Pm[i], Pm[i])) @ Jo[i] for i in range(N))
ok = rank_ok
for lam, (gfun, f2) in [(1.0, (g_rkl, 1.0)), (0.5, (g_fkl, 1.0)), (0.0, (make_g_jsd(0.3), 0.3 * 0.7))]:
    Dv = np.array([(v_nl(th_o + 1e-5 * e, lam, gfun) - v_nl(th_o - 1e-5 * e, lam, gfun)) / 2e-5 for e in np.eye(Pdim)]).T
    ok &= np.abs(Dv - f2 * H).max() < 1e-5
Fseq = lambda th: tree.seq_rkl(softmax(logits(th)), Pm)
h = 1e-3
HF = np.array([[(Fseq(th_o + h * (ea + ec)) - Fseq(th_o + h * (ea - ec)) - Fseq(th_o - h * (ea - ec)) + Fseq(th_o - h * (ea + ec)))
                / (4 * h * h) for ec in np.eye(Pdim)] for ea in np.eye(Pdim)])
ok &= np.abs(HF - H).max() < 1e-4
# (iii) local linear convergence of the exact iteration; the asymptotic factor of F is rho^2 (F ~ dist^2)
evH = np.linalg.eigvalsh(H); pos = evH[evH > 1e-8]
eta = 1.0 / pos.max()
rho = max(abs(1 - eta * hh) for hh in pos)
th = th_o + 0.05 * rng_nl.normal(size=Pdim)
Fs = []
for _ in range(1500):
    th = th - eta * v_nl(th, 1.0, g_rkl)
    Fs.append(Fseq(th))
r_emp = (Fs[-1] / Fs[-501]) ** (1 / 500)
ok &= len(pos) == (V - 1) * N and Fs[-1] < 1e-5 * Fs[0] and abs(r_emp - rho**2) < 5e-4
print("      nonlinear student: %d params, positive spectrum of H has %d elements; asymptotic factor of F %.6f vs rho^2 %.6f"
      % (Pdim, len(pos), r_emp, rho**2))
results.append(report("prop:local-conv: surjective Jacobian; Dv = f''(1) H (3 losses, 3 mixtures) = Hess F; local convergence", ok))

# ---------------------------------------------------------------------------------------------
# prop:contraction -- ||Phi(t1) - Phi(t2)|| <= kappa ||t1 - t2|| (forward KL + weight decay, misspecified)
# ---------------------------------------------------------------------------------------------
tree = Tree(3, 3); N, V, T = tree.N, tree.V, tree.T
Pd, gam, lam = 3, 0.5, 1.0
Psi = rng.normal(size=(N, V, Pd)) * 0.6; b0 = rng.normal(size=(N, V)) * 0.3
Pm = softmax(rng.normal(size=(N, V)))


def Lreg(thp, th):
    nu = lam * tree.prefix_law(softmax(np.einsum("svp,p->sv", Psi, th) + b0)) + (1 - lam) * tree.prefix_law(Pm)
    Qp = softmax(np.einsum("svp,p->sv", Psi, thp) + b0)
    return (sum(nu[i] * kl(Pm[i], Qp[i]) for i in range(N)) + gam / 2 * thp @ thp,
            sum(nu[i] * Psi[i].T @ (Qp[i] - Pm[i]) for i in range(N)) + gam * thp)


Phi = lambda th: minimize(lambda x: Lreg(x, th), np.zeros(Pd), jac=True, method="BFGS", options={"gtol": 1e-12}).x
LamTV = max(np.linalg.norm(Psi[s], 2) for s in range(N)) / (2 * np.sqrt(2))
worst = 0.0
for _ in range(12):
    t1, t2 = rng.normal(size=Pd) * 2, rng.normal(size=Pd) * 2
    p1, p2 = Phi(t1), Phi(t2)
    Qp = softmax(np.einsum("svp,p->sv", Psi, p2) + b0)
    G = np.array([Psi[i].T @ (Qp[i] - Pm[i]) for i in range(N)])
    rad = [np.linalg.norm(G[tree.depth == t] - G[tree.depth == t].mean(0), axis=1).max() for t in range(T)]
    kap = 2 * lam * LamTV * sum(t * rad[t] for t in range(T)) / gam          # (t-1) in 1-based indexing
    worst = max(worst, np.linalg.norm(p1 - p2) / (kap * np.linalg.norm(t1 - t2)))
results.append(report("prop:contraction: ||Phi(t1)-Phi(t2)|| <= kappa ||t1-t2|| (worst ratio %.3f)" % worst, worst <= 1.0))

# ---------------------------------------------------------------------------------------------
# prop:separable(iii) -- the stable point is far from the F-minimiser, yet better in TV
# ---------------------------------------------------------------------------------------------
ok = True
for logdelta, tau in [(-6.0, 0.1), (-40.0, 0.1)]:
    delta = 10.0 ** logdelta
    pb = np.array([1 - delta, delta]); qb = np.array([1 - tau, tau])
    eps = (1 - tau) * (np.log(1 - tau) - np.log1p(-delta)) + tau * (np.log(tau) - logdelta * np.log(10))
    F_stab = 0.5 * eps
    Fw = lambda w: w * np.log(2 * w) + (1 - w) * np.log(2 * (1 - w)) + (1 - w) * eps     # step-1 law (w, 1-w) on (a, b)
    wopt = minimize_scalar(Fw, bounds=(1e-12, 1 - 1e-12), method="bounded", options={"xatol": 1e-14}).x
    ok &= abs(Fw(wopt) - np.log(2 / (1 + np.exp(-eps)))) < 1e-9
    tv_stab = 0.25 * np.abs(pb - qb).sum()
    tau_exact = 0.5 * np.abs(pb - qb).sum()                                    # TV(q_b, p_b)
    tv_star = 0.5 * (abs(0.5 - wopt) + np.abs(0.5 * pb - (1 - wopt) * qb).sum())
    ok &= abs(tv_stab - tau_exact / 2) < 1e-15 and tv_star >= 0.5 * np.tanh(eps / 2) - 1e-9 and F_stab > Fw(wopt)
    if logdelta == -40.0:
        ok &= abs(eps - 8.885) < 1e-3 and abs(F_stab - 4.443) < 1e-3 and abs(Fw(wopt) - 0.693) < 1e-3 and tv_star > 0.4998
        print("      delta=1e-40: eps=%.4f F(stable)=%.4f minF=%.4f TV(stable)=%.4f TV(F-min)=%.4f"
              % (eps, F_stab, Fw(wopt), tv_stab, tv_star))
results.append(report("prop:separable(iii): F(stable)=eps/2 > minF=log(2/(1+e^-eps)); TV(stable)=tau/2 < TV(F-min)", ok))

# ---------------------------------------------------------------------------------------------
# prop:rps -- three students, cyclic "beats on its own prefixes", no stable point in {A,B,C}
# ---------------------------------------------------------------------------------------------
tree = Tree(3, 2); N = tree.N                         # prefixes: root, (0,), (1,), (2,); V = 3; teacher uniform
e_, gam = 0.05, 0.3
Pt3 = np.full((N, 3), 1 / 3.0)


def cond(val):                                         # (r, (1-r)/2, (1-r)/2) with reverse KL to uniform equal to val
    f = lambda r: kl(np.array([r, (1 - r) / 2, (1 - r) / 2]), np.full(3, 1 / 3.0)) - val
    r = 1 / 3.0 if val == 0 else brentq(f, 1 / 3.0, 1 - 1e-15)
    return np.array([r, (1 - r) / 2, (1 - r) / 2])


def student(first, errs):
    Qs = np.zeros((N, 3)); Qs[0] = first
    for j in range(3):
        Qs[tree.index[(j,)]] = cond(errs[j])
    return Qs


def rkl_rows(Qs):
    return np.array([kl(Qs[i], Pt3[i]) for i in range(N)])


W = {"A": [1 - 2 * e_, e_, e_], "B": [e_, 1 - 2 * e_, e_], "C": [e_, e_, 1 - 2 * e_]}
EV = {"A": gam * np.array([1, 0, 2]), "B": gam * np.array([2, 1, 0]), "C": gam * np.array([0, 2, 1])}
S = {X: student(np.array(W[X]), EV[X]) for X in "ABC"}
ell = {X: rkl_rows(S[X]) for X in "ABC"}
dlaw = {X: tree.prefix_law(S[X]) for X in "ABC"}
Lv = {(Y, X): dlaw[X] @ ell[Y] for X in "ABC" for Y in "ABC"}
stable = lambda X, cls: all(Lv[(X, X)] <= Lv[(Y, X)] + 1e-15 for Y in cls)
ok = ([X for X in "AB" if stable(X, "AB")] == ["A"] and [X for X in "BC" if stable(X, "BC")] == ["B"]
      and [X for X in "CA" if stable(X, "CA")] == ["C"] and [X for X in "ABC" if stable(X, "ABC")] == []
      and {X: min("ABC", key=lambda Y: Lv[(Y, X)]) for X in "ABC"} == {"A": "C", "C": "B", "B": "A"})
ok &= all(abs(ell[X][1:] - EV[X]).max() < 1e-12 for X in "ABC")
results.append(report("prop:rps: pairwise stable points A, B, C; none in {A,B,C}; exact DAgger cycles A->C->B->A", ok))

# ---------------------------------------------------------------------------------------------
# prop:oscillation(a) -- exact DAgger overshoots: Phi'(0) = -kM/(2(1+k^2)), 2-cycle
# ---------------------------------------------------------------------------------------------
sig = lambda z: 1 / (1 + np.exp(-z))
bern = lambda z: np.array([1 - sig(z), sig(z)])


def L1(thp, th, kap, M):
    w = sig(-kap * th)                                          # probability of first token a
    q1 = np.array([sig(-kap * thp), sig(kap * thp)])
    return kl(q1, np.array([0.5, 0.5])) + w * kl(bern(thp), bern(M)) + (1 - w) * kl(bern(thp), bern(-M))


def Phi1(th, kap, M):
    grid = np.linspace(-3 * M, 3 * M, 6001)
    g0 = grid[int(np.argmin([L1(g, th, kap, M) for g in grid]))]
    return minimize_scalar(lambda t: L1(t, th, kap, M), bracket=(g0 - 0.01, g0, g0 + 0.01), tol=1e-12).x


kap, M = 1.0, 8.0
dPhi = (Phi1(1e-4, kap, M) - Phi1(-1e-4, kap, M)) / 2e-4
vfun = lambda t: (L1(t + 1e-6, t, kap, M) - L1(t - 1e-6, t, kap, M)) / 2e-6
dv = (vfun(1e-4) - vfun(-1e-4)) / 2e-4
th = 0.01
for _ in range(40):
    th = Phi1(th, kap, M)
th2 = Phi1(th, kap, M)
Fcyc, F0 = L1(th, th, kap, M), L1(0.0, 0.0, kap, M)
results.append(report("prop:oscillation(a): Phi'(0)=%.4f (formula %.4f), v'(0)=%.4f (formula %.4f); 2-cycle +-%.3f with "
                      "F=%.3f > F(0)=%.3f" % (dPhi, -kap * M / (2 * (1 + kap**2)), dv, (1 + kap**2) / 4 + kap * M / 8,
                                             abs(th), Fcyc, F0),
                      abs(dPhi + 2.0) < 1e-3 and abs(dv - 1.5) < 1e-4 and abs(th + th2) < 1e-6 and abs(abs(th) - 3.830) < 1e-3
                      and abs(Fcyc - 8.155) < 1e-3 and abs(F0 - 3.307) < 1e-3))

# ---------------------------------------------------------------------------------------------
# prop:oscillation(b) -- the stop-gradient flow has an attracting periodic orbit
# ---------------------------------------------------------------------------------------------
e2 = lambda a: np.array([np.cos(a), np.sin(a)])
kap, om, rho, psi_ = 1.0, 3.0, 6.0, np.pi / 3
phis = 2 * np.pi * np.arange(3) / 3
U = np.array([e2(p - psi_) for p in phis]); Bm = np.array([e2(p) for p in phis])
p2 = [softmax(om * Bm @ (rho * e2(p))) for p in phis]; u3 = np.ones(3) / 3


def vb(th):
    w = softmax(kap * U @ th); q2 = softmax(om * Bm @ th)
    return kap * U.T @ g_rkl(w, u3) + sum(w[j] * om * Bm.T @ g_rkl(q2, p2[j]) for j in range(3))


def Fb(th):
    w = softmax(kap * U @ th); q2 = softmax(om * Bm @ th)
    return kl(w, u3) + sum(w[j] * kl(q2, p2[j]) for j in range(3))


def gradFb(th):
    w = softmax(kap * U @ th); q2 = softmax(om * Bm @ th)
    return vb(th) + kap * U.T @ (np.diag(w) - np.outer(w, w)) @ np.array([kl(q2, p2[j]) for j in range(3)])


Dv0 = np.array([(vb(1e-6 * e) - vb(-1e-6 * e)) / 2e-6 for e in np.eye(2)]).T
Dv_formula = (kap**2 + om**2) / 2 * np.eye(2) - om**2 * kap * rho / 4 * np.array([[np.cos(psi_), -np.sin(psi_)],
                                                                              [np.sin(psi_), np.cos(psi_)]])
eig = np.linalg.eigvals(-Dv0)
results.append(report("prop:oscillation(b): v(0)=0, Dv(0) formula, eig(-Dv(0)) = %.2f +- %.2fi"
                      % (eig[0].real, abs(eig[0].imag)),
                      np.abs(vb(np.zeros(2))).max() < 1e-12 and np.abs(Dv0 - Dv_formula).max() < 1e-6
                      and abs(eig[0].real - 1.75) < 1e-4 and abs(abs(eig[0].imag) - 11.691) < 1e-3))


def first_return(r):
    ev = lambda t, y: y[1]
    ev.direction = 1
    sol = solve_ivp(lambda t, y: -vb(y), (0, 20), np.array([r, 0.0]), events=ev, rtol=1e-11, atol=1e-12, max_step=0.004)
    k = [i for i, t in enumerate(sol.t_events[0]) if t > 1e-3][0]
    tr = sol.t_events[0][k]
    rad = np.linalg.norm(sol.y[:, sol.t <= tr], axis=0)
    return sol.y_events[0][k][0], tr, rad.min(), rad.max()


r_in, r_out = 0.2, 1.05
angv = min((lambda th, d: (th[0] * d[1] - th[1] * d[0]) / (th @ th))(r * e2(a), -vb(r * e2(a)))
           for r in np.linspace(r_in, r_out, 120) for a in np.linspace(0, 2 * np.pi, 720, endpoint=False))
turns = [first_return(r) for r in np.linspace(0.3, 1.0, 8)]
in_annulus = all(r_in <= tt[2] and tt[3] <= r_out for tt in turns)
P03, P10 = turns[0][0], turns[-1][0]
rstar = brentq(lambda r: first_return(r)[0] - r, 0.3, 1.0, xtol=1e-11)
mult = (first_return(rstar + 1e-5)[0] - first_return(rstar - 1e-5)[0]) / 2e-5
Pr, per, _, _ = first_return(rstar)
orbit = solve_ivp(lambda t, y: -vb(y), (0, per), np.array([rstar, 0.0]), rtol=1e-11, atol=1e-12, dense_output=True,
                  max_step=0.004).sol(np.linspace(0, per, 1001))
radii = np.linalg.norm(orbit, axis=0); Fo = [Fb(orbit[:, i]) for i in range(1001)]
vmin = min(np.linalg.norm(vb(orbit[:, i])) for i in range(1001))
gf = solve_ivp(lambda t, y: -gradFb(y), (0, 60), np.array([rstar, 0.0]), rtol=1e-10, atol=1e-12)
print("      annulus [%.2f, %.2f]: min angular velocity %.3f; P(0.3)=%.3f, P(1)=%.3f; r*=%.4f, period %.3f, "
      "multiplier %.3f" % (r_in, r_out, angv, P03, P10, rstar, per, mult))
print("      orbit radius in [%.3f, %.3f], F in [%.2f, %.2f], |v| >= %.2f; gradient flow of F from r*: F -> %.3f"
      % (radii.min(), radii.max(), min(Fo), max(Fo), vmin, Fb(gf.y[:, -1])))
results.append(report("prop:oscillation(b): return map sends [0.3,1] into itself inside an annulus with positive "
                      "angular velocity; attracting periodic orbit (multiplier < 1)",
                      angv > 0 and in_annulus and P03 > 0.3 and P10 < 1.0 and 0 < mult < 1 and vmin > 0.5
                      and abs(per - 1.607) < 2e-3 and abs(mult - 0.146) < 2e-3
                      and abs(radii.min() - 0.447) < 2e-3 and abs(radii.max() - 0.714) < 2e-3
                      and abs(min(Fo) - 14.68) < 1e-2 and abs(max(Fo) - 17.16) < 1e-2 and Fb(gf.y[:, -1]) < 1.2))

print("\nALL PASS" if all(results) else "\nSOME CHECKS FAILED")
