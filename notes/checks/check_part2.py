"""Part 2: horizon experiment in an exactly solvable (exchangeable) model.

Teacher  p = (1-eps) Ber(mu0)^T + eps Ber(mu1)^T      (stochastic, misspecified)
Class    Q = { Ber(mu)^T : mu in (0,1) },  one shared logit theta, mu = sigmoid(theta)

Everything depends on x only through k = #ones, so all expectations are exact
sums over k = 0..T (no Monte Carlo), and T can be large.

(A) agnostic factor TV(p, q_hat)/OPT of the population minimiser of
    forward KL (log-loss BC), JSD, TV, D_{1/2}
(B) gradient-estimator variance in Fisher units  Var / tr I_T(theta)
    - D_beta : x ~ p (teacher samples), g = -beta * w(x) * S(x)
    - SFT    : x ~ p, g = -S(x)                    (log-loss / teacher forcing)
    - RKL    : x ~ q, g = (R - b) S, R = log q/p   (sequence-level REINFORCE)
    - JSD    : x ~ q, g = (rho - b) S, rho = 0.5 log(2q/(p+q))
    (b = variance-optimal constant baseline for the REINFORCE ones)
(C) Doob identity  sum_t E[A_t^2] = Var_q(R)  for R = 1{q > p}
"""
import numpy as np
from scipy.special import gammaln, logsumexp

MU0, MU1, EPS = 0.5, 0.7, 0.01


def log_binom(T):
    k = np.arange(T + 1)
    return gammaln(T + 1) - gammaln(k + 1) - gammaln(T - k + 1)


def log_seq_prob_iid(T, mu):
    k = np.arange(T + 1)
    return k * np.log(mu) + (T - k) * np.log1p(-mu)


def log_seq_prob_teacher(T):
    return np.logaddexp(np.log1p(-EPS) + log_seq_prob_iid(T, MU0),
                        np.log(EPS) + log_seq_prob_iid(T, MU1))


def objectives(T, mu, lb, lp):
    """Return TV, KL(p||q), JSD, D_1/2 for q = Ber(mu)^T (exact)."""
    lq = log_seq_prob_iid(T, mu)
    Pp = np.exp(lb + lp)          # law of k under p
    Pq = np.exp(lb + lq)          # law of k under q
    tv = 0.5 * np.abs(Pp - Pq).sum()
    kl = np.sum(Pp * (lp - lq))
    lm = np.logaddexp(lp, lq) - np.log(2)
    jsd = 0.5 * np.sum(Pp * (lp - lm)) + 0.5 * np.sum(Pq * (lq - lm))
    u = np.exp(np.minimum(lq - lp, 0.0))  # min(q/p, 1)
    d_half = np.sum(Pp * (1 - np.sqrt(u)))
    return tv, kl, jsd, d_half


def argmin_mu(fun, lo=1e-3, hi=1 - 1e-3):
    grid = np.linspace(lo, hi, 2001)
    vals = np.array([fun(m) for m in grid])
    i = int(np.argmin(vals))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    g = (np.sqrt(5) - 1) / 2
    for _ in range(80):  # golden-section refinement
        c, d = b - g * (b - a), a + g * (b - a)
        if fun(c) < fun(d):
            b = d
        else:
            a = c
    return 0.5 * (a + b)


print("(A) agnostic factor TV(p, q_hat) / OPT, teacher = 0.99 Ber(.5)^T + 0.01 Ber(.7)^T")
print(f"{'T':>7} {'OPT':>8} | {'fwd-KL':>8} {'JSD':>8} {'TV':>8} {'D_1/2':>8}")
for T in (10, 100, 1000, 3000, 10000, 30000):
    lb, lp = log_binom(T), log_seq_prob_teacher(T)
    res = {}
    for j, name in enumerate(("TV", "KL", "JSD", "D_half")):
        mu_hat = argmin_mu(lambda m: objectives(T, m, lb, lp)[j])
        res[name] = objectives(T, mu_hat, lb, lp)[0]
    opt = res["TV"]
    print(f"{T:>7} {opt:8.5f} | {res['KL'] / opt:8.2f} {res['JSD'] / opt:8.2f} "
          f"{res['TV'] / opt:8.2f} {res['D_half'] / opt:8.2f}")


def variances(T, mu, beta=1.0):
    lb, lp = log_binom(T), log_seq_prob_teacher(T)
    lq = log_seq_prob_iid(T, mu)
    k = np.arange(T + 1)
    S = k - T * mu                       # d/dtheta log q(x), theta = logit(mu)
    Pp, Pq = np.exp(lb + lp), np.exp(lb + lq)
    fisher = T * mu * (1 - mu)           # = E_q S^2
    out = {}
    # D_beta, teacher samples
    w = np.where(lq < lp, np.exp(beta * np.minimum(lq - lp, 0.0)), 0.0)
    g = -beta * w * S
    mean = np.sum(Pp * g)
    out["D_beta"] = (np.sum(Pp * g ** 2) - mean ** 2) / fisher
    # grad check of D_beta by finite differences in theta
    def D(m):
        u = np.exp(np.minimum(log_seq_prob_iid(T, m) - lp, 0.0))
        return np.sum(Pp * (1 - u ** beta))
    th = np.log(mu / (1 - mu))
    h = 1e-5
    sig = lambda z: 1 / (1 + np.exp(-z))
    fd = (D(sig(th + h)) - D(sig(th - h))) / (2 * h)
    out["gradcheck"] = abs(fd - mean) / max(abs(fd), 1e-12)
    # SFT, teacher samples
    out["SFT"] = (np.sum(Pp * S ** 2) - np.sum(Pp * S) ** 2) / fisher
    # REINFORCE with optimal constant baseline
    for name, R in (("RKL", lq - lp),
                    ("JSD", 0.5 * (np.log(2) + lq - np.logaddexp(lp, lq)))):
        b = np.sum(Pq * R * S ** 2) / np.sum(Pq * S ** 2)
        g = (R - b) * S
        out[name] = (np.sum(Pq * g ** 2) - np.sum(Pq * g) ** 2) / fisher
    tv = 0.5 * np.abs(Pp - Pq).sum()
    return out, tv


for regime in ("local: mu = mu0 + 1/sqrt(T)", "fixed per-token gap: mu = mu0 + 0.05"):
    print(f"\n(B) Var / tr I_T  [{regime}]   (D_beta with beta = 1 and 1/2)")
    print(f"{'T':>7} {'TV(p,q)':>8} | {'D_1':>7} {'D_1/2':>7} {'SFT':>9} {'RKL':>9} {'JSD':>7} | gradcheck")
    for T in (10, 100, 1000, 10000):
        mu = MU0 + (1 / np.sqrt(T) if regime.startswith("local") else 0.05)
        o1, tv = variances(T, mu, 1.0)
        o2, _ = variances(T, mu, 0.5)
        print(f"{T:>7} {tv:8.4f} | {o1['D_beta']:7.4f} {o2['D_beta']:7.4f} {o1['SFT']:9.3f} "
              f"{o1['RKL']:9.2f} {o1['JSD']:7.4f} | {max(o1['gradcheck'], o2['gradcheck']):.1e}")


print("\n(C) Doob identity for R = 1{q>p}: sum_t E[A_t^2] = Var_q(R); naive energy T*Var(R)")
for T in (10, 100, 1000):
    mu = MU0 + 1 / np.sqrt(T)
    lp, lq = log_seq_prob_teacher(T), log_seq_prob_iid(T, mu)
    R = (lq > lp).astype(float)          # indexed by final count k
    # M_t(j) = P_q(R = 1 | j ones among first t tokens); backward recursion
    M = [None] * (T + 1)
    M[T] = R.copy()
    for t in range(T - 1, -1, -1):
        M[t] = mu * M[t + 1][1:t + 2] + (1 - mu) * M[t + 1][:t + 1]
    energy = 0.0
    for t in range(1, T + 1):            # law of j ones among first t-1 tokens
        j = np.arange(t)
        Pj = np.exp(log_binom(t - 1) + j * np.log(mu) + (t - 1 - j) * np.log1p(-mu))
        a1 = M[t][1:t + 1] - M[t - 1]    # A_t if x_t = 1
        a0 = M[t][:t] - M[t - 1]         # A_t if x_t = 0
        energy += np.sum(Pj * (mu * a1 ** 2 + (1 - mu) * a0 ** 2))
    Pq = np.exp(log_binom(T) + lq)
    varR = np.sum(Pq * R) * (1 - np.sum(Pq * R))
    print(f"T = {T:5d}: Doob energy = {energy:.6f}, Var_q(R) = {varR:.6f}, naive T*Var(R) = {T * varR:9.3f}")
