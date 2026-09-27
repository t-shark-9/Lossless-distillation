"""Exact-enumeration checks of the sequence-level results (Sections 1, 2 and 4).

A tiny autoregressive teacher/student pair (vocabulary V, horizon T) is
enumerated exactly, so every expectation below is computed without Monte Carlo.
Run:  python3 verification/check_sequences.py
"""
import itertools
import numpy as np

rng = np.random.default_rng(1)
V, T = 3, 4
TOL = 1e-9
prefixes = [p for t in range(T) for p in itertools.product(range(V), repeat=t)]
index = {p: i for i, p in enumerate(prefixes)}
seqs = list(itertools.product(range(V), repeat=T))


def softmax_rows(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def seq_prob(P, y):
    return float(np.prod([P[index[y[:t]], y[t]] for t in range(T)]))


def prefix_dist(P, t):
    """Law d_t of the prefix s_t = y_{<t} (length t-1 in 1-based notation)."""
    out = {}
    for y in itertools.product(range(V), repeat=t):
        out[y] = float(np.prod([P[index[y[:k]], y[k]] for k in range(t)]))
    return out


def kl(p, q):
    return float(np.sum(p * (np.log(p) - np.log(q))))


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def report(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return ok


theta_T = rng.normal(scale=1.5, size=(len(prefixes), V))
theta_S = theta_T + rng.normal(scale=0.7, size=theta_T.shape)
Pt, Ps = softmax_rows(theta_T), softmax_rows(theta_S)
pseq = np.array([seq_prob(Pt, y) for y in seqs])
qseq = np.array([seq_prob(Ps, y) for y in seqs])
results = []

# thm:chain: chain rules
fwd = sum(w * kl(Pt[index[s]], Ps[index[s]]) for t in range(T) for s, w in prefix_dist(Pt, t).items())
rev = sum(w * kl(Ps[index[s]], Pt[index[s]]) for t in range(T) for s, w in prefix_dist(Ps, t).items())
results += [report("KL(p||q)_seq = sum_t E_{d^p_t} KL_t (teacher prefixes)", abs(fwd - kl(pseq, qseq)) < TOL),
            report("KL(q||p)_seq = sum_t E_{d^q_t} KL(q_t||p_t) (student prefixes)", abs(rev - kl(qseq, pseq)) < TOL)]

# prop:hybrid: hybrid bounds for TV and the converse
e_off = [sum(w * tv(Pt[index[s]], Ps[index[s]]) for s, w in prefix_dist(Pt, t).items()) for t in range(T)]
e_on = [sum(w * tv(Pt[index[s]], Ps[index[s]]) for s, w in prefix_dist(Ps, t).items()) for t in range(T)]
TVs = tv(pseq, qseq)
results += [report("TV_seq <= sum_t E_{d^p} TV_t and <= sum_t E_{d^q} TV_t", TVs <= sum(e_off) + TOL and TVs <= sum(e_on) + TOL),
            report("max_t E_{d^p_t} TV_t <= 2 TV_seq and max_t E_{d^q_t} TV_t <= 2 TV_seq",
                   max(e_off) <= 2 * TVs + TOL and max(e_on) <= 2 * TVs + TOL)]

# thm:quadratic / thm:pdl: cumulative per-step costs, off-policy (quadratic) and PDL (on-policy)
C = rng.random((len(prefixes), V))  # c_t(s, a) in [0,1]


def J(P):
    return sum(seq_prob(P, y) * sum(C[index[y[:t]], y[t]] for t in range(T)) for y in seqs)


def Qfun(P):
    """Teacher cost-to-go Q_t(s,a) = c(s,a) + E_P[sum_{k>t} c_k | s, a]."""
    Q = np.zeros((len(prefixes), V))
    for s in sorted(prefixes, key=len, reverse=True):
        for a in range(V):
            nxt = s + (a,)
            Q[index[s], a] = C[index[s], a] + (P[index[nxt]] @ Q[index[nxt]] if len(nxt) < T else 0.0)
    return Q


Qp = Qfun(Pt)
pdl = sum(w * (Ps[index[s]] - Pt[index[s]]) @ Qp[index[s]] for t in range(T) for s, w in prefix_dist(Ps, t).items())
dJ = J(Ps) - J(Pt)
u = max(Qp[i].max() - Qp[i].min() for i in range(len(prefixes)))
results += [report("performance-difference identity", abs(pdl - dJ) < 1e-9),
            report("|dJ| <= sum_t (T-t+1) e_off_t (off-policy)", abs(dJ) <= sum((T - t) * e_off[t] for t in range(T)) + TOL),
            report("|dJ| <= u * sum_t e_on_t (on-policy, recoverability)", abs(dJ) <= u * sum(e_on) + TOL)]

# prop:exposure: exposure-bias gap for a bounded per-state loss g in [0, G]
g = np.array([tv(Pt[i], Ps[i]) for i in range(len(prefixes))])  # G = 1
gap = sum(sum(w * g[index[s]] for s, w in prefix_dist(Ps, t).items()) -
          sum(w * g[index[s]] for s, w in prefix_dist(Pt, t).items()) for t in range(T))
bound = sum(sum(e_off[:t]) for t in range(T))
results.append(report("|exposure gap| <= G sum_t sum_{k<t} e_off_k", abs(gap) <= bound + TOL))

# thm:quadratic, tightness: the cliff example grows quadratically under off-policy error eps
eps = 1e-3
for TT in [10, 100, 1000]:
    on_track = np.array([(1 - eps) ** t for t in range(TT)])  # P(on track before step t)
    Jq = sum(1 - on_track[t] * (1 - eps) for t in range(TT))  # cost 1 per off-track step (incl. the deviating one)
    print(f"      cliff T={TT:5d}: J(q)={Jq:9.3f}  eps*T(T+1)/2={eps*TT*(TT+1)/2:9.3f}")

# prop:rb-reverse: unbiased gradient of the sequence-level reverse KL (tabular student)


def seq_rkl(theta):
    P = softmax_rows(theta)
    q = np.array([seq_prob(P, y) for y in seqs])
    return kl(q, pseq)


def seq_jsd(theta, b):
    P = softmax_rows(theta)
    q = np.array([seq_prob(P, y) for y in seqs])
    m = b * pseq + (1 - b) * q
    return b * kl(pseq, m) + (1 - b) * kl(q, m)


def num_grad(f, theta, h=1e-6):
    G = np.zeros_like(theta)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            E = np.zeros_like(theta); E[i, j] = h
            G[i, j] = (f(theta + E) - f(theta - E)) / (2 * h)
    return G


def score(P, s, a):
    """d log q(a|s) / d theta: nonzero only in row index[s]."""
    G = np.zeros((len(prefixes), V)); G[index[s]] = -P[index[s]]; G[index[s], a] += 1.0
    return G


def local_rkl_grad(P, s):
    q, p = P[index[s]], Pt[index[s]]
    G = np.zeros((len(prefixes), V)); G[index[s]] = q * (np.log(q / p) - kl(q, p))
    return G


def estimator_mean(theta, include_score=True, baseline=True):
    P = softmax_rows(theta)
    klrow = np.array([kl(P[i], Pt[i]) for i in range(len(prefixes))])
    tot = np.zeros_like(theta)
    for y in seqs:
        w = seq_prob(P, y)
        est = np.zeros_like(theta)
        for t in range(T):
            s = y[:t]
            est += local_rkl_grad(P, s)
            if include_score:
                future = sum(klrow[index[y[:k]]] for k in range(t + 1, T))
                b = klrow[index[s]] * 0.37 if baseline else 0.0  # any function of s_t is a valid baseline
                est += score(P, s, y[t]) * (future - b)
        tot += w * est
    return tot


g_true = num_grad(seq_rkl, theta_S)
g_est = estimator_mean(theta_S)
g_sg = estimator_mean(theta_S, include_score=False)
results += [report("reverse-KL estimator (RB local + reward-to-go + baseline) is unbiased",
                   np.allclose(g_true, g_est, atol=1e-6)),
            report("stop-gradient (local term only) estimator is biased",
                   np.abs(g_true - g_sg).max() > 1e-3)]

# prop:seq-grad: grad of sequence-level JSD_beta = (1-beta) E_q[grad log q * log(q/M)]
b = 0.4
P = softmax_rows(theta_S)
G = np.zeros_like(theta_S)
for y, qy, py in zip(seqs, qseq, pseq):
    sc = sum(score(P, y[:t], y[t]) for t in range(T))
    G += qy * sc * (1 - b) * np.log(qy / (b * py + (1 - b) * qy))
results.append(report("on-policy score-function gradient of sequence JSD_beta",
                      np.allclose(G, num_grad(lambda th: seq_jsd(th, b), theta_S), atol=1e-6)))

# prop:mix-est: bounded unbiased mixture estimator of sequence-level JSD_beta
for b in [0.2, 0.5, 0.8]:
    M = b * pseq + (1 - b) * qseq
    r, s_ = pseq / M, qseq / M
    phi = b * r * np.log(r) + (1 - b) * s_ * np.log(s_)
    jsd_exact = b * kl(pseq, M) + (1 - b) * kl(qseq, M)
    results.append(report(f"mixture estimator (beta={b}): unbiased and in [0, log(1/min(b,1-b))]",
                          abs(np.sum(M * phi) - jsd_exact) < TOL and phi.min() >= -TOL
                          and phi.max() <= np.log(1 / min(b, 1 - b)) + TOL))

# def:pajsd: the pathwise/score split is unbiased; adding the teacher pathwise term to prop:jsd-est is biased
b = 0.3
P_ = softmax_rows(theta_S)
M = b * pseq + (1 - b) * qseq
dQ = [sum(score(P_, y[:t], y[t]) for t in range(T)) * qy for y, qy in zip(seqs, qseq)]   # grad Q(y)
teacher_path = sum(-b * (1 - b) * py * g / My for py, g, My in zip(pseq, dQ, M))
student_part = sum((1 - b) * (g * np.log(qy / My) - (1 - b) * qy * g / My)
                   for qy, g, My in zip(qseq, dQ, M))
g_true = num_grad(lambda th: seq_jsd(th, b), theta_S)
g_score = sum((1 - b) * g * np.log(qy / My) for qy, g, My in zip(qseq, dQ, M))
results += [report("teacher-pathwise + student score/pathwise split is unbiased",
                   np.allclose(teacher_path + student_part, g_true, atol=1e-6)),
            report("adding the teacher pathwise term to the score estimator is biased",
                   np.abs(g_score + teacher_path - g_true).max() > 1e-3)]

# Unbiased TV estimator: TV = E_{y~p}[(1 - q(y)/p(y))_+]
results.append(report("TV_seq = E_p[(1-q/p)_+]", abs(np.sum(pseq * np.clip(1 - qseq / pseq, 0, None)) - TVs) < TOL))

print("\nALL PASS" if all(results) else "\nSOME CHECKS FAILED")
