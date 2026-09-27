"""Checks for Section 5 (representation matching) in the linear-Gaussian model.

Run:  python3 verification/check_representations.py
"""
import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import minimize

rng = np.random.default_rng(3)


def report(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def psd_sqrt(A):
    w, U = np.linalg.eigh((A + A.T) / 2)
    return (U * np.sqrt(np.clip(w, 0, None))) @ U.T


results = []
dT, dS, V = 5, 2, 7
X = rng.normal(size=(dT, dT)); Sigma = X @ X.T / dT + 0.05 * np.eye(dT)
W = rng.normal(size=(V, dT))
p = rng.dirichlet(np.ones(V)); F = np.diag(p) - np.outer(p, p)
Fh, Sh = psd_sqrt(F), psd_sqrt(Sigma)
G = Fh @ W @ Sh
s = np.linalg.svd(G, compute_uv=False)
lam, U = np.linalg.eigh(Sigma); lam, U = lam[::-1], U[:, ::-1]


def Lout(Ws, B):
    D = Ws @ B - W
    return 0.5 * np.trace(Fh @ D @ Sigma @ D.T @ Fh)


def best_readout(B):
    """min over W_S of L_out for fixed B (weighted least squares, closed form via lstsq on whitened problem)."""
    # minimise ||Fh (Ws B - W) Sh||_F^2 over Ws:  Ws = W Sigma B^T (B Sigma B^T)^{-1} is optimal for any F
    Ws = W @ Sigma @ B.T @ np.linalg.inv(B @ Sigma @ B.T)
    return Lout(Ws, B)


# thm:overconstraint (a): output-only optimum equals the Eckart-Young tail
def obj(x):
    Ws = x[:V * dS].reshape(V, dS); B = x[V * dS:].reshape(dS, dT)
    return Lout(Ws, B)


best = min(minimize(obj, rng.normal(size=V * dS + dS * dT), method="BFGS",
                    options={"gtol": 1e-10, "maxiter": 20000}).fun for _ in range(5))
ey = 0.5 * np.sum(s[dS:] ** 2)
results.append(report(f"(a) output-only optimum {best:.6f} = EY tail {ey:.6f}", abs(best - ey) < 1e-6))

# thm:overconstraint (b): PCA-constrained optimum
B_pca = U[:, :dS].T
val_b = best_readout(B_pca)
pred_b = 0.5 * sum(lam[i] * np.linalg.norm(Fh @ W @ U[:, i]) ** 2 for i in range(dS, dT))
results.append(report(f"(b) PCA-constrained optimum {val_b:.6f} = formula {pred_b:.6f}", abs(val_b - pred_b) < 1e-9))
results.append(report("(c) over-constraint cost is non-negative", val_b >= ey - 1e-12))

# thm:overconstraint (c): the 2-d example with arbitrary cost
Sig2 = np.diag([4.0, 1.0]); W2 = np.outer(rng.normal(size=V), [0.0, 1.0])
B2 = np.array([[1.0, 0.0]])
Ws2 = W2 @ Sig2 @ B2.T @ np.linalg.inv(B2 @ Sig2 @ B2.T)
D2 = Ws2 @ B2 - W2
cost2 = 0.5 * np.trace(Fh @ D2 @ Sig2 @ D2.T @ Fh)
w2 = W2[:, 1]
results.append(report("(c) example: cost = lambda_2 w^T F w / 2", abs(cost2 - 0.5 * 1.0 * w2 @ F @ w2) < 1e-12))

# prop:fisher-rep: the Fisher-aligned target is output-optimal
_, _, Vt = np.linalg.svd(G)
UF = Vt[:dS].T
B_F = UF.T @ np.linalg.inv(Sh)
results.append(report(f"Fisher-aligned B attains the output optimum ({best_readout(B_F):.6f})",
                      abs(best_readout(B_F) - ey) < 1e-9))

# prop:cca: R_LS = tr S_TT - tr(S_TS S_SS^-1 S_ST) = d_T - sum rho^2 when teacher whitened
n = 200000
hT = rng.normal(size=(n, dT)) @ Sh
hS = hT @ rng.normal(size=(dT, dS)) + 0.3 * rng.normal(size=(n, dS))
hT -= hT.mean(0); hS -= hS.mean(0)
STT, SSS, STS = hT.T @ hT / n, hS.T @ hS / n, hT.T @ hS / n
A = STS @ np.linalg.inv(SSS)
R_emp = np.mean(np.sum((hS @ A.T - hT) ** 2, axis=1))
R_formula = np.trace(STT) - np.trace(STS @ np.linalg.inv(SSS) @ STS.T)
Wh = np.linalg.inv(psd_sqrt(STT))
hTw = hT @ Wh.T
STSw = hTw.T @ hS / n
rho2 = np.linalg.eigvalsh(np.linalg.inv(psd_sqrt(SSS)) @ STSw.T @ STSw @ np.linalg.inv(psd_sqrt(SSS)))
results += [report("R_LS closed form", abs(R_emp - R_formula) < 1e-8),
            report("whitened R_LS = d_T - sum rho_i^2",
                   abs((np.mean(np.sum((hS @ (STSw @ np.linalg.inv(SSS)).T - hTw) ** 2, 1))) - (dT - rho2.sum())) < 1e-6)]

# prop:cca CKA example: h_S = Sigma^{-1/2} h_T gives R_LS = 0 but CKA < 1
d = 4
Xc = rng.normal(size=(d, d)); Sc = Xc @ Xc.T
Sc_isqrt = np.linalg.inv(psd_sqrt(Sc))
S_SS = Sc_isqrt @ Sc @ Sc_isqrt; S_ST = Sc_isqrt @ Sc
cka = np.linalg.norm(S_ST, "fro") ** 2 / (np.linalg.norm(S_SS, "fro") * np.linalg.norm(Sc, "fro"))
results.append(report(f"CKA example: {cka:.4f} = tr/(sqrt(d)||.||_F) = "
                      f"{np.trace(Sc) / (np.sqrt(d) * np.linalg.norm(Sc, 'fro')):.4f} < 1",
                      abs(cka - np.trace(Sc) / (np.sqrt(d) * np.linalg.norm(Sc, "fro"))) < 1e-10 and cka < 1))

print("\nALL PASS" if all(results) else "\nSOME CHECKS FAILED")
