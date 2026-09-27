"""A 'realistic prior': the distribution of small MLP teachers produced by a
training procedure (random init + random minibatch order, optionally a freshly
sampled training set).  Trains N teachers in parallel with numpy (vectorised
over the ensemble), evaluates them on a query sample from D, and saves the
function-space samples for the analysis script.
"""
import numpy as np
import argparse

p = argparse.ArgumentParser()
p.add_argument("--N", type=int, default=400)
p.add_argument("--width", type=int, default=32)
p.add_argument("--steps", type=int, default=4000)
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--ntrain", type=int, default=256)
p.add_argument("--nquery", type=int, default=2000)
p.add_argument("--resample_data", action="store_true")
p.add_argument("--out", default="../results/mlp_seed.npz")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

N, H = args.N, args.width
rng_data = np.random.default_rng(12345)          # D and the (fixed) training set


def target(x):
    return np.sin(np.pi * x[..., 0]) * np.cos(0.5 * np.pi * x[..., 1]) + 0.5 * x[..., 0] * x[..., 1]


def sample_D(rng, n):
    return rng.uniform(-1, 1, size=(n, 2))


Xq = sample_D(rng_data, args.nquery)             # query sample from D (shared)
if args.resample_data:
    rng_s = np.random.default_rng(10_000 + args.seed)
    Xtr = sample_D(rng_s, N * args.ntrain).reshape(N, args.ntrain, 2)
    Ytr = target(Xtr) + 0.1 * rng_s.standard_normal((N, args.ntrain))
else:
    X0 = sample_D(rng_data, args.ntrain)
    Y0 = target(X0) + 0.1 * rng_data.standard_normal(args.ntrain)
    Xtr = np.broadcast_to(X0, (N, args.ntrain, 2))
    Ytr = np.broadcast_to(Y0, (N, args.ntrain))

rng = np.random.default_rng(args.seed)           # seeds: init + minibatch order
shapes = {"W1": (2, H), "b1": (1, H), "W2": (H, H), "b2": (1, H), "W3": (H, 1), "b3": (1, 1)}
params = {}
for k, s in shapes.items():
    if k.startswith("W"):
        params[k] = rng.standard_normal((N,) + s) / np.sqrt(s[0])
    else:
        params[k] = np.zeros((N,) + s)
m_ = {k: np.zeros_like(v) for k, v in params.items()}
v_ = {k: np.zeros_like(v) for k, v in params.items()}
b1_, b2_, eps_ = 0.9, 0.999, 1e-8


def forward(P, X):
    a1 = np.tanh(X @ P["W1"] + P["b1"])
    a2 = np.tanh(a1 @ P["W2"] + P["b2"])
    return a1, a2, (a2 @ P["W3"] + P["b3"])[..., 0]


ar = np.arange(N)[:, None]
for t in range(1, args.steps + 1):
    idx = rng.integers(0, args.ntrain, size=(N, args.batch))
    X = Xtr[ar, idx]
    Y = Ytr[ar, idx]
    a1, a2, out = forward(params, X)
    g_out = (2.0 / args.batch) * (out - Y)[..., None]            # (N,B,1)
    grads = {"W3": np.swapaxes(a2, 1, 2) @ g_out, "b3": g_out.sum(1, keepdims=True)}
    g2 = (g_out @ np.swapaxes(params["W3"], 1, 2)) * (1 - a2 ** 2)
    grads["W2"] = np.swapaxes(a1, 1, 2) @ g2
    grads["b2"] = g2.sum(1, keepdims=True)
    g1 = (g2 @ np.swapaxes(params["W2"], 1, 2)) * (1 - a1 ** 2)
    grads["W1"] = np.swapaxes(X, 1, 2) @ g1
    grads["b1"] = g1.sum(1, keepdims=True)
    lr = args.lr * (0.1 ** (t / args.steps))                   # exponential decay to lr/10
    for k in params:
        m_[k] = b1_ * m_[k] + (1 - b1_) * grads[k]
        v_[k] = b2_ * v_[k] + (1 - b2_) * grads[k] ** 2
        mh = m_[k] / (1 - b1_ ** t)
        vh = v_[k] / (1 - b2_ ** t)
        params[k] -= lr * mh / (np.sqrt(vh) + eps_)
    if t % 1000 == 0:
        _, _, o = forward(params, Xtr)
        tr = np.mean((o - Ytr) ** 2)
        _, _, oq = forward(params, Xq[None])
        te = np.mean((oq - target(Xq)) ** 2)
        print(f"step {t}: train mse {tr:.4f}  test mse (vs clean target) {te:.4f}", flush=True)

_, _, Fq = forward(params, Xq[None])
np.savez(args.out, F=Fq, Xq=Xq, **{k: v for k, v in params.items()},
         P=sum(int(np.prod(s)) for s in shapes.values()), width=H)
print("saved", args.out, Fq.shape)
