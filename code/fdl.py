"""Core formulas for functional description length (FDL) under a Gaussian
(second-order) teacher prior with squared L2(D) distortion.

All quantities are in bits unless a name ends in `_nats`.

Notation (see notes/functional_information.md):
  lam   : eigenvalues of the teacher-prior covariance operator C on L2(D)
  z     : coordinates <f - m, phi_i> of a teacher f in the eigenbasis of C
  eps   : squared-L2(D) distortion tolerance
  theta : reverse water-filling level, sum_i min(lam_i, theta) = eps
"""
import numpy as np

LN2 = np.log(2.0)


def water_level(lam, eps, iters=200):
    """theta such that sum_i min(lam_i, theta) = eps (reverse water-filling)."""
    lam = np.asarray(lam, dtype=float)
    if eps >= lam.sum():
        return np.inf
    lo, hi = 0.0, float(lam.max())
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if np.minimum(lam, mid).sum() > eps:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def rate_gaussian(lam, eps):
    """Kolmogorov's formula R_C(eps) = sum_i 1/2 log2+(lam_i/theta)."""
    th = water_level(lam, eps)
    if not np.isfinite(th):
        return 0.0
    lam = np.asarray(lam, dtype=float)
    a = lam > th
    return 0.5 * np.sum(np.log2(lam[a] / th))


def dispersion_gaussian_nats2(lam, eps):
    """Rate-dispersion V(eps) = Var[j(F, eps)] in nats^2."""
    th = water_level(lam, eps)
    lam = np.asarray(lam, dtype=float)
    a = lam > th
    return 0.5 * a.sum() + np.sum(lam[~a] ** 2) / (2 * th ** 2)


def tilted_info(z, lam, eps):
    """eps-tilted information j(f, eps) of teacher coordinates z (bits).

    j = sum_{lam_i>theta} [1/2 ln(lam_i/theta) + (z_i^2/lam_i - 1)/2]
        + sum_{lam_i<=theta} (z_i^2 - lam_i)/(2 theta)          (nats)
    """
    th = water_level(lam, eps)
    z = np.asarray(z, dtype=float)
    lam = np.asarray(lam, dtype=float)
    if not np.isfinite(th):
        return 0.0
    a = lam > th
    j = (0.5 * np.sum(np.log(lam[a] / th))
         + 0.5 * np.sum(z[..., a] ** 2 / lam[a] - 1.0, axis=-1)
         + np.sum(z[..., ~a] ** 2 - lam[~a], axis=-1) / (2 * th))
    return j / LN2


def achievability_gap(z, lam, eps, n_mc=200_000, rng=None):
    """Delta_f(eps) = -log2 E_{Q_f}[1{d<=eps} exp(lam*(d-eps))] >= 0 (bits).

    Q_f is the tilted reproduction law; for the Gaussian source it is Gaussian:
    above water G_i ~ N(z_i (lam_i-theta)/lam_i, theta (lam_i-theta)/lam_i),
    below water G_i = 0.  Then -log2 P_{G*}(B_eps(f)) = j(f,eps) + Delta_f(eps).
    """
    rng = np.random.default_rng(rng)
    th = water_level(lam, eps)
    lam = np.asarray(lam, dtype=float)
    z = np.asarray(z, dtype=float)
    a = lam > th
    lam_star = 1.0 / (2.0 * th)
    mu = z[a] * (lam[a] - th) / lam[a]
    v = th * (lam[a] - th) / lam[a]
    below = np.sum(z[~a] ** 2)
    out = 0.0
    done = 0
    chunk = 20_000
    while done < n_mc:
        m = min(chunk, n_mc - done)
        G = mu + np.sqrt(v) * rng.standard_normal((m, a.sum()))
        d = np.sum((z[a] - G) ** 2, axis=1) + below
        w = np.where(d <= eps, np.exp(lam_star * (d - eps)), 0.0)
        out += w.sum()
        done += m
    return -np.log2(out / n_mc)


def ball_logmass_direct(z, lam, eps, n_mc=2_000_000, rng=None):
    """-log2 P_{G*}(B_eps(f)) by direct Monte Carlo (only for small problems)."""
    rng = np.random.default_rng(rng)
    th = water_level(lam, eps)
    lam = np.asarray(lam, dtype=float)
    s = np.sqrt(np.maximum(lam - th, 0.0))
    hits = 0
    done = 0
    chunk = 200_000
    while done < n_mc:
        m = min(chunk, n_mc - done)
        G = s * rng.standard_normal((m, len(lam)))
        d = np.sum((z - G) ** 2, axis=1)
        hits += int(np.sum(d <= eps))
        done += m
    return -np.log2(max(hits, 1) / n_mc), hits


def n_above_water(lam, eps):
    th = water_level(lam, eps)
    return int(np.sum(np.asarray(lam) > th))


def rate_diag_gaussian(sig2, eps):
    """Gaussian upper bound in a FIXED orthonormal basis with per-coordinate
    second moments sig2 (valid upper bound on R_pi for any pi with these
    second moments, correlations ignored)."""
    return rate_gaussian(np.sort(np.asarray(sig2))[::-1], eps)


def ball_logmass_gauss(z, sig2, eps, n_mc=100_000, rng=None, iters=100):
    """-log2 P_{G ~ N(0, diag sig2)}( ||z - G||^2 <= eps ) by exponential
    tilting: tilt s chosen so that the tilted mean of d equals eps."""
    rng = np.random.default_rng(rng)
    z = np.asarray(z, float)
    sig2 = np.asarray(sig2, float)

    def tilted_mean_d(s):
        v = sig2 / (1 + 2 * s * sig2)
        mu = z * 2 * s * sig2 / (1 + 2 * s * sig2)
        return np.sum((z - mu) ** 2 + v)

    if tilted_mean_d(0.0) <= eps:
        s = 0.0
    else:
        lo, hi = 0.0, 1.0
        while tilted_mean_d(hi) > eps:
            hi *= 2
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if tilted_mean_d(mid) > eps:
                lo = mid
            else:
                hi = mid
        s = 0.5 * (lo + hi)
    v = sig2 / (1 + 2 * s * sig2)
    mu = z * 2 * s * sig2 / (1 + 2 * s * sig2)
    logZ = np.sum(-0.5 * np.log1p(2 * s * sig2) - s * z ** 2 / (1 + 2 * s * sig2))
    acc = []
    done = 0
    while done < n_mc:
        m = min(20_000, n_mc - done)
        G = mu + np.sqrt(v) * rng.standard_normal((m, z.size))
        d = np.sum((z - G) ** 2, 1)
        acc.append(np.where(d <= eps, s * (d - eps), -np.inf) + s * eps)
        done += m
    acc = np.concatenate(acc)
    mx = acc.max()
    if not np.isfinite(mx):
        return np.inf
    log_mean = mx + np.log(np.mean(np.exp(acc - mx)))
    return -(logZ + log_mean) / LN2
