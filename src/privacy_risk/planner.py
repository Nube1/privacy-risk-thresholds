"""A planner trained on integer counts (Chapter 22).

A small tanh network, or logistic regression when there are no hidden layers, is fitted to
successes and trials by the penalised binomial loss (22.1). Gradients are written by hand so a
reviewer can check them against finite differences. The planner predicts; it never certifies.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar


def _check_counts(successes, trials):
    for name, v in (("successes", successes), ("trials", trials)):
        arr = np.asarray(v)
        if arr.dtype.kind not in "iu":
            raise TypeError("successes and trials must be integer counts, not rates")
    k, b = np.asarray(successes, int), np.asarray(trials, int)
    if np.any(b <= 0) or np.any(k < 0) or np.any(k > b):
        raise ValueError("need 0 <= successes <= trials and trials > 0")
    return k, b


def _shapes(n_in, hidden):
    sizes = [n_in, *hidden, 1]
    return [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]


def _unpack(theta, shapes):
    out, pos = [], 0
    for r, c in shapes:
        W = theta[pos:pos + r * c].reshape(r, c); pos += r * c
        bvec = theta[pos:pos + c]; pos += c
        out.append((W, bvec))
    return out


def forward(theta, x, hidden):
    layers = _unpack(theta, _shapes(x.shape[1], hidden))
    acts = [x]
    for W, bvec in layers[:-1]:
        acts.append(np.tanh(acts[-1] @ W + bvec))
    W, bvec = layers[-1]
    return (acts[-1] @ W + bvec)[:, 0], acts


def objective(theta, x, k, b, hidden, penalty):
    """Loss (22.1) and its gradient. The data term is divided by the total number of trials."""
    n = b.sum()
    h, acts = forward(theta, x, hidden)
    loss = (b * np.logaddexp(0, h) - k * h).sum() / n
    p = 1 / (1 + np.exp(-h))
    layers = _unpack(theta, _shapes(x.shape[1], hidden))
    loss += 0.5 * penalty * float(theta @ theta)
    delta = ((b * p - k) / n)[:, None]
    grads = []
    for li in range(len(layers) - 1, -1, -1):
        W, _ = layers[li]
        gW = acts[li].T @ delta
        gb = delta.sum(axis=0)
        grads.append((gW, gb))
        if li > 0:
            delta = (delta @ W.T) * (1 - acts[li] ** 2)
    grads.reverse()
    g = np.concatenate([np.concatenate([gW.ravel(), gb]) for gW, gb in grads]) + penalty * theta
    return loss, g


def fit(x, successes, trials, hidden=(), penalty=1e-4, seed=0):
    """Fit by L-BFGS from a seeded start. Returns the parameters, hidden sizes and convergence flag."""
    k, b = _check_counts(successes, trials)
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    nparam = sum(r * c + c for r, c in _shapes(x.shape[1], hidden))
    theta0 = rng.normal(0, 0.3, nparam) if hidden else np.zeros(nparam)
    res = minimize(lambda t: objective(t, x, k, b, hidden, penalty), theta0, jac=True,
                   method="L-BFGS-B", options={"maxiter": 2000})
    return {"theta": res.x, "hidden": tuple(hidden), "converged": bool(res.success)}


def logits(model, x):
    return forward(model["theta"], np.asarray(x, float), model["hidden"])[0]


def binomial_cross_entropy(z, successes, trials):
    """Mean negative log likelihood per trial (nats), dropping the binomial coefficient."""
    k, b = _check_counts(successes, trials)
    z = np.asarray(z, float)
    return float((b * np.logaddexp(0, z) - k * z).sum() / b.sum())


def calibrate_temperature(z, successes, trials, bounds=(0.2, 5.0)):
    """Fit one temperature T on a calibration partition (Proposition 22.3). Needs some z != 0."""
    z = np.asarray(z, float)
    if not np.any(z != 0):
        raise ValueError("all logits are zero; a temperature is not identified")
    res = minimize_scalar(lambda lt: binomial_cross_entropy(z / np.exp(lt), successes, trials),
                          bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded")
    return float(np.exp(res.x))


def grouped_split(groups, fractions=(0.6, 0.2, 0.2), seed=0):
    """Assign whole groups to training, calibration and test at random."""
    groups = np.asarray(groups)
    if abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("fractions must sum to one")
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    c1 = int(round(fractions[0] * n)); c2 = c1 + int(round(fractions[1] * n))
    parts = (uniq[:c1], uniq[c1:c2], uniq[c2:])
    return tuple(np.flatnonzero(np.isin(groups, p)) for p in parts)


def recover_count(rate: float, trials: int, decimals: int) -> int:
    """Rebuild a count from a rounded rate. Safe only when trials < 10**decimals (Prop 22.1)."""
    if trials >= 10 ** decimals:
        raise ValueError("a rate rounded this coarsely cannot be trusted to recover the count")
    return int(round(trials * rate))
