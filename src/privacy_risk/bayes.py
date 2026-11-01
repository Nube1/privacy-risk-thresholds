"""Bayesian inference over a finite prior and the sensitivity spectrum (Chapter 20)."""
from __future__ import annotations

import numpy as np

from .transport import transport


def observation_points(n: int = 16, per_side: int = 4) -> np.ndarray:
    """A per_side x per_side lattice of cell indices on an n x n grid (16 points by default)."""
    idx = (np.arange(per_side) * (n // per_side) + n // (2 * per_side)).astype(int)
    return np.array([(i, j) for i in idx for j in idx])


def transport_forward(dispersion: float, amplitude: float, n: int = 16, horizon: float = 0.2,
                      points: np.ndarray | None = None) -> np.ndarray:
    pts = observation_points(n) if points is None else points
    c = transport(n=n, horizon=horizon, dispersion=dispersion, amplitude=amplitude)["density"]
    return c[pts[:, 0], pts[:, 1]]


def grid_posterior(y: np.ndarray, predictions: np.ndarray, sigma: float,
                   prior: np.ndarray | None = None) -> np.ndarray:
    """Equation (20.12): exact posterior weights over G parameter vectors.

    `predictions` has one row of predicted observations per parameter vector. The
    largest log likelihood is subtracted before exponentiating.
    """
    y = np.asarray(y, float)
    pred = np.atleast_2d(np.asarray(predictions, float))
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    p = np.full(pred.shape[0], 1.0 / pred.shape[0]) if prior is None else np.asarray(prior, float)
    if p.shape != (pred.shape[0],) or np.any(p < 0) or p.sum() <= 0:
        raise ValueError("the prior must be nonnegative with one weight per parameter vector")
    ll = -0.5 * np.sum((y - pred) ** 2, axis=1) / sigma ** 2
    w = np.exp(ll - ll.max()) * p
    return w / w.sum()


def predictive_pmf(weights: np.ndarray, risks: np.ndarray, B: int) -> np.ndarray:
    """Equation (20.13): the mixture law of the number of successes in B fresh attacks."""
    from scipy.stats import binom
    k = np.arange(B + 1)
    return sum(w * binom.pmf(k, B, r) for w, r in zip(weights, risks))


def sensitivity_spectrum(dispersion: float, amplitude: float, n: int = 16, horizon: float = 0.2,
                         step: float = 1e-4) -> np.ndarray:
    """Singular values of the finite-difference Jacobian of the observations with respect to
    (dispersion, amplitude), largest first."""
    base = transport_forward(dispersion, amplitude, n, horizon)
    cols = [(transport_forward(dispersion + step, amplitude, n, horizon) - base) / step,
            (transport_forward(dispersion, amplitude + step, n, horizon) - base) / step]
    return np.linalg.svd(np.column_stack(cols), compute_uv=False)
