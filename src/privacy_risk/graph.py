"""The graph baseline of Chapter 19: a continuous-time Markov chain on a graph of states."""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def validate_generator(Q, tol: float = 1e-9) -> np.ndarray:
    Q = np.asarray(Q, float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("a generator is a square matrix")
    off = Q - np.diag(np.diag(Q))
    if np.any(off < -tol):
        raise ValueError("off-diagonal rates must be nonnegative")
    if np.any(np.abs(Q.sum(axis=1)) > tol):
        raise ValueError("rows of a generator sum to zero")
    return Q


def propagate(Q, c0, t: float) -> np.ndarray:
    """Masses at time t: exp(t Q^T) c(0)."""
    Q = validate_generator(Q)
    c0 = np.asarray(c0, float)
    if np.any(c0 < 0):
        raise ValueError("masses must be nonnegative")
    return expm(t * Q.T) @ c0


def node_risk(g, c) -> float:
    c = np.asarray(c, float)
    if c.sum() <= 0:
        raise ValueError("total mass must be positive")
    return float(np.asarray(g, float) @ c / c.sum())


def estimate_generator(transitions, holding_time) -> np.ndarray:
    """Maximum likelihood: Q_ij = n_ij / T_i, with the diagonal fixed by the row sums."""
    n = np.asarray(transitions, float).copy()
    T = np.asarray(holding_time, float)
    if np.any(T <= 0):
        raise ValueError("every state needs positive holding time")
    np.fill_diagonal(n, 0.0)
    Q = n / T[:, None]
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q
