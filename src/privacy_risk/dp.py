"""What a formal privacy budget permits, stated as a posterior bound.

Differential privacy bounds the movement of posterior odds. It does not name a
linkage probability, and the probability it permits depends on the prior the
adversary brings, so the two regimes must not be compared on one numerical scale.
"""
from __future__ import annotations

import math

import numpy as np


def posterior_bound(epsilon: float, prior: float) -> float:
    """Largest posterior an (epsilon, 0)-differentially private release permits.

    With delta > 0 no uniform posterior bound exists, so this function covers the
    pure case only.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    if not 0 <= prior <= 1:
        raise ValueError("the prior must be a probability")
    factor = math.exp(epsilon)
    return float(factor * prior / (factor * prior + (1 - prior)))


def posterior_bound_table(epsilons: np.ndarray, priors: np.ndarray) -> list[dict[str, float]]:
    return [{"epsilon": float(e), "prior": float(p), "bound": posterior_bound(e, p)}
            for e in np.asarray(epsilons, float) for p in np.asarray(priors, float)]


def basic_composition(epsilons: np.ndarray, deltas: np.ndarray | None = None
                      ) -> dict[str, float]:
    """Basic adaptive composition, which needs no assumption about the adversary."""
    epsilons = np.asarray(epsilons, dtype=float)
    if np.any(epsilons < 0):
        raise ValueError("epsilons must be nonnegative")
    if deltas is None:
        total_delta = 0.0
    else:
        deltas = np.asarray(deltas, dtype=float)
        if deltas.shape != epsilons.shape:
            raise ValueError("supply one delta per epsilon")
        if np.any(deltas < 0) or np.any(deltas >= 1):
            raise ValueError("deltas must lie in [0, 1)")
        total_delta = float(deltas.sum())
    return {"epsilon_total": float(epsilons.sum()), "delta_total": total_delta}
