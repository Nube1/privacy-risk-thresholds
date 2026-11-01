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


def attack_ceiling(epsilon: float, baseline: float, delta: float = 0.0) -> float:
    """Ceiling on the success of any attack at once: min(1, e^epsilon * baseline + delta).

    `baseline` is the success the attack would have without the release (the largest
    prior probability of the right answer). The ceiling holds for every attack that
    sees only an (epsilon, delta)-differentially private release of the data, but it
    is a ceiling, not the success of a particular attack.
    """
    if epsilon < 0 or not 0 <= baseline <= 1 or not 0 <= delta <= 1:
        raise ValueError("need epsilon >= 0 and baseline and delta in [0, 1]")
    return float(min(1.0, math.exp(epsilon) * baseline + delta))


def advanced_composition(epsilon: float, k: int, delta_slack: float) -> float:
    """Total epsilon for k adaptive (epsilon, 0) releases, with slack delta_slack added to delta.

    sqrt(2 k ln(1/delta')) * epsilon + k * epsilon * (exp(epsilon) - 1); never worse than
    the basic total k * epsilon.
    """
    if epsilon < 0 or k < 1 or not 0 < delta_slack < 1:
        raise ValueError("need epsilon >= 0, k >= 1 and 0 < delta_slack < 1")
    tight = (math.sqrt(2 * k * math.log(1 / delta_slack)) * epsilon
             + k * epsilon * (math.exp(epsilon) - 1))
    return float(min(k * epsilon, tight))


def laplace_epsilon(scale: float, sensitivity: float = 1.0) -> float:
    """Epsilon of Laplace noise of the given scale for a query of the given sensitivity.

    A count changes by 1 when a record is added or removed and by up to 2 when one
    record is replaced, so the same noise of scale 1/epsilon gives epsilon under
    add-or-remove and 2 * epsilon under replace-one. The guarantee is per record: a
    borrower with several loans needs the sensitivity to count all of them.
    """
    if scale <= 0 or sensitivity < 0:
        raise ValueError("need scale > 0 and sensitivity >= 0")
    return float(sensitivity / scale)


def postprocessing_bound(p_d, p_dprime, h, epsilon: float, delta: float = 0.0) -> tuple[float, float, bool]:
    """Proposition 21.8 for a finite output set.

    ``p_d`` and ``p_dprime`` are the mechanism's output distributions on neighbouring
    datasets and ``h`` is the probability of the event B after a data-independent
    computation. Returns the two sides of (21.7) and whether the inequality holds.
    """
    p_d = np.asarray(p_d, float)
    p_dprime = np.asarray(p_dprime, float)
    h = np.asarray(h, float)
    if not (p_d.shape == p_dprime.shape == h.shape):
        raise ValueError("shapes must match")
    if np.any(h < 0) or np.any(h > 1):
        raise ValueError("h must lie in [0, 1]")
    left = float(p_d @ h)
    right = float(math.exp(epsilon) * (p_dprime @ h) + delta)
    return left, right, left <= right + 1e-12
