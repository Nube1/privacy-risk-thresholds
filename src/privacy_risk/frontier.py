"""The risk--utility feasible set, its frontier and a fitted value curve."""
from __future__ import annotations

import numpy as np
from scipy import optimize


def nondominated(risk: np.ndarray, utility: np.ndarray) -> np.ndarray:
    risk, utility = np.asarray(risk), np.asarray(utility)
    if risk.shape != utility.shape:
        raise ValueError("risk and utility must have the same shape")
    keep = np.ones(len(risk), dtype=bool)
    for i in range(len(risk)):
        dominates = ((risk <= risk[i]) & (utility >= utility[i])
                     & ((risk < risk[i]) | (utility > utility[i])))
        keep[i] = not np.any(dominates)
    return keep


def choose_max_utility(settings: np.ndarray, upper_risk: np.ndarray,
                       utility: np.ndarray, tolerance: float):
    feasible = np.asarray(upper_risk) <= tolerance
    if not np.any(feasible):
        return None
    candidates = np.flatnonzero(feasible)
    return settings[candidates[np.argmax(np.asarray(utility)[candidates])]]


def value_curve(risk: np.ndarray, a: float, b: float, gamma: float) -> np.ndarray:
    return a + b * np.asarray(risk, dtype=float) ** gamma


def fit_value_curve(risk: np.ndarray, utility: np.ndarray) -> dict[str, float]:
    """Fit an increasing concave power curve to the frontier points."""
    risk, utility = np.asarray(risk, float), np.asarray(utility, float)
    if risk.size < 4:
        raise ValueError("the fit needs at least four frontier points")
    parameters, _ = optimize.curve_fit(value_curve, risk, utility,
                                       p0=[-0.05, -0.001, -1.0], maxfev=400_000)
    predicted = value_curve(risk, *parameters)
    residual = float(np.sum((utility - predicted) ** 2))
    total = float(np.sum((utility - utility.mean()) ** 2))
    return {"a": float(parameters[0]), "b": float(parameters[1]),
            "gamma": float(parameters[2]),
            "r_squared": 1 - residual / total if total else float("nan"),
            "points": int(risk.size)}


def concavity_diagnostic(risk: np.ndarray, utility: np.ndarray) -> dict:
    """Successive slopes of the frontier, and whether they decline as concavity needs.

    A discrete mechanism family generally fails this, which is why the existence
    argument is stated with semicontinuity rather than concavity.
    """
    order = np.argsort(np.asarray(risk, float))
    r, u = np.asarray(risk, float)[order], np.asarray(utility, float)[order]
    slopes = np.diff(u) / np.diff(r)
    declining = bool(np.all(np.diff(slopes) <= 1e-12))
    return {"slopes": slopes.tolist(), "is_concave": declining,
            "largest_slope": float(slopes.max()), "smallest_slope": float(slopes.min())}
