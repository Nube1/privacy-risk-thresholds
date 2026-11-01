"""Falsification tests. Each one can reject the model that the risk figure rests on."""
from __future__ import annotations

from itertools import pairwise

import numpy as np


def expected_calibration_error(probability: np.ndarray, outcome: np.ndarray,
                               bins: int = 10) -> float:
    p, y = np.asarray(probability, float), np.asarray(outcome, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in pairwise(edges):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def calibration_table(probability: np.ndarray, outcome: np.ndarray,
                      edges: np.ndarray | None = None,
                      minimum: int = 100) -> list[dict[str, float]]:
    """Binned posterior against realised success, for the calibration figure."""
    p, y = np.asarray(probability, float), np.asarray(outcome, float)
    edges = (np.array([0, .05, .1, .2, .3, .5, .7, .9, 1.0001])
             if edges is None else np.asarray(edges, float))
    rows = []
    for lo, hi in pairwise(edges):
        mask = (p >= lo) & (p < hi)
        if mask.sum() >= minimum:
            rows.append({"low": float(lo), "high": float(hi), "count": int(mask.sum()),
                         "mean_posterior": float(p[mask].mean()),
                         "empirical": float(y[mask].mean())})
    return rows


def calibration_gap(rows: list[dict[str, float]]) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([abs(r["empirical"] - r["mean_posterior"]) for r in rows]))


def monotonicity_violations(protection: np.ndarray, risk: np.ndarray,
                            tolerance: float = 0.0):
    order = np.argsort(protection)
    deltas = np.diff(np.asarray(risk)[order])
    return np.flatnonzero(deltas > tolerance).tolist()


def frequency_transport_error(fitted_mean: float, true_mean: float) -> float:
    """Absolute error of the fitted population cell size against the benchmark."""
    return float(abs(fitted_mean - true_mean))


def stress_ratios(declared: np.ndarray, stronger: np.ndarray,
                  tolerance: float = 0.0) -> dict[str, float]:
    """How far a materially stronger adversary moves the risk of each setting.

    The test asks whether the more capable adversary is ever materially weaker, which
    would mean the capability was misdeclared or the simulation is wrong. `tolerance`
    is the allowance for simulation error, so a small negative movement is not read as
    a failure.
    """
    declared, stronger = np.asarray(declared, float), np.asarray(stronger, float)
    if declared.shape != stronger.shape or declared.size == 0:
        raise ValueError("declared and stronger risks must be same-shaped and nonempty")
    usable = declared > 0
    ratios = stronger[usable] / declared[usable]
    return {"minimum": float(ratios.min()), "median": float(np.median(ratios)),
            "maximum": float(ratios.max()), "tolerance": float(tolerance),
            "stronger_always_larger": bool(np.all(stronger >= declared)),
            "stronger_never_materially_lower": bool(ratios.min() >= 1 - tolerance)}


def composition_test(pooled: float, union_upper: float,
                     programme_tolerance: float | None = None,
                     pooled_upper: float | None = None) -> dict[str, float | bool]:
    """Compare the pooled attack with the interval the marginals imply.

    A pooled risk outside that interval does not invalidate the release-level model.
    It establishes that the programme, and not the single release, is the object that
    needs governing, so it is reported against a programme tolerance of its own.
    """
    report: dict[str, float | bool] = {
        "pooled": float(pooled), "union_upper": float(union_upper),
        "ratio": float(pooled / union_upper) if union_upper else float("inf"),
        "inside_marginal_bounds": bool(pooled <= union_upper + 1e-12),
        "marginals_bound_the_programme": bool(pooled <= union_upper + 1e-12)}
    if pooled_upper is not None:
        report["pooled_upper"] = float(pooled_upper)
    if programme_tolerance is not None:
        # judged by the upper bound when one is supplied, never by the point estimate
        judged = float(pooled_upper) if pooled_upper is not None else float(pooled)
        report["programme_tolerance"] = float(programme_tolerance)
        report["upper_within_programme_tolerance"] = bool(judged <= programme_tolerance)
    return report


def validation_report(ece: float, ece_limit: float, frequency_mae: float,
                      frequency_mae_limit: float, violations: list[int]) -> dict:
    checks = {"linkage_calibration": ece <= ece_limit,
              "frequency_transport": frequency_mae <= frequency_mae_limit,
              "mechanism_monotonicity": len(violations) == 0}
    return {"passed": all(checks.values()), "checks": checks,
            "metrics": {"ece": ece, "frequency_mae": frequency_mae,
                        "monotonicity_violations": violations}}


def full_validation_report(ece: float, ece_limit: float, frequency_mae: float,
                           frequency_mae_limit: float, violations: list[int],
                           stress: dict, composition: dict,
                           misspecified_ece: float | None = None) -> dict:
    """Release-level validation, plus the programme evidence recorded beside it.

    The composition result is reported, not folded into release validity: a pooled
    attack that beats the marginal bound says the programme needs its own tolerance,
    which is a different decision from whether this release may go out.
    """
    report = validation_report(ece, ece_limit, frequency_mae, frequency_mae_limit, violations)
    report["checks"]["adversary_stress"] = bool(stress["stronger_never_materially_lower"])
    if misspecified_ece is not None:
        # a band wide enough to pass a known misspecified model cannot show the declared
        # model wrong either, so the test is only valid if it rejects that model
        report["checks"]["calibration_band_has_power"] = bool(misspecified_ece > ece_limit)
        report["metrics"]["ece_misspecified"] = float(misspecified_ece)
    report["metrics"]["ece_limit"] = float(ece_limit)
    report["passed"] = all(report["checks"].values())
    report["metrics"]["stress"] = stress
    report["programme"] = composition
    return report
