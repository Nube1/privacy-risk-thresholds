"""Governance thresholds: the decision record, the implied tolerance and the budget."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime

import numpy as np


@dataclass(frozen=True)
class GovernanceThreshold:
    tolerance: float
    owner: str
    expiry_date: str
    adversary: str
    harm_class: str
    alpha: float


def decide(upper_risk: float, threshold: GovernanceThreshold,
           validation_passed: bool, assessment_date: str | None = None) -> dict:
    assessed = (
        date.fromisoformat(assessment_date)
        if assessment_date
        else datetime.now(tz=UTC).date()
    )
    expiry = date.fromisoformat(threshold.expiry_date)
    if not validation_passed:
        status, reason = "INVALID", "prespecified validation failed"
    elif assessed >= expiry:
        status, reason = "FAIL", "assessment expired"
    elif upper_risk > threshold.tolerance:
        status, reason = "FAIL", "upper risk bound exceeds tolerance"
    else:
        status, reason = "PASS", "upper risk bound is within tolerance"
    return {"status": status, "reason": reason, "upper_risk": upper_risk,
            "assessment_date": assessed.isoformat(), "threshold": asdict(threshold)}


def implied_tolerance(upper_risk: np.ndarray, utility: np.ndarray, kappa: float) -> int:
    """Index of the setting that maximises utility minus `kappa` times the risk bound.

    The tolerance implied by a stated relative harm weight is the risk coordinate
    of that setting. The weight is decision utility per unit of attack risk.
    """
    upper_risk, utility = np.asarray(upper_risk, float), np.asarray(utility, float)
    if upper_risk.shape != utility.shape or upper_risk.size == 0:
        raise ValueError("risk and utility must be same-shaped and nonempty")
    if kappa <= 0:
        raise ValueError("the relative harm weight must be positive")
    return int(np.argmax(utility - kappa * upper_risk))


def risk_budget(weights: np.ndarray, programme_tolerance: float, gamma: float) -> np.ndarray:
    """Allocate a programme tolerance across releases of differing value.

    For the power value curve with exponent `gamma` below zero the first-order
    conditions give shares proportional to the weights raised to 1 / (1 - gamma).
    """
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0 or np.any(weights <= 0):
        raise ValueError("value weights must be positive")
    if not 0 < programme_tolerance <= 1:
        raise ValueError("the programme tolerance must lie in (0, 1]")
    if gamma >= 0:
        raise ValueError("the value curve exponent must be negative")
    exponent = 1.0 / (1.0 - gamma)
    shares = weights ** exponent
    return programme_tolerance * shares / shares.sum()


def budget_value(allocation: np.ndarray, weights: np.ndarray,
                 curve: tuple[float, float, float]) -> float:
    """Total weighted value of an allocation under a fitted power value curve."""
    a, b, gamma = curve
    allocation = np.asarray(allocation, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if np.any(allocation <= 0):
        raise ValueError("every release needs a positive share of the budget")
    return float(np.sum(weights * (a + b * allocation ** gamma)))
