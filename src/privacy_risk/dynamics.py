"""Changing exposure and activity, and the pooled attack of Chapter 19.

Log exposure follows an autoregression around a long-run level and activity follows a
logistic Markov transition. With phi = 1, sigma = 0 and no dependence on the past the
panel is frozen. Every loan is live in the first vintage.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class PanelModel:
    n_loans: int = 3000
    n_classes: int = 12
    vintages: int = 6
    level: float = 11.0
    start_sd: float = 1.2
    mu: float = 0.0
    phi: float = 0.95
    sigma: float = 0.15
    gamma0: float = 0.0
    gamma1: float = 1.5
    gamma2: float = 0.0

    @classmethod
    def static(cls, **kw) -> "PanelModel":
        """The frozen limit: exposure never moves, activity is independent of the past."""
        base = dict(phi=1.0, sigma=0.0, gamma0=2.0, gamma1=0.0, gamma2=0.0)
        base.update(kw)
        return cls(**base)


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_panel(model: PanelModel, rng: np.random.Generator) -> dict:
    n, v = model.n_loans, model.vintages
    loge = np.empty((n, v))
    live = np.zeros((n, v), dtype=bool)
    loge[:, 0] = rng.normal(model.level, model.start_sd, n)
    live[:, 0] = True
    for t in range(1, v):
        loge[:, t] = (model.level + model.mu + model.phi * (loge[:, t - 1] - model.level)
                      + model.sigma * rng.standard_normal(n))
        p = _expit(model.gamma0 + model.gamma1 * live[:, t - 1]
                   + model.gamma2 * (loge[:, t - 1] - model.level))
        live[:, t] = rng.random(n) < p
    return {"loge": loge, "live": live, "cls": rng.integers(0, model.n_classes, n)}


def pooled_attack(panel: dict, w: int, targets, know_activity: bool, s: float,
                  rng: np.random.Generator) -> tuple[int, int]:
    """Return (successes, targets). Presence and absence are both used."""
    loge, live, cls = panel["loge"], panel["live"], panel["cls"]
    v = loge.shape[1]
    if not 1 <= w <= v:
        raise ValueError("need 1 <= w <= number of vintages")
    targets = list(targets)
    hits = 0
    for j in targets:
        if not live[j, 0]:
            raise ValueError("a target must be live in the first vintage")
        cand = np.flatnonzero((cls == cls[j]) & live[:, 0])
        if know_activity:
            cand = cand[np.all(live[cand, :w] == live[j, :w], axis=1)]
        if s > 0:
            ts = np.flatnonzero(live[j, :w])
            y = loge[j, ts] + s * rng.standard_normal(ts.size)
            resid = np.where(live[np.ix_(cand, ts)], loge[np.ix_(cand, ts)] - y, 1e6)
            score = -0.5 * np.sum(resid ** 2, axis=1) / s ** 2
            cand = cand[score >= score.max()]
        hits += int(rng.choice(cand) == j)
    return hits, len(targets)
