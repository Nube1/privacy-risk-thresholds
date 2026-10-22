"""Neural proposal and independent simultaneous audit for release decisions.

The model ranks candidates only. Audit successes must come from fresh attack
draws, independent of development data conditional on each fixed scenario.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .uncertainty import binomial_upper


@dataclass(frozen=True)
class ReleaseEvidence:
    name: str
    successes: np.ndarray  # one independent audit per scenario and harm class
    trials: np.ndarray
    tolerance: np.ndarray
    utility: float


def upper_binomial(successes: np.ndarray, trials: np.ndarray, alpha: float) -> np.ndarray:
    """Exact one-sided Clopper--Pearson bound, including the zero-success case."""
    s = np.asarray(successes)
    n = np.asarray(trials)
    if (s.shape != n.shape or not np.issubdtype(s.dtype, np.integer)
            or not np.issubdtype(n.dtype, np.integer)
            or np.any(s < 0) or np.any(n < 1) or np.any(s > n)):
        raise ValueError("successes/trials must have equal shapes and 0 <= s <= n")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    return np.array([binomial_upper(int(k), int(b), alpha)
                     for k, b in zip(s.flat, n.flat)]).reshape(s.shape)


def zero_success_budget(tolerance: float, alpha: float, total_tests: int) -> int:
    """Smallest trial count whose exact zero-success upper bound passes.

    This budget only applies if no attack succeeds; it is not a promised
    sample size for a coordinate with a positive count.
    """
    if (not 0 < tolerance < 1 or not 0 < alpha < 1
            or isinstance(total_tests, bool) or not isinstance(total_tests, int)
            or total_tests < 1):
        raise ValueError("require 0 < tolerance, alpha < 1 and positive test count")
    raw = math.log(alpha / total_tests) / math.log1p(-tolerance)
    budget = max(1, math.ceil(raw))
    # Floating-point rounding at an integer boundary cannot under-budget.
    while -math.expm1(math.log(alpha / total_tests) / budget) > tolerance:
        budget += 1
    return budget


def certify(evidence: ReleaseEvidence, alpha: float, total_tests: int) -> dict:
    """Simultaneous finite-family gate. Never uses neural output to certify."""
    if total_tests < evidence.successes.size:
        raise ValueError("total_tests must cover every inspected coordinate")
    if (evidence.tolerance.shape != evidence.successes.shape
            or evidence.successes.ndim != 1
            or np.any((evidence.tolerance <= 0) | (evidence.tolerance >= 1))
            or not np.isfinite(evidence.tolerance).all()):
        raise ValueError("tolerance dimension mismatch")
    upper = upper_binomial(evidence.successes, evidence.trials, alpha / total_tests)
    position = (evidence.tolerance - upper) / evidence.tolerance
    return {"name": evidence.name, "upper": upper.tolist(),
            "trust_position": position.tolist(), "certified": bool(np.all(position >= 0)),
            "utility": evidence.utility}


def propose_and_audit(
    model: MLP, candidate_features: np.ndarray, candidate_names: list[str],
    utilities: np.ndarray, tolerances: np.ndarray,
    audit: Callable[[str], tuple[np.ndarray, np.ndarray]], *,
    shortlist_size: int, alpha: float = .05,
) -> dict:
    """Rank with the model, audit the fixed shortlist, and choose by utility.

    ``audit(name)`` returns independent successes and trials for *every*
    declared coordinate. The caller owns independence and scenario design.
    No audit result is permitted to influence model fitting or shortlist size.
    """
    x = np.asarray(candidate_features, dtype=float)
    utility = np.asarray(utilities, dtype=float)
    tau = np.asarray(tolerances, dtype=float)
    if (x.ndim != 2 or x.shape[0] != len(candidate_names)
            or utility.shape != (len(candidate_names),)
            or tau.shape != (model.b2.size,) or np.any((tau <= 0) | (tau >= 1))
            or not np.isfinite(x).all() or not np.isfinite(utility).all()
            or len(set(candidate_names)) != len(candidate_names)
            or not 1 <= shortlist_size <= len(candidate_names)):
        raise ValueError("invalid candidates, utilities, tolerances or shortlist size")
    predictions = model.predict(x)
    rank = np.argsort(np.max(predictions / tau[None, :], axis=1), kind="stable")
    shortlist = rank[:shortlist_size]
    reports = []
    for j in shortlist:
        s, n = audit(candidate_names[j])
        evidence = ReleaseEvidence(candidate_names[j], np.asarray(s),
                                   np.asarray(n), tau, float(utility[j]))
        report = certify(evidence, alpha, shortlist_size * tau.size)
        report["neural_estimate_for_ranking_only"] = predictions[j].tolist()
        report["observed_successes"] = evidence.successes.tolist()
        report["trials"] = evidence.trials.tolist()
        reports.append(report)
    eligible = [r for r in reports if r["certified"]]
    return {"selected": max(eligible, key=lambda r: r["utility"])["name"]
            if eligible else None, "alpha_family": alpha,
            "audited_coordinates": shortlist_size * tau.size, "reports": reports}


class MLP:
    """One-hidden-layer ReLU network (depth L=1) with explicit numpy backpropagation.

    ``sentry_r.DeepRiskNet`` is the depth-configurable network.
    """

    def __init__(self, dimension: int, width: int, seed: int, outputs: int = 1):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, np.sqrt(2 / dimension), (dimension, width))
        self.b1 = np.zeros(width)
        self.w2 = rng.normal(0, np.sqrt(2 / width), (width, outputs))
        self.b2 = np.zeros(outputs)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return 1 / (1 + np.exp(-np.clip((np.maximum(x @ self.w1 + self.b1, 0)
                                                @ self.w2 + self.b2), -40, 40)))

    def fit(self, x: np.ndarray, y: np.ndarray, epochs=250, lr=0.02,
            weights: np.ndarray | None = None):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.ndim != 2 or y.shape != (len(x), self.b2.size) or not np.isfinite(x).all():
            raise ValueError("invalid training shape or nonfinite features")
        if np.any((y < 0) | (y > 1)):
            raise ValueError("labels must be probabilities in [0,1]")
        w = np.ones_like(y) if weights is None else np.asarray(weights, float)
        if (w.shape != y.shape or not np.isfinite(w).all() or np.any(w < 0)
                or np.any(w.sum(axis=0) == 0)):
            raise ValueError("weights must be nonnegative with observations per output")
        parameters = [self.w1, self.b1, self.w2, self.b2]
        first = [np.zeros_like(p) for p in parameters]
        second = [np.zeros_like(p) for p in parameters]
        for t in range(1, epochs + 1):
            z = x @ self.w1 + self.b1
            h = np.maximum(z, 0)
            logit = np.clip(h @ self.w2 + self.b2, -40, 40)
            p = 1 / (1 + np.exp(-logit))
            dz2 = (p - y) * w / len(x)  # weighted Bernoulli cross entropy
            grads = [x.T @ ((dz2 @ self.w2.T) * (z > 0)),
                     np.sum((dz2 @ self.w2.T) * (z > 0), axis=0),
                     h.T @ dz2, np.sum(dz2, axis=0)]
            for i, (param, grad) in enumerate(zip(parameters, grads)):
                first[i] = .9 * first[i] + .1 * grad
                second[i] = .999 * second[i] + .001 * grad * grad
                param -= lr * (first[i] / (1 - .9 ** t)) / (
                    np.sqrt(second[i] / (1 - .999 ** t)) + 1e-8)
        return self


def synthetic_demo(seed: int = 19, dimension: int = 8) -> dict:
    """Disjoint toy development and final audits; no claims about real data."""
    if dimension < 2:
        raise ValueError("dimension must be >= 2")
    rng = np.random.default_rng(seed)
    x_train = rng.normal(size=(5000, dimension))
    # Unknown nonlinear attack success process for demonstration only.
    def risk(x):
        z = -3.0 + .9*x[:, 0] - .6*x[:, 1] + .4*x[:, 0]*x[:, 1]
        if dimension > 2:
            z += .35 * np.sin(x[:, 2])
        return 1 / (1 + np.exp(-np.clip(z, -40, 40)))
    base_train = risk(x_train)
    y_train = rng.binomial(1, np.column_stack([base_train,
                                                np.minimum(1, 1.25*base_train)]))
    net = MLP(dimension, min(32, 4*dimension), seed, outputs=2).fit(x_train, y_train)
    candidate_x = rng.normal(size=(12, dimension))
    # Candidate selection uses development only; final Bernoulli audit is fresh.
    def audit(name: str) -> tuple[np.ndarray, np.ndarray]:
        j = int(name.split("_")[-1])
        base = risk(candidate_x[j:j+1])[0]
        probabilities = np.array([base, min(1, 1.25*base)])
        n = np.full(2, 1200)
        return rng.binomial(n, probabilities), n

    result = propose_and_audit(net, candidate_x,
                               [f"setting_{j}" for j in range(len(candidate_x))],
                               candidate_x[:, 1], np.array([.12, .18]), audit,
                               shortlist_size=4)
    return {"seed": seed, "dimension": dimension, **result,
            "status": "synthetic demonstration; no empirical validation"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output", type=Path, default=Path("neural_demo.json"))
    args = parser.parse_args()
    if args.dimension < 2:
        parser.error("--dimension must be at least 2")
    result = synthetic_demo(args.seed, args.dimension)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("selected", "audited_coordinates", "status")}))


if __name__ == "__main__":
    main()
