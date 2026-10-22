"""Certification that survives optional stopping, and an adversary that stops when paid.

Two things in the fixed-sample account of Section \\ref{sec:est} do not hold in an
operation. The institution watches its simulation and is tempted to stop when the
bound first passes a tolerance, and the adversary watches the release programme and
stops when its posterior is worth acting on. This module implements the three
devices the paper uses for that setting:

* a beta--binomial e-process and the confidence sequence it induces, valid at every
  monitoring time rather than at one prearranged sample size;
* the posterior threshold at which an adversary with a declared payoff acts, and the
  precision and coverage of an attack read at that threshold;
* a transport bound from one attack distribution to a declared neighbourhood of it.

Nothing here is a claim about future adversaries. Each device converts an
assumption that is usually left in prose into a number the institution has to state.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import betaln


def log_e_process(successes: int, trials: int, risk: float,
                  prior: tuple[float, float] = (1.0, 1.0)) -> float:
    """Log of the mixture e-value at a candidate risk, from counts alone.

    The Bernoulli likelihood at ``risk`` is the null. The alternative mixes over
    the success probability with a beta prior, so the ratio is a nonnegative
    martingale under the null and Ville's inequality applies to its supremum.
    """
    if not 0.0 < risk < 1.0:
        return float("inf")
    a, b = prior
    successes, trials = int(successes), int(trials)
    failures = trials - successes
    null = successes * np.log(risk) + failures * np.log1p(-risk)
    return float(betaln(a + successes, b + failures) - betaln(a, b) - null)


def in_sequence(successes: int, trials: int, risk: float, alpha: float,
                prior: tuple[float, float] = (1.0, 1.0)) -> bool:
    """Whether a candidate risk is still in the confidence sequence."""
    return log_e_process(successes, trials, risk, prior) < np.log(1.0 / alpha)


def _boundary(successes: int, trials: int, alpha: float, lower: float, upper: float,
              prior: tuple[float, float], tolerance: float = 1e-10) -> float:
    """Bisect for the point where the e-process crosses the level."""
    threshold = np.log(1.0 / alpha)
    for _ in range(200):
        middle = 0.5 * (lower + upper)
        if log_e_process(successes, trials, middle, prior) < threshold:
            lower = middle
        else:
            upper = middle
        if upper - lower < tolerance:
            break
    # ``lower`` is the side still in the sequence and ``upper`` the excluded side.
    # Returning the excluded side keeps the interval conservative at both ends.
    return upper


def confidence_sequence(successes: int, trials: int, alpha: float = 0.05,
                        prior: tuple[float, float] = (1.0, 1.0)) -> tuple[float, float]:
    """The anytime-valid interval for the risk, given the counts so far.

    Coverage holds simultaneously at every monitoring time, so the institution may
    look after every replicate and stop when it likes.
    """
    trials = int(trials)
    if trials == 0:
        return 0.0, 1.0
    centre = min(max(successes / trials, 1e-9), 1 - 1e-9)
    if not in_sequence(successes, trials, centre, alpha, prior):
        return 0.0, 1.0
    low = _boundary(successes, trials, alpha, centre, 1e-12, prior)
    high = _boundary(successes, trials, alpha, centre, 1 - 1e-12, prior)
    return float(min(low, centre)), float(max(high, centre))


def anytime_upper(successes: int, trials: int, alpha: float = 0.05,
                  prior: tuple[float, float] = (1.0, 1.0)) -> float:
    """The upper end of the confidence sequence, the figure a gate would use."""
    return confidence_sequence(successes, trials, alpha, prior)[1]


def selected_anytime_upper(successes: int, trials: int, selected: int,
                           alpha: float = 0.05,
                           prior: tuple[float, float] = (1.0, 1.0)) -> float:
    """The same bound for a maximum chosen after inspection, by a union bound."""
    return anytime_upper(successes, trials, alpha / max(int(selected), 1), prior)


def certification_replicates(tolerance: float, alpha: float = 0.05,
                             prior: tuple[float, float] = (1.0, 1.0),
                             cap: int = 5_000_000) -> int:
    """Replicates needed to certify a tolerance while monitoring throughout.

    The fixed-sample answer solves $(1-\\tau)^B \\le \\alpha$. Here the run is
    watched, so the certificate has to hold at every look, and the count is the
    smallest number of successless attacks whose anytime bound falls below the
    tolerance.
    """
    low, high = 1, 1024
    while anytime_upper(0, high, alpha, prior) > tolerance:
        high *= 2
        if high > cap:
            return cap
    while low < high:
        middle = (low + high) // 2
        if anytime_upper(0, middle, alpha, prior) <= tolerance:
            high = middle
        else:
            low = middle + 1
    return int(low)


def attack_threshold(benefit: float, false_cost: float, action_cost: float) -> float:
    r"""The posterior at which an adversary with these payoffs acts.

    A correct allegation is worth ``benefit``, a false one costs ``false_cost``, and
    making any allegation costs ``action_cost``. The expected payoff at posterior
    $p$ is $ph-(1-p)c_f-c_a$, so the adversary acts above
    $(c_f+c_a)/(h+c_f)$. The threshold is one half whenever $h=c_f+2c_a$, so equal
    stakes with free allegations is one case among many. When $c_a\ge h$ the payoff
    is not positive even at $p=1$, and the adversary never acts: the threshold is
    then infinite.
    """
    if action_cost >= benefit:
        return math.inf
    denominator = benefit + false_cost
    return float((false_cost + action_cost) / denominator)


def precision_coverage(posteriors, correct, thresholds=None) -> list[dict]:
    """The precision and coverage of an attack as the acting threshold moves."""
    posteriors = np.asarray(posteriors, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if thresholds is None:
        thresholds = np.round(np.arange(0.1, 1.0, 0.1), 2)
    out = []
    for threshold in np.asarray(thresholds, dtype=float).tolist():
        acts = posteriors >= threshold
        decisions = int(acts.sum())
        out.append({
            "threshold": float(threshold),
            "coverage": decisions / posteriors.size if posteriors.size else float("nan"),
            "decisions": decisions,
            "precision": (float(correct[acts].mean()) if decisions else float("nan")),
            "yield": (float(correct[acts].sum()) / posteriors.size
                      if posteriors.size else float("nan"))})
    return out


def at_threshold(curve: list[dict], threshold: float) -> dict:
    """The point of a precision--coverage curve an adversary's payoff selects."""
    if not math.isfinite(threshold):
        return {"threshold": threshold, "coverage": 0.0, "precision": float("nan")}
    eligible = [row for row in curve if row["threshold"] >= threshold]
    return eligible[0] if eligible else curve[-1]


def transport_bound(risk: float, divergence: float) -> float:
    """Risk under any deployment law within a declared KL neighbourhood.

    Pinsker's inequality turns the divergence into a total variation distance, and
    the success indicator is bounded, so the risk moves by at most that distance.
    """
    return float(min(1.0, risk + np.sqrt(max(divergence, 0.0) / 2.0)))


def transport_divergence(risk: float, ceiling: float) -> float:
    """The divergence a stated risk ceiling can absorb, the inverse of the bound."""
    slack = max(ceiling - risk, 0.0)
    return float(2.0 * slack * slack)


def bernoulli_kl(a: float, b: float) -> float:
    """kl(a || b) for Bernoulli laws, in nats."""
    a = min(max(float(a), 0.0), 1.0)
    b = min(max(float(b), 1e-15), 1 - 1e-15)
    out = 0.0
    if a > 0:
        out += a * np.log(a / b)
    if a < 1:
        out += (1 - a) * np.log((1 - a) / (1 - b))
    return float(out)


def transport_bound_kl(risk: float, divergence: float) -> float:
    """The largest risk inside a KL neighbourhood, by data processing.

    If D_KL(Q || P) <= rho then kl(R_Q || R_P) <= rho for the binary success
    indicator, so R_Q is at most the largest r >= R_P with kl(r || R_P) <= rho. This is
    never looser than the Pinsker bound and is much tighter when R_P is small.
    """
    risk = float(risk)
    rho = max(float(divergence), 0.0)
    if risk >= 1.0:
        return 1.0
    low, high = risk, 1.0
    if bernoulli_kl(high, risk) <= rho:
        return 1.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if bernoulli_kl(middle, risk) <= rho:
            low = middle
        else:
            high = middle
    return float(low)


def transport_divergence_kl(risk: float, ceiling: float) -> float:
    """The divergence a risk ceiling can absorb under the data-processing bound."""
    return bernoulli_kl(ceiling, risk) if ceiling > risk else 0.0


def monitoring_study(true_risk: float, replicates: int, looks: int, runs: int,
                     alpha: float, rng: np.random.Generator,
                     prior: tuple[float, float] = (1.0, 1.0)) -> dict:
    """What peeking costs, measured rather than asserted.

    Each run simulates a stream of attacks against a release whose true risk is
    known. An institution that looks after every batch and stops the moment its
    fixed-sample bound falls below the truth is wrong more often than its stated
    level; the same rule applied to the confidence sequence is not.
    """
    from .uncertainty import binomial_upper

    schedule = np.linspace(replicates / looks, replicates, looks).astype(int)
    fixed_failures = 0
    anytime_failures = 0
    fixed_widths = []
    anytime_widths = []
    for _ in range(int(runs)):
        draws = rng.random(int(replicates)) < true_risk
        successes = np.cumsum(draws)
        fixed_bad = False
        anytime_bad = False
        for trials in schedule.tolist():
            hits = int(successes[trials - 1])
            if binomial_upper(hits, trials, alpha) < true_risk:
                fixed_bad = True
            if anytime_upper(hits, trials, alpha, prior) < true_risk:
                anytime_bad = True
        fixed_failures += int(fixed_bad)
        anytime_failures += int(anytime_bad)
        final = int(successes[-1])
        fixed_widths.append(binomial_upper(final, int(replicates), alpha))
        anytime_widths.append(anytime_upper(final, int(replicates), alpha, prior))
    return {"true_risk": float(true_risk), "replicates": int(replicates),
            "looks": int(looks), "runs": int(runs), "alpha": float(alpha),
            "fixed_error_rate": fixed_failures / int(runs),
            "anytime_error_rate": anytime_failures / int(runs),
            "fixed_upper_mean": float(np.mean(fixed_widths)),
            "anytime_upper_mean": float(np.mean(anytime_widths)),
            "width_ratio": float(np.mean(anytime_widths) / np.mean(fixed_widths))}
