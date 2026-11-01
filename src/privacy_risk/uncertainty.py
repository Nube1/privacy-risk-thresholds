"""Monte Carlo error, exact one-sided bounds and the cost of certifying a tolerance."""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.stats import beta, chi2


def binomial_upper(successes: int, trials: int, alpha: float = 0.05) -> float:
    """One-sided Clopper--Pearson upper confidence bound."""
    if trials <= 0 or not 0 <= successes <= trials or not 0 < alpha < 1:
        raise ValueError("invalid binomial inputs")
    if successes == trials:
        return 1.0
    return float(beta.ppf(1 - alpha, successes + 1, trials - successes))


def binomial_lower(successes: int, trials: int, alpha: float = 0.05) -> float:
    """One-sided Clopper--Pearson lower confidence bound."""
    if trials <= 0 or not 0 <= successes <= trials or not 0 < alpha < 1:
        raise ValueError("invalid binomial inputs")
    if successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, trials - successes + 1))


def monte_carlo_summary(success: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    success = np.asarray(success, dtype=bool)
    k, n = int(success.sum()), len(success)
    return {"successes": k, "trials": n, "estimate": k / n,
            "standard_error": float(np.sqrt((k / n) * (1 - k / n) / n)),
            "upper": binomial_upper(k, n, alpha)}


def certification_replicates(tolerance: float, alpha: float = 0.05) -> int:
    """Replicates needed to certify the bound at `tolerance` with no observed success."""
    if not 0 < tolerance < 1 or not 0 < alpha < 1:
        raise ValueError("tolerance and alpha must lie in (0, 1)")
    return math.ceil(math.log(alpha) / math.log(1 - tolerance))


def selected_maximum_bound(risks: np.ndarray, replicates: int,
                           alpha: float = 0.05,
                           counts: np.ndarray | None = None) -> dict[str, float]:
    """The worst target's risk with a lower bound corrected for selection.

    The worst target is chosen on its own estimate, so its limit is taken at
    `alpha` divided by the number of targets examined. Pass the success `counts`
    when they are available: rebuilding them from rates by rounding is exact only
    when every rate was computed from exactly `replicates` attacks.
    """
    risks = np.asarray(risks, dtype=float)
    if risks.size == 0:
        raise ValueError("no per-target risks supplied")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if counts is None:
        warnings.warn("selected_maximum_bound rebuilt success counts from rates; pass the "
                      "integer counts instead", RuntimeWarning, stacklevel=2)
        counts = np.rint(risks * replicates).astype(int)
    else:
        counts = np.asarray(counts, dtype=int)
        if counts.shape != risks.shape or np.any((counts < 0) | (counts > replicates)):
            raise ValueError("counts must match risks and lie in [0, replicates]")
    per_target_alpha = alpha / risks.size
    corrected = np.array([binomial_lower(c, replicates, per_target_alpha) for c in counts])
    individual = np.array([binomial_lower(c, replicates, alpha) for c in counts])
    return {"maximum": float(risks.max()), "maximum_lower_bound": float(corrected.max()),
            "targets": int(risks.size), "alpha_per_target": float(per_target_alpha),
            "mean": float(risks.mean()),
            "individual_lower_bounds": [float(x) for x in individual]}


def nested_variance(estimates: np.ndarray, replicates: int) -> dict[str, float]:
    """Split the variance of a nested run into its Monte Carlo and design parts.

    The interval on the design share is a chi-square interval for the total variance,
    which assumes the outer estimates are roughly normal; read it as approximate.

    `estimates` holds one risk estimate per outer replicate, each computed from
    `replicates` inner attacks. The sample variance of those estimates carries both
    sources, so the within part is subtracted to leave the between part, truncated
    at zero when the finite estimate falls slightly negative.
    """
    estimates = np.asarray(estimates, dtype=float)
    if estimates.size < 2 or replicates < 1:
        raise ValueError("need at least two outer replicates and one inner replicate")
    within = float(np.mean(estimates * (1 - estimates) / replicates))
    total = float(np.var(estimates, ddof=1))
    between = max(total - within, 0.0)
    mean = float(estimates.mean())
    naive = math.sqrt(mean * (1 - mean) / (replicates * estimates.size))
    honest = math.sqrt((between + within) / estimates.size)
    # a chi-square interval for the total variance, carried through to the share
    # the design contributes, so the split is not read as a point fact
    degrees = estimates.size - 1
    lower_total = degrees * total / chi2.ppf(0.975, degrees)
    upper_total = degrees * total / chi2.ppf(0.025, degrees)
    share_low = max(0.0, (lower_total - within) / lower_total) if lower_total > 0 else 0.0
    share_high = max(0.0, (upper_total - within) / upper_total) if upper_total > 0 else 0.0
    return {"outer_replicates": int(estimates.size), "inner_replicates": int(replicates),
            "between_share": between / total if total > 0 else 0.0,
            "between_share_interval": [float(share_low), float(share_high)],
            "mean": mean, "within_variance": within, "between_variance": between,
            "total_variance": total,
            "within_sd": math.sqrt(within), "between_sd": math.sqrt(between),
            "naive_standard_error": naive, "nested_standard_error": honest,
            "understatement_factor": honest / naive if naive > 0 else float("inf")}


def plugin_ordering(sample_frequency_risk: float, population_frequency_risk: float,
                    reciprocal_of_mean: float) -> dict[str, float | bool]:
    """Check the ordering the paper's plug-in proposition asserts."""
    return {"sample_frequency": sample_frequency_risk,
            "population_frequency": population_frequency_risk,
            "reciprocal_of_mean": reciprocal_of_mean,
            "ratio": (sample_frequency_risk / population_frequency_risk
                      if population_frequency_risk else float("inf")),
            "dominance_holds": sample_frequency_risk >= population_frequency_risk,
            "jensen_holds": sample_frequency_risk >= reciprocal_of_mean}


def simultaneous_upper_bounds(successes, trials, alpha: float = 0.05) -> dict:
    """Exact one-sided bounds for M frozen coordinates at level alpha / M each.

    By the union bound the chance that any coordinate's true risk exceeds its bound is
    at most alpha, without independence between coordinates. Choosing among candidates
    that pass every required coordinate stays valid; retrying a failed coordinate is a
    new attempt and needs its own share of a programme budget.
    """
    successes = [int(k) for k in successes]
    trials = [int(n) for n in trials]
    if len(successes) != len(trials) or not successes:
        raise ValueError("supply one trial count per success count")
    per = alpha / len(successes)
    bounds = [binomial_upper(k, n, per) for k, n in zip(successes, trials)]
    return {"alpha": alpha, "alpha_per_coordinate": per, "upper_bounds": bounds}


def beta_posterior_upper(successes: int, trials: int, prior_a: float = 1.0,
                         prior_b: float = 1.0, level: float = 0.95) -> float:
    """Upper posterior quantile under a Beta prior: a planning figure, not a certificate.

    It is conditional on the prior and the model. Certification uses
    `binomial_upper` on fresh audit draws.
    """
    if prior_a <= 0 or prior_b <= 0 or not 0 < level < 1:
        raise ValueError("invalid prior or level")
    if not 0 <= successes <= trials:
        raise ValueError("invalid counts")
    return float(beta.ppf(level, prior_a + successes, prior_b + trials - successes))


def posterior_predictive_counts(risk_values, weights, trials: int) -> np.ndarray:
    """P(K_new = k) for k = 0..trials under a weighted mixture of binomials.

    Plugging the posterior mean risk into one binomial omits parameter uncertainty;
    this keeps it.
    """
    from scipy.stats import binom
    risk_values = np.asarray(risk_values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if risk_values.shape != weights.shape or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("risk values and nonnegative weights must match")
    if np.any((risk_values < 0) | (risk_values > 1)):
        raise ValueError("risk values must lie in [0, 1]")
    weights = weights / weights.sum()
    k = np.arange(int(trials) + 1)
    return np.sum(weights[:, None] * binom.pmf(k[None, :], int(trials), risk_values[:, None]),
                  axis=0)
