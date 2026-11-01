"""Monte Carlo error, exact one-sided bounds and the cost of certifying a tolerance."""
from __future__ import annotations

import math

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
                           successes: np.ndarray | None = None) -> dict[str, float]:
    """The worst target's risk with a lower bound corrected for selection.

    The worst target is chosen on its own estimate, so its limit is taken at
    `alpha` divided by the number of targets examined. Pass the integer
    `successes` out of `replicates` for each target when they are known; without
    them the counts are rebuilt by rounding `risks * replicates`, which is exact
    only when each risk is a count over `replicates` attacks.
    """
    risks = np.asarray(risks, dtype=float)
    if risks.size == 0:
        raise ValueError("no per-target risks supplied")
    if successes is None:
        counts = np.rint(risks * replicates).astype(int)
    else:
        counts = np.asarray(successes)
        if counts.shape != risks.shape or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError("successes must be integers with one count per target")
        if np.any(counts < 0) or np.any(counts > replicates):
            raise ValueError("each success count must lie between 0 and replicates")
    per_target_alpha = alpha / risks.size
    corrected = np.array([binomial_lower(c, replicates, per_target_alpha) for c in counts])
    individual = np.array([binomial_lower(c, replicates, alpha) for c in counts])
    return {"maximum": float(risks.max()), "maximum_lower_bound": float(corrected.max()),
            "targets": int(risks.size), "alpha_per_target": float(per_target_alpha),
            "mean": float(risks.mean()),
            "individual_lower_bounds": [float(x) for x in individual]}


def nested_variance(estimates: np.ndarray, replicates: int) -> dict[str, float]:
    """Split the variance of a nested run into its Monte Carlo and design parts.

    `estimates` holds one risk estimate per outer replicate, each computed from
    `replicates` inner attacks. The sample variance of those estimates carries both
    sources, so the within part is subtracted to leave the between part, truncated
    at zero when the finite estimate falls slightly negative. Any interval built
    from the result by a chi-square argument assumes the outer estimates are roughly
    normal, which is a poor description when the risk is near zero or the number of
    outer replicates is small.
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
