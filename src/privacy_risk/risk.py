"""Scenario-conditional risk measures and the population-frequency model."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize
from scipy.stats import nbinom

from .keys import key_frequencies


def prosecutor_risk(sample_keys: np.ndarray) -> np.ndarray:
    counts = key_frequencies(np.asarray(sample_keys))
    return np.array([1.0 / counts[int(k)] for k in sample_keys])


def marketer_sample_risk(sample_keys: np.ndarray) -> float:
    keys = np.asarray(sample_keys)
    return len(np.unique(keys)) / len(keys)


def marketer_population_frequency_risk(sample_keys: np.ndarray,
                                       population_keys: np.ndarray) -> float:
    frequencies = key_frequencies(np.asarray(population_keys))
    return float(np.mean([1.0 / frequencies[int(k)] for k in sample_keys]))


def marketer_unconditional_risk(sample_keys: np.ndarray, sampling_fraction: float) -> float:
    """The measure of the unconditional marketer experiment: pi times the in-file risk."""
    if not 0 < sampling_fraction <= 1:
        raise ValueError("sampling fraction must be in (0, 1]")
    return float(sampling_fraction * prosecutor_risk(sample_keys).mean())


def journalist_unique_probability(pi: float, frequency_prior: dict[int, float]) -> float:
    """Journalist risk under a declared prior probability mass function for F_k."""
    if not 0 < pi <= 1 or not math.isclose(sum(frequency_prior.values()), 1.0, abs_tol=1e-9):
        raise ValueError("invalid sampling probability or frequency prior")
    numerator = pi * frequency_prior.get(1, 0.0)
    denominator = sum(m * pi * (1 - pi) ** (m - 1) * p
                      for m, p in frequency_prior.items() if m >= 1)
    return numerator / denominator if denominator else 0.0


@dataclass(frozen=True)
class FrequencyFit:
    """A negative binomial law for the population key frequencies.

    Binomial thinning of a negative binomial law is negative binomial with the same
    dispersion, so the population law is identified from the sample cell counts.
    """

    mean: float
    dispersion: float

    def probabilities(self, upper: int = 600) -> tuple[np.ndarray, np.ndarray]:
        support = np.arange(1, upper)
        p = self.dispersion / (self.dispersion + self.mean)
        return support, nbinom.pmf(support, self.dispersion, p)


def fit_frequency_model(sample_counts: np.ndarray, cells: int,
                        sampling_fraction: float) -> FrequencyFit:
    """Fit the population frequency law to observed sample cell counts, zeros included."""
    counts = np.asarray(sample_counts, dtype=int)
    if counts.size > cells:
        raise ValueError("more occupied cells than the key space allows")
    observed = np.concatenate([counts, np.zeros(cells - counts.size, dtype=int)])

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        mean, dispersion = math.exp(parameters[0]), math.exp(parameters[1])
        return -float(np.sum(nbinom.logpmf(observed, dispersion,
                                           dispersion / (dispersion + mean))))

    start = [math.log(max(observed.mean(), 1e-3)), 0.5]
    fit = optimize.minimize(negative_log_likelihood, start, method="Nelder-Mead",
                            options={"maxiter": 6000, "fatol": 1e-10})
    return FrequencyFit(mean=math.exp(fit.x[0]) / sampling_fraction,
                        dispersion=math.exp(fit.x[1]))


def journalist_from_fit(pi: float, fit: FrequencyFit) -> float:
    """Journalist risk evaluated under a fitted frequency law rather than a stated prior."""
    support, mass = fit.probabilities()
    numerator = pi * float(mass[0])
    denominator = float(np.sum(support * pi * (1 - pi) ** (support - 1) * mass))
    return numerator / denominator if denominator else 0.0


def journalist_truth(sample_keys: np.ndarray, population_keys: np.ndarray) -> float:
    """The quantity the model estimates, computed directly because the population is known."""
    sample_counts = key_frequencies(np.asarray(sample_keys))
    population_counts = key_frequencies(np.asarray(population_keys))
    unique = [k for k, count in sample_counts.items() if count == 1]
    if not unique:
        return float("nan")
    return float(np.mean([population_counts[k] == 1 for k in unique]))
