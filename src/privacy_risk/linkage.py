"""Posterior record linkage and the Monte Carlo attack of the declared scenario."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from .synthetic import Register, Release


def posterior_linkage(likelihoods: np.ndarray, priors: np.ndarray,
                      no_match_likelihood: float = 0.0, no_match_prior: float = 0.0) -> np.ndarray:
    likelihoods = np.asarray(likelihoods, dtype=float)
    priors = np.asarray(priors, dtype=float)
    if likelihoods.shape != priors.shape or np.any(likelihoods < 0) or np.any(priors < 0):
        raise ValueError("likelihoods and priors must be same-shaped and nonnegative")
    weights = likelihoods * priors
    denominator = weights.sum() + no_match_likelihood * no_match_prior
    if denominator <= 0:
        raise ValueError("posterior denominator must be positive")
    return weights / denominator


def maximum_posterior_success(posteriors: np.ndarray) -> float:
    posteriors = np.asarray(posteriors, dtype=float)
    return float(posteriors.max(initial=0.0))


def membership_conditional_success(posteriors: np.ndarray, no_match_posterior: float) -> float:
    """Success given membership (Proposition 2.4): the largest posterior divided by 1 - p0.

    ``posteriors`` are the released-candidate posteriors of equation (2.5) and
    ``no_match_posterior`` is p_t0. The plain maximum understates success by 1 - p0.
    """
    if not 0 <= no_match_posterior < 1:
        raise ValueError("no_match_posterior must lie in [0, 1)")
    return maximum_posterior_success(posteriors) / (1.0 - no_match_posterior)


def exact_match_probabilities(target_keys: np.ndarray, release_keys: np.ndarray) -> np.ndarray:
    release_keys = np.asarray(release_keys)
    target_keys = np.asarray(target_keys)
    counts = {k: int(np.sum(release_keys == k)) for k in np.unique(release_keys)}
    return np.array([1.0 / counts[k] if counts.get(k, 0) else 0.0 for k in target_keys])


@dataclass(frozen=True)
class ErrorModel:
    """The adversary's declared information and the process that generates it.

    `covariate_sd` is the true noise in the adversary's reading of the confidential
    variable; `assumed_sd` is the noise its own model assumes. Setting the two apart
    is how a misspecified adversary is simulated.
    """

    age_error: float = 0.15
    occupation_error: float = 0.10
    covariate_sd: float = 0.75
    assumed_sd: float | None = None

    def __post_init__(self) -> None:
        for value in (self.age_error, self.occupation_error):
            if not 0 <= value < 1:
                raise ValueError("key error rates must be in [0, 1)")
        if self.covariate_sd <= 0 or (self.assumed_sd is not None and self.assumed_sd <= 0):
            raise ValueError("covariate standard deviations must be positive")

    @property
    def declared_sd(self) -> float:
        return self.covariate_sd if self.assumed_sd is None else self.assumed_sd

    def scaled(self, factor: float, matched: bool = True) -> ErrorModel:
        """A more or less capable adversary. `matched` keeps its model correct."""
        true_sd = self.covariate_sd * factor
        return ErrorModel(self.age_error, self.occupation_error, true_sd,
                          true_sd if matched else self.declared_sd)

    def misspecified(self, factor: float) -> ErrorModel:
        """The same information, wrongly modelled."""
        return ErrorModel(self.age_error, self.occupation_error, self.covariate_sd,
                          self.covariate_sd * factor)


@dataclass
class AttackResult:
    rate: float
    successes: int
    trials: int
    posterior: np.ndarray = field(default_factory=lambda: np.empty(0))
    success: np.ndarray = field(default_factory=lambda: np.empty(0))
    targets: np.ndarray = field(default_factory=lambda: np.empty(0))

    def as_dict(self) -> dict[str, float]:
        return {"rate": self.rate, "successes": self.successes, "trials": self.trials}


def _neighbours(band: int, bands: int) -> list[int]:
    return [b for b in (band - 1, band + 1) if 0 <= b <= bands - 1]


def _blocks(release: Release) -> dict[int, tuple[int, int]]:
    block = release.district * 2 + release.sex
    order = np.argsort(block, kind="stable")
    sorted_block = block[order]
    values, first = np.unique(sorted_block, return_index=True)
    ends = np.append(first[1:], sorted_block.size)
    return order, {int(b): (int(a), int(e)) for b, a, e in zip(values, first, ends)}


def attack(register: Register, release: Release, model: ErrorModel, replicates: int,
           rng: np.random.Generator, targets: np.ndarray | None = None,
           use_covariate: bool = True) -> AttackResult:
    """Simulate the declared adversary against one release.

    The adversary blocks on district and sex, which it observes without error, then
    scores every candidate in that block by the likelihood of the observed age band
    and occupation group under the same error model that generated them, multiplied by
    the density of its noisy covariate reading. It selects the maximum-posterior
    candidate and abstains only when the total likelihood vanishes.
    """
    if release.size == 0:
        return AttackResult(0.0, 0, 0)
    order, spans = _blocks(release)
    index, band, group = release.index[order], release.band[order], release.group[order]
    position = {int(r): i for i, r in enumerate(release.index.tolist())}
    outcome = register.frame.outcome.to_numpy()

    drawn = (rng.choice(release.index, size=int(replicates), replace=True)
             if targets is None else np.asarray(targets))
    trials = int(drawn.size)
    success = np.zeros(trials)
    posterior = np.zeros(trials)
    for i, target in enumerate(drawn.tolist()):
        slot = position[int(target)]
        block = int(release.district[slot]) * 2 + int(release.sex[slot])
        start, end = spans[block]
        candidates, cand_band, cand_group = index[start:end], band[start:end], group[start:end]
        seen_band, seen_group = int(release.band[slot]), int(release.group[slot])
        if rng.random() < model.age_error:
            options = _neighbours(seen_band, release.bands)
            if options:
                seen_band = options[int(rng.integers(0, len(options)))]
        if rng.random() < model.occupation_error and release.groups > 1:
            seen_group = int((seen_group + rng.integers(1, release.groups)) % release.groups)
        neighbours = np.maximum(2 - (cand_band == 0).astype(int)
                                - (cand_band == release.bands - 1).astype(int), 1)
        age_likelihood = np.where(cand_band == seen_band, 1 - model.age_error,
                                  np.where(np.abs(cand_band - seen_band) == 1,
                                           model.age_error / neighbours, 1e-12))
        group_likelihood = np.where(cand_group == seen_group, 1 - model.occupation_error,
                                    model.occupation_error / max(release.groups - 1, 1))
        likelihood = age_likelihood * group_likelihood
        if use_covariate:
            reading = outcome[int(target)] + rng.normal(0, model.covariate_sd)
            likelihood = likelihood * norm.pdf(reading, loc=outcome[candidates],
                                               scale=model.declared_sd)
        total = likelihood.sum()
        if total <= 0:
            continue
        weights = likelihood / total
        best = int(np.argmax(weights))
        posterior[i] = weights[best]
        success[i] = float(candidates[best] == target)
    return AttackResult(rate=float(success.mean()), successes=int(success.sum()), trials=trials,
                        posterior=posterior, success=success, targets=drawn)


def per_target_risk(register: Register, release: Release, model: ErrorModel,
                    targets: np.ndarray, replicates: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Risk of each named target, each estimated from `replicates` replicates."""
    targets = np.asarray(targets)
    if targets.size == 0:
        return np.empty(0)
    repeated = np.repeat(targets, int(replicates))
    result = attack(register, release, model, 0, rng, targets=repeated)
    return result.success.reshape(targets.size, int(replicates)).mean(axis=1)


def minimum_cell_targets(release: Release, minimum: int, cap: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Released records sitting in the smallest key classes, where 1/k is tightest."""
    key = release.key
    values, counts = np.unique(key, return_counts=True)
    size = dict(zip(values.tolist(), counts.tolist()))
    smallest = max(minimum, int(min(size.values())))
    chosen = release.index[np.array([size[int(k)] == smallest for k in key.tolist()])]
    if chosen.size > cap:
        chosen = rng.choice(chosen, size=cap, replace=False)
    return chosen
