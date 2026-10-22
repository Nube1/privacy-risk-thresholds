"""An adversarial evaluation of a credit-release system.

The rest of the package measures risk for one release under one rule. This module
does what a security evaluation asks for instead: it fixes a set of attacks, runs
them against the protection policies a statistical office actually operates, and
reports the outcome in the terms an attack is judged by -- precision, recall,
posterior success probability, worst case over subgroups, and what the attacker
must know to mount it.

The policies compared are:

* ``none`` -- loan-level release with no suppression, the isolated-release baseline;
* ``min_contributors`` -- the operational rule that a cell must have at least three
  contributing lenders;
* ``k_anonymity`` -- suppression of key classes with fewer than k borrowers;
* ``complementary`` -- primary suppression by contributor count, with the smallest
  remaining cell in each region suppressed as well, so that a suppressed cell
  cannot be recovered from the published row total;
* ``dp_counts`` -- no microdata: counts per key class released with Laplace noise;
* ``programme`` -- the rule of this paper, whose parameter is chosen so that the
  *pooled* attack over the whole programme of vintages meets the tolerance.

The distinction that matters is the last one. Every operational policy above is
calibrated against a single release. The pooled attack combines the vintages, and
the evaluation reports how far a per-release calibration is from a programme-level
one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from .credit import (CreditRegister, central_bank_utility, relationship_disclosure,
                     status_inference)
from .sequential import precision_coverage
from .uncertainty import binomial_upper

DECISION_THRESHOLD = 0.9        # the attribute rule: act when the class is near-homogeneous
IDENTIFICATION_THRESHOLD = 0.5  # linkage: act when the best candidate is more likely than not


@dataclass(frozen=True)
class Policy:
    """A release rule, and what it publishes."""

    name: str
    kind: str  # "microdata" or "counts"
    parameter: float = 0.0
    label: str = ""
    epsilon: float = 0.0


@dataclass
class Outcome:
    """One attack against one policy."""

    policy: str
    attack: str
    decisions: int = 0
    true_positive: int = 0
    positives: int = 0
    trials: int = 0
    posterior: float = float("nan")
    detail: dict = field(default_factory=dict)

    @property
    def precision(self) -> float:
        return self.true_positive / self.decisions if self.decisions else float("nan")

    @property
    def recall(self) -> float:
        return self.true_positive / self.positives if self.positives else float("nan")

    @property
    def success(self) -> float:
        return self.true_positive / self.trials if self.trials else float("nan")

    def record(self, alpha: float = 0.05) -> dict:
        out = {"policy": self.policy, "attack": self.attack,
               "decisions": self.decisions, "true_positive": self.true_positive,
               "positives": self.positives, "trials": self.trials,
               "precision": self.precision, "recall": self.recall,
               "success": self.success, "posterior": self.posterior}
        if self.trials:
            out["success_upper"] = binomial_upper(self.true_positive, self.trials, alpha)
        out.update(self.detail)
        return out


def policies(settings: dict) -> list[Policy]:
    """The comparison set, in the order the paper's table reports it."""
    k_levels = [int(k) for k in settings.get("k_levels", [5, 10])]
    epsilons = [float(e) for e in settings.get("epsilons", [1.0])]
    out = [Policy("none", "microdata", 1, "no suppression"),
           Policy("min_contributors", "microdata", 3, "three contributors"),
           Policy("complementary", "microdata", 3, "complementary suppression")]
    out += [Policy(f"k_anonymity_{k}", "microdata", k, f"$k$-anonymity, $k={k}$")
            for k in k_levels]
    out += [Policy(f"dp_counts_{e:g}", "counts", 0, f"noisy counts, $\\varepsilon={e:g}$",
                   epsilon=e) for e in epsilons]
    return out


def released_rows(register: CreditRegister, policy: Policy, month: int) -> np.ndarray:
    """The loans a policy publishes in one month."""
    live = np.flatnonzero(register.live[:, month])
    if live.size == 0 or policy.kind == "counts":
        return live if policy.kind == "counts" else live
    key = register.key()[live]
    values, counts = np.unique(key, return_counts=True)
    if policy.name.startswith("k_anonymity") or policy.name == "none":
        keep = values[counts >= max(int(policy.parameter), 1)]
        return live[np.isin(key, keep)]
    # contributor rules count distinct lenders, not records
    lender = register.lender[live]
    contributors = {int(cell): int(np.unique(lender[key == cell]).size)
                    for cell in values.tolist()}
    keep = [cell for cell, n in contributors.items() if n >= int(policy.parameter)]
    if policy.name == "complementary":
        # a suppressed cell can be recovered from a published row total, so the
        # smallest surviving cell in each region is withheld as well
        by_region: dict[int, list[tuple[int, int]]] = {}
        size = dict(zip(values.tolist(), counts.tolist()))
        for cell in keep:
            members = key == cell
            region = int(register.region[live][members][0])
            by_region.setdefault(region, []).append((size[cell], cell))
        secondary = {min(group)[1] for group in by_region.values() if len(group) > 1}
        keep = [cell for cell in keep if cell not in secondary]
    return live[np.isin(key, np.array(keep, dtype=key.dtype))]


def _readings(rng: np.random.Generator, truth: np.ndarray, sd: float,
              months: int) -> np.ndarray:
    """One noisy exposure reading per month, the adversary's auxiliary information."""
    return truth[None, :] + rng.normal(0.0, sd, size=(months, truth.size))


def identification_attack(register: CreditRegister, policy: Policy, covariate_sd: float,
                          replicates: int, rng: np.random.Generator,
                          assumed_sd: float | None = None, pooled: bool = False,
                          fields: tuple[str, ...] = ("sector", "region", "size", "exposure"),
                          threshold: float = IDENTIFICATION_THRESHOLD,
                          alpha: float = 0.05,
                          aux_readings: int | None = None,
                          return_hits: bool = False) -> Outcome:
    """Maximum-posterior linkage, per release or pooled over the programme.

    The adversary holds the target's key and one noisy reading of its exposure per
    vintage. It blocks on region, scores every released loan by the likelihood of
    the key and the density of the reading, normalises to a posterior, and names a
    record only when that posterior reaches the decision threshold. Pooling
    multiplies the evidence across the vintages the target appears in, which is
    what a per-release rule does not bound.

    By default the adversary holds one exposure reading per vintage it attacks, so the
    pooled attack also holds more auxiliary information than the per-release attack.
    ``aux_readings`` separates the two effects: it fixes the number of independent
    readings whatever the number of vintages, and membership of the released files
    across vintages is then scored as a separate filter.
    """
    assumed = float(assumed_sd or covariate_sd)
    months = register.months if pooled else 1
    if aux_readings is not None:
        return _identification_fixed_readings(register, policy, covariate_sd, replicates,
                                              rng, assumed, months, fields, threshold,
                                              alpha, int(aux_readings))
    log_exposure = np.log(register.exposure)
    key = _key_from_fields(register, fields)
    published = [released_rows(register, policy, m) for m in range(months)]
    universe = published[0]
    if universe.size == 0:
        return Outcome(policy.name, "identification", trials=int(replicates))
    targets = rng.choice(universe, size=int(replicates), replace=True)
    readings = _readings(rng, log_exposure[targets], covariate_sd, months)
    member = [np.zeros(register.loans, dtype=bool) for _ in range(months)]
    for m in range(months):
        member[m][published[m]] = True
    out = Outcome(policy.name, "identification", trials=int(replicates),
                  positives=int(replicates))
    posteriors = []
    hits = []
    subgroup: dict[str, list[int]] = {}
    named: dict[str, list[int]] = {}
    for i, target in enumerate(targets.tolist()):
        block = np.flatnonzero((register.region == register.region[target])
                               & (key == key[target]) & member[0])
        if block.size == 0:
            continue
        score = np.zeros(block.size)
        seen = 0
        for m in range(months):
            live = member[m][block]
            if not member[m][target]:
                # the adversary knows the target's history, so a candidate that is
                # published in a vintage where the target is not is excluded too
                if m > 0:
                    score += np.where(live, -30.0, 0.0)
                continue
            seen += 1
            if "exposure" in fields:
                density = norm.logpdf(readings[m, i], loc=log_exposure[block],
                                      scale=assumed)
            else:
                density = np.zeros(block.size)
            score += np.where(live, density, -30.0)
        if seen == 0:
            continue
        weights = np.exp(score - score.max())
        weights /= weights.sum()
        chosen = int(np.argmax(weights))
        posterior = float(weights[chosen])
        posteriors.append(posterior)
        correct = bool(block[chosen] == target)
        hits.append(correct)
        band = str(int(register.size_class[target]))
        subgroup.setdefault(band, [0, 0])
        subgroup[band][1] += 1
        named.setdefault(band, [0, 0])
        named[band][0] += int(correct)
        named[band][1] += 1
        if posterior >= threshold:
            out.decisions += 1
            if correct:
                out.true_positive += 1
                subgroup[band][0] += 1
    out.posterior = float(np.mean(posteriors)) if posteriors else float("nan")
    worst = {band: hit / total for band, (hit, total) in sorted(subgroup.items())
             if total >= 30}
    bound = {band: binomial_upper(int(round(rate * subgroup[band][1])),
                                  subgroup[band][1], alpha / max(len(worst), 1))
             for band, rate in worst.items()}
    # The same success event as the average: the named loan is the target. A release
    # passes the subgroup control when every class's own bound meets the tolerance,
    # an intersection-union test that needs no correction for the number of classes.
    class_rate = {band: hit / total for band, (hit, total) in sorted(named.items())
                  if total >= 30}
    class_upper = {band: binomial_upper(named[band][0], named[band][1], alpha)
                   for band in class_rate}
    out.detail = {"attempted": len(hits), "matched": int(np.sum(hits)),
                  "by_size_class_named": class_rate,
                  "worst_class_rate": max(class_rate.values()) if class_rate else float("nan"),
                  "worst_class_upper": (max(class_upper.values()) if class_upper
                                        else float("nan")),
                  "curve": precision_coverage(posteriors, hits) if hits else [],
                  "match_rate": float(np.mean(hits)) if hits else float("nan"),
                  "mean_posterior": out.posterior,
                  "by_size_class": worst,
                  "worst_size_class": max(worst.values()) if worst else float("nan"),
                  "worst_size_class_upper": max(bound.values()) if bound else float("nan"),
                  "threshold": threshold, "pooled": pooled, "vintages": months,
                  "auxiliary_fields": list(fields),
                  "assumed_sd": assumed, "true_sd": float(covariate_sd),
                  "calibration_error": (abs(float(np.mean(posteriors))
                                            - float(np.mean(hits)))
                                        if hits else float("nan"))}
    if return_hits:
        out.detail["hits"] = [int(h) for h in hits]
    return out


def _identification_fixed_readings(register: CreditRegister, policy: Policy,
                                   covariate_sd: float, replicates: int,
                                   rng: np.random.Generator, assumed: float, months: int,
                                   fields: tuple[str, ...], threshold: float, alpha: float,
                                   aux_readings: int) -> Outcome:
    """The linkage attack with a fixed number of auxiliary readings.

    The target draw and the first readings use the same random stream as the default
    attack with the same generator, so the two can be compared target by target.
    """
    log_exposure = np.log(register.exposure)
    key = _key_from_fields(register, fields)
    published = [released_rows(register, policy, m) for m in range(months)]
    universe = published[0]
    if universe.size == 0:
        return Outcome(policy.name, "identification", trials=int(replicates))
    targets = rng.choice(universe, size=int(replicates), replace=True)
    readings = _readings(rng, log_exposure[targets], covariate_sd, max(aux_readings, 1))
    member = [np.zeros(register.loans, dtype=bool) for _ in range(months)]
    for m in range(months):
        member[m][published[m]] = True
    out = Outcome(policy.name, "identification", trials=int(replicates),
                  positives=int(replicates))
    hits, posteriors = [], []
    for i, target in enumerate(targets.tolist()):
        block = np.flatnonzero((register.region == register.region[target])
                               & (key == key[target]) & member[0])
        if block.size == 0:
            continue
        score = np.zeros(block.size)
        if "exposure" in fields:
            for r in range(aux_readings):
                score += norm.logpdf(readings[r, i], loc=log_exposure[block], scale=assumed)
        for m in range(1, months):
            # exclusion in both directions: the candidate's published pattern must
            # match the target's, which the adversary can work out from its history
            score += np.where(member[m][block] == member[m][target], 0.0, -30.0)
        weights = np.exp(score - score.max())
        weights /= weights.sum()
        chosen = int(np.argmax(weights))
        posterior = float(weights[chosen])
        correct = bool(block[chosen] == target)
        posteriors.append(posterior)
        hits.append(correct)
        if posterior >= threshold:
            out.decisions += 1
            out.true_positive += int(correct)
    out.posterior = float(np.mean(posteriors)) if posteriors else float("nan")
    out.detail = {"attempted": len(hits), "matched": int(np.sum(hits)),
                  "match_rate": float(np.mean(hits)) if hits else float("nan"),
                  "mean_posterior": out.posterior, "vintages": months,
                  "aux_readings": aux_readings, "pooled": months > 1,
                  "threshold": threshold, "auxiliary_fields": list(fields)}
    return out


def _key_from_fields(register: CreditRegister, fields: tuple[str, ...]) -> np.ndarray:
    """The key the adversary can match on, given the auxiliary information it holds."""
    parts = []
    if "sector" in fields:
        parts.append((register.sector, register.n_sectors))
    if "size" in fields:
        parts.append((register.size_class, register.n_size_classes))
    if not parts:
        return np.zeros(register.loans, dtype=int)
    value = np.zeros(register.loans, dtype=int)
    for column, width in parts:
        value = value * width + column
    return value


def counts_attack(register: CreditRegister, policy: Policy, rng: np.random.Generator,
                  threshold: float = IDENTIFICATION_THRESHOLD, alpha: float = 0.05,
                  pooled: bool = False) -> Outcome:
    """Identification from published cell counts alone, under Laplace noise.

    With no microdata the adversary can still single a target out: a key class of
    size one identifies its member. It reads the noisy count and declares
    uniqueness when the posterior that the true count is one reaches the threshold.

    Pooled over the programme, each vintage publishes its own noisy counts with a
    fresh draw of noise, so the programme spends ``months * epsilon`` under basic
    composition. The adversary treats a cell's count as unchanged across vintages and
    multiplies the Laplace likelihoods of the readings; the target is still the cell's
    size in the first vintage.
    """
    live = np.flatnonzero(register.live[:, 0])
    key = register.key()[live]
    values, counts = np.unique(key, return_counts=True)
    scale = 1.0 / max(policy.epsilon, 1e-9)
    months = register.months if pooled else 1
    full_key = register.key()
    monthly = np.empty((months, counts.size))
    for m in range(months):
        month_key = full_key[register.live[:, m]]
        month_values, month_counts = np.unique(month_key, return_counts=True)
        lookup = dict(zip(month_values.tolist(), month_counts.tolist()))
        monthly[m] = [lookup.get(v, 0) for v in values.tolist()]
    monthly[0] = counts
    readings = monthly + rng.laplace(0.0, scale, size=monthly.shape)
    # posterior that the count is one, against a geometric-tailed prior on sizes
    support = np.arange(1, 60)
    prior = np.array([(counts == s).sum() for s in support], dtype=float)
    prior = prior / prior.sum() if prior.sum() else np.ones_like(prior) / prior.size
    out = Outcome(policy.name, "identification", trials=int(counts.size),
                  positives=int((counts == 1).sum()))
    posteriors = []
    for column, truth in enumerate(counts.tolist()):
        log_like = -np.abs(readings[:, column][:, None] - support[None, :]).sum(axis=0) / scale
        like = np.exp(log_like - log_like.max()) * prior
        posterior = float(like[0] / like.sum()) if like.sum() > 0 else 0.0
        posteriors.append(posterior)
        if posterior >= threshold:
            out.decisions += 1
            out.true_positive += int(truth == 1)
    out.posterior = float(np.mean(posteriors))
    out.detail = {"cells": int(counts.size), "unique_cells": int((counts == 1).sum()),
                  "match_rate": out.success,
                  "epsilon": policy.epsilon, "noise_scale": scale,
                  "vintages": months, "epsilon_total": months * policy.epsilon,
                  "mean_posterior": out.posterior,
                  "worst_size_class": float("nan"), "worst_size_class_upper": float("nan"),
                  "success_upper": binomial_upper(out.true_positive,
                                                  max(out.trials, 1), alpha)}
    return out


def relationship_attack(register: CreditRegister, policy: Policy) -> Outcome:
    """Naming the lender of a released borrower, scored as an allegation."""
    rows = released_rows(register, policy, 0)
    if policy.kind == "counts" or rows.size == 0:
        return Outcome(policy.name, "relationship")
    result = relationship_disclosure(register, rows)
    out = Outcome(policy.name, "relationship",
                  decisions=int(result["allegations"]),
                  true_positive=int(result["true_positive"]),
                  positives=int(result["true_positive"] + result["false_negative"]),
                  trials=int(result["allegations"]),
                  posterior=float(result["positive_predictive_value"]))
    out.detail = {"lift_over_market_share": result["lift_over_market_share"],
                  "prevalence": result["prevalence"],
                  "specificity": result["specificity"]}
    return out


def attribute_attack(register: CreditRegister, policy: Policy,
                     threshold: float = DECISION_THRESHOLD) -> Outcome:
    """Inferring arrears status from the released classification."""
    rows = released_rows(register, policy, 0)
    if policy.kind == "counts" or rows.size == 0:
        return Outcome(policy.name, "attribute")
    result = status_inference(register, rows, threshold)
    decisions = int(round(result["confident_share"] * rows.size))
    correct = int(round(result["confident_accuracy"] * decisions)) if decisions else 0
    out = Outcome(policy.name, "attribute", decisions=decisions, true_positive=correct,
                  positives=int(rows.size), trials=int(rows.size),
                  posterior=float(result["confident_accuracy"]))
    base = float(result["base_rate"])
    majority = max(base, 1.0 - base)
    out.detail = {"homogeneous_share": result["homogeneous_share"],
                  "base_rate": result["base_rate"],
                  # guessing the majority status for everyone, the accuracy an
                  # adversary gets from the base rate alone
                  "majority_accuracy": majority,
                  "gain_over_majority": float(result["confident_accuracy"]) - majority,
                  "lift_over_base_rate": (result["confident_accuracy"]
                                          / max(result["base_rate"], 1e-9))}
    return out


def differencing_attack(register: CreditRegister, policy: Policy) -> Outcome:
    """Reading one loan off the change between two vintages of the same cell."""
    if policy.kind == "counts":
        return Outcome(policy.name, "differencing")
    key = register.key()
    minimum = max(int(policy.parameter), 1)
    decisions = 0
    correct = 0
    comparisons = 0
    for month in range(register.months - 1):
        before, after = register.live[:, month], register.live[:, month + 1]
        for cell in np.unique(key).tolist():
            members = key == cell
            count_before = int((members & before).sum())
            count_after = int((members & after).sum())
            if min(count_before, count_after) < minimum:
                continue
            comparisons += 1
            if abs(count_after - count_before) != 1:
                continue
            decisions += 1
            correct += int((members & (before ^ after)).sum() == 1)
    out = Outcome(policy.name, "differencing", decisions=decisions,
                  true_positive=correct, positives=decisions, trials=max(comparisons, 0))
    out.posterior = correct / decisions if decisions else float("nan")
    out.detail = {"comparisons": comparisons, "unit_changes": decisions,
                  "minimum_cell": minimum}
    return out


def utility_of(register: CreditRegister, policy: Policy, weights: dict | None) -> float:
    """Utility for the four analyses, on what the policy actually publishes."""
    if policy.kind == "counts":
        # counts support growth and stability aggregates but not lender-level work
        rows = np.flatnonzero(register.live[:, 0])
        value = central_bank_utility(register, rows, weights)["utility"]
        return float(value - 0.35 - 0.05 / max(policy.epsilon, 1e-9))
    rows = released_rows(register, policy, 0)
    if rows.size == 0:
        return -10.0
    return float(central_bank_utility(register, rows, weights)["utility"])


def programme_parameter(register: CreditRegister, covariate_sd: float, replicates: int,
                        tolerance: float, rng: np.random.Generator,
                        levels: list[int], alpha: float = 0.05) -> dict:
    """The smallest k whose *pooled* risk meets the tolerance, and the per-release k.

    Both are computed on the same attack. The gap between them is the cost of
    governing a programme with a rule written for one release.
    """
    single, pooled = {}, {}
    for k in levels:
        policy = Policy(f"k_anonymity_{k}", "microdata", k)
        single[k] = identification_attack(register, policy, covariate_sd, replicates,
                                          np.random.default_rng(rng.integers(1 << 30)))
        pooled[k] = identification_attack(register, policy, covariate_sd, replicates,
                                          np.random.default_rng(rng.integers(1 << 30)),
                                          pooled=True)

    def smallest(table: dict) -> int | None:
        for k in sorted(levels):
            if table[k].detail.get("match_rate", 1.0) <= tolerance:
                return k
        return None

    def upper(outcome: Outcome) -> float:
        attempted = int(outcome.detail.get("attempted", 0))
        if attempted == 0:
            return float("nan")
        return float(binomial_upper(int(outcome.detail.get("matched", 0)), attempted, alpha))

    def smallest_upper(table: dict) -> int | None:
        for k in sorted(levels):
            if upper(table[k]) <= tolerance:
                return k
        return None

    return {"levels": levels, "tolerance": tolerance, "alpha": alpha,
            "per_release_upper": {str(k): upper(single[k]) for k in levels},
            "pooled_upper": {str(k): upper(pooled[k]) for k in levels},
            "per_release_choice_upper": smallest_upper(single),
            "pooled_choice_upper": smallest_upper(pooled),
            "per_release_rate": {str(k): single[k].detail.get("match_rate")
                                 for k in levels},
            "pooled_rate": {str(k): pooled[k].detail.get("match_rate") for k in levels},
            "per_release_choice": smallest(single), "pooled_choice": smallest(pooled),
            "per_release_records": {str(k): int(released_rows(
                register, Policy(f"k_anonymity_{k}", "microdata", k), 0).size)
                for k in levels}}


def evaluate(config: dict) -> dict:
    """The whole matrix: every attack against every policy, plus the ablations."""
    from .credit import make_credit_register  # local import keeps the module light

    settings = config.get("evaluation", {})
    credit_settings = config.get("credit", {})
    register = make_credit_register(config)
    seed = int(config["seed"])
    alpha = float(config.get("alpha", 0.05))
    covariate_sd = float(credit_settings.get("covariate_sd", 0.6))
    replicates = int(settings.get("replicates", 1500))
    tolerance = float(settings.get("tolerance", config.get("risk_tolerance", 0.05)))
    weights = credit_settings.get("weights")
    rows = []
    for policy in policies(settings):
        if policy.kind == "counts":
            single = counts_attack(register, policy,
                                   np.random.default_rng(seed + 8_200))
            pooled = counts_attack(register, policy,
                                   np.random.default_rng(seed + 8_250), pooled=True)
        else:
            single = identification_attack(register, policy, covariate_sd, replicates,
                                           np.random.default_rng(seed + 8_300),
                                           alpha=alpha)
            pooled = identification_attack(register, policy, covariate_sd, replicates,
                                           np.random.default_rng(seed + 8_400),
                                           pooled=True, alpha=alpha)
        released = (int(released_rows(register, policy, 0).size)
                    if policy.kind == "microdata" else 0)
        rows.append({
            "policy": policy.name, "label": policy.label, "kind": policy.kind,
            "released": released,
            "utility": utility_of(register, policy, weights),
            "identification": single.record(alpha),
            "identification_pooled": pooled.record(alpha),
            "relationship": relationship_attack(register, policy).record(alpha),
            "attribute": attribute_attack(register, policy).record(alpha),
            "differencing": differencing_attack(register, policy).record(alpha),
            "pooled_excess": (pooled.detail.get("match_rate", float("nan"))
                              - single.detail.get("match_rate", float("nan"))),
            "meets_tolerance_single": bool(
                single.detail.get("match_rate", 1.0) <= tolerance),
            "meets_tolerance_pooled": bool(
                pooled.detail.get("match_rate", 1.0) <= tolerance)})

    baseline = Policy("k_anonymity_5", "microdata", 5)
    ablations = []
    for fields in [("sector", "region", "size", "exposure"),
                   ("sector", "region", "size"),
                   ("sector", "region", "exposure"),
                   ("region", "exposure")]:
        outcome = identification_attack(register, baseline, covariate_sd, replicates,
                                        np.random.default_rng(seed + 8_500),
                                        fields=fields, alpha=alpha)
        ablations.append({"fields": list(fields), "held": len(fields),
                          **outcome.record(alpha)})

    shifts = []
    for factor in [float(f) for f in settings.get("shift_factors", [0.5, 1.0, 2.0])]:
        outcome = identification_attack(register, baseline, covariate_sd, replicates,
                                        np.random.default_rng(seed + 8_600),
                                        assumed_sd=covariate_sd * factor, alpha=alpha)
        shifts.append({"factor": factor, **outcome.record(alpha)})

    programme = programme_parameter(
        register, covariate_sd, replicates, tolerance,
        np.random.default_rng(seed + 8_700),
        [int(k) for k in settings.get("programme_levels", [1, 5, 10, 20, 50])],
        alpha=alpha)
    # A finer grid above the coarse one, drawn from its own stream so that the coarse
    # results are unchanged.
    fine_levels = [int(k) for k in settings.get("programme_levels_fine", [])]
    programme_fine = (programme_parameter(register, covariate_sd, replicates, tolerance,
                                          np.random.default_rng(seed + 8_750),
                                          fine_levels, alpha=alpha)
                      if fine_levels else None)

    # How much of the pooled excess is more auxiliary information, and how much is the
    # joint content of the releases. Paired with the main runs through their streams.
    decomposition = []
    for name in settings.get("decomposition_policies", []):
        policy = next(p for p in policies(settings) if p.name == name)
        months = register.months
        rates = {}
        for label, pooled_flag, readings, stream in [
                ("one_release_one_reading", False, None, 8_300),
                ("one_release_all_readings", False, months, 8_300),
                ("all_releases_one_reading", True, 1, 8_400),
                ("all_releases_all_readings", True, None, 8_400)]:
            outcome = identification_attack(register, policy, covariate_sd, replicates,
                                            np.random.default_rng(seed + stream),
                                            pooled=pooled_flag, alpha=alpha,
                                            aux_readings=readings)
            matched = int(outcome.detail.get("matched", 0))
            attempted = max(int(outcome.detail.get("attempted", 0)), 1)
            rates[label] = {"match_rate": outcome.detail.get("match_rate"),
                            "matched": matched, "attempted": attempted,
                            "upper": float(binomial_upper(matched, attempted, alpha))}
        decomposition.append({"policy": name, "label": policy.label, **rates})

    frontier = [{"policy": row["policy"], "risk": row["identification_pooled"]["success"],
                 "match_rate": row["identification_pooled"].get("match_rate"),
                 "utility": row["utility"]} for row in rows]
    # Count releases measure class uniqueness, not loan identification, so they are
    # not compared with the loan tolerance and cannot be chosen by it.
    best = max((row for row in rows
                if row["kind"] == "microdata" and row["meets_tolerance_pooled"]),
               key=lambda row: row["utility"], default=None)
    return {"policies": rows, "ablations": ablations, "shift": shifts,
            "programme": programme, "programme_fine": programme_fine,
            "decomposition": decomposition, "frontier": frontier,
            "tolerance": tolerance, "replicates": replicates,
            "admissible_best": best["label"] if best else None,
            "register": {"loans": register.loans, "borrowers": register.n_borrowers,
                         "lenders": register.n_lenders, "months": register.months}}


def programme_attack(register: CreditRegister, policy: Policy, covariate_sd: float,
                     replicates: int, rng: np.random.Generator, vintages: int = 1,
                     readings: int | str = 1, target_history: bool = True,
                     assumed_sd: float | None = None,
                     target_class: int | None = None,
                     exposure_digits: int | None = None) -> dict:
    """The linkage attack with each piece of the adversary's knowledge set separately.

    ``vintages`` is the number of monthly releases the adversary holds (1 or all).
    ``readings`` is the number of independent noisy exposure readings it holds,
    whatever the number of vintages; ``"live"`` gives one reading for each vintage in
    which the target is live, the attack of the main evaluation. ``target_history`` says whether it knows in which
    vintages the target's loan is live. Without that knowledge it knows only that the
    target is in the first release, and the other vintages add nothing on this
    register, because a loan's published key and exposure do not change between
    vintages and whether a loan is live is drawn independently of both.

    ``target_class`` restricts the target law to one size class, the stratified draw
    a subgroup control needs when the class is a small share of the release.
    ``exposure_digits`` publishes exposure rounded to that many significant figures;
    the adversary scores candidates against the published, rounded values. Ties
    between equally likely candidates are broken at random.

    Returns per-target hits and size classes, so that averages, subgroup rates and
    their bounds can all be computed from one set of draws.
    """
    assumed = float(assumed_sd or covariate_sd)
    months = register.months if vintages > 1 else 1
    true_log_exposure = np.log(register.exposure)
    log_exposure = (np.log(round_significant(register.exposure, exposure_digits))
                    if exposure_digits else true_log_exposure)
    key = register.key()
    published = [released_rows(register, policy, m) for m in range(months)]
    universe = published[0]
    released = int(universe.size)
    if target_class is not None:
        universe = universe[register.size_class[universe] == int(target_class)]
    if universe.size == 0:
        return {"hits": np.zeros(0, dtype=int), "size_class": np.zeros(0, dtype=int),
                "released": released}
    targets = rng.choice(universe, size=int(replicates), replace=True)
    per_live = readings == "live"
    rows = months if per_live else max(int(readings), 1)
    obs = _readings(rng, true_log_exposure[targets], covariate_sd, rows)
    ties = np.random.default_rng(int(rng.integers(1 << 32)))
    member = [np.zeros(register.loans, dtype=bool) for _ in range(months)]
    for m in range(months):
        member[m][published[m]] = True
    hits = np.zeros(targets.size, dtype=int)
    for i, target in enumerate(targets.tolist()):
        block = np.flatnonzero((key == key[target]) & member[0])
        score = np.zeros(block.size)
        used = ([m for m in range(months) if member[m][target]] if per_live
                else range(rows))
        for r in used:
            score += norm.logpdf(obs[r, i], loc=log_exposure[block], scale=assumed)
        if target_history:
            for m in range(1, months):
                score += np.where(member[m][block] == member[m][target], 0.0, -30.0)
        best = np.flatnonzero(score >= score.max() - 1e-12)
        chosen = int(best[0]) if best.size == 1 else int(ties.choice(best))
        hits[i] = int(block[chosen] == target)
    return {"hits": hits, "size_class": register.size_class[targets].astype(int),
            "released": released}


def round_significant(values: np.ndarray, digits: int | None) -> np.ndarray:
    """Round positive values to a number of significant figures."""
    values = np.asarray(values, dtype=float)
    if not digits:
        return values
    magnitude = np.floor(np.log10(values))
    scale = 10.0 ** (magnitude - int(digits) + 1)
    return np.round(values / scale) * scale


def relink_share(register: CreditRegister, policy: Policy,
                 exposure_digits: int | None = None) -> float:
    """Share of first-release loans that published attributes alone relink later.

    With rotating pseudonyms an adversary can still join two vintages on the fields
    that do not change. On this register key and exposure do not change between
    vintages, so a loan whose (key, published exposure) pair is unique in the first
    release can be found again in every later release that contains it. With exact
    exposure the pair is unique for almost every loan; rounding creates ties.
    """
    key = register.key()
    first = released_rows(register, policy, 0)
    published = round_significant(register.exposure[first], exposure_digits)
    pairs = np.stack([key[first], np.log(published)], axis=1)
    _, inverse, counts = np.unique(pairs, axis=0, return_inverse=True,
                                   return_counts=True)
    return float(np.mean(counts[inverse.ravel()] == 1))
