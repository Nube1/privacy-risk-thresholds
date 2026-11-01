"""The paper's experiments, each returning the numbers it reports.

Four kinds of work are kept apart: proof checks that verify algebra, simulation
checks that estimate attack success and its uncertainty, falsification tests that
can reject the model, and the governance gate that applies a tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from . import dp
from .credit import (
    borrower_identification,
    central_bank_utility,
    differencing_risk,
    make_credit_register,
    reconstruction_from_totals,
    relationship_disclosure,
    released_loans,
    status_inference,
)
from .composition import pooled_attack
from .evaluation import evaluate
from .sequential import (
    anytime_upper,
    at_threshold,
    attack_threshold,
    certification_replicates as anytime_replicates,
    monitoring_study,
    transport_bound,
    transport_bound_kl,
    transport_divergence,
    transport_divergence_kl,
)
from .frontier import concavity_diagnostic, fit_value_curve, nondominated, value_curve
from .governance import budget_value, implied_tolerance, risk_budget
from .keys import key_frequencies
from .linkage import ErrorModel, attack, minimum_cell_targets, per_target_counts
from .risk import (
    fit_frequency_model,
    journalist_from_fit,
    journalist_truth,
    marketer_population_frequency_risk,
    marketer_sample_risk,
    marketer_unconditional_risk,
    prosecutor_risk,
)
from .synthetic import (
    Mechanism,
    Register,
    Release,
    apply_mechanism,
    identity_release,
    make_register,
    pack_key,
    resample,
    suppress_small_cells,
)
from .uncertainty import (
    binomial_lower,
    binomial_upper,
    certification_replicates,
    nested_variance,
    plugin_ordering,
    selected_maximum_bound,
)
from .validation import calibration_gap, calibration_table


def replicates(config: dict, name: str, default: int) -> int:
    """Read a replicate count, accepting either a mapping or a single number."""
    value = config.get("attack_replicates", default)
    if isinstance(value, dict):
        return int(value.get(name, default))
    return int(value)


def error_model(config: dict) -> ErrorModel:
    return ErrorModel(age_error=float(config.get("error_age", 0.15)),
                      occupation_error=float(config.get("error_occupation", 0.10)),
                      covariate_sd=float(config.get("covariate_sd", 0.75)))


def population_key(register: Register) -> np.ndarray:
    frame = register.frame
    return pack_key(frame.district.to_numpy(), frame.sex.to_numpy(),
                    register.bands(frame.age.to_numpy(), register.fine_band_width),
                    frame.occupation.to_numpy(), register.fine_bands, register.occupations)


def key_structure(register: Register) -> dict:
    """Proof-adjacent bookkeeping: how much of the population the key alone separates."""
    population = population_key(register)
    release = identity_release(register)
    _, population_counts = np.unique(population, return_counts=True)
    _, sample_counts = np.unique(release.key, return_counts=True)
    cells = register.districts * 2 * register.fine_bands * register.occupations
    return {"population": register.size, "released": release.size,
            "sampling_fraction": register.sampling_fraction,
            "cells_possible": int(cells),
            "cells_occupied": int(population_counts.size),
            "population_unique_cells": int((population_counts == 1).sum()),
            "population_unique_share": float((population_counts == 1).sum() / register.size),
            "released_cells": int(sample_counts.size),
            "released_unique_cells": int((sample_counts == 1).sum()),
            "released_unique_share": float((sample_counts == 1).sum() / release.size),
            "mean_population_cell": float(register.size / population_counts.size)}


def conventional_measures(register: Register, config: dict) -> dict:
    """Table of the standard summaries, each labelled by the experiment it answers."""
    release = identity_release(register)
    population = population_key(register)
    sample_keys = release.key
    per_record = prosecutor_risk(sample_keys)
    fine = register.fine_band_width
    fractions = list(config.get("journalist_fractions", [0.02, 0.05, 0.10, 0.20, 0.50]))
    cells = register.districts * 2 * register.fine_bands * register.occupations
    journalist = []
    for i, fraction in enumerate(fractions):
        rng = np.random.default_rng(int(config["seed"]) + 7000 + i)
        drawn = np.flatnonzero(rng.random(register.size) < fraction)
        counts = np.array(list(key_frequencies(population[drawn]).values()))
        fit = fit_frequency_model(counts, cells, fraction)
        journalist.append({"pi": float(fraction),
                           "truth": journalist_truth(population[drawn], population),
                           "model": journalist_from_fit(fraction, fit),
                           "fitted_mean_cell": fit.mean,
                           "dispersion": fit.dispersion})
    plugin = marketer_population_frequency_risk(sample_keys, population)
    return {"fine_band_width": fine,
            "prosecutor_mean": float(per_record.mean()),
            "prosecutor_max": float(per_record.max()),
            "released_unique_share": float(np.mean(per_record == 1.0)),
            "marketer_sample": marketer_sample_risk(sample_keys),
            "plugin_population_frequency": plugin,
            "marketer_unconditional": marketer_unconditional_risk(
                sample_keys, register.sampling_fraction),
            "pi_times_plugin": float(register.sampling_fraction * plugin),
            "true_mean_population_cell": float(register.size / cells),
            "mean_occupied_cell": float(register.size / np.unique(population).size),
            "journalist": journalist,
            "plugin_ordering": plugin_ordering(float(per_record.mean()), plugin,
                                               float(1 / (1 / per_record).mean()))}


def attack_sequence(register: Register, config: dict) -> dict:
    """Key error alone, then the same adversary with one noisy covariate."""
    release = identity_release(register)
    model = error_model(config)
    alpha = float(config.get("alpha", 0.05))
    seed = int(config["seed"])
    key_only = attack(register, release, model, replicates(config, "key_only", 5000),
                      np.random.default_rng(seed + 101), use_covariate=False)
    headline = attack(register, release, model, replicates(config, "headline", 20000),
                      np.random.default_rng(seed + 102))
    wrong = attack(register, release,
                   model.misspecified(float(config.get("misspecified_covariate_factor", 2.5))),
                   replicates(config, "headline", 20000), np.random.default_rng(seed + 103))
    correct_rows = calibration_table(headline.posterior, headline.success)
    wrong_rows = calibration_table(wrong.posterior, wrong.success)
    theory = float(prosecutor_risk(release.key).mean())
    return {"theory_exact_matching": theory,
            "key_error_only": key_only.rate, "key_error_trials": key_only.trials,
            "with_covariate": headline.rate, "trials": headline.trials,
            "upper_bound": binomial_upper(headline.successes, headline.trials, alpha),
            "understatement": (headline.rate - key_only.rate) / headline.rate
            if headline.rate else float("nan"),
            "misspecified_rate": wrong.rate,
            "calibration": [{"name": "declared model correct", "bins": correct_rows,
                             "gap": calibration_gap(correct_rows)},
                            {"name": "covariate precision misspecified", "bins": wrong_rows,
                             "gap": calibration_gap(wrong_rows)}]}


def k_anonymity(register: Register, config: dict) -> list[dict]:
    """Both controls: the average over a uniform target, and the worst target."""
    model = error_model(config)
    alpha = float(config.get("alpha", 0.05))
    seed = int(config["seed"])
    reps = replicates(config, "per_target", 250)
    cap = int(config.get("per_target_cap", 250))
    rows = []
    for minimum in config.get("kanonymity_levels", [3, 5, 10, 20]):
        mechanism = Mechanism(register.fine_band_width, register.occupations, int(minimum))
        release = apply_mechanism(register, mechanism)
        if release.size == 0:
            continue
        average = attack(register, release, model, replicates(config, "kanon", 12000),
                         np.random.default_rng(seed + 300 + minimum))
        key_only = attack(register, release, model, replicates(config, "kanon_key", 6000),
                          np.random.default_rng(seed + 200 + minimum), use_covariate=False)
        targets = minimum_cell_targets(release, int(minimum), cap,
                                       np.random.default_rng(seed + 7))
        counts = per_target_counts(register, release, model, targets, reps,
                                   np.random.default_rng(seed + 400 + minimum))
        risks = counts / float(reps) if counts.size else np.empty(0)
        worst = selected_maximum_bound(risks, reps, alpha, counts=counts)
        above = float(np.mean(np.asarray(worst["individual_lower_bounds"])
                              > 1 / minimum))
        rows.append({"k": int(minimum), "released": release.size,
                     "retained": release.size / identity_release(register).size,
                     "bound": 1 / minimum, "key_only": key_only.rate,
                     "average": average.rate,
                     "average_trials": average.trials,
                     "average_upper": binomial_upper(average.successes, average.trials, alpha),
                     "targets": worst["targets"], "replicates_per_target": reps,
                     "worst": worst["maximum"], "worst_lower": worst["maximum_lower_bound"],
                     "alpha_per_target": worst["alpha_per_target"],
                     "mean_minimum_cell_risk": worst["mean"],
                     "share_above_bound": float(np.mean(risks > 1 / minimum)),
                     "share_lower_above_bound": above})
    return rows


@dataclass(frozen=True)
class Truths:
    district: np.ndarray
    age: np.ndarray
    occupation: np.ndarray
    outcome_sd: float


def population_truths(register: Register) -> Truths:
    frame = register.frame
    outcome = frame.outcome.to_numpy()
    fine = register.bands(frame.age.to_numpy(), register.fine_band_width)
    district = np.array([outcome[frame.district.to_numpy() == d].mean()
                         for d in range(register.districts)])
    age = np.array([outcome[fine == b].mean() for b in range(register.fine_bands)])
    occupation = np.array([outcome[frame.occupation.to_numpy() == o].mean()
                           for o in range(register.occupations)])
    return Truths(district, age, occupation, float(outcome.std()))


def _group_rmse(labels: np.ndarray, values: np.ndarray, target_group: np.ndarray,
                truth: np.ndarray, minimum: int = 3) -> tuple[float, float]:
    estimate: dict[int, float] = {}
    for group in np.unique(labels).tolist():
        mask = labels == group
        if mask.sum() >= minimum:
            estimate[int(group)] = float(values[mask].mean())
    reported = np.array([estimate.get(int(g), np.nan) for g in target_group])
    usable = ~np.isnan(reported)
    if not usable.any():
        return 10.0, 0.0
    error = float(np.sqrt(np.mean((reported[usable] - truth[usable]) ** 2)))
    return error, float(usable.mean())


def utility(register: Register, release: Release, mechanism: Mechanism,
            truths: Truths) -> dict:
    """Negative average normalised error over the three declared analyses.

    Each analysis is stated at its own finest classification, so a release can only
    answer it as well as its released classification allows.
    """
    values = register.frame.outcome.to_numpy()[release.index]
    fine_midpoints = (register.age_min + register.fine_band_width
                      * np.arange(register.fine_bands) + register.fine_band_width // 2)
    district_error, district_cover = _group_rmse(
        release.district, values, np.arange(register.districts), truths.district)
    age_error, age_cover = _group_rmse(
        release.band, values, register.bands(fine_midpoints, mechanism.age_width), truths.age)
    occupation_error, occupation_cover = _group_rmse(
        release.group, values,
        register.collapse(np.arange(register.occupations), mechanism.occupation_groups),
        truths.occupation)
    cover = (district_cover + age_cover + occupation_cover) / 3
    loss = ((district_error + age_error + occupation_error) / (3 * truths.outcome_sd)
            + 0.5 * (1 - cover))
    return {"utility": -float(loss), "rmse_district": district_error,
            "rmse_age": age_error, "rmse_occupation": occupation_error, "coverage": cover}


def mechanism_grid(register: Register, config: dict) -> dict:
    """Risk and utility of every setting, the frontier, and the implied tolerance."""
    model = error_model(config)
    stronger_factor = float(config.get("stronger_covariate_factor", 0.5))
    alpha = float(config.get("alpha", 0.05))
    seed = int(config["seed"])
    reps = replicates(config, "grid", 1200)
    truths = population_truths(register)
    grid_config = config.get("grid", {})
    settings = []
    for width in grid_config.get("age_widths", [5, 10, 20]):
        for groups in grid_config.get("occupation_groups", [12, 6, 3]):
            for minimum in grid_config.get("min_cell_counts", [1, 3, 5, 10, 20]):
                mechanism = Mechanism(int(width), int(groups), int(minimum))
                release = apply_mechanism(register, mechanism)
                if release.size < int(config.get("minimum_release_size", 300)):
                    continue
                counts = key_frequencies(release.key)
                frequencies = np.array([counts[int(k)] for k in release.key.tolist()], float)
                stream = abs(hash((width, groups, minimum))) % 2**31
                declared = attack(register, release, model, reps,
                                  np.random.default_rng(seed + stream))
                capable = attack(register, release, model.scaled(stronger_factor), reps,
                                 np.random.default_rng(seed + stream + 1))
                robust = declared if declared.rate >= capable.rate else capable
                measures = utility(register, release, mechanism, truths)
                settings.append({"age_width": mechanism.age_width,
                                 "occupation_groups": mechanism.occupation_groups,
                                 "min_cell_count": mechanism.min_cell_count,
                                 "label": mechanism.label(), "released": release.size,
                                 "retained": release.size / identity_release(register).size,
                                 "prosecutor_max": float(1 / frequencies.min()),
                                 "prosecutor_mean": float(np.mean(1 / frequencies)),
                                 "risk_declared": declared.rate, "risk_stronger": capable.rate,
                                 "risk": robust.rate, "replicates": reps,
                                 "risk_upper": binomial_upper(robust.successes,
                                                              robust.trials, alpha),
                                 **measures})
    risk = np.array([s["risk_upper"] for s in settings])
    value = np.array([s["utility"] for s in settings])
    frontier = np.flatnonzero(nondominated(risk, value))
    frontier = frontier[np.argsort(risk[frontier])]
    fit = fit_value_curve(risk[frontier], value[frontier])
    tolerances = []
    for kappa in config.get("kappas", [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]):
        pick = frontier[implied_tolerance(risk[frontier], value[frontier], float(kappa))]
        tolerances.append({"kappa": float(kappa), "tolerance": float(risk[pick]),
                           "utility": float(value[pick]), "setting": settings[pick]["label"],
                           "retained": settings[pick]["retained"], "index": int(pick)})
    return {"settings": settings, "frontier": frontier.tolist(), "fit": fit,
            "concavity": concavity_diagnostic(risk[frontier], value[frontier]),
            "implied_tolerances": tolerances,
            "risk_span": [float(risk[frontier].min()), float(risk[frontier].max())],
            "utility_span": [float(value[frontier].max()), float(value[frontier].min())]}


def programme(register: Register, config: dict) -> dict:
    """Four releases from one sample, attacked separately and then pooled."""
    settings = config.get("programme", {})
    minimum = int(settings.get("kmin", 3))
    released = register.released
    zeros = np.zeros(len(released), dtype=int)
    ten_year = register.bands(released.age.to_numpy(), 10)
    fine = register.bands(released.age.to_numpy(), register.fine_band_width)
    district = released.district.to_numpy()
    sex = released.sex.to_numpy()
    occupation = released.occupation.to_numpy()
    plans = [
        ("A (district, sex, fine age)",
         Release(register.sample_index, district, sex, fine, zeros, register.fine_bands, 1)),
        ("B (district, sex, occupation)",
         Release(register.sample_index, district, sex, zeros, occupation, 1,
                 register.occupations)),
        ("C (district, ten-year age, six groups)",
         Release(register.sample_index, district, zeros, ten_year,
                 register.collapse(occupation, 6), (register.age_max - register.age_min) // 10,
                 6)),
        ("D (sex, ten-year age, occupation)",
         Release(register.sample_index, zeros, sex, ten_year, occupation,
                 (register.age_max - register.age_min) // 10, register.occupations)),
    ]
    releases = [(name, apply_suppression(plan, minimum)) for name, plan in plans]
    rng = np.random.default_rng(int(config["seed"]) + 909)
    count = replicates(config, "composition_targets", 20000)
    targets = rng.choice(register.sample_index, size=count, replace=True)
    result = pooled_attack([release for _, release in releases], targets)
    payload = result.as_dict()
    payload["kmin"] = minimum
    payload["targets"] = int(count)
    # the programme decision uses the exact upper bound, as the release decision does
    payload["pooled_upper"] = float(binomial_upper(int(round(result.pooled[-1] * count)),
                                                   int(count),
                                                   float(config.get("alpha", 0.05))))
    payload["releases"] = [{"name": name, "released": release.size,
                            "marginal": result.marginal[i], "coverage": result.coverage[i],
                            "unique_share": result.marginal_unique_share[i]}
                           for i, (name, release) in enumerate(releases)]
    payload["population_size"] = register.size
    return payload


def programme_scaling(config: dict) -> list[dict]:
    """Does the pooling gap survive at other population sizes?

    A reasonable question about a synthetic study is whether the effect is an
    artefact of the size chosen. The same four releases, the same keys and the
    same minimum cell count are rebuilt on registers of several sizes, and the
    pooled risk is compared with the bound the marginals imply at each.
    """
    sizes = config.get("scaling", {}).get("population_sizes", [])
    if not sizes:
        return []
    targets = int(config.get("scaling", {}).get("targets", 5000))
    rows = []
    for size in sizes:
        local = dict(config)
        local["population_size"] = int(size)
        local["programme"] = dict(config.get("programme", {}))
        local["replicates"] = {**config.get("replicates", {}),
                               "composition_targets": targets}
        register = make_register(local)
        result = programme(register, local)
        rows.append({
            "population_size": int(size),
            "sample_size": int(register.released.shape[0]),
            "largest_marginal": float(max(result["marginal"])),
            "pooled": float(result["pooled"][-1]),
            "pooled_two": float(result["pooled"][1]),
            "union_bound": float(result["union_upper"]),
            "excess": float(result["pooled"][-1] - max(result["marginal"])),
            "ratio_to_bound": (float(result["pooled"][-1] / result["union_upper"])
                               if result["union_upper"] > 0 else float("nan")),
            "unique_share": float(max(row["unique_share"]
                                      for row in result["releases"])),
        })
    return rows


def apply_suppression(release: Release, minimum: int) -> Release:
    return suppress_small_cells(release, minimum)


def programme_budget(config: dict, fit: dict) -> dict:
    """Allocate a programme tolerance across releases of differing value."""
    settings = config.get("programme", {})
    weights = np.asarray(settings.get("weights", [1.0, 0.7, 0.5, 0.3]), float)
    total = float(settings.get("tau_programme", 0.5))
    curve = (fit["a"], fit["b"], fit["gamma"])
    allocation = risk_budget(weights, total, fit["gamma"])
    equal = np.full(weights.size, total / weights.size)
    optimal_value = budget_value(allocation, weights, curve)
    equal_value = budget_value(equal, weights, curve)
    # the closed form is checked against a constrained numerical optimum
    numerical = optimize.minimize(
        lambda shares: -budget_value(np.maximum(shares, 1e-4), weights, curve),
        equal, method="SLSQP", bounds=[(1e-3, total)] * weights.size,
        constraints=[{"type": "eq", "fun": lambda shares: total - float(np.sum(shares))}])
    agreement = float(np.max(np.abs(numerical.x - allocation)))
    return {"weights": weights.tolist(), "tau_programme": total,
            "exponent": 1 / (1 - fit["gamma"]),
            "allocation": allocation.tolist(), "equal_split": equal.tolist(),
            "numerical_allocation": [float(x) for x in numerical.x],
            "closed_form_agreement": agreement,
            "agreement_decimals": int(max(0, -np.floor(np.log10(agreement))))
            if agreement > 0 else 12,
            "proportional": (total * weights / weights.sum()).tolist(),
            "optimal_value": optimal_value, "equal_value": equal_value,
            "gain": optimal_value - equal_value,
            "loss_reduction": (optimal_value - equal_value) / abs(equal_value)
            if equal_value else float("nan")}


def design_variance(register: Register, config: dict) -> dict:
    """Estimate how much of the risk estimate's variance the binomial term misses.

    Each outer replicate redraws the sample from the same population, applies the
    same mechanism and runs a fresh block of attacks. The binomial expression covers
    only the variation inside a block.
    """
    settings = config.get("nested", {})
    outer = int(settings.get("outer_replicates", 40))
    inner = int(settings.get("inner_replicates", 500))
    mechanism = Mechanism(int(settings.get("age_width", 20)),
                          int(settings.get("occupation_groups", 3)),
                          int(settings.get("min_cell_count", 10)))
    model = error_model(config)
    seed = int(config["seed"])
    estimates = []
    for replicate in range(outer):
        drawn = resample(register, seed + 90_000 + replicate)
        release = apply_mechanism(drawn, mechanism)
        if release.size == 0:
            continue
        result = attack(drawn, release, model, inner,
                        np.random.default_rng(seed + 91_000 + replicate))
        estimates.append(result.rate)
    summary = nested_variance(np.asarray(estimates), inner)
    summary["mechanism"] = mechanism.label()
    summary["estimates"] = estimates
    return summary


def risk_distribution(register: Register, config: dict) -> dict[str, list[float]]:
    """Prosecutor risk of every released record under a range of minimum cell counts."""
    out: dict[str, list[float]] = {}
    for minimum in config.get("risk_distribution_levels", [1, 3, 5, 10, 20]):
        release = apply_mechanism(register, Mechanism(register.fine_band_width,
                                                      register.occupations, int(minimum)))
        if release.size == 0:
            continue
        out[str(int(minimum))] = np.sort(prosecutor_risk(release.key)).tolist()
    return out


def certification(config: dict) -> list[dict]:
    settings = config.get("certification", {})
    taus = settings.get("taus", [0.05, 0.01, 1e-3, 1e-4, 1e-5])
    alphas = settings.get("alphas", [0.05, 0.01])
    return [{"tolerance": float(t), "alpha": float(a),
             "replicates": certification_replicates(float(t), float(a))}
            for t in taus for a in alphas]


def _class_bounds(identification: dict, band: str, alpha: float) -> list[float] | None:
    """A two-sided interval for one size class, which the subgroup claim needs."""
    counts = identification.get("counts_by_size_class", {}).get(band)
    if not counts:
        return None
    successes, trials = int(counts["successes"]), int(counts["trials"])
    return [binomial_lower(successes, trials, alpha / 2),
            binomial_upper(successes, trials, alpha / 2)]


def credit_register(config: dict) -> dict:
    """The central bank case: five risks and one utility on a credit register."""
    settings = config.get("credit", {})
    register = make_credit_register(config)
    seed = int(config["seed"])
    alpha = float(config.get("alpha", 0.05))
    covariate_sd = float(settings.get("covariate_sd", 0.6))
    reps = replicates(config, "credit", 4000)
    rows_out = []
    for minimum in settings.get("min_cell_counts", [1, 5, 10, 20]):
        rows = released_loans(register, int(minimum))
        if rows.size == 0:
            continue
        identification = borrower_identification(register, rows, covariate_sd, reps,
                                                 np.random.default_rng(seed + 6100 + minimum))
        relationship = relationship_disclosure(register, rows)
        status = status_inference(register, rows)
        value = central_bank_utility(register, rows, settings.get("weights"))
        largest = str(register.n_size_classes - 1)
        rate = identification["rate"]
        rows_out.append({
            "min_cell_count": int(minimum), "released": int(rows.size),
            "retained": value["retained"],
            "identification": rate,
            "identification_borrower": identification["borrower_rate"],
            "identification_upper": binomial_upper(int(round(rate * reps)), reps, alpha),
            "identification_largest": identification["rate_by_size_class"].get(largest),
            "identification_by_size_class": identification["rate_by_size_class"],
            "identification_largest_bounds": _class_bounds(identification, largest, alpha),
            "reciprocal_frequency": identification["mean_reciprocal_frequency"],
            "relationship_ppv": relationship["positive_predictive_value"],
            "relationship_table": {k: relationship[k] for k in
                                   ("prevalence", "sensitivity", "specificity",
                                    "base_rate_identity", "lift_over_market_share")},
            "status_homogeneous": status["homogeneous_share"],
            "status_homogeneous_arrears": status["homogeneous_arrears_share"],
            "status_homogeneous_largest":
                status["homogeneous_share_by_size_class"].get(largest),
            "status_confident_share": status["confident_share"],
            "status_confident_accuracy": status["confident_accuracy"],
            "utility": value["utility"], "losses": value["losses"]})
    reconstruction = [reconstruction_from_totals(register, 0, int(minimum))
                      for minimum in settings.get("minimum_contributors", [3, 5])]
    differencing = [differencing_risk(register, int(minimum))
                    for minimum in settings.get("min_cell_counts", [1, 5, 10, 20])[:3]]
    return {"register": {"borrowers": register.n_borrowers, "lenders": register.n_lenders,
                         "loans": register.loans, "sectors": register.n_sectors,
                         "regions": register.n_regions,
                         "size_classes": register.n_size_classes,
                         "months": register.months,
                         "key_cells": register.n_sectors * register.n_regions
                         * register.n_size_classes},
            "settings": rows_out, "reconstruction": reconstruction,
            "differencing": differencing}


def real_data(config: dict) -> dict:
    """The framework run on two real public microdata files."""
    from . import realdata as rd

    settings = config.get("real", {})
    seed = int(config["seed"])
    alpha = float(config.get("alpha", 0.05))
    reps = replicates(config, "real", 1500)

    census = rd.load_census2000()
    analyses = [{"name": "income by state", "by": "state_code", "outcome": "income"},
                {"name": "income by education", "by": "education", "outcome": "income"},
                {"name": "income by experience", "by": "experience", "outcome": "income"}]
    full = rd.build_release(census, {}, 1)
    error = float(settings.get("key_error", 0.10))
    covariate_sd = float(settings.get("covariate_sd", 0.5))
    stronger = float(settings.get("stronger_covariate_factor", 0.5))

    grid = []
    # One family covers both adversary precisions at every prespecified setting.
    # This remains conservative when a setting is skipped for insufficient coverage.
    grid_size = (len(settings.get("puma_widths", [1, 10, 100]))
                 * len(settings.get("education_widths", [1, 4]))
                 * len(settings.get("experience_widths", [1, 10]))
                 * len(settings.get("min_cell_counts", [1, 5, 10])))
    family_alpha = alpha / (2 * grid_size)
    for puma_width in settings.get("puma_widths", [1, 10, 100]):
        for education_width in settings.get("education_widths", [1, 4]):
            for experience_width in settings.get("experience_widths", [1, 10]):
                for minimum in settings.get("min_cell_counts", [1, 5, 10]):
                    coarsening = {"puma": puma_width, "education": education_width,
                                  "experience": experience_width}
                    release = rd.build_release(census, coarsening, minimum)
                    if release.size < int(settings.get("minimum_release_size", 500)):
                        continue
                    stream = int(np.random.SeedSequence(
                        [seed, int(puma_width), int(education_width),
                         int(experience_width), int(minimum)]).generate_state(1)[0])
                    declared = rd.attack(census, release, error, covariate_sd, reps,
                                         np.random.default_rng(seed + stream))
                    capable = rd.attack(census, release, error,
                                        covariate_sd * stronger, reps,
                                        np.random.default_rng(seed + stream + 1))
                    robust = declared if declared["rate"] >= capable["rate"] else capable
                    value = rd.utility(census, release, analyses)
                    grid.append({"label": release.label, "released": release.size,
                                 "retained": value["retained"],
                                 "coarsening": coarsening, "min_cell_count": minimum,
                                 "exact_matching": float(np.mean(
                                     1 / np.unique(release.key, return_counts=True)[1]
                                     [np.unique(release.key, return_inverse=True)[1]])),
                                 "risk_declared": declared["rate"],
                                 "risk_stronger": capable["rate"],
                                 "risk": robust["rate"],
                                 "risk_upper": max(binomial_upper(
                                     declared["successes"], declared["trials"], family_alpha),
                                     binomial_upper(capable["successes"],
                                                    capable["trials"], family_alpha)),
                                 "utility": value["utility"], "losses": value["losses"]})
    risk = np.array([g["risk_upper"] for g in grid])
    value = np.array([g["utility"] for g in grid])
    frontier = np.flatnonzero(nondominated(risk, value))
    frontier = frontier[np.argsort(risk[frontier])]
    tolerances = []
    for kappa in config.get("kappas", [0.25, 1.0, 5.0]):
        pick = frontier[implied_tolerance(risk[frontier], value[frontier], float(kappa))]
        tolerances.append({"kappa": float(kappa), "tolerance": float(risk[pick]),
                           "setting": grid[pick]["label"],
                           "retained": grid[pick]["retained"]})

    chosen = grid[int(frontier[len(frontier) // 2])]
    chosen_release = rd.build_release(census, chosen["coarsening"],
                                      chosen["min_cell_count"])
    worst = rd.per_target(census, chosen_release, error, covariate_sd,
                          int(settings.get("per_target_replicates", 200)),
                          int(settings.get("per_target_cap", 80)),
                          np.random.default_rng(seed + 61), alpha)

    rng = np.random.default_rng(seed + 62)
    plans = [{"name": "geography and schooling", "drop": ("experience",)},
             {"name": "geography and experience", "drop": ("education",)},
             {"name": "state, schooling and experience", "drop": ("puma",)},
             {"name": "area detail and schooling",
              "coarsening": {"experience": 10}, "drop": ()}]
    pooled_targets = rng.choice(np.arange(census.size),
                                size=int(settings.get("composition_targets", 5000)),
                                replace=True)
    pooled = rd.composition(census, plans, int(settings.get("composition_kmin", 3)),
                            pooled_targets)

    loans = rd.load_loan_applications()
    loan_release = rd.build_release(loans, {}, 1)
    loan_attack = rd.attack(loans, loan_release, error, 40.0, reps,
                            np.random.default_rng(seed + 63))
    return {"census": {"file": census.name, "source": census.source,
                       "full_key": rd.key_structure(census, full),
                       "key_space": rd.key_space(census, (("state_code", "puma"),)),
                       "journalist": rd.journalist_estimate(
                           full, float(settings.get("sampling_fraction", 0.05)),
                           rd.key_space(census, (("state_code", "puma"),))),
                       "grid": grid, "frontier": frontier.tolist(),
                       "grid_bound": {"method": "Bonferroni one-sided exact binomial",
                                      "family_alpha": alpha,
                                      "comparisons": 2 * grid_size,
                                      "per_comparison_alpha": family_alpha},
                       "implied_tolerances": tolerances,
                       "chosen": chosen["label"], "worst_target": worst,
                       "composition": pooled},
            "loans": {"file": loans.name, "source": loans.source,
                      "structure": rd.key_structure(loans, loan_release),
                      "identification": {**loan_attack,
                                         "upper": rd.bound(loan_attack["rate"],
                                                           loan_attack["trials"], alpha)},
                      "denied": rd.attribute_disclosure(loans, loan_release, "denied"),
                      "late": rd.attribute_disclosure(loans, loan_release, "late"),
                      "residual": [rd.residual_disclosure(loans, tuple(cols))
                                   for cols in settings.get(
                                       "loan_aggregates",
                                       [["area", "occupancy", "income_band"],
                                        ["area", "occupancy"]])]}}


def formal_privacy(config: dict) -> dict:
    settings = config.get("dp", {})
    epsilons = settings.get("epsilons", [0.1, 0.5, 1.0, 2.0, 5.0])
    priors = settings.get("priors", [0.01, 0.1, 1 / 3])
    return {"posterior_bounds": dp.posterior_bound_table(epsilons, priors),
            "composition": dp.basic_composition(epsilons)}


def frontier_curve_points(fit: dict, span: list[float], points: int = 200) -> dict:
    grid = np.linspace(span[0], span[1], points)
    return {"risk": grid.tolist(),
            "utility": value_curve(grid, fit["a"], fit["b"], fit["gamma"]).tolist()}


def adaptive(config: dict, evaluation: dict) -> dict:
    """Certification under monitoring, a paid adversary, and declared shift.

    Three quantities the fixed-sample account leaves out. What peeking costs an
    institution that stops when its bound first passes; where an adversary with a
    stated payoff actually acts, rather than at one half; and how far the risk can
    move inside a declared neighbourhood of the attack distribution.
    """
    settings = config.get("sequential", {})
    alpha = float(config.get("alpha", 0.05))
    tolerance = float(config.get("risk_tolerance", 0.05))
    reference = next((row for row in evaluation["policies"]
                      if row["policy"] == "k_anonymity_5"), evaluation["policies"][0])
    pooled = reference["identification_pooled"]
    curve = pooled.get("curve", [])
    risk = float(pooled.get("match_rate") or pooled["success"])
    successes = int(round(risk * pooled["trials"]))

    payoffs = [{"benefit": float(h), "false_cost": float(cf), "action_cost": float(ca),
                "threshold": attack_threshold(float(h), float(cf), float(ca)),
                **{key: value for key, value in
                   at_threshold(curve, attack_threshold(float(h), float(cf),
                                                        float(ca))).items()
                   if key != "threshold"}}
               for h, cf, ca in settings.get("payoffs", [[1, 1, 0], [10, 1, 0.5],
                                                         [1, 5, 0.5]])]
    fixed_upper = binomial_upper(successes, int(pooled["trials"]), alpha)
    # The shift bound is applied to the upper confidence bound, not the point
    # estimate, and by data processing on the success indicator; Pinsker is kept
    # beside it for comparison.
    divergences = [{"divergence": float(rho), "bound": transport_bound_kl(fixed_upper,
                                                                           float(rho)),
                    "bound_pinsker": transport_bound(fixed_upper, float(rho))}
                   for rho in settings.get("divergences", [0.005, 0.02, 0.05])]
    certification_rows = [
        {"tolerance": float(tau), "alpha": alpha,
         "fixed": certification_replicates(float(tau), alpha),
         "anytime": anytime_replicates(float(tau), alpha)}
        for tau in settings.get("tolerances", [0.01, 0.001])]
    study = monitoring_study(
        risk, int(settings.get("replicates", 2000)), int(settings.get("looks", 20)),
        int(settings.get("runs", 400)), alpha,
        np.random.default_rng(int(config["seed"]) + 9_100))
    # The floor chosen for the programme, judged by its upper bound on the finer grid
    fine = evaluation.get("programme_fine") or evaluation["programme"]
    floor = fine.get("pooled_choice_upper")
    admissible_risk = (float(fine["pooled_upper"][str(floor)]) if floor is not None
                       else None)
    admissible = {"label": f"$k$-anonymity, $k={floor}$"} if floor is not None else None
    return {"reference_policy": reference["policy"], "risk": risk,
            "admissible_policy": admissible["label"] if admissible else None,
            "admissible_risk": admissible_risk,
            "admissible_room": (transport_divergence_kl(admissible_risk,
                                                        evaluation["tolerance"])
                                if admissible_risk is not None else None),
            "admissible_room_pinsker": (transport_divergence(admissible_risk,
                                                             evaluation["tolerance"])
                                        if admissible_risk is not None else None),
            "anytime_upper": anytime_upper(successes, int(pooled["trials"]), alpha),
            "fixed_upper": fixed_upper,
            "payoffs": payoffs, "transport": divergences,
            "admissible_divergence": transport_divergence_kl(fixed_upper, tolerance),
            "certification": certification_rows, "monitoring": study,
            "curve": curve, "tolerance": tolerance}


def run_all(config: dict) -> dict:
    """Every experiment in the paper, in the order the paper reports them."""
    register = make_register(config)
    grid = mechanism_grid(register, config)
    evaluation = evaluate(config)
    return {"keys": key_structure(register),
            "risk_distribution": risk_distribution(register, config),
            "design_variance": design_variance(register, config),
            "measures": conventional_measures(register, config),
            "attack": attack_sequence(register, config),
            "k_anonymity": k_anonymity(register, config),
            "grid": grid,
            "programme": programme(register, config),
            "scaling": programme_scaling(config),
            "budget": programme_budget(config, grid["fit"]),
            "credit": credit_register(config),
            "evaluation": evaluation,
            "adaptive": adaptive(config, evaluation),
            "real": real_data(config),
            "certification": certification(config),
            "formal_privacy": formal_privacy(config)}
