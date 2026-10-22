import numpy as np
import pytest

from privacy_risk.experiments import (  # noqa: I001
    programme_scaling,
    certification,
    conventional_measures,
    error_model,
    formal_privacy,
    k_anonymity,
    key_structure,
    mechanism_grid,
    population_truths,
    programme,
    programme_budget,
    replicates,
    risk_distribution,
    utility,
)
from privacy_risk.synthetic import Mechanism, apply_mechanism, make_register

CONFIG = {
    "seed": 20260904, "population_size": 6000, "sample_fraction": 0.2,
    "journalist_fractions": [0.1, 0.2], "kanonymity_levels": [3, 5],
    "risk_distribution_levels": [1, 3],
    "grid": {"age_widths": [10, 20], "occupation_groups": [12, 3], "min_cell_counts": [1, 5]},
    "attack_replicates": {"headline": 200, "key_only": 100, "kanon": 150, "kanon_key": 100,
                          "grid": 60, "per_target": 20, "composition_targets": 200},
    "per_target_cap": 10, "minimum_release_size": 20, "kappas": [0.5, 5.0],
    "programme": {"kmin": 3, "weights": [1.0, 0.5], "tau_programme": 0.4},
    "certification": {"taus": [0.01], "alphas": [0.05]},
    "dp": {"epsilons": [1.0], "priors": [0.1]},
}
REGISTER = make_register(CONFIG)


def test_replicates_accepts_a_number_or_a_mapping():
    assert replicates({"attack_replicates": 42}, "grid", 7) == 42
    assert replicates({"attack_replicates": {"grid": 9}}, "grid", 7) == 9
    assert replicates({}, "grid", 7) == 7


def test_key_structure_counts_uniques():
    keys = key_structure(REGISTER)
    assert keys["cells_occupied"] <= keys["cells_possible"]
    assert 0 <= keys["population_unique_share"] <= 1
    assert keys["released"] < keys["population"]


def test_marketer_identity_and_plugin_ordering_hold():
    measures = conventional_measures(REGISTER, CONFIG)
    assert measures["prosecutor_mean"] == pytest.approx(measures["marketer_sample"])
    ordering = measures["plugin_ordering"]
    assert ordering["dominance_holds"] and ordering["jensen_holds"]
    assert measures["marketer_unconditional"] < measures["prosecutor_mean"]
    assert len(measures["journalist"]) == 2


def test_utility_falls_when_the_key_is_coarsened():
    truths = population_truths(REGISTER)
    fine = Mechanism(REGISTER.fine_band_width, REGISTER.occupations, 1)
    coarse = Mechanism(20, 3, 1)
    sharp = utility(REGISTER, apply_mechanism(REGISTER, fine), fine, truths)
    blunt = utility(REGISTER, apply_mechanism(REGISTER, coarse), coarse, truths)
    assert blunt["rmse_occupation"] > sharp["rmse_occupation"]
    assert blunt["utility"] < sharp["utility"]


def test_k_anonymity_reports_both_controls():
    rows = k_anonymity(REGISTER, CONFIG)
    assert rows, "expected at least one minimum cell count to leave a release"
    for row in rows:
        assert row["worst"] >= row["average"]
        assert 0 <= row["worst_lower"] <= row["worst"]
        assert row["alpha_per_target"] <= 0.05


def test_grid_returns_a_frontier_and_a_tolerance_for_every_price():
    grid = mechanism_grid(REGISTER, CONFIG)
    assert grid["settings"] and grid["frontier"]
    assert len(grid["implied_tolerances"]) == len(CONFIG["kappas"])
    assert grid["risk_span"][0] <= grid["risk_span"][1]
    assert set(grid["concavity"]) == {"slopes", "is_concave", "largest_slope",
                                      "smallest_slope"}


def test_pooled_attack_can_leave_the_marginal_bounds():
    result = programme(REGISTER, CONFIG)
    assert result["pooled"][-1] >= max(result["marginal"]) - 1e-12
    assert result["union_upper"] >= result["marginal_lower"]
    assert len(result["releases"]) == 4


def test_budget_matches_the_closed_form_and_beats_an_equal_split():
    fit = {"a": 0.4, "b": -0.4, "gamma": -0.3, "r_squared": 0.9, "points": 8}
    budget = programme_budget(CONFIG, fit)
    assert budget["optimal_value"] >= budget["equal_value"]
    assert sum(budget["allocation"]) == pytest.approx(CONFIG["programme"]["tau_programme"])


def test_certification_and_formal_privacy_tables():
    assert certification(CONFIG)[0]["replicates"] == 299
    bounds = formal_privacy(CONFIG)["posterior_bounds"]
    assert bounds[0]["bound"] > bounds[0]["prior"]


def test_risk_distribution_shrinks_with_the_minimum_cell_count():
    spread = risk_distribution(REGISTER, CONFIG)
    assert max(spread["3"]) <= 1 / 3 + 1e-12 < max(spread["1"])


def test_error_model_scaling_and_misspecification():
    model = error_model(CONFIG)
    stronger = model.scaled(0.5)
    assert stronger.covariate_sd == pytest.approx(model.covariate_sd / 2)
    assert stronger.declared_sd == stronger.covariate_sd
    wrong = model.misspecified(2.0)
    assert wrong.covariate_sd == model.covariate_sd
    assert wrong.declared_sd == pytest.approx(2 * model.covariate_sd)
    with pytest.raises(ValueError):
        error_model({"error_age": 1.5})
    assert np.isfinite(model.covariate_sd)


def test_the_pooling_gap_does_not_dissolve_with_population_size():
    config = dict(CONFIG)
    config["scaling"] = {"population_sizes": [2000, 8000], "targets": 800}
    rows = programme_scaling(config)
    assert len(rows) == 2
    for row in rows:
        assert row["pooled"] >= row["largest_marginal"]
        assert row["sample_size"] < row["population_size"]
    # larger registers are safer release by release, and the marginals guide worse
    assert rows[-1]["largest_marginal"] <= rows[0]["largest_marginal"]
    assert rows[-1]["ratio_to_bound"] > rows[0]["ratio_to_bound"]


def test_scaling_is_skipped_when_no_sizes_are_configured():
    assert programme_scaling({**CONFIG, "scaling": {}}) == []
