import numpy as np
import pytest

from privacy_risk.realdata import (
    attack,
    attribute_disclosure,
    build_release,
    composition,
    encode,
    journalist_estimate,
    key_space,
    key_structure,
    load_census2000,
    load_loan_applications,
    per_target,
    residual_disclosure,
    utility,
)
from privacy_risk.experiments import real_data
from privacy_risk.uncertainty import binomial_upper

CENSUS = load_census2000()
LOANS = load_loan_applications()


def test_files_load_with_their_declared_columns():
    for file in (CENSUS, LOANS):
        assert file.size > 0
        assert file.block in file.frame.columns
        assert set(file.key_columns) <= set(file.frame.columns)
        assert file.covariate in file.frame.columns
        assert file.source


def test_the_full_census_key_is_mostly_unique():
    release = build_release(CENSUS, {}, 1)
    structure = key_structure(CENSUS, release)
    assert structure["unique_share"] > 0.5
    assert structure["max_reciprocal_frequency"] == 1.0


def test_coarsening_and_suppression_reduce_risk_and_coverage():
    fine = build_release(CENSUS, {}, 1)
    coarse = build_release(CENSUS, {"puma": 100, "education": 4, "experience": 10}, 5)
    assert coarse.size < fine.size
    assert (key_structure(CENSUS, coarse)["mean_reciprocal_frequency"]
            < key_structure(CENSUS, fine)["mean_reciprocal_frequency"])
    _, counts = np.unique(coarse.key, return_counts=True)
    assert counts.min() >= 5


def test_encode_is_injective():
    codes = encode([np.array([0, 0, 1]), np.array([0, 1, 0])])
    assert len(set(codes.tolist())) == 3


def test_key_space_counts_nested_columns_by_observed_pairs():
    nested = key_space(CENSUS, (("state_code", "puma"),))
    flat = key_space(CENSUS, ())
    assert nested < flat


def test_journalist_estimate_reports_no_truth():
    release = build_release(CENSUS, {}, 1)
    estimate = journalist_estimate(release, 0.05, key_space(CENSUS,
                                                            (("state_code", "puma"),)))
    assert estimate["truth_available"] is False
    assert 0 < estimate["rho_model"] < 1


def test_the_covariate_lifts_the_attack_above_exact_matching():
    release = build_release(CENSUS, {"puma": 100, "education": 4, "experience": 10}, 5)
    structure = key_structure(CENSUS, release)
    result = attack(CENSUS, release, 0.10, 0.5, 600, np.random.default_rng(1))
    assert result["trials"] == 600
    assert result["rate"] > structure["mean_reciprocal_frequency"]


def test_per_target_finds_a_worse_record_than_the_average():
    release = build_release(CENSUS, {"puma": 100, "education": 4, "experience": 10}, 5)
    worst = per_target(CENSUS, release, 0.10, 0.5, 40, 12, np.random.default_rng(2))
    assert worst["maximum"] >= worst["mean"]
    assert worst["maximum_lower_bound"] <= worst["maximum"]
    assert worst["smallest_cell"] == 5


def test_attribute_and_residual_disclosure_on_the_loan_file():
    release = build_release(LOANS, {}, 1)
    denied = attribute_disclosure(LOANS, release, "denied")
    assert 0 <= denied["homogeneous_positive_share"] <= denied["homogeneous_share"] <= 1
    residual = residual_disclosure(LOANS, ("area", "occupancy", "income_band"))
    assert residual["single_record_cells"] <= residual["cells"]
    assert 0 <= residual["exposed_amount_share"] <= 1


def test_utility_falls_when_records_are_suppressed():
    analyses = [{"name": "income by state", "by": "state_code", "outcome": "income"}]
    loose = utility(CENSUS, build_release(CENSUS, {}, 1), analyses)
    tight = utility(CENSUS, build_release(CENSUS, {"puma": 100}, 20), analyses)
    assert loose["utility"] >= tight["utility"]
    assert loose["retained"] == 1.0


def test_pooled_attack_on_one_real_file_beats_the_union_bound():
    plans = [{"name": "geography and schooling", "drop": ("experience",)},
             {"name": "geography and experience", "drop": ("education",)},
             {"name": "state, schooling and experience", "drop": ("puma",)}]
    rng = np.random.default_rng(3)
    targets = rng.choice(np.arange(CENSUS.size), size=800, replace=True)
    result = composition(CENSUS, plans, 3, targets)
    assert result["pooled"][-1] >= result["marginal_lower"]
    assert len(result["releases"]) == 3


def test_attack_on_an_empty_release_returns_nothing():
    empty = build_release(CENSUS, {}, 10_000_000)
    assert empty.size == 0
    assert attack(CENSUS, empty, 0.1, 0.5, 10, np.random.default_rng(4))["trials"] == 0
    with pytest.raises(ValueError):
        composition(CENSUS, [], 3, np.arange(5))


def test_real_grid_uses_familywise_bounds_and_stable_streams():
    config = {"seed": 20, "alpha": .05, "attack_replicates": {"real": 40},
              "real": {"puma_widths": [100], "education_widths": [4],
                       "experience_widths": [10], "min_cell_counts": [5],
                       "per_target_replicates": 5, "per_target_cap": 3,
                       "composition_targets": 50}}
    a, b = real_data(config), real_data(config)
    assert a == b
    row = a["census"]["grid"][0]
    assert a["census"]["grid_bound"]["comparisons"] == 2
    assert row["risk_upper"] >= binomial_upper(round(row["risk"] * 40), 40, .025)
    assert a["census"]["full_key"]["unique_share"] == (
        a["census"]["full_key"]["unique_cells"] / CENSUS.size)
