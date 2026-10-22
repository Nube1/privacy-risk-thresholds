import numpy as np

from privacy_risk.credit import make_credit_register
from privacy_risk.evaluation import (
    Policy,
    attribute_attack,
    counts_attack,
    differencing_attack,
    evaluate,
    identification_attack,
    policies,
    programme_parameter,
    relationship_attack,
    released_rows,
    utility_of,
)

CONFIG = {"seed": 20260904, "risk_tolerance": 0.2,
          "credit": {"borrowers": 2500, "lenders": 20, "months": 4},
          "evaluation": {"replicates": 200, "k_levels": [5], "epsilons": [1.0],
                         "programme_levels": [1, 5, 20], "shift_factors": [1.0, 2.0]}}
REGISTER = make_credit_register(CONFIG)


def test_policy_set_names_every_operational_baseline():
    names = [p.name for p in policies(CONFIG["evaluation"])]
    assert names[:3] == ["none", "min_contributors", "complementary"]
    assert "k_anonymity_5" in names
    assert any(name.startswith("dp_counts") for name in names)


def test_suppression_rules_only_remove_records_and_order_as_expected():
    loose = released_rows(REGISTER, Policy("none", "microdata", 1), 0)
    contributors = released_rows(REGISTER, Policy("min_contributors", "microdata", 3), 0)
    complementary = released_rows(REGISTER, Policy("complementary", "microdata", 3), 0)
    strict = released_rows(REGISTER, Policy("k_anonymity_20", "microdata", 20), 0)
    assert set(strict).issubset(set(loose))
    assert set(complementary).issubset(set(contributors))
    assert contributors.size <= loose.size
    assert strict.size < loose.size


def test_k_anonymity_leaves_no_cell_below_k():
    rows = released_rows(REGISTER, Policy("k_anonymity_10", "microdata", 10), 0)
    _, counts = np.unique(REGISTER.key()[rows], return_counts=True)
    assert counts.min() >= 10


def test_pooling_vintages_raises_risk_above_the_per_release_figure():
    policy = Policy("k_anonymity_5", "microdata", 5)
    single = identification_attack(REGISTER, policy, 0.6, 400,
                                   np.random.default_rng(11))
    pooled = identification_attack(REGISTER, policy, 0.6, 400,
                                   np.random.default_rng(11), pooled=True)
    assert pooled.detail["vintages"] == REGISTER.months
    assert single.detail["vintages"] == 1
    assert pooled.detail["match_rate"] > single.detail["match_rate"]
    assert 0.0 <= single.precision <= 1.0


def test_withdrawing_the_exposure_reading_disables_the_attack():
    policy = Policy("k_anonymity_5", "microdata", 5)
    full = identification_attack(REGISTER, policy, 0.6, 400, np.random.default_rng(3))
    without = identification_attack(REGISTER, policy, 0.6, 400,
                                    np.random.default_rng(3),
                                    fields=("sector", "region", "size"))
    assert without.detail["match_rate"] < full.detail["match_rate"]


def test_a_misspecified_adversary_is_worse_calibrated_than_a_correct_one():
    policy = Policy("k_anonymity_5", "microdata", 5)
    correct = identification_attack(REGISTER, policy, 0.6, 500, np.random.default_rng(5))
    wrong = identification_attack(REGISTER, policy, 0.6, 500, np.random.default_rng(5),
                                  assumed_sd=0.15)
    assert wrong.detail["calibration_error"] > correct.detail["calibration_error"]


def test_noisier_counts_leave_the_adversary_less_certain():
    loose = counts_attack(REGISTER, Policy("dp", "counts", 0, epsilon=10.0),
                          np.random.default_rng(7))
    tight = counts_attack(REGISTER, Policy("dp", "counts", 0, epsilon=0.2),
                          np.random.default_rng(7))
    assert loose.posterior > tight.posterior
    assert loose.true_positive >= tight.true_positive


def test_the_other_three_attacks_report_usable_rates():
    policy = Policy("k_anonymity_5", "microdata", 5)
    relationship = relationship_attack(REGISTER, policy)
    attribute = attribute_attack(REGISTER, policy)
    differencing = differencing_attack(REGISTER, policy)
    assert 0.0 < relationship.posterior <= 1.0
    assert relationship.detail["lift_over_market_share"] > 1.0
    assert 0.0 < attribute.posterior <= 1.0
    assert differencing.decisions <= differencing.trials


def test_utility_falls_as_suppression_tightens():
    weights = None
    loose = utility_of(REGISTER, Policy("none", "microdata", 1), weights)
    strict = utility_of(REGISTER, Policy("k_anonymity_20", "microdata", 20), weights)
    assert loose > strict


def test_the_programme_needs_a_stricter_parameter_than_one_release():
    levels = [1, 5, 20]
    result = programme_parameter(REGISTER, 0.6, 400, 0.2, np.random.default_rng(9), levels)
    assert result["per_release_choice"] is not None
    # None means no tested level meets the tolerance, which is stricter than any
    if result["pooled_choice"] is not None:
        assert levels.index(result["pooled_choice"]) >= levels.index(
            result["per_release_choice"])
    for k in levels:
        assert result["pooled_rate"][str(k)] >= result["per_release_rate"][str(k)]


def test_evaluate_returns_the_whole_matrix():
    result = evaluate(CONFIG)
    assert {"policies", "ablations", "shift", "programme", "frontier"} <= set(result)
    assert len(result["ablations"]) == 4
    assert len(result["shift"]) == 2
    for row in result["policies"]:
        for attack in ("identification", "identification_pooled", "relationship",
                       "attribute", "differencing"):
            assert attack in row
