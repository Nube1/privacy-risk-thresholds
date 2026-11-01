import math

import pytest

from privacy_risk.uncertainty import binomial_upper, monte_carlo_summary


def test_zero_success_exact_bound_and_rule_of_three():
    upper = binomial_upper(0, 1000, .05)
    assert upper == pytest.approx(1 - .05 ** (1/1000))
    assert upper == pytest.approx(-math.log(.05)/1000, rel=.01)
    assert monte_carlo_summary([False] * 100)["estimate"] == 0



def test_certification_replicates_and_rule_of_three():
    from privacy_risk.uncertainty import certification_replicates
    assert certification_replicates(0.001, 0.05) == 2995
    assert certification_replicates(1e-5, 0.01) == 460515
    with pytest.raises(ValueError):
        certification_replicates(0, 0.05)


def test_selected_maximum_bound_corrects_for_selection():
    import numpy as np

    from privacy_risk.uncertainty import binomial_lower, selected_maximum_bound
    risks = np.array([0.8, 0.5, 0.2])
    summary = selected_maximum_bound(risks, 250, 0.05, counts=np.array([200, 125, 50]))
    assert summary["maximum"] == 0.8
    assert summary["alpha_per_target"] == pytest.approx(0.05 / 3)
    assert summary["maximum_lower_bound"] < binomial_lower(200, 250, 0.05)
    assert binomial_lower(0, 10) == 0.0


def test_plugin_ordering_reports_both_directions():
    from privacy_risk.uncertainty import plugin_ordering
    report = plugin_ordering(0.47, 0.064, 0.19)
    assert report["dominance_holds"] and report["jensen_holds"]
    assert report["ratio"] == pytest.approx(0.47 / 0.064)


def test_nested_variance_separates_the_two_sources():
    import numpy as np

    from privacy_risk.uncertainty import nested_variance
    rng = np.random.default_rng(5)
    estimates = 0.2 + 0.05 * rng.standard_normal(50)
    summary = nested_variance(estimates, 400)
    assert summary["between_variance"] > summary["within_variance"]
    assert summary["nested_standard_error"] > summary["naive_standard_error"]
    assert summary["understatement_factor"] > 1
    flat = nested_variance(np.full(20, 0.3), 100)
    assert flat["between_variance"] == 0.0
    with pytest.raises(ValueError):
        nested_variance(np.array([0.2]), 100)
