from privacy_risk.governance import GovernanceThreshold, decide

T = GovernanceThreshold(.1, "Owner", "2027-01-01", "prosecutor", "high", .05)

def test_gate_pass_fail_invalid_and_expiry():
    assert decide(.08, T, True, "2026-01-01")["status"] == "PASS"
    assert decide(.11, T, True, "2026-01-01")["status"] == "FAIL"
    assert decide(.01, T, False, "2026-01-01")["status"] == "INVALID"
    assert decide(.01, T, True, "2027-01-01")["status"] == "FAIL"



def test_implied_tolerance_falls_as_harm_gets_dearer():
    import numpy as np
    import pytest

    from privacy_risk.governance import implied_tolerance
    risk = np.array([0.05, 0.20, 0.40])
    utility = np.array([-0.40, -0.15, -0.10])
    cheap = implied_tolerance(risk, utility, 0.25)
    dear = implied_tolerance(risk, utility, 5.0)
    assert risk[cheap] >= risk[dear]
    with pytest.raises(ValueError):
        implied_tolerance(risk, utility, 0)


def test_risk_budget_matches_the_closed_form_and_sums_to_the_tolerance():
    import numpy as np
    import pytest

    from privacy_risk.governance import budget_value, risk_budget
    weights = np.array([1.0, 0.5])
    allocation = risk_budget(weights, 0.4, -1.0)
    assert allocation.sum() == pytest.approx(0.4)
    assert allocation[0] > allocation[1]
    expected = 0.4 * np.sqrt(weights) / np.sqrt(weights).sum()
    assert allocation == pytest.approx(expected)
    equal = np.full(2, 0.2)
    curve = (0.4, -0.4, -0.5)
    assert budget_value(allocation, weights, curve) >= budget_value(equal, weights, curve)
    with pytest.raises(ValueError):
        risk_budget(weights, 0.4, 0.5)
    with pytest.raises(ValueError):
        budget_value(np.array([0.0, 0.4]), weights, curve)
