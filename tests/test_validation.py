import pytest

from privacy_risk.validation import (
    expected_calibration_error,
    monotonicity_violations,
    validation_report,
)


def test_validation_can_fail():
    ece = expected_calibration_error([.1, .9], [0, 1], bins=2)
    assert ece == pytest.approx(.1)
    assert monotonicity_violations([0, 1, 2], [.5, .4, .45], .01) == [1]
    assert not validation_report(ece, .05, 0, 1, [1])["passed"]


def test_calibration_table_and_gap():
    import numpy as np

    from privacy_risk.validation import calibration_gap, calibration_table
    posterior = np.concatenate([np.full(200, 0.15), np.full(200, 0.85)])
    outcome = np.concatenate([np.zeros(200), np.ones(200)])
    rows = calibration_table(posterior, outcome, minimum=50)
    assert len(rows) == 2
    assert calibration_gap(rows) == pytest.approx(0.15, abs=0.01)
    assert calibration_gap([]) != calibration_gap([])  # nan


def test_stress_and_composition_tests_can_fail():
    from privacy_risk.validation import composition_test, stress_ratios
    weaker = stress_ratios([0.2, 0.2], [0.1, 0.3], tolerance=0.05)
    assert not weaker["stronger_never_materially_lower"]
    stronger = stress_ratios([0.2, 0.2], [0.3, 0.3], tolerance=0.05)
    assert stronger["stronger_always_larger"]
    outside = composition_test(0.4, 0.25, 0.3)
    assert not outside["marginals_bound_the_programme"]
    assert not outside["upper_within_programme_tolerance"]
    inside = composition_test(0.2, 0.25)
    assert inside["inside_marginal_bounds"] and "upper_within_programme_tolerance" not in inside


def test_full_report_keeps_release_validity_separate_from_the_programme():
    from privacy_risk.validation import composition_test, full_validation_report, stress_ratios
    report = full_validation_report(0.01, 0.06, 0.5, 1.5, [],
                                    stress_ratios([0.2], [0.3], 0.05),
                                    composition_test(0.4, 0.25, 0.3))
    assert report["passed"]
    assert report["programme"]["marginals_bound_the_programme"] is False


def test_frequency_transport_error():
    from privacy_risk.validation import frequency_transport_error
    assert frequency_transport_error(14.8, 14.88) == pytest.approx(0.08)
