import numpy as np
import pytest

from privacy_risk.sequential import (
    anytime_upper,
    at_threshold,
    attack_threshold,
    certification_replicates,
    confidence_sequence,
    in_sequence,
    log_e_process,
    monitoring_study,
    precision_coverage,
    selected_anytime_upper,
    transport_bound,
    transport_divergence,
)
from privacy_risk.uncertainty import binomial_upper
from privacy_risk.uncertainty import certification_replicates as fixed_replicates


def test_the_e_value_has_mean_one_under_the_null():
    # summed exactly over the binomial law, which is what the martingale property says
    from scipy.stats import binom

    for risk, trials in [(0.2, 20), (0.05, 40), (0.5, 12)]:
        expectation = sum(binom.pmf(s, trials, risk)
                          * np.exp(log_e_process(s, trials, risk))
                          for s in range(trials + 1))
        assert expectation == pytest.approx(1.0, abs=1e-9)


def test_the_sequence_covers_the_truth_under_repeated_looks():
    rng = np.random.default_rng(6)
    risk = 0.1
    failures = 0
    for _ in range(200):
        draws = rng.random(600) < risk
        running = np.cumsum(draws)
        if any(not in_sequence(int(running[b - 1]), b, risk, 0.05)
               for b in range(50, 601, 50)):
            failures += 1
    assert failures / 200 <= 0.05


def test_the_sequence_is_wider_than_the_fixed_sample_bound():
    assert anytime_upper(0, 1000, 0.05) > binomial_upper(0, 1000, 0.05)
    assert anytime_upper(60, 1000, 0.05) > binomial_upper(60, 1000, 0.05)


def test_the_interval_brackets_the_observed_rate_and_narrows_with_evidence():
    low, high = confidence_sequence(60, 1000, 0.05)
    assert low < 0.06 < high
    wide_low, wide_high = confidence_sequence(6, 100, 0.05)
    assert (high - low) < (wide_high - wide_low)


def test_a_selected_maximum_costs_width():
    assert selected_anytime_upper(0, 500, 8, 0.05) > anytime_upper(0, 500, 0.05)


def test_monitoring_costs_more_replicates_than_one_prearranged_look():
    for tolerance in (0.01, 0.001):
        assert certification_replicates(tolerance) > fixed_replicates(tolerance)


def test_the_attack_threshold_is_one_half_when_benefit_is_false_cost_plus_twice_action():
    assert attack_threshold(1.0, 1.0, 0.0) == pytest.approx(0.5)
    assert attack_threshold(3.0, 1.0, 1.0) == pytest.approx(0.5)
    assert attack_threshold(10.0, 1.0, 0.5) < 0.5
    assert attack_threshold(1.0, 5.0, 0.5) > 0.5


def test_an_adversary_whose_action_cost_exceeds_the_benefit_never_acts():
    import math
    assert attack_threshold(1.0, 100.0, 100.0) == math.inf
    assert attack_threshold(1.0, 0.0, 1.0) == math.inf


def test_the_kl_transport_bound_is_tighter_than_pinsker():
    from privacy_risk.sequential import (bernoulli_kl, transport_bound_kl,
                                         transport_divergence_kl)
    room = transport_divergence_kl(0.004, 0.12)
    assert room == pytest.approx(bernoulli_kl(0.12, 0.004))
    assert room > 10 * transport_divergence(0.004, 0.12)
    assert transport_bound_kl(0.004, room) == pytest.approx(0.12, abs=1e-6)
    for risk, rho in [(0.05, 0.01), (0.2, 0.05), (0.01, 0.001)]:
        assert transport_bound_kl(risk, rho) <= transport_bound(risk, rho) + 1e-12


def test_precision_rises_and_coverage_falls_along_the_curve():
    rng = np.random.default_rng(8)
    posteriors = rng.random(2000)
    correct = rng.random(2000) < posteriors
    curve = precision_coverage(posteriors, correct)
    coverage = [row["coverage"] for row in curve]
    precision = [row["precision"] for row in curve]
    assert coverage == sorted(coverage, reverse=True)
    assert precision[0] < precision[-1]
    assert at_threshold(curve, 0.42)["threshold"] == pytest.approx(0.5)


def test_the_transport_bound_and_its_inverse_agree():
    assert transport_bound(0.06, 0.0) == pytest.approx(0.06)
    assert transport_bound(0.5, 2.0) == 1.0
    divergence = transport_divergence(0.06, 0.12)
    assert transport_bound(0.06, divergence) == pytest.approx(0.12)


def test_peeking_breaks_the_fixed_bound_and_not_the_sequence():
    study = monitoring_study(0.1, 800, 16, 120, 0.05, np.random.default_rng(12))
    assert study["fixed_error_rate"] > 0.05
    assert study["anytime_error_rate"] <= 0.05
    assert study["width_ratio"] > 1.0
