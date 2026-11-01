"""Membership renormalisation (Expansion, Proposition 2.1) and counts kept as counts."""
import warnings

import numpy as np
import pytest

from privacy_risk.linkage import (
    decision_success,
    fixed_target_success,
    maximum_posterior_success,
    membership_success,
    posterior_linkage,
    posterior_with_no_match,
)
from privacy_risk.uncertainty import (
    beta_posterior_upper,
    binomial_upper,
    posterior_predictive_counts,
    selected_maximum_bound,
    simultaneous_upper_bounds,
)


def test_two_candidate_counterexample_needs_the_ratio():
    p = posterior_with_no_match(np.ones(2), np.array([0.25, 0.25]), 1.0, 0.5)
    assert p == pytest.approx([0.5, 0.25, 0.25])
    assert membership_success(p) == pytest.approx(0.5)
    # the unnormalised maximum understates success by the membership probability
    assert p[1:].max() == pytest.approx(0.25)


def test_known_membership_reduces_to_the_plain_maximum():
    p = posterior_with_no_match(np.ones(4), np.ones(4), 1.0, 0.0)
    assert membership_success(p) == pytest.approx(0.25)
    plain = posterior_linkage(np.ones(4), np.ones(4))
    assert maximum_posterior_success(plain) == pytest.approx(0.25)


def test_zero_membership_and_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        membership_success([1.0, 0.0])
    with pytest.raises(ValueError):
        membership_success([0.6, 0.6])
    with pytest.raises(ValueError):
        posterior_with_no_match([1.0], [1.0, 2.0], 1.0, 0.5)


def test_abstention_changes_success_and_precision():
    p = np.array([0.2, 0.5, 0.3])
    forced = decision_success(p)
    assert forced["success"] == pytest.approx(0.5 / 0.8)
    strict = decision_success(p, threshold=0.7)
    assert strict["abstains"] == 1.0 and strict["success"] == 0.0
    declines = decision_success(np.array([0.6, 0.25, 0.15]), may_choose_no_match=True)
    assert declines["names_a_candidate"] == 0.0


def test_fixed_target_success_is_a_count_over_draws():
    rng = np.random.default_rng(1)
    # target 0 is observed with noise; the rule names whichever of 0/1 is nearer
    result = fixed_target_success(lambda z: int(z > 0.5), lambda t, r: t + r.normal(0, 0.3),
                                  0, 2000, rng)
    assert isinstance(result["successes"], int) and result["trials"] == 2000
    assert 0.9 < result["rate"] < 0.99


def test_counts_are_not_rebuilt_from_rates():
    counts = np.array([3, 7, 10])
    risks = counts / 50
    summary = selected_maximum_bound(risks, 50, 0.05, counts=counts)
    assert summary["maximum"] == pytest.approx(0.2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        selected_maximum_bound(risks, 50, 0.05)
    assert any("integer counts" in str(w.message) for w in caught)
    with pytest.raises(ValueError):
        selected_maximum_bound(risks, 50, 0.05, counts=np.array([3, 7, 60]))


def test_simultaneous_bounds_split_alpha_and_cover():
    from scipy.stats import binom
    out = simultaneous_upper_bounds([0, 5], [100, 100], 0.05)
    assert out["alpha_per_coordinate"] == pytest.approx(0.025)
    assert out["upper_bounds"][0] == pytest.approx(binomial_upper(0, 100, 0.025))
    for p in (0.02, 0.1, 0.4):
        miss = sum(binom.pmf(k, 60, p) for k in range(61) if binomial_upper(k, 60, 0.025) < p)
        assert miss <= 0.025 + 1e-12


def test_bayesian_quantities_are_planning_figures():
    assert beta_posterior_upper(0, 100) < binomial_upper(0, 100, 0.05) + 0.01
    pmf = posterior_predictive_counts([0.05, 0.15], [0.5, 0.5], 40)
    assert pmf.sum() == pytest.approx(1.0)
    k = np.arange(41)
    from scipy.stats import binom
    plug = binom.pmf(k, 40, 0.10)
    variance = lambda f: f @ (k - f @ k) ** 2  # noqa: E731
    assert variance(pmf) > variance(plug)
