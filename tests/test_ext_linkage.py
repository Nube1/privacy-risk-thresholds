import numpy as np
import pytest
from privacy_risk.linkage import posterior_linkage, maximum_posterior_success, membership_conditional_success


def test_renormalisation_example():
    p = np.array([0.25, 0.25])
    assert maximum_posterior_success(p) == 0.25
    assert membership_conditional_success(p, 0.5) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        membership_conditional_success(p, 1.0)


def test_example_2_5_pieter_posteriors():
    lv = np.log([100.0, 140.0, 250.0])
    for sd, want in ((0.2, (0.15, 0.83, 0.02)), (0.5, (0.33, 0.43, 0.24))):
        lik = np.exp(-0.5 * ((np.log(145) - lv) / sd) ** 2)
        p = posterior_linkage(lik, np.ones(3) / 3)
        assert p == pytest.approx(want, abs=0.006)
    lik = np.exp(-0.5 * ((np.log(145) - lv) / 0.05) ** 2)
    assert 1 - posterior_linkage(lik, np.ones(3) / 3)[1] < 1e-11


def test_example_2_3_prosecutor_risks():
    keys = list("AAAABBBCCD")
    f = {k: keys.count(k) for k in set(keys)}
    risks = [1 / f[k] for k in keys]
    assert np.mean(risks) == pytest.approx(0.4) and max(risks) == 1
    assert 0.05 * 0.25 == pytest.approx(0.0125)
