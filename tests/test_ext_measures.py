import numpy as np
import pytest
from privacy_risk.risk import journalist_unique_probability as rho, fit_frequency_model, journalist_from_fit, FrequencyFit

G = {1: 0.2, 2: 0.3, 4: 0.5}


def test_example_3_3_five_figures():
    F = {"A": 12, "B": 10, "C": 3, "D": 5, "E": 10}
    keys = list("AAABBCDE")
    f = {k: keys.count(k) for k in set(keys)}
    pros = np.mean([1 / f[k] for k in keys]); pop = np.mean([1 / F[k] for k in keys])
    assert pros == pytest.approx(0.625) and len(set(keys)) / 8 == 0.625
    assert pop == pytest.approx(0.1354, abs=1e-4) and pros / pop == pytest.approx(4.6, abs=0.02)
    assert 1 / np.mean([f[k] for k in keys]) == 0.5
    assert 0.2 * pros == pytest.approx(0.125) and 0.2 * pop == pytest.approx(0.027, abs=5e-4)


def test_example_3_5_and_algorithm_3_1_checks():
    assert rho(0.1, G) == pytest.approx(0.0910, abs=5e-5)
    assert rho(0.5, G) == pytest.approx(0.2667, abs=5e-5)
    assert rho(1.0, G) == pytest.approx(1.0)
    assert rho(0.1, {2: 0.5, 4: 0.5}) == 0.0
    assert 0.5 / rho(0.5, G) == pytest.approx(1.9, abs=0.03)
    with pytest.raises(ValueError):
        rho(0.1, {1: 0.5, 2: 0.4})
    with pytest.raises(ValueError):
        rho(0.0, G)


def test_algorithm_3_2_checks():
    fit = fit_frequency_model(np.array([3, 1, 1, 2, 5, 1]), 10, 0.1)
    assert fit.mean == pytest.approx(13.00, abs=1e-3)
    assert journalist_from_fit(0.1, fit) == pytest.approx(0.0193, abs=5e-5)
    assert journalist_from_fit(1.0, FrequencyFit(mean=2.0, dispersion=0.5)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        fit_frequency_model(np.array([3, 1, 1, 2, 5, 1]), 4, 0.1)
