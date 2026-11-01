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


def test_chapter4_examples():
    import math
    from privacy_risk.uncertainty import binomial_upper, certification_replicates
    assert round(binomial_upper(127, 1200), 4) == 0.1216
    assert max(x for x in range(100, 140) if binomial_upper(x, 1200) <= 0.12) == 125
    assert round(binomial_upper(337, 4000), 3) == 0.092
    assert round(binomial_upper(458, 4000), 3) == 0.123
    assert certification_replicates(0.12) == 24
    assert round(1 - 0.05 ** (1 / 24), 3) == 0.117 and round(1 - 0.05 ** (1 / 23), 3) == 0.122
    assert round(binomial_upper(1, 24), 3) == 0.183


def test_chapter5_examples():
    from privacy_risk.uncertainty import binomial_upper as u
    assert [round(1 - 0.95 ** k, 3) for k in (2, 5, 10)] == [0.098, 0.226, 0.401]
    assert [round(u(c, 400), 3) for c in (30, 36, 44, 35, 52, 60)] == [0.1, 0.117, 0.139, 0.114, 0.161, 0.183]
    assert max(x for x in range(80) if u(x, 400) <= 0.12) == 37
    assert round(u(384, 4000), 3) == 0.104


def test_chapter6_examples():
    import math
    from scipy.stats import binom
    from privacy_risk.uncertainty import binomial_upper as u
    from privacy_risk.sequential import anytime_upper as a
    assert round(u(84, 500), 4) == 0.1979 and round(u(85, 500), 4) == 0.2001
    f = binom.cdf(84, 500, 0.201)
    assert round(f, 4) == 0.0350 and round(1 - (1 - f) ** 20, 3) == 0.509
    k = max(x for x in range(200) if u(x, 500, 0.0025) <= 0.2)
    g = binom.cdf(k, 500, 0.201)
    assert k == 75 and round(g, 4) == 0.0020 and round(1 - (1 - g) ** 20, 3) == 0.039
    up = 0.97 / 0.88
    assert math.ceil(math.log(20) / math.log(up)) == 31
    w = up ** 9 * 0.25
    assert round(w, 2) == 0.60 and math.ceil(math.log(20 / w) / math.log(up)) == 37
    assert round(a(20, 300), 3) == 0.130 and round(a(40, 600), 2) == 0.11
