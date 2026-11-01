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


def test_chapters7_8_examples():
    from privacy_risk.uncertainty import binomial_upper as u
    cells = ((30, 400), (9, 150), (2, 60))
    assert [round(u(k, n), 3) for k, n in cells] == [0.1, 0.102, 0.101]
    assert [round(u(k, n, 0.05 / 3), 3) for k, n in cells] == [0.108, 0.116, 0.123]
    ps = lambda h, cf, ca: (cf + ca) / (h + cf)
    assert (ps(1, 1, 0), round(ps(10, 1, .5), 3), round(ps(1, 5, .5), 3)) == (0.5, 0.136, 0.917)
    pts = {"A": (.08, .40), "B": (.15, .70), "C": (.35, .86), "D": (.46, .90)}
    s = lambda i, j: (pts[j][1] - pts[i][1]) / (pts[j][0] - pts[i][0])
    assert [round(x, 2) for x in (s("C", "D"), s("B", "C"), s("A", "B"), s("B", "D"))] == [0.36, 0.8, 4.29, 0.65]
    e = 1 / 1.23
    a = 0.2 / (1 + 0.25 ** e)
    S = lambda r1, r2: r1 ** -0.23 + 0.25 * r2 ** -0.23
    assert round(a, 3) == 0.151 and round(0.2 - a, 3) == 0.049
    assert [round(S(*x), 3) for x in ((a, 0.2 - a), (.16, .04), (.1, .1))] == [2.045, 2.048, 2.123]


def test_chapter9_examples():
    import itertools
    assert round(1 - .96 * .98 * .99, 3) == 0.069
    loans = list(itertools.product([1, 2, 3], repeat=2))
    r1, r2 = {}, {}
    for l in loans:
        r1.setdefault(l[1], []).append(l)
        r2.setdefault(l[0], []).append(l)
    hits = sum(set(r1[t[1]]) & set(r2[t[0]]) == {t} for t in loans)
    assert hits == 9 and all(len(v) == 3 for v in list(r1.values()) + list(r2.values()))


def test_chapter10_examples():
    import math
    e = math.e
    assert (round(e * .02, 3), round(e * .05, 3)) == (0.054, 0.136)
    post = lambda eps, p: math.exp(eps) * p / (math.exp(eps) * p + 1 - p)
    assert [round(post(1, .1), 3), round(post(4, .1), 3), round(post(1, 1 / 3), 3), round(post(4, 1 / 3), 3)] == [0.232, 0.858, 0.576, 0.965]


def test_chapter11_examples():
    import numpy as np
    from scipy.stats import norm
    from privacy_risk.uncertainty import binomial_lower as lo
    y = np.array([8.2, 8.9, 9.6])
    for s, want in ((.75, [0.216, 0.428, 0.355]), (.375, [0.042, 0.65, 0.308])):
        L = norm.pdf(9.1, y, s)
        assert list((L / L.sum()).round(3)) == want
    assert round(lo(186, 250, .05), 3) == 0.695 and round(lo(186, 250, .05 / 230), 3) == 0.638
    assert sum(1 / k for k in (3, 2, 1, 4) for _ in range(k)) / 10 == 0.4


def test_chapter12_examples():
    import math
    from privacy_risk.uncertainty import binomial_upper as u
    assert round(u(359, 1500), 3) == 0.258 and round(u(359, 1500, .05 / 72), 3) == 0.276
    assert round(math.exp(-10 * .29), 3) == 0.055 and round(math.exp(-2 * .29), 2) == 0.56
