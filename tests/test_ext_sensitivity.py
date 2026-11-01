import math
import numpy as np
import pytest
from privacy_risk.sensitivity import (finite_time_growth, lorenz, lorenz_jacobian, diagnostic_horizon,
                                      flip_rate_formula, decision_flip_rate, robust_gate, _rk4)
from privacy_risk.dp import postprocessing_bound


def test_linear_growth_rate_is_exact_for_an_eigenvector():
    A = np.diag([0.5, -1.0])
    assert finite_time_growth(lambda z: A @ z, lambda z: A, [1, 1], [1, 0], 4.0) == pytest.approx(0.5, abs=1e-10)


def test_lorenz_rates_match_chapter():
    z = np.ones(3)
    for _ in range(2000):
        z = _rk4(lorenz, z, 0.005)
    got = [round(finite_time_growth(lorenz, lorenz_jacobian, z, e, 60.0), 3) for e in np.eye(3)]
    assert got == [0.882, 0.893, 0.861]


def test_diagnostic_horizon_example():
    assert diagnostic_horizon(0.88, 0.5, 0.002, 0.02) == pytest.approx(3.40, abs=0.01)
    assert diagnostic_horizon(0.88, 0.5, 0.0002, 0.02) == pytest.approx(6.02, abs=0.01)
    assert diagnostic_horizon(-0.1, 0.5, 0.002, 0.02) == math.inf
    with pytest.raises(ValueError):
        diagnostic_horizon(0.88, 0.5, 0.05, 0.02)


def test_flip_rate_formula_and_simulation_agree():
    assert flip_rate_formula(0.11, 0.12, 0.01) == pytest.approx(0.1587, abs=1e-4)
    assert flip_rate_formula(0.11, 0.12, 0.02) == pytest.approx(0.3085, abs=1e-4)
    r = decision_flip_rate(lambda x: x[0] > 0.12, [0.11], [0.01], 20000, seed=1)
    assert abs(r["rate"] - 0.1587) < 0.01


def test_flip_rate_refusals_and_zero_sd():
    rule = lambda x: x[0] > 0.12
    assert decision_flip_rate(rule, [0.11], [0.0], 100)["rate"] == 0.0
    with pytest.raises(ValueError):
        decision_flip_rate(rule, [0.11], [-0.01], 100)
    with pytest.raises(ValueError):
        decision_flip_rate(rule, [0.11], [0.01, 0.01], 100)
    with pytest.raises(ValueError):
        decision_flip_rate(rule, [0.11], [0.01], 0)


def test_stress_gate_example_21_6():
    tol = {"ext": 0.12}
    full = {"ext": {"base": 0.104, "precise": 0.112, "merger": 0.131}}
    r = robust_gate(tol, full)
    assert r["status"] == "FAIL" and r["failures"][0]["scenario"] == "merger"
    assert robust_gate(tol, {"ext": {"base": 0.104, "precise": 0.112}})["failures"] == []
    r = robust_gate({"ext": 0.12, "ins": 0.12}, {"ext": {"base": 0.104}})
    assert r["failures"][0] == {"attack": "ins", "reason": "missing"}


def test_postprocessing_inequality_for_randomised_response():
    p, q = [0.75, 0.25], [0.25, 0.75]
    l, r, ok = postprocessing_bound(p, q, [1, 0], math.log(3))
    assert ok and l == pytest.approx(0.75) and r == pytest.approx(0.75)
    assert postprocessing_bound(p, q, [0.3, 0.9], math.log(3))[:2] == pytest.approx((0.45, 2.25))
    assert postprocessing_bound(p, q, [0.6, 0.6], math.log(3))[:2] == pytest.approx((0.6, 1.8))
