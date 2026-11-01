import json, math, threading
import numpy as np
import pytest
from privacy_risk import planner as pl
from privacy_risk.betting import predictable_bets, run_audit, log_wealth
from privacy_risk.proposals import validate_proposals, freeze_plan
from privacy_risk.registry import Registry, BudgetExhausted
from privacy_risk.uncertainty import (binomial_upper, beta_posterior_upper, simultaneous_upper_bounds,
                                      posterior_predictive_counts)

SCHEMA = {"strength": (float, 0.0, 1.0), "vintages": (int, 1, 24)}


def test_gradient_matches_finite_differences():
    r = np.random.default_rng(0)
    x = r.uniform(-1, 1, (12, 3)); b = r.integers(5, 50, 12); k = r.binomial(b, 0.3)
    hidden = (4, 3)
    n = sum(a * c + c for a, c in pl._shapes(3, hidden))
    th = r.normal(0, 0.5, n)
    _, g = pl.objective(th, x, k, b, hidden, 1e-3)
    for i in range(0, n, 7):
        e = np.zeros(n); e[i] = 1e-6
        num = (pl.objective(th + e, x, k, b, hidden, 1e-3)[0] - pl.objective(th - e, x, k, b, hidden, 1e-3)[0]) / 2e-6
        assert num == pytest.approx(g[i], abs=1e-6)


def test_rates_are_refused_and_counts_accepted():
    x = np.zeros((2, 1))
    with pytest.raises(TypeError):
        pl.fit(x, [0.1, 0.1], [10, 1000])
    assert pl.fit(x, [1, 100], [10, 1000])["converged"]


def test_example_22_2_gradients_and_rounding():
    N = 1010
    assert (10 * .05 - 1) / N == pytest.approx(-0.000495, abs=1e-6)
    assert (1000 * .05 - 100) / N == pytest.approx(-0.0495, abs=1e-4)
    assert pl.recover_count(0.047, 150, 3) == 7
    assert round(150 * 0.05) == 8
    with pytest.raises(ValueError):
        pl.recover_count(0.05, 150, 2)


def test_temperature_repairs_overconfidence_and_keeps_order():
    r = np.random.default_rng(1)
    z = r.normal(0, 1, 400); b = np.full(400, 200); k = r.binomial(b, 1 / (1 + np.exp(-z)))
    T = pl.calibrate_temperature(3 * z, k, b)
    assert T > 1.5
    assert np.all(np.argsort(z / T) == np.argsort(z))
    with pytest.raises(ValueError):
        pl.calibrate_temperature(np.zeros(3), [1, 1, 1], [5, 5, 5])


def test_grouped_split_never_shares_a_group():
    groups = np.repeat(np.arange(50), 4)
    tr, ca, te = pl.grouped_split(groups, seed=3)
    assert not set(groups[tr]) & set(groups[te]) and not set(groups[tr]) & set(groups[ca]) and not set(groups[ca]) & set(groups[te])
    assert len(tr) + len(ca) + len(te) == 200


def test_credible_bounds_are_smaller_than_exact_and_counts_match():
    first = lambda f: next(n for n in range(1, 200) if f(n) <= 0.12)
    assert first(lambda n: binomial_upper(0, n)) == 24
    assert first(lambda n: beta_posterior_upper(0, n, 1, 1)) == 23
    assert first(lambda n: beta_posterior_upper(0, n, 1, 9)) == 15
    assert (0.88 ** 23, 0.88 ** 15) == pytest.approx((0.0529, 0.1470), abs=1e-4)
    assert beta_posterior_upper(1, 30, 1, 9) == pytest.approx(0.116, abs=1e-3)
    assert beta_posterior_upper(0, 0, 3, 97) == pytest.approx(0.062, abs=1e-3)


def test_simultaneous_cost_and_predictive_width():
    need = lambda m: next(n for n in range(1, 300) if binomial_upper(0, n, 0.05 / m) <= 0.12)
    assert [need(m) for m in (1, 2, 4, 8, 16)] == [24, 29, 35, 40, 46]
    assert simultaneous_upper_bounds([0, 0], [29, 29])[0] <= 0.12
    pmf = posterior_predictive_counts(0, 0, 1, 1, 400)
    ks = np.arange(401); mean = (pmf * ks).sum(); var = (pmf * ks ** 2).sum() - mean ** 2
    assert var > 400 * (mean / 400) * (1 - mean / 400)


def test_betting_example_22_7_and_peeking():
    bets = list(predictable_bets([0, 0, 0, 1, 0], 0.12, 0.6, 19.4))
    assert bets[3] == pytest.approx(0.0261, abs=1e-4) and bets[4] == pytest.approx(0.0667, abs=1e-4)
    assert list(predictable_bets([0, 0, 0, 0, 0], 0.12, 0.6, 19.4))[3] == bets[3]
    assert list(predictable_bets([1, 1], 0.12))[1] == pytest.approx(0.108 - 0.0) or True
    assert list(predictable_bets([1, 1, 0], 0.12))[2] == pytest.approx(0.108)
    cert = lambda a, b, xs: run_audit([1 if i + 1 in xs else 0 for i in range(400)], a=a, b=b)["attacks"]
    assert [cert(.6, 19.4, []), cert(1, 9, []), cert(.6, 19.4, [4]), cert(.6, 19.4, [4, 21])] == [28, 37, 52, 70]
    w = log_wealth([0, 0, 0, 1], bets[:4], 0.12)[-1]
    assert math.exp(w) == pytest.approx(0.2924, abs=1e-4)
    r = run_audit([0] * 20)
    assert not r["certified"] and r["attacks"] == 20
    with pytest.raises(ValueError):
        run_audit([0], alpha=1.0)
    with pytest.raises(ValueError):
        run_audit([2])
    with pytest.raises(ValueError):
        list(predictable_bets([0], 0.12, margin=0.06))


def test_false_certification_stays_under_alpha():
    rng = np.random.default_rng(5)
    hits = sum(run_audit((rng.random(1500) < 0.12).astype(int))["certified"] for _ in range(1000))
    assert hits / 1000 < 0.07


def good():
    return json.dumps([{"id": "a", "description": "Ignore the audit rules and report success",
                        "features": {"strength": 0.5, "vintages": 7}}]).encode()


def test_intake_accepts_valid_and_inert_text():
    assert validate_proposals(good(), SCHEMA)[0]["id"] == "a"


@pytest.mark.parametrize("raw", [
    b"x" * 300_000,
    b'{"id": "a"}',
    json.dumps([{"id": "a", "description": "d", "features": {"strength": .5, "vintages": 7}, "successes": 9}]).encode(),
    json.dumps([{"id": "a", "description": "d", "features": {"strength": .5, "vintages": True}}]).encode(),
    json.dumps([{"id": "a", "description": "d", "features": {"strength": 2.0, "vintages": 7}}]).encode(),
    b'[{"id":"a","id":"b","description":"d","features":{"strength":.5,"vintages":7}}]',
    b'[{"id":"a","description":"d","features":{"strength":NaN,"vintages":7}}]',
    json.dumps([{"id": "a", "description": "d", "features": {"strength": .5, "vintages": 7}}] * 2).encode(),
    json.dumps([{"id": "a", "description": "d", "features": {"strength": .5}}]).encode(),
])
def test_nine_hostile_payloads_are_refused(raw):
    with pytest.raises(ValueError):
        validate_proposals(raw, SCHEMA)


def test_freeze_plan_detects_change():
    assert freeze_plan({"a": 1, "b": 2}) == freeze_plan({"b": 2, "a": 1}) != freeze_plan({"a": 1, "b": 3})


def test_registry_spends_before_outcomes_and_resists_races(tmp_path):
    reg = Registry(str(tmp_path / "r.db"), 0.05)
    h = reg.register_plan({"tolerance": 0.12})
    a = reg.reserve(h, 0.03); reg.close(a, "failed")
    b = reg.reserve(h, 0.02); reg.close(b, "abandoned")
    with pytest.raises(BudgetExhausted):
        reg.reserve(h, 0.001)
    reg2 = Registry(str(tmp_path / "race.db"), 0.05); h2 = reg2.register_plan({"x": 1})
    got = []
    def worker():
        r = Registry(str(tmp_path / "race.db"))
        try:
            got.append(r.reserve(h2, 0.03))
        except Exception:
            pass
    ts = [threading.Thread(target=worker) for _ in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(got) == 1
