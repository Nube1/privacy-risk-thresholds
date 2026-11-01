"""Checks from Chapter 19 (changing panels and the graph baseline)."""
import numpy as np
import pytest

from privacy_risk.dynamics import PanelModel, pooled_attack, simulate_panel
from privacy_risk.graph import estimate_generator, node_risk, propagate, validate_generator

Q = np.array([[-.15, .10, .05], [.05, -.10, .05], [.20, 0, -.20]])
G = [0.02, 0.05, 0.30]


def test_example_19_6_masses_and_risk():
    c0 = [0.6, 0.3, 0.1]
    assert node_risk(G, c0) == pytest.approx(0.057)
    for t, masses, risk in ((1, (.552, .326, .122), .064), (5, (.449, .379, .171), .079)):
        c = propagate(Q, c0, t)
        assert c.sum() == pytest.approx(1.0) and np.all(c >= 0)
        assert c == pytest.approx(masses, abs=6e-4)
        assert node_risk(G, c) == pytest.approx(risk, abs=6e-4)
    assert node_risk(G, propagate(Q, c0, 200)) == pytest.approx(0.088, abs=5e-4)


def test_risk_stays_between_state_risks_and_generator_checks():
    c = propagate(Q, [1, 0, 0], 3.0)
    assert min(G) <= node_risk(G, c) <= max(G)
    with pytest.raises(ValueError):
        validate_generator([[-1, 2], [1, -1]])


def test_estimated_generator_from_counts_is_example_19_6():
    n = np.array([[0, 6, 3], [2, 0, 2], [4, 0, 0]])
    assert np.allclose(estimate_generator(n, [60, 40, 20]), Q)


def test_graph_estimator_recovers_a_chain_from_simulated_paths():
    rng = np.random.default_rng(1)
    k = 4
    true = rng.uniform(0.05, 0.15, (k, k)); np.fill_diagonal(true, 0); np.fill_diagonal(true, -true.sum(1))
    counts, time = np.zeros((k, k)), np.zeros(k)
    for _ in range(400):
        s, t = int(rng.integers(k)), 0.0
        while True:
            q = -true[s, s]
            wait = rng.exponential(1 / q)
            if t + wait > 10:
                time[s] += 10 - t; break
            time[s] += wait; t += wait
            p = true[s].copy(); p[s] = 0
            nxt = int(rng.choice(k, p=p / p.sum())); counts[s, nxt] += 1; s = nxt
    assert np.abs(estimate_generator(counts, time) - true).max() < 0.05


def test_panel_is_live_in_vintage_one_and_frozen_limit_does_not_move():
    rng = np.random.default_rng(5)
    f = simulate_panel(PanelModel.static(n_loans=500), rng)
    assert f["live"][:, 0].all() and np.allclose(f["loge"], f["loge"][:, :1])
    assert abs(f["live"][:, 1:].mean() - 0.881) < 0.02


def test_changing_activity_rates():
    rng = np.random.default_rng(6)
    c = simulate_panel(PanelModel(n_loans=20000), rng)
    live = c["live"]
    stay = (live[:, 1:] & live[:, :-1]).sum() / live[:, :-1].sum()
    back = (live[:, 1:] & ~live[:, :-1]).sum() / (~live[:, :-1]).sum()
    assert stay == pytest.approx(0.818, abs=0.01) and back == pytest.approx(0.5, abs=0.02)


def test_two_loans_one_absent_in_vintage_two():
    live = np.array([[1, 1, 1], [1, 0, 1]], bool)
    loge = np.full((2, 3), 11.6)
    panel = {"loge": loge, "live": live, "cls": np.zeros(2, int)}
    rng = np.random.default_rng(0)
    hit = sum(pooled_attack(panel, 3, [0], False, 0.3, rng)[0] for _ in range(200))
    assert hit == 200
    hit1 = sum(pooled_attack(panel, 1, [0], False, 0.3, rng)[0] for _ in range(200))
    assert 0.35 < hit1 / 200 < 0.65


def test_target_must_be_live_and_w_must_fit():
    panel = {"loge": np.zeros((2, 3)), "live": np.array([[0, 1, 1], [1, 1, 1]], bool), "cls": np.zeros(2, int)}
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        pooled_attack(panel, 2, [0], True, 0.0, rng)
    with pytest.raises(ValueError):
        pooled_attack(panel, 4, [1], True, 0.0, rng)


def test_changing_panel_reveals_more_than_frozen():
    rng = np.random.default_rng(11)
    out = {}
    for name, mk in (("frozen", PanelModel.static), ("changing", PanelModel)):
        panels = [simulate_panel(mk(), rng) for _ in range(3)]
        out[name] = sum(pooled_attack(p, 6, np.flatnonzero(p["live"][:, 0])[:400], True, 0.3, rng)[0] for p in panels)
    assert out["changing"] > out["frozen"] + 100
