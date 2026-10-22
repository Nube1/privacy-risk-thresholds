import math

import numpy as np
import pytest

from privacy_risk.adaptive_audit import AdaptiveAudit
from privacy_risk.credit import make_credit_register
from privacy_risk.neural_pipeline import (bet_from, fit, floor_features, release_chain,
                                          write_table)

CONFIG = {"seed": 20260904, "credit": {"borrowers": 1500, "lenders": 15, "months": 4}}


def test_floor_features_need_no_attack_and_move_with_the_floor():
    register = make_credit_register(CONFIG)
    loose, strict = floor_features(register, 1), floor_features(register, 20)
    assert loose[0] == pytest.approx(0.0) and strict[0] == pytest.approx(math.log(20))
    assert loose[1] == pytest.approx(1.0) and strict[1] < loose[1]


def test_bets_stay_below_the_tolerance_and_above_zero():
    for prediction in (0.0, 0.05, 0.119999, 0.5):
        q = bet_from(prediction, 0.12, 0.11)
        assert 0 < q <= 0.11


def test_the_network_fits_counts_and_predicts_probabilities():
    x = np.linspace(0, 1, 12)[:, None]
    k = np.round(200 * (0.3 - 0.2 * x)).astype(float)
    b = np.full_like(k, 200.0)
    positions = np.arange(12)
    net, mean, scale, metrics = fit(x, k, b, positions[positions % 3 != 1],
                                    positions[positions % 3 == 1], depth=2, width=4, seed=1)
    p = net.predict((x - mean) / scale)
    assert metrics["best_epoch"] >= 1
    assert np.all((p > 0) & (p < 1)) and p[0, 0] > p[-1, 0]


def test_a_release_chain_with_frozen_bets_stops_at_the_first_failing_floor():
    levels, names = [50, 10, 1], ["a", "b"]
    hits = {("a", 50): [0] * 300, ("b", 50): [0] * 300,
            ("a", 10): [0] * 300, ("b", 10): [1] * 300,
            ("a", 1): [0] * 300, ("b", 1): [0] * 300}
    bets = {(n, k): 0.03 for n in names for k in levels}
    result = release_chain(AdaptiveAudit(0.05), "c", hits, levels, names, 0.12, 1.0,
                           300, bets, 0.11)
    assert result["floor"] == 50
    assert [a["k"] for a in result["audited"]] == [50, 10]


def test_the_table_lists_every_design(tmp_path):
    designs = {d: {"floors": [150, 100], "median_attacks": 10.0}
               for d in ("plug_in", "logistic_bets", "network_bets", "logistic_plan",
                         "network_plan")}
    path = tmp_path / "t.tex"
    write_table({"designs": designs, "seeds": 2, "tolerance": 0.12, "alpha": 0.05}, path)
    text = path.read_text()
    assert "tab:neural-pipeline" in text and text.count("150 (1)") == 5
