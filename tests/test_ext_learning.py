"""Checks from Chapter 18 (the release gate, the betting audit, the registry)."""
import math

import numpy as np
import pytest

from privacy_risk.betting import predictable_bets, run_audit
from privacy_risk.evidence import MeasurementRecord, training_rows
from privacy_risk.gate import Coordinate, decide_release
from privacy_risk.registry import BudgetExhausted, Registry

UTIL = {"75": 0.45, "150": 0.20}
BOUNDS = [Coordinate("75", "external", 0.104), Coordinate("150", "external", 0.109),
          Coordinate("150", "insider", 0.42)]


def test_external_only_releases_the_floor_of_75():
    r = decide_release(UTIL, {"external": 0.12}, BOUNDS)
    assert r["status"] == "RELEASE" and r["candidate"] == "75"


def test_insider_required_withholds_and_names_the_gaps():
    r = decide_release(UTIL, {"external": 0.12, "insider": 0.12}, BOUNDS)
    assert r["status"] == "WITHHOLD"
    assert r["missing"]["75"] == ["insider"] and r["failed"]["150"] == ["insider"]


def test_bound_equal_to_tolerance_passes():
    r = decide_release({"75": 0.45}, {"external": 0.12}, [Coordinate("75", "external", 0.12)])
    assert r["candidate"] == "75"


def test_planner_prediction_and_empty_inputs_are_refused():
    with pytest.raises(ValueError):
        decide_release(UTIL, {"external": 0.12}, [Coordinate("75", "external", 0.1, "planner")])
    with pytest.raises(ValueError):
        decide_release(UTIL, {}, BOUNDS)


def test_the_first_bets_of_example_18_4():
    wealth, trace = 1.0, []
    outcomes = [0, 0, 0, 1]
    for x, bet in zip(outcomes, predictable_bets(outcomes, 0.12)):
        wealth *= bet / 0.12 if x else (1 - bet) / 0.88
        trace.append(round(wealth, 3))
    assert trace == [1.023, 1.057, 1.101, 0.705]


def test_clean_run_certifies_and_zero_success_audit_needs_24():
    assert run_audit([0] * 100)["attacks"] == 37
    assert math.ceil(math.log(0.05) / math.log(0.88)) == 24
    assert run_audit([0, 0, 0, 1] + [0] * 200)["attacks"] == 55


def test_error_at_the_tolerance_stays_below_alpha():
    rng = np.random.default_rng(3)
    n = 1000
    cert = sum(run_audit(rng.random(1500) < 0.12)["certified"] for _ in range(n))
    assert cert / n < 0.08


def test_example_18_3_four_reservations_against_one_budget():
    reg = Registry(total_budget=0.05)
    p50 = reg.register_plan({"floor": 50})
    a = reg.reserve(p50, 0.0125); assert reg.remaining() == 0.0375; reg.close(a, "failed")
    a = reg.reserve(p50, 0.0125); assert reg.remaining() == 0.025; reg.close(a, "failed")
    p75 = reg.register_plan({"floor": 75})
    a = reg.reserve(p75, 0.0125); assert reg.remaining() == 0.0125; reg.close(a, "passed")
    a = reg.reserve(p75, 0.0125); assert reg.remaining() == 0.0; reg.close(a, "abandoned")
    for share in (0.0125, 0.001):
        with pytest.raises(BudgetExhausted):
            reg.reserve(p75, share)
    assert reg.verify_hashes() == []


def test_unregistered_plan_cannot_reserve_and_edits_are_detected():
    reg = Registry(total_budget=0.05)
    with pytest.raises(KeyError):
        reg.reserve("nothing", 0.01)
    h = reg.register_plan({"floor": 75})
    reg.db.execute("UPDATE plan SET body='{\"floor\":1}' WHERE hash=?", (h,))
    assert reg.verify_hashes() == [h]


def _row(split, **kw):
    base = dict(scenario="s", mechanism="m", target_law="loans", conditioning="live at v1",
                split=split, seed_stream="a", data_digest="d", evaluator_digest="e",
                successes=3, trials=50)
    base.update(kw)
    return MeasurementRecord(**base)


def test_records_need_integer_counts_and_final_audit_never_trains():
    with pytest.raises(TypeError):
        _row("development", successes=0.06)
    with pytest.raises(ValueError):
        _row("final-audit")
    final = _row("final-audit", plan_hash="h", error_share=0.0125)
    dev = _row("development")
    assert training_rows([final, dev]) == [dev]
    with pytest.raises(PermissionError):
        training_rows([final, dev], splits=("development", "final-audit"))
