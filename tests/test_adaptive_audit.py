"""Check conditional validity where optional monitoring actually matters."""
import math

import numpy as np
import pytest

from privacy_risk.adaptive_audit import AdaptiveAudit, planned_weights, zero_success_budget
from scripts.adaptive_audit_comparison import simulate


def test_zero_success_boundary_and_late_registration():
    gate = AdaptiveAudit(.05)
    first = gate.register("initial", .12, .5)
    b = zero_success_budget(.12, .05, .5)
    for _ in range(b - 1):
        assert not first.observe(0, 0)
    assert first.observe(0, 0)
    later = gate.register("llm-proposed-pooled-attack", .18, .5)
    assert not gate.eligible(["initial", "llm-proposed-pooled-attack"])
    assert later.observe(0, 0) is False
    assert gate.eligible(["initial", "missing"]) is False
    with pytest.raises(ValueError):
        gate.register("third", .2, .01)


def test_predictable_bets_and_invalid_inputs():
    gate = AdaptiveAudit()
    coord = gate.register("A", .2, 1)
    with pytest.raises(ValueError):
        coord.observe(0, .2)
    with pytest.raises(ValueError):
        coord.observe(1, float("nan"))
    with pytest.raises(ValueError):
        coord.observe(True, .1)
    assert not coord.observe(1, 0)
    assert coord.log_e == -math.inf
    with pytest.raises(ValueError):
        gate.register("A", .2, .1)


def test_adaptive_stopping_null_error_monte_carlo():
    # Optional stopping over up to 250 looks under risk exactly at tolerance.
    # This is a regression check on the implementation, not proof of validity.
    rng = np.random.default_rng(731)
    false_passes = 0
    runs = 2000
    for _ in range(runs):
        audit = AdaptiveAudit(.05)
        coordinate = audit.register("one", .1, 1)
        for hit in rng.binomial(1, .1, 250):
            if coordinate.observe(int(hit), .035):
                false_passes += 1
                break
    assert false_passes / runs <= .05 + 3 * math.sqrt(.05 * .95 / runs)


def test_continuous_cost_allocation_prefers_expensive_low_tolerance():
    weights = planned_weights([.05, .2], [2, 1])
    assert math.isclose(sum(weights), 1)
    assert weights[0] > weights[1]
    with pytest.raises(ValueError):
        planned_weights([.05, .2], [1])


def test_toy_comparison_is_seeded_and_separates_safe_from_boundary():
    result = simulate(seed=2709, replications=500, cap=300)
    assert result == simulate(seed=2709, replications=500, cap=300)
    assert result["scenario_results"]["safe"]["anytime_pass_rate"] > .85
    assert result["scenario_results"]["boundary"]["anytime_pass_rate"] < .03


def test_supermartingale_factor_is_at_most_one_on_the_composite_null():
    for tau in (.01, .05, .12, .3, .7):
        for p in np.linspace(tau, 1, 41):
            for q in np.linspace(0, tau, 41, endpoint=False):
                assert p * q / tau + (1 - p) * (1 - q) / (1 - tau) <= 1 + 1e-12


def test_certification_holds_at_first_crossing_and_stops_the_stream():
    gate = AdaptiveAudit(.05)
    coord = gate.register("A", .12, 1)
    needed = zero_success_budget(.12, .05, 1)
    for _ in range(needed):
        coord.observe(0, 0)
    assert coord.certified and gate.eligible(["A"])
    with pytest.raises(RuntimeError):
        coord.observe(1, .05)
    assert gate.eligible(["A"])
    assert coord.log == [(0, 0.0)] * needed


def test_registered_weight_cannot_be_changed():
    gate = AdaptiveAudit(.05)
    coord = gate.register("A", .12, .01)
    with pytest.raises(AttributeError):
        coord.weight = 1.0


@pytest.mark.parametrize("tau,alpha,w", [(.12, .05, .5), (.12, .05, 1 / 8), (.001, .05, 1),
                                         (1 - 2 ** -.5, .125, 1), (.3, .01, .2)])
def test_zero_success_budget_is_the_crossing_trial(tau, alpha, w):
    gate = AdaptiveAudit(alpha)
    coord = gate.register("A", tau, w)
    trials = 0
    while not coord.observe(0, 0):
        trials += 1
    assert trials + 1 == zero_success_budget(tau, alpha, w)


def test_chain_shares_one_weight_and_stops_at_first_failure():
    from privacy_risk.adaptive_audit import AdaptiveAudit
    gate = AdaptiveAudit(0.05)
    chain = gate.register_chain("pooled", 0.12, 0.5, ["50", "40", "30"])
    assert gate.spent == 0.5
    first = chain.open_next()
    assert first.threshold == pytest.approx(-math.log(0.05 * 0.5))
    with pytest.raises(RuntimeError):
        chain.open_next()
    while not first.certified:
        first.observe(0, 0.0)
    second = chain.open_next()
    second.observe(1, 0.0)
    with pytest.raises(RuntimeError):
        chain.open_next()
    assert chain.floor() == "50"


def test_state_cannot_be_set_from_outside():
    from privacy_risk.adaptive_audit import AdaptiveAudit
    gate = AdaptiveAudit(0.05)
    coord = gate.register("a", 0.12, 0.5)
    for name in ("crossed", "certified", "log_e", "trials"):
        with pytest.raises(AttributeError):
            setattr(coord, name, True)
    with pytest.raises(AttributeError):
        gate.spent = 0.0


def test_drift_weights_reduce_to_zero_success_weights():
    from privacy_risk.adaptive_audit import (drift_weights, expected_length_bound,
                                             planned_weights)
    taus, costs = [0.12, 0.05], [1.0, 3.0]
    assert drift_weights(taus, costs, [0, 0], [0, 0]) == pytest.approx(
        planned_weights(taus, costs))
    near = expected_length_bound(0.10, 0.10, 0.12, 0.05, 1 / 3)
    far = expected_length_bound(0.06, 0.06, 0.12, 0.05, 1 / 3)
    assert near > 5 * far


def test_committed_coordinate_logs_bet_before_outcome_and_ledger_detects_edits():
    import numpy as np
    from privacy_risk.adaptive_audit import (AdaptiveAudit, AuditLedger,
                                             CommittedCoordinate, commit)
    gate = AdaptiveAudit(0.05)
    ledger = AuditLedger()
    secret = b"operator secret"
    coord = CommittedCoordinate(gate.register("c", 0.12, 1.0), ledger, commit(secret),
                                "beacon-round-1")
    with pytest.raises(RuntimeError):
        coord.run(secret, lambda seed: 0)
    for _ in range(10):
        coord.commit_bet(0.0)
        coord.run(secret, lambda seed: int(np.random.default_rng(seed).random() < 0.0))
    kinds = [e["kind"] for e in ledger.entries]
    assert kinds[0] == "register" and kinds[1:3] == ["bet", "outcome"]
    assert AuditLedger.verify(ledger.entries)
    tampered = ledger.entries
    tampered[2]["payload"]["outcome"] = 1
    assert not AuditLedger.verify(tampered)
    with pytest.raises(ValueError):
        coord.commit_bet(0.0)
        coord.run(b"wrong secret", lambda seed: 0)
