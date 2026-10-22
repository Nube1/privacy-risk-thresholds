"""The neural score cannot bypass the independently audited release gate."""
import math
from unittest.mock import patch

import numpy as np
import pytest

from privacy_risk.neural_trust import (
    MLP,
    ReleaseEvidence,
    certify,
    propose_and_audit,
    synthetic_demo,
    upper_binomial,
    zero_success_budget,
)
from privacy_risk.neural_trust import main as neural_main


def test_zero_success_bound_and_family_penalty():
    single = upper_binomial(np.array([0]), np.array([100]), .05)[0]
    assert single == pytest.approx(1 - .05 ** (1 / 100))
    evidence = ReleaseEvidence("candidate", np.array([0, 0]),
                               np.array([100, 100]), np.array([.04, .04]), 1.)
    result = certify(evidence, .05, total_tests=4)
    assert not result["certified"]
    assert result["upper"][0] > single
    assert result["trust_position"][0] < 0


def test_paper_budget_is_the_smallest_zero_success_audit():
    budget = zero_success_budget(.12, .05, 8)
    assert budget == 40
    assert upper_binomial(np.array([0]), np.array([budget]), .05 / 8)[0] <= .12
    assert upper_binomial(np.array([0]), np.array([budget - 1]), .05 / 8)[0] > .12
    with pytest.raises(ValueError):
        zero_success_budget(.12, .05, 0)


@pytest.mark.parametrize("tolerance,alpha,tests", [
    (.12, .05, 8), (.04, .05, 4), (.05, .05, 1), (.01, .05, 20), (.20, .05, 4),
    (.18, .05, 8), (.3, .01, 50), (.001, .1, 3), (.5, .2, 1)])
def test_zero_success_budget_matches_closed_form(tolerance, alpha, tests):
    closed = math.ceil(math.log(alpha / tests) / math.log(1 - tolerance))
    budget = zero_success_budget(tolerance, alpha, tests)
    assert budget == max(1, closed)
    assert upper_binomial(np.array([0]), np.array([budget]), alpha / tests)[0] <= tolerance
    if budget > 1:
        assert upper_binomial(np.array([0]), np.array([budget - 1]),
                              alpha / tests)[0] > tolerance


def test_cli_rejects_dimension_one_with_usage_error(capsys):
    with patch("sys.argv", ["privacy-risk-neural-demo", "--dimension", "1"]):
        with pytest.raises(SystemExit) as stop:
            neural_main()
    assert stop.value.code == 2
    assert "--dimension must be at least 2" in capsys.readouterr().err


def test_adverse_audit_blocks_even_if_network_recommends_candidate():
    model = MLP(2, 4, seed=2, outputs=2)
    # Keep the model fixed. The second coordinate is certainly unsafe in audit.
    def audit(_name):
        return np.array([0, 100]), np.array([100, 100])

    report = propose_and_audit(model, np.array([[0., 0.]]), ["A"],
                               np.array([10.]), np.array([.2, .2]), audit,
                               shortlist_size=1)
    assert report["selected"] is None
    assert report["reports"][0]["upper"][1] == 1.


def test_dimensions_seed_and_integer_audit_validation():
    assert synthetic_demo(19, 8) == synthetic_demo(19, 8)
    report = synthetic_demo(19, 32)
    assert report["dimension"] == 32
    assert report["audited_coordinates"] == 8
    assert all(len(r["neural_estimate_for_ranking_only"]) == 2 for r in report["reports"])
    with pytest.raises(ValueError):
        upper_binomial(np.array([.5]), np.array([100]), .05)


def test_unobserved_output_cannot_be_trained_as_zero_risk():
    model = MLP(2, 4, seed=3, outputs=2)
    with pytest.raises(ValueError, match="observations per output"):
        model.fit(np.ones((3, 2)), np.zeros((3, 2)),
                  weights=np.array([[1., 0.], [1., 0.], [1., 0.]]))
