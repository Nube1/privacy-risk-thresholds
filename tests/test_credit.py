import numpy as np
import pytest

from privacy_risk.credit import (
    borrower_identification,
    central_bank_utility,
    differencing_risk,
    make_credit_register,
    reconstruction_from_totals,
    relationship_disclosure,
    released_loans,
    status_inference,
)

CONFIG = {"seed": 20260904,
          "credit": {"borrowers": 2500, "lenders": 20, "months": 4}}
REGISTER = make_credit_register(CONFIG)


def test_register_shape_and_relationships():
    assert REGISTER.loans > REGISTER.n_borrowers
    # loans that leave do not return, and some loans enter after the first month
    live = REGISTER.live
    for m in range(1, live.shape[1] - 1):
        assert not (live[:, m - 1] & ~live[:, m] & live[:, m + 1]).any()
    assert (~live[:, 0] & live[:, -1]).any()
    assert REGISTER.key().max() < (REGISTER.n_sectors * REGISTER.n_regions
                                   * REGISTER.n_size_classes)
    assert (REGISTER.exposure > 0).all()


def test_minimum_cell_count_only_removes_records():
    loose = released_loans(REGISTER, 1)
    tight = released_loans(REGISTER, 20)
    assert tight.size <= loose.size
    _, counts = np.unique(REGISTER.key()[tight], return_counts=True)
    assert counts.min() >= 20


def test_identification_rises_with_borrower_size():
    rows = released_loans(REGISTER, 1)
    result = borrower_identification(REGISTER, rows, 0.6, 1500,
                                     np.random.default_rng(3))
    by_class = result["rate_by_size_class"]
    assert result["rate"] > 0
    if "0" in by_class and "3" in by_class:
        assert by_class["3"] >= by_class["0"]
    assert borrower_identification(REGISTER, np.array([], dtype=int), 0.6, 10,
                                   np.random.default_rng(4))["trials"] == 0


def test_relationship_predictive_value_satisfies_the_base_rate_identity():
    rows = released_loans(REGISTER, 1)
    result = relationship_disclosure(REGISTER, rows)
    assert result["positive_predictive_value"] == pytest.approx(
        result["base_rate_identity"], abs=1e-6)
    assert 0 < result["prevalence"] < 0.2
    assert result["specificity"] > 0.9
    assert result["true_positive"] + result["false_positive"] == result["allegations"]
    assert relationship_disclosure(REGISTER, np.array([], dtype=int))["allegations"] == 0


def test_status_inference_finds_homogeneous_classes():
    rows = released_loans(REGISTER, 1)
    result = status_inference(REGISTER, rows)
    assert 0 <= result["homogeneous_share"] <= 1
    assert result["confident_accuracy"] > 0.5
    classes = result["homogeneous_share_by_size_class"]
    assert classes and all(0 <= v <= 1 for v in classes.values())


def test_reconstruction_share_falls_as_contributors_are_required():
    loose = reconstruction_from_totals(REGISTER, 0, 2)
    tight = reconstruction_from_totals(REGISTER, 0, 6)
    assert tight["cells_below_minimum"] >= loose["cells_below_minimum"]
    assert 0 <= loose["exposed_amount_share"] <= 1
    assert loose["single_contributor_cells"] <= loose["cells"]


def test_differencing_shares_are_probabilities():
    for minimum in (1, 10):
        result = differencing_risk(REGISTER, minimum)
        assert 0 <= result["unit_change_share"] <= 1
        assert 0 < result["posterior_single_move"] <= 1


def test_utility_falls_as_records_are_suppressed():
    loose = central_bank_utility(REGISTER, released_loans(REGISTER, 1))
    tight = central_bank_utility(REGISTER, released_loans(REGISTER, 20))
    assert loose["utility"] >= tight["utility"]
    assert set(loose["losses"]) == {"growth", "concentration", "stability",
                                    "supervision"}
    assert central_bank_utility(REGISTER, np.array([], dtype=int))["utility"] == -10.0
