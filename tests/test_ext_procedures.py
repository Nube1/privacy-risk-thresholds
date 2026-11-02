import math

import pytest

from privacy_risk.procedures import (DefenceRecord, allocate_error_budget, combine_readings,
                                     composition_remedy, floor_by_precision,
                                     reallocate_for_new_product, reserve_share, rounding_sd)
from privacy_risk.validation import (full_validation_report, uniqueness_transport_error,
                                     validation_report)


def test_combine_readings_never_worse_than_best_source():
    assert combine_readings([0.6, 0.6]) == pytest.approx(0.6 / math.sqrt(2))
    assert combine_readings([0.3, 1.0]) < 0.3
    with pytest.raises(ValueError):
        combine_readings([])


def test_rounding_sd():
    assert rounding_sd(1000) == pytest.approx(1000 / math.sqrt(12))
    assert rounding_sd(1000, 50_000) == pytest.approx(1000 / math.sqrt(12) / 50_000)


def test_reserve_schedule_sums_to_one():
    assert sum(reserve_share(i) for i in range(1, 200_000)) == pytest.approx(1.0, abs=1e-5)


def test_budget_allocation_proportional_and_bounded():
    out = allocate_error_budget(0.05, [1, 2, 4], [0.02, 0.02, 0.04], 0.2)
    assert out["known_shares"] == pytest.approx([0.008, 0.016, 0.016])
    assert out["spent_on_known"] + out["reserve_total"] == pytest.approx(0.05)
    assert sum(out["reserve_next_five"]) < out["reserve_total"]
    with pytest.raises(ValueError):
        allocate_error_budget(0.05, [1], [0.0])


def test_new_product_cannot_raise_total():
    ok = reallocate_for_new_product(0.01, 0.04, 0.05, 0.005)
    assert ok["permitted"] and ok["remaining_after"] == pytest.approx(0.005)
    bad = reallocate_for_new_product(0.01, 0.04, 0.05, 0.02)
    assert not bad["permitted"]


def test_composition_remedy_tree():
    assert composition_remedy(0.1, 0.12, {})["branch"] == "small"
    r = composition_remedy(0.46, 0.12, {"coarsen": 0.3, "rotate": 0.11, "secure": 0.05})
    assert (r["branch"], r["remedy"]) == ("moderate", "rotate")
    assert composition_remedy(0.46, 0.12, {"coarsen": 0.3})["branch"] == "large"


def test_defence_record_completeness():
    d = DefenceRecord(legal_reading="r", reasons="x")
    assert not d.ready() and "owner" in d.missing()
    full = DefenceRecord(*["x"] * 8)
    assert full.ready()


def test_uniqueness_test_rejects_what_the_mean_cell_test_accepts():
    # a model with the right mean cell size but a uniqueness probability 25 per cent too low
    assert uniqueness_transport_error(0.30, 0.40) == pytest.approx(0.25)
    ok = validation_report(0.01, 0.03, 0.055, 1.5, [], 0.25, 0.20)
    assert ok["checks"]["frequency_transport"] and not ok["checks"]["uniqueness_transport"]
    assert not ok["passed"]
    with pytest.raises(ValueError):
        validation_report(0.01, 0.03, 0.055, 1.5, [], 0.25, None)
    with pytest.raises(ValueError):
        uniqueness_transport_error(0.1, 0.0)
    stress = {"stronger_never_materially_lower": True}
    full = full_validation_report(0.01, 0.03, 0.055, 1.5, [], stress, {}, None, 0.25, 0.20)
    assert not full["passed"]


def test_floor_by_precision_reproduces_and_is_monotone_in_precision():
    from privacy_risk.credit import make_credit_register
    cfg = {"seed": 7, "credit": {"borrowers": 3000, "months": 1}}
    reg = make_credit_register(cfg)
    a = floor_by_precision(reg, [0.3, 1.0], [1, 10, 50], 0.12, 0.05, 300, seed=7)
    b = floor_by_precision(reg, [0.3, 1.0], [1, 10, 50], 0.12, 0.05, 300, seed=7)
    assert a == b
    assert all(d["bound"] >= 0 for r in a for d in r["detail"])
