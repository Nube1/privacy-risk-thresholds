import json
from pathlib import Path

import numpy as np
import pytest

from privacy_risk.synthetic import (
    Mechanism,
    apply_mechanism,
    identity_release,
    make_population,
    make_register,
    pack_key,
    sample_population,
    suppress_small_cells,
)

CONFIG = {"seed": 20260904, "population_size": 4000, "sample_fraction": 0.1}


def test_register_is_deterministic_and_sampled():
    first, second = make_register(CONFIG), make_register(CONFIG)
    assert first.size == 4000 == second.size
    assert np.array_equal(first.sample_index, second.sample_index)
    assert 0 < first.sample_index.size < first.size


def test_pack_key_is_injective_on_its_range():
    keys = pack_key(np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([2, 2, 2]),
                    np.array([3, 3, 3]), bands=14, groups=12)
    assert len(set(keys.tolist())) == 3
    with pytest.raises(ValueError):
        pack_key(np.array([0]), np.array([0]), np.array([0]), np.array([0]), 0, 12)


def test_coarsening_and_suppression_shrink_the_release():
    register = make_register(CONFIG)
    fine = identity_release(register)
    coarse = apply_mechanism(register, Mechanism(20, 3, 1))
    assert np.unique(coarse.key).size < np.unique(fine.key).size
    assert coarse.size == fine.size
    suppressed = apply_mechanism(register, Mechanism(20, 3, 5))
    assert suppressed.size < coarse.size
    _, counts = np.unique(suppressed.key, return_counts=True)
    assert counts.min() >= 5
    assert suppress_small_cells(coarse, 1) is coarse


def test_sample_fraction_is_checked():
    with pytest.raises(ValueError):
        make_register({**CONFIG, "sample_fraction": 0})


def test_earlier_public_interface_still_works(tmp_path: Path):
    frame = make_population(500, 7)
    assert {"unit", "key", "outcome"} <= set(frame.columns)
    drawn = sample_population(frame, 0.5, 7)
    assert len(drawn) == 250
    with pytest.raises(ValueError):
        sample_population(frame, 0, 7)
    (tmp_path / "frame.json").write_text(json.dumps({"rows": len(frame)}))
