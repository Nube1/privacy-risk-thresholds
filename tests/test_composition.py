import numpy as np
import pytest

from privacy_risk.composition import empirical_joint_success, independent_union, marginal_bounds


def test_composition_bounds_and_independence():
    assert marginal_bounds([.1, .2, .3]) == pytest.approx((.3, .6))
    assert independent_union([.1, .2]) == pytest.approx(.28)
    matrix = np.array([[1, 0], [0, 0], [0, 1], [0, 0]])
    assert empirical_joint_success(matrix) == .5



def test_pooled_attack_intersects_candidate_sets():
    from privacy_risk.composition import candidate_sets, pooled_attack
    from privacy_risk.synthetic import Release
    # nine records labelled by a pair of coordinates in {0, 1, 2}. Release one keys on
    # the second coordinate, release two on the first, so each release is 3-anonymous
    # while the intersection of the two candidate sets is a singleton for every record
    index = np.arange(9)
    first_coordinate = index // 3
    second_coordinate = index % 3
    zeros = np.zeros(9, dtype=int)

    def release(key: np.ndarray) -> Release:
        return Release(index, zeros, zeros, key, zeros, 3, 1)

    first, second = release(second_coordinate), release(first_coordinate)
    assert candidate_sets(first)[0].size == 3
    result = pooled_attack([first, second], index)
    assert result.marginal == pytest.approx([1 / 3, 1 / 3])
    assert result.pooled[1] == 1.0
    assert result.unique_share[1] == 1.0
    assert result.marginal_unique_share == [0.0, 0.0]
    lower, upper = result.bounds()
    assert lower == pytest.approx(1 / 3) and upper == pytest.approx(2 / 3)
    assert result.pooled[1] > upper


def test_pooled_attack_refuses_an_empty_programme():
    from privacy_risk.composition import pooled_attack
    with pytest.raises(ValueError):
        pooled_attack([], np.arange(2))
