import math

import pytest

from privacy_risk.dp import basic_composition, posterior_bound, posterior_bound_table


def test_epsilon_zero_leaves_the_prior_alone():
    assert posterior_bound(0.0, 0.3) == pytest.approx(0.3)


def test_bound_grows_with_epsilon_and_with_the_prior():
    assert posterior_bound(1.0, 0.1) == pytest.approx(math.e * 0.1 / (math.e * 0.1 + 0.9))
    assert posterior_bound(2.0, 0.1) > posterior_bound(1.0, 0.1)
    assert posterior_bound(1.0, 0.3) > posterior_bound(1.0, 0.1)


def test_certain_and_impossible_priors_are_fixed_points():
    assert posterior_bound(5.0, 1.0) == 1.0
    assert posterior_bound(5.0, 0.0) == 0.0


def test_table_and_composition():
    table = posterior_bound_table([0.5, 1.0], [0.1])
    assert [row["epsilon"] for row in table] == [0.5, 1.0]
    assert basic_composition([0.5, 0.5], [1e-6, 1e-6]) == {"epsilon_total": 1.0,
                                                           "delta_total": 2e-6}
    assert basic_composition([1.0])["delta_total"] == 0.0


def test_refuses_impossible_inputs():
    with pytest.raises(ValueError):
        posterior_bound(-1, 0.5)
    with pytest.raises(ValueError):
        posterior_bound(1, 1.5)
    with pytest.raises(ValueError):
        basic_composition([-1.0])


def test_attack_ceiling_and_accountants():
    import math

    from privacy_risk.dp import advanced_composition, attack_ceiling, laplace_epsilon
    assert attack_ceiling(0.0, 0.1) == pytest.approx(0.1)
    assert attack_ceiling(1.0, 0.1, 0.01) == pytest.approx(math.e * 0.1 + 0.01)
    assert attack_ceiling(10.0, 0.5) == 1.0
    assert advanced_composition(0.01, 10_000, 1e-6) < 10_000 * 0.01
    assert advanced_composition(1.0, 2, 1e-6) == 2.0
    assert laplace_epsilon(2.0, 1.0) == 0.5 and laplace_epsilon(2.0, 2.0) == 1.0
