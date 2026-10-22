import numpy as np

from privacy_risk.frontier import choose_max_utility, nondominated


def test_frontier_and_choice():
    assert nondominated([.2, .1, .3], [.8, .7, .6]).tolist() == [True, True, False]
    assert choose_max_utility(np.array([0, 1]), [.2, .1], [.9, .7], .15) == 1
    assert choose_max_utility(np.array([0]), [.2], [.9], .1) is None



def test_value_curve_fit_and_concavity_diagnostic():
    import numpy as np
    import pytest

    from privacy_risk.frontier import concavity_diagnostic, fit_value_curve, value_curve
    risk = np.linspace(0.1, 0.5, 12)
    utility = value_curve(risk, -0.05, -0.001, -1.5)
    fit = fit_value_curve(risk, utility)
    assert fit["r_squared"] > 0.99 and fit["gamma"] < 0
    assert concavity_diagnostic(risk, utility)["is_concave"]
    jagged = concavity_diagnostic([0.1, 0.2, 0.3], [-0.5, -0.45, -0.1])
    assert not jagged["is_concave"]
    with pytest.raises(ValueError):
        fit_value_curve([0.1, 0.2], [-0.1, -0.2])
