import numpy as np
import pytest
from privacy_risk.multiscale import (MultiscaleFeatures, network_jacobian,
                                    proposal_sensitivity, shifted_audit, zero_success_plan)
from privacy_risk.sentry_r import DeepRiskNet


def test_raw_chain_rule_and_bound():
    f = MultiscaleFeatures([-2, 0, 1], [2, 3, 5], 3)
    net = DeepRiskNet(f.output_dimension, 2, width=9, depth=3)
    x = np.array([[.17, 1.23, 2.19], [.89, .67, 3.42]])
    result = proposal_sensitivity(net, f, x)
    for j in range(3):
        delta = np.zeros_like(x)
        delta[:, j] = 1e-6
        fd = (net.predict(f.transform(x+delta))-net.predict(f.transform(x-delta)))/2e-6
        np.testing.assert_allclose(result['jacobian'][:, :, j], fd, atol=2e-8)
    assert max(np.linalg.norm(j, 2) for j in result['jacobian']) <= result['model_lipschitz_bound']
    with pytest.raises(ValueError):
        f.transform([[3, 1, 2]])


def test_clipped_predictor_has_zero_derivative():
    net = DeepRiskNet(2, 1, depth=0)
    net.biases[-1][:] = 100
    assert np.all(network_jacobian(net, np.ones((2, 2))) == 0)


def test_exact_zero_formula_and_shift_monotonicity():
    n = np.array([120, 80])
    base = shifted_audit([0, 0], n, [.1, .2], [0, 0])
    np.testing.assert_allclose(base['audit_upper'], -np.expm1(np.log(.025)/n))
    shifted = shifted_audit([0, 0], n, [.1, .2], [.08, .01])
    assert not shifted['eligible']
    assert np.all(shifted['target_upper'] >= base['target_upper'])
    full = shifted_audit([120, 80], n, [.1, .2], [0, 0])
    assert np.all(full['audit_upper'] == 1)


def test_plan_passes_and_minimum_integer_counts():
    tau, rho = np.array([.08, .12]), np.array([.02, .01])
    p = zero_success_plan(tau, rho, costs=[1, 3])
    n, w = p['trials_if_zero_successes'], p['weights']
    assert shifted_audit([0, 0], n, tau, rho, weights=w)['eligible']
    assert np.all(-np.expm1(np.log(.05*w)/(n-1))+rho > tau)
    for _ in range(30):
        other = np.random.default_rng(_).dirichlet([1, 1])
        cost = np.sum(np.array([1, 3])/-np.log1p(-(tau-rho))*-np.log(.05*other))
        assert cost >= p['continuous_cost']-1e-10


@pytest.mark.parametrize('kwargs', [dict(radii=[.2]), dict(weights=[1.1]),
                                   dict(successes=[.5]), dict(alpha=float('nan'))])
def test_refuses_invalid(kwargs):
    args = dict(successes=[0], trials=[100], tolerances=[.12], radii=[0], alpha=.05)
    args.update(kwargs)
    if kwargs == dict(radii=[.2]):
        assert not shifted_audit(**args)['eligible']
    else:
        with pytest.raises(ValueError):
            shifted_audit(**args)
