"""Checks from Chapter 20 (the transport solver and the finite-prior posterior)."""
import numpy as np
import pytest
from scipy.stats import binom

from privacy_risk.bayes import grid_posterior, predictive_pmf, sensitivity_spectrum, transport_forward
from privacy_risk.transport import exact_diffusion, transport


def _single_cell():
    c0 = np.zeros((8, 8)); c0[3, 2] = 1.0
    return c0


def test_example_20_4_one_step_from_a_single_cell():
    r = transport(n=8, horizon=0.0667, dispersion=0.03, amplitude=0.2, smoothing=0.05,
                  initial=_single_cell(), steps=1)
    c = r["density"]
    assert c[3, 2] == pytest.approx(0.3891, abs=1e-4)
    assert c[4, 2] == pytest.approx(0.2267, abs=1e-4)
    assert c[2, 2] == pytest.approx(0.1281, abs=1e-4)
    assert r["stability"] == pytest.approx(0.619, abs=1e-3)
    assert c.sum() == pytest.approx(1.0)


def test_unstable_requests_are_refused():
    with pytest.raises(ValueError):
        transport(n=8, horizon=0.2, dispersion=0.03, amplitude=0.2, initial=_single_cell(), steps=1)
    with pytest.raises(ValueError):
        transport(n=32, horizon=0.2, dispersion=0.03, amplitude=0.2, steps=2)
    with pytest.raises(ValueError):
        transport(n=4)


def test_refinement_against_the_exact_diffusion_solution():
    expected = {16: (8, 1.03e-4), 32: (31, 3.23e-5), 64: (123, 8.51e-6)}
    for n, (steps, rmse) in expected.items():
        r = transport(n=n, horizon=0.2, dispersion=0.03, amplitude=0.0)
        err = np.sqrt(np.mean((r["density"] - exact_diffusion(n, 0.2, 0.03)) ** 2))
        assert r["steps"] == steps and err == pytest.approx(rmse, rel=0.01)
        assert r["mass_error"] == pytest.approx(0.0, abs=1e-12) and r["minimum"] > 0.56
        assert r["stability"] <= 0.8


def test_shear_run_conserves_mass_and_stays_positive():
    r = transport(n=32, horizon=0.2, dispersion=0.03, amplitude=0.2, smoothing=0.05)
    assert r["steps"] == 33 and r["stability"] == pytest.approx(0.784, abs=1e-3)
    assert r["mass_error"] == pytest.approx(0.0, abs=1e-12) and r["minimum"] > 0.57


def test_zero_horizon_takes_one_step():
    assert transport(n=16, horizon=0.0)["steps"] == 1


def test_posterior_weights_sum_to_one_and_survive_large_likelihood_gaps():
    pred = np.array([[0.0, 0.0], [10.0, 10.0]])
    w = grid_posterior(np.array([0.0, 0.0]), pred, 0.01)
    assert w.sum() == pytest.approx(1.0) and w[0] == pytest.approx(1.0)


def test_example_20_6_predictive_mixture_is_wider_than_the_plug_in():
    w, r, B = np.array([.5, .5]), np.array([.05, .10]), 400
    rbar = float(w @ r)
    var_r = float(w @ (r - rbar) ** 2)
    plug = B * rbar * (1 - rbar)
    assert plug == pytest.approx(27.75)
    assert plug + B * (B - 1) * var_r == pytest.approx(127.5)
    pm = predictive_pmf(w, r, B)
    assert pm.sum() == pytest.approx(1.0)
    assert binom.sf(39, B, rbar) == pytest.approx(0.04, abs=0.001)
    assert pm[40:].sum() == pytest.approx(0.26, abs=0.005)


def test_sensitivity_spectrum_shows_a_weak_amplitude_direction():
    s = sensitivity_spectrum(0.03, 0.2)
    assert s[0] == pytest.approx(7.1, abs=0.1) and s[1] == pytest.approx(0.516, abs=0.01)
    assert 13 < s[0] / s[1] < 14.5


def test_amplitude_posterior_is_about_twenty_times_wider_than_dispersion():
    grid = [(d, a) for d in np.linspace(.01, .05, 9) for a in np.linspace(.1, .3, 9)]
    pred = np.array([transport_forward(d, a) for d, a in grid])
    y = transport_forward(.03, .2) + np.random.default_rng(3).normal(0, .01, 16)
    w = grid_posterior(y, pred, .01)
    d, a = np.array(grid).T
    sd = np.sqrt(w @ (d - w @ d) ** 2); sa = np.sqrt(w @ (a - w @ a) ** 2)
    assert sa > 10 * sd
