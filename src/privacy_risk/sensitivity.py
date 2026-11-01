"""Sensitivity tools of Chapter 21: growth rates, flip rates and the stress gate.

These tools stress-test a decision. They do not forecast, and they are not a
security control. A growth rate is a finite-time diagnostic for one trajectory,
one starting direction and one norm; it does not show that a system is chaotic.
"""
from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.stats import norm


def _rk4(f: Callable[[np.ndarray], np.ndarray], z: np.ndarray, dt: float) -> np.ndarray:
    k1 = f(z)
    k2 = f(z + 0.5 * dt * k1)
    k3 = f(z + 0.5 * dt * k2)
    k4 = f(z + dt * k3)
    return z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def lorenz(z: np.ndarray, sigma: float = 10.0, rho: float = 28.0, beta: float = 8 / 3) -> np.ndarray:
    """Right-hand side of equation (21.3). A benchmark, not a financial model."""
    return np.array([sigma * (z[1] - z[0]), z[0] * (rho - z[2]) - z[1], z[0] * z[1] - beta * z[2]])


def lorenz_jacobian(z: np.ndarray, sigma: float = 10.0, rho: float = 28.0, beta: float = 8 / 3) -> np.ndarray:
    return np.array([[-sigma, sigma, 0.0], [rho - z[2], -1.0, -z[0]], [z[1], z[0], -beta]])


def finite_time_growth(f, jac, z0, v0, horizon: float, dt: float = 0.005,
                       renorm_every: int = 20) -> float:
    """Equation (21.2) by integrating the state and the tangent together.

    The tangent vector is renormalised every ``renorm_every`` steps and the logarithms
    of the growth are added up, as in Benettin et al. (1980).
    """
    if horizon <= 0 or dt <= 0:
        raise ValueError("horizon and dt must be positive")
    z = np.asarray(z0, float).copy()
    v = np.asarray(v0, float).copy()
    nv = np.linalg.norm(v)
    if nv == 0:
        raise ValueError("the starting perturbation must be nonzero")
    v /= nv
    steps = int(round(horizon / dt))
    total = 0.0

    def joint(w):
        return np.concatenate([f(w[:len(z)]), jac(w[:len(z)]) @ w[len(z):]])

    w = np.concatenate([z, v])
    for k in range(1, steps + 1):
        w = _rk4(joint, w, dt)
        if k % renorm_every == 0 or k == steps:
            g = np.linalg.norm(w[len(z):])
            total += math.log(g)
            w[len(z):] /= g
    return total / (steps * dt)


def diagnostic_horizon(rate: float, lipschitz: float, eps0: float, delta: float) -> float:
    """Equation (21.4). Infinite for a nonpositive rate; refuses delta <= L * eps0."""
    if lipschitz <= 0 or eps0 <= 0 or delta <= 0:
        raise ValueError("lipschitz, eps0 and delta must be positive")
    if delta <= lipschitz * eps0:
        raise ValueError("delta must exceed lipschitz * eps0")
    if rate <= 0:
        return math.inf
    return math.log(delta / (lipschitz * eps0)) / rate


def flip_rate_formula(b: float, tau: float, s: float) -> float:
    """Proposition 21.3: Phi(-|tau - b| / s) for a threshold decision."""
    if s <= 0:
        raise ValueError("s must be positive")
    return float(norm.cdf(-abs(tau - b) / s))


def decision_flip_rate(rule: Callable[[np.ndarray], bool], base: Sequence[float],
                       sd: Sequence[float], draws: int, seed: int = 0) -> dict:
    """Algorithm 21.1: count the share of perturbed runs whose decision differs."""
    base = np.asarray(base, float)
    sd = np.asarray(sd, float)
    if sd.shape != base.shape:
        raise ValueError("one standard deviation is needed for each input")
    if np.any(sd < 0):
        raise ValueError("standard deviations must not be negative")
    if draws < 1:
        raise ValueError("draws must be at least 1")
    ref = bool(rule(base))
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((draws, base.size))
    flips = sum(bool(rule(base + sd * row)) != ref for row in z)
    return {"reference": ref, "flips": int(flips), "draws": draws, "rate": flips / draws}


def robust_gate(tolerances: Mapping[str, float],
                bounds: Mapping[str, Mapping[str, float]]) -> dict:
    """Algorithm 21.2: pass only if the worst preregistered scenario meets every tolerance.

    ``bounds[attack][scenario]`` is an upper confidence bound. An attack with no bound
    is recorded as missing, which fails the gate. Scenarios outside the set are not covered.
    """
    if not tolerances:
        raise ValueError("no required attack")
    failures = []
    for attack, tol in tolerances.items():
        got = bounds.get(attack) or {}
        if not got:
            failures.append({"attack": attack, "reason": "missing"})
            continue
        s, b = max(got.items(), key=lambda kv: kv[1])
        if b > tol:
            failures.append({"attack": attack, "reason": "failed", "scenario": s, "bound": b})
    return {"status": "PASS" if not failures else "FAIL", "failures": failures}
