"""A finite-volume solver for advection-diffusion on a periodic square (Chapter 20).

The drift is the decaying shear of Proposition 20.2, horizontal and a function of the
row. The update is upwind advection with centred diffusion and an explicit step, and
the step must satisfy the stability bound of Proposition 20.3. The solver refuses a step
that breaks the bound and never clips.
"""
from __future__ import annotations

import math

import numpy as np

K = 2 * math.pi
BOUND = 0.8


def exact_diffusion(n: int, t: float, dispersion: float) -> np.ndarray:
    """Equation (20.8) sampled at cell centres: no drift, unit mass."""
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1 + math.exp(-dispersion * K ** 2 * t) * (0.35 * np.cos(K * X) + 0.20 * np.sin(K * Y))


def _rate(amplitude: float, dispersion: float, h: float) -> float:
    return abs(amplitude) / h + 4 * dispersion / h ** 2


def transport(n: int = 24, horizon: float = 0.2, dispersion: float = 0.03,
              amplitude: float = 0.2, smoothing: float = 0.05,
              initial: np.ndarray | None = None, steps: int | None = None) -> dict:
    """One run of Algorithm 20.1. Returns the density and its diagnostics."""
    if not isinstance(n, (int, np.integer)) or n < 8:
        raise ValueError("n must be a whole number of at least 8")
    for name, v in (("horizon", horizon), ("dispersion", dispersion),
                    ("amplitude", amplitude), ("smoothing", smoothing)):
        if not math.isfinite(v):
            raise ValueError(f"{name} must be finite")
    if horizon < 0 or dispersion < 0 or smoothing < 0:
        raise ValueError("horizon, dispersion and smoothing must not be negative")
    c = exact_diffusion(n, 0.0, dispersion) if initial is None else np.array(initial, float)
    if c.shape != (n, n) or not np.all(np.isfinite(c)) or np.any(c < 0):
        raise ValueError("the initial density must be n x n, finite and nonnegative")
    h = 1.0 / n
    R = _rate(amplitude, dispersion, h)
    if steps is None:
        steps = max(1, math.ceil(horizon * R / BOUND))
    elif steps < 1:
        raise ValueError("steps must be at least 1")
    dt = horizon / steps
    stability = dt * R
    if stability > BOUND + 1e-12:
        raise ValueError(f"step violates the stability bound: dt*R = {stability:.3f} > {BOUND}")
    y = (np.arange(n) + 0.5) / n
    mass0 = c.mean()
    for m in range(steps):
        u = amplitude * math.exp(-smoothing * K ** 2 * m * dt) * np.sin(K * y)   # one value per row j
        up, um = np.maximum(u, 0.0)[None, :], np.minimum(u, 0.0)[None, :]
        flux = up * c + um * np.roll(c, -1, axis=0)           # right face of cell (i, j)
        c = (c - dt / h * (flux - np.roll(flux, 1, axis=0))
             + dispersion * dt / h ** 2 * (np.roll(c, -1, 0) + np.roll(c, 1, 0)
                                           + np.roll(c, -1, 1) + np.roll(c, 1, 1) - 4 * c))
    return {"density": c, "mass_error": float(c.mean() - mass0), "minimum": float(c.min()),
            "steps": int(steps), "dt": dt, "stability": stability}


def risk_from_density(c: np.ndarray, response=None) -> float:
    """Equation (20.3) on a grid: the density-weighted mean of an attack response.

    Without a fitted response a declared toy response is used, the density's own
    normalised value, which says nothing about a real attack.
    """
    c = np.asarray(c, float)
    g = np.clip(c / c.max(), 0, 1) if response is None else np.asarray(response, float)
    return float((g * c).sum() / c.sum())
