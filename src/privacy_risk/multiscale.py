"""Explicit proposal sensitivities and fixed-family distribution-shift audit.

Predictions and their Lipschitz bounds describe the fitted model, not the true
attack law. Only independent, fixed-size binomial audits enter the audit gate.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta

from .sentry_r import DeepRiskNet


def _matrix(x, dimension):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != dimension or not np.isfinite(x).all():
        raise ValueError("finite sample-by-feature matrix required")
    return x


class MultiscaleFeatures:
    """Identity plus coordinatewise Mexican-hat features on a declared box.

    This is a finite wavelet dictionary, not an orthogonal wavelet transform or
    an asserted interpolation basis. Dimension grows linearly in input size.
    """

    def __init__(self, lower, upper, levels=2):
        self.lower = np.asarray(lower, float).copy()
        self.upper = np.asarray(upper, float).copy()
        if (self.lower.ndim != 1 or not self.lower.size
                or self.upper.shape != self.lower.shape
                or not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all()
                or np.any(self.upper <= self.lower)
                or type(levels) is not int or not 1 <= levels <= 8):
            raise ValueError("finite ordered bounds and 1..8 integer levels required")
        self.levels = levels
        self.dimension = self.lower.size
        self.atoms = [(2.**j, (k + .5)/2**j)
                      for j in range(levels) for k in range(2**j)]
        self.output_dimension = self.dimension * (1 + len(self.atoms))

    def _normalise(self, x):
        x = _matrix(x, self.dimension)
        if np.any(x < self.lower) or np.any(x > self.upper):
            raise ValueError("input outside the registered feature box")
        return (x - self.lower)/(self.upper - self.lower)

    def transform(self, x):
        z = self._normalise(x)
        blocks = [z]
        for scale, centre in self.atoms:
            u = scale*(z - centre)
            blocks.append((1 - u*u)*np.exp(-u*u/2))
        return np.concatenate(blocks, axis=1)

    def jacobian(self, x):
        z = self._normalise(x)
        inv = 1/(self.upper - self.lower)
        blocks = [np.broadcast_to(inv, z.shape)]
        for scale, centre in self.atoms:
            u = scale*(z - centre)
            blocks.append(scale*(u**3 - 3*u)*np.exp(-u*u/2)*inv)
        diagonal = np.eye(self.dimension)
        return np.concatenate([a[:, :, None]*diagonal for a in blocks], axis=1)

    def lipschitz_bound(self):
        # |(u^3-3u)exp(-u^2/2)| <= 3sqrt(3)exp(-3/2)+3exp(-1/2).
        c = 3*np.sqrt(3)*np.exp(-1.5) + 3*np.exp(-.5)
        return float(np.max(1/(self.upper-self.lower))
                     * np.sqrt(1 + c*c*sum(s*s for s, _ in self.atoms)))


def network_jacobian(net: DeepRiskNet, x):
    """Derivative of the actual clipped predictor, zero convention at ReLU kinks."""
    x = _matrix(x, net.dimension)
    p, acts, preacts = net.forward(x)
    derivative = np.broadcast_to(np.eye(net.dimension),
                                 (len(x), net.dimension, net.dimension)).copy()
    for w, z in zip(net.weights[:-1], preacts):
        derivative = np.einsum('bhd,hk->bkd', derivative, w)
        derivative *= (z > 0)[:, :, None]
    derivative = np.einsum('bhd,hk->bkd', derivative, net.weights[-1])
    logits = acts[-1] @ net.weights[-1] + net.biases[-1]
    derivative *= (p*(1-p)* (np.abs(logits) < 35))[:, :, None]
    return derivative


def proposal_sensitivity(net: DeepRiskNet, features: MultiscaleFeatures, x):
    """Prediction, raw-unit Jacobian, and a global model-only Lipschitz bound."""
    z = features.transform(x)
    jac = np.einsum('bkh,bhd->bkd', network_jacobian(net, z), features.jacobian(x))
    bound = .25*features.lipschitz_bound()
    for w in net.weights:
        bound *= np.linalg.norm(w, 2)
    return {"prediction": net.predict(z), "jacobian": jac,
            "model_lipschitz_bound": float(bound)}


def shifted_audit(successes, trials, tolerances, radii, *, alpha=.05, weights=None):
    """CP union-bound certificate under EXTERNALLY justified TV radii.

    Fixed declared family, fixed sample sizes and independent Bernoulli draws
    within each coordinate are required. Dependence between coordinates is OK.
    A radius rho bounds total variation between target and audit attack laws.
    Counts do not establish that radius. Do not repeatedly inspect this gate.
    """
    k, n = np.asarray(successes), np.asarray(trials)
    tau, rho = np.asarray(tolerances, float), np.asarray(radii, float)
    if (k.ndim != 1 or not k.size or n.shape != k.shape or tau.shape != k.shape
            or rho.shape != k.shape or not np.issubdtype(k.dtype, np.integer)
            or not np.issubdtype(n.dtype, np.integer) or np.any(n < 1)
            or np.any(k < 0) or np.any(k > n)
            or not np.isfinite(tau).all() or not np.isfinite(rho).all()
            or np.any((tau <= 0) | (tau >= 1)) or np.any((rho < 0) | (rho > 1))
            or not np.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError("invalid audit inputs")
    w = np.full(k.size, 1/k.size) if weights is None else np.asarray(weights, float)
    if (w.shape != k.shape or not np.isfinite(w).all() or np.any(w <= 0)
            or float(w.sum()) > 1):
        raise ValueError("positive predeclared weights must sum to at most one")
    upper = np.ones(k.size)
    partial = k < n
    upper[partial] = beta.isf(alpha*w[partial], k[partial]+1, n[partial]-k[partial])
    target_upper = np.minimum(1., upper+rho)
    position = 1-target_upper/tau
    return {"audit_upper": upper, "target_upper": target_upper,
            "trust_position": position, "eligible": bool(np.all(position >= 0)),
            "alpha": float(alpha), "weights": w,
            "assumption": "external total-variation bound; fixed-size Bernoulli audits"}


def zero_success_plan(tolerances, radii, *, alpha=.05, costs=None):
    """Continuous cost-optimal weights, then conservative integer budgets.

    Minimises sum c_i log(1/(alpha*w_i))/[-log(1-tau_i+rho_i)] before rounding.
    Does not promise passage when attacks succeed; freeze before final audits.
    """
    tau, rho = np.asarray(tolerances, float), np.asarray(radii, float)
    if (tau.ndim != 1 or not tau.size or rho.shape != tau.shape
            or not np.isfinite(tau).all() or not np.isfinite(rho).all()
            or np.any((tau <= 0) | (tau >= 1)) or np.any(rho < 0)
            or np.any(rho >= tau) or not np.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError("require 0 <= radius < tolerance < 1")
    c = np.ones(tau.size) if costs is None else np.asarray(costs, float)
    if c.shape != tau.shape or not np.isfinite(c).all() or np.any(c <= 0):
        raise ValueError("positive finite costs required")
    drift = -np.log1p(-(tau-rho))
    a = c/drift
    weights = a/a.sum()
    if weights.sum() > 1:
        weights /= np.nextafter(weights.sum(), np.inf)
    budgets = np.ceil(-np.log(alpha*weights)/drift).astype(np.int64)
    for i in range(len(budgets)):
        while -np.expm1(np.log(alpha*weights[i])/budgets[i])+rho[i] > tau[i]:
            budgets[i] += 1
    return {"weights": weights, "trials_if_zero_successes": budgets,
            "continuous_cost": float(np.sum(a*-np.log(alpha*weights))),
            "rounded_cost": float(c@budgets)}
