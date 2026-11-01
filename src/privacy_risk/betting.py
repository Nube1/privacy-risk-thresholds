"""A betting audit with predictable bets (Chapter 6, Example 18.4).

Wealth starts at one. Before each attack the bet is the forecast mean of a Beta
prior updated on the outcomes so far, clipped below the tolerance. The audit
certifies when wealth reaches 1/alpha. Ville's inequality bounds the chance of a
false certificate at a risk equal to the tolerance by alpha.
"""
from __future__ import annotations

from typing import Iterable


def _margin(tolerance: float, margin: float | None) -> float:
    m = tolerance / 10 if margin is None else margin
    if not 0 < tolerance < 1 or not 0 < m < tolerance / 2:
        raise ValueError("need 0 < tolerance < 1 and 0 < margin < tolerance / 2")
    return m


def predictable_bets(outcomes: Iterable[int], tolerance: float, a: float = 1.0,
                     b: float = 9.0, margin: float | None = None):
    """Yield the bet placed before each outcome, using only earlier outcomes (Algorithm 22.1).

    The bet is min(tau - m, max(m, posterior mean)) for a Beta(a, b) prior. The outcome of
    attack k is never read before bet k is yielded.
    """
    m = _margin(tolerance, margin)
    successes = 0
    for i, x in enumerate(outcomes):
        yield min(tolerance - m, max(m, (a + successes) / (a + b + i)))
        successes += int(x)


def log_wealth(outcomes: Iterable[int], bets: Iterable[float], tolerance: float) -> list[float]:
    """Running log wealth, the logarithm of equation (6.1)."""
    import math
    outcomes, bets = list(outcomes), list(bets)
    if len(outcomes) != len(bets):
        raise ValueError("one bet per outcome")
    out, total = [], 0.0
    for x, q in zip(outcomes, bets):
        if x not in (0, 1):
            raise ValueError("outcomes must be 0 or 1")
        if not 0 < q < tolerance:
            raise ValueError("every bet must lie strictly between 0 and the tolerance")
        total += math.log(q / tolerance) if x else math.log((1 - q) / (1 - tolerance))
        out.append(total)
    return out


def run_audit(outcomes: Iterable[int], tolerance: float = 0.12, alpha: float = 0.05,
              max_attacks: int = 1500, a: float = 1.0, b: float = 9.0,
              margin: float | None = None) -> dict:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    m = _margin(tolerance, margin)
    wealth, successes, n = 1.0, 0, 0
    for x in outcomes:
        if n >= max_attacks:
            break
        if x not in (0, 1):
            raise ValueError("outcomes must be 0 or 1")
        bet = min(tolerance - m, max(m, (a + successes) / (a + b + n)))
        wealth *= bet / tolerance if x else (1 - bet) / (1 - tolerance)
        successes += int(x)
        n += 1
        if wealth >= 1 / alpha:
            return {"certified": True, "attacks": n, "successes": successes, "wealth": wealth}
    return {"certified": False, "attacks": n, "successes": successes, "wealth": wealth}
