"""A betting audit with predictable bets (Chapter 6, Example 18.4).

Wealth starts at one. Before each attack the bet is the forecast mean of a Beta
prior updated on the outcomes so far, clipped below the tolerance. The audit
certifies when wealth reaches 1/alpha. Ville's inequality bounds the chance of a
false certificate at a risk equal to the tolerance by alpha.
"""
from __future__ import annotations

from typing import Iterable


def predictable_bets(outcomes: Iterable[int], tolerance: float, a: float = 1.0,
                     b: float = 9.0, margin: float = 0.012):
    """Yield the bet placed before each outcome, using only earlier outcomes."""
    successes = 0
    for i, x in enumerate(outcomes):
        yield min((a + successes) / (a + b + i), tolerance - margin)
        successes += int(x)


def run_audit(outcomes: Iterable[int], tolerance: float = 0.12, alpha: float = 0.05,
              max_attacks: int = 1500, a: float = 1.0, b: float = 9.0,
              margin: float = 0.012) -> dict:
    wealth, successes, n = 1.0, 0, 0
    for x in outcomes:
        if n >= max_attacks:
            break
        bet = min((a + successes) / (a + b + n), tolerance - margin)
        wealth *= bet / tolerance if x else (1 - bet) / (1 - tolerance)
        successes += int(x)
        n += 1
        if wealth >= 1 / alpha:
            return {"certified": True, "attacks": n, "successes": successes, "wealth": wealth}
    return {"certified": False, "attacks": n, "successes": successes, "wealth": wealth}
