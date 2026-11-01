"""The release gate of Chapter 18: learning proposes, independent audit authorises.

A candidate is released only if every required attack has an audit bound at or below
its tolerance. A missing bound counts as a failure, not as zero risk. A bound that
comes from anything but an independent audit (for example a planner's prediction) is
refused.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Coordinate:
    candidate: str
    attack: str
    bound: float
    source: str = "audit"


def decide_release(utilities: Mapping[str, float], tolerances: Mapping[str, float],
                   coordinates: Sequence[Coordinate]) -> dict:
    """Return RELEASE with the feasible candidate of largest utility, or WITHHOLD."""
    if not utilities:
        raise ValueError("no candidate")
    if not tolerances:
        raise ValueError("no required attack")
    for attack, tol in tolerances.items():
        if not 0 < tol <= 1:
            raise ValueError(f"tolerance for {attack!r} must lie in (0, 1]")
    bounds: dict[tuple[str, str], float] = {}
    for c in coordinates:
        if c.source != "audit":
            raise ValueError(f"bound for {c.candidate!r}/{c.attack!r} is not from an independent audit")
        if c.candidate not in utilities:
            raise ValueError(f"bound names {c.candidate!r}, which was not frozen")
        if not 0 <= c.bound <= 1:
            raise ValueError("a bound must lie in [0, 1]")
        bounds[(c.candidate, c.attack)] = c.bound
    failed = {t: [] for t in utilities}
    missing = {t: [] for t in utilities}
    for t in utilities:
        for attack, tol in tolerances.items():
            b = bounds.get((t, attack))
            if b is None:
                missing[t].append(attack)
            elif b > tol:
                failed[t].append(attack)
    feasible = [t for t in utilities if not failed[t] and not missing[t]]
    if not feasible:
        return {"status": "WITHHOLD", "candidate": None, "failed": failed, "missing": missing,
                "next_steps": ["redesign the release", "move to a more restricted access route"]}
    best = max(feasible, key=lambda t: utilities[t])
    return {"status": "RELEASE", "candidate": best, "failed": failed, "missing": missing}
