"""Release programmes: marginal bounds, the union event and the pooled attack."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .synthetic import Release


def marginal_bounds(risks: np.ndarray) -> tuple[float, float]:
    risks = np.asarray(risks, dtype=float)
    if risks.size == 0 or np.any((risks < 0) | (risks > 1)):
        raise ValueError("risks must be a nonempty vector in [0, 1]")
    return float(risks.max()), float(min(1.0, risks.sum()))


def independent_union(risks: np.ndarray) -> float:
    risks = np.asarray(risks, dtype=float)
    return float(1 - np.prod(1 - risks))


def empirical_joint_success(success_matrix: np.ndarray) -> float:
    success_matrix = np.asarray(success_matrix, dtype=bool)
    if success_matrix.ndim != 2:
        raise ValueError("expected attacks x releases matrix")
    return float(success_matrix.any(axis=1).mean())


def candidate_sets(release: Release) -> dict[int, np.ndarray]:
    """For each released record, the records sharing its released key."""
    key = release.key
    members: dict[int, np.ndarray] = {}
    for cell in np.unique(key).tolist():
        group = release.index[key == cell]
        for record in group.tolist():
            members[int(record)] = group
    return members


@dataclass
class PooledResult:
    marginal: list[float]
    coverage: list[float]
    pooled: list[float]
    unique_share: list[float]
    marginal_unique_share: list[float]
    targets: int

    def bounds(self) -> tuple[float, float]:
        return marginal_bounds(np.asarray(self.marginal))

    def as_dict(self) -> dict:
        lower, upper = self.bounds()
        return {"targets": self.targets, "marginal": self.marginal,
                "coverage": self.coverage, "pooled": self.pooled,
                "unique_share": self.unique_share,
                "marginal_unique_share": self.marginal_unique_share,
                "marginal_lower": lower, "union_upper": upper,
                "independence": independent_union(np.asarray(self.marginal))}


def candidate_sets_from_keys(index: np.ndarray, key: np.ndarray) -> dict[int, np.ndarray]:
    """Candidate sets for a release given as an index array and a key array."""
    index, key = np.asarray(index), np.asarray(key)
    members: dict[int, np.ndarray] = {}
    for cell in np.unique(key).tolist():
        group = index[key == cell]
        for record in group.tolist():
            members[int(record)] = group
    return members


def pooled_attack_from_sets(sets: list[dict[int, np.ndarray]],
                            targets: np.ndarray) -> PooledResult:
    """The pooled attack given candidate sets, so any release format can use it."""
    if not sets:
        raise ValueError("no releases supplied")
    targets = np.asarray(targets)
    marginal = np.zeros((targets.size, len(sets)))
    covered = np.zeros(len(sets))
    pooled = np.zeros((targets.size, len(sets)))
    unique = np.zeros(len(sets))
    unique_alone = np.zeros(len(sets))
    for row, target in enumerate(targets.tolist()):
        intersection: set[int] | None = None
        for column, membership in enumerate(sets):
            group = membership.get(int(target))
            if group is not None:
                covered[column] += 1
                marginal[row, column] = 1.0 / group.size
                if group.size == 1:
                    unique_alone[column] += 1
                current = set(group.tolist())
                intersection = current if intersection is None else intersection & current
            if intersection:
                pooled[row, column] = 1.0 / len(intersection)
                if len(intersection) == 1:
                    unique[column] += 1
    return PooledResult(marginal=marginal.mean(axis=0).tolist(),
                        coverage=(covered / targets.size).tolist(),
                        pooled=pooled.mean(axis=0).tolist(),
                        unique_share=(unique / targets.size).tolist(),
                        marginal_unique_share=(unique_alone / targets.size).tolist(),
                        targets=int(targets.size))


def pooled_attack(releases: list[Release], targets: np.ndarray) -> PooledResult:
    """Risk of an adversary that intersects the candidate sets of releases 1 to m.

    Marginal risks are unconditional: a target suppressed from a release contributes
    zero to that release and does not narrow the intersection. The pooled figure is
    therefore what an adversary achieves with the releases that contain the target,
    and it is not bounded by the marginals.
    """
    if not releases:
        raise ValueError("no releases supplied")
    return pooled_attack_from_sets([candidate_sets(release) for release in releases],
                                   targets)

