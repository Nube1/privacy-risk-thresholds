"""Penetration-test findings as declared adversary knowledge, audited with fresh attacks.

A penetration test of a dissemination system reports software findings: an
access-control bypass, a credential leak on redirect, an injection flaw. Each
one, if exploited, changes what the adversary can see, and so changes the
disclosure risk of a release that was reviewed on the assumption that the
adversary sees only the published table. This module makes that change
explicit and measurable.

The method has four steps.

1. Declare. Each finding names the capabilities it grants, from a closed list
   whose members map onto parameters of the declared attack: a suppression
   bypass lets the adversary read cells below the floor; a query-log or
   credential leak sharpens the adversary's auxiliary reading; microdata access
   removes the release mechanism altogether.
2. Compose. Capabilities only add. The adversary conditioned on a set of
   findings is the release setting with every granted relaxation applied, so
   the view is monotone: more findings never make the adversary weaker.
3. Measure. The same executable attacker that audits the release runs on the
   relaxed view with fresh targets. Nothing is predicted from a severity score.
4. Certify. Each finding-conditioned adversary gets an exact one-sided upper
   bound on its success probability at a share of the programme's error
   budget, so the verdict is of the same kind as the release certificate: with
   this finding exploited, is the bound still below the tolerance? The pooled
   adversary, with every confirmed finding at once, is its own coordinate,
   because marginal risks do not bound a pooled attack.

Every step is written to a hash-chained ledger so the plan is fixed before the
outcomes exist. What this cannot do is also stated: it measures the declared
attack on the relaxed view, not whether the flaw is exploitable in the
deployment, which is the penetration tester's finding, not ours.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.stats import beta

from .adaptive_audit import AuditLedger
from .injector_search import LEVELS, Scenario, execute

SEVERITIES = ('low', 'medium', 'high', 'critical')
STATUSES = ('confirmed', 'theoretical', 'fixed')

# Capability -> how the adversary's view of the release changes. Each entry is
# a relaxation of the release action (puma_width, education_width, minimum_cell,
# covariate_sd); None leaves a component as published.
CAPABILITIES = {
    'read_suppressed_cells': dict(minimum_cell=1),
    'read_fine_geography': dict(puma_width=min(LEVELS[0])),
    'read_fine_education': dict(education_width=min(LEVELS[1])),
    'sharpen_auxiliary': dict(covariate_sd_factor=.5),
    'read_microdata': dict(puma_width=min(LEVELS[0]), education_width=min(LEVELS[1]),
                           minimum_cell=1, covariate_sd=min(LEVELS[3])),
}


@dataclass(frozen=True)
class Finding:
    """One penetration-test finding and the capabilities it grants."""
    identifier: str
    title: str
    component: str
    severity: str
    status: str
    grants: tuple[str, ...]
    reference: str = ''

    def __post_init__(self):
        if self.severity not in SEVERITIES:
            raise ValueError(f'severity must be one of {SEVERITIES}')
        if self.status not in STATUSES:
            raise ValueError(f'status must be one of {STATUSES}')
        unknown = [g for g in self.grants if g not in CAPABILITIES]
        if unknown:
            raise ValueError(f'unknown capabilities {unknown}; declare them in CAPABILITIES')
        if not self.identifier or not self.title:
            raise ValueError('a finding needs an identifier and a title')


@dataclass(frozen=True)
class Release:
    """The published setting of one release, as the review approved it."""
    puma_width: int
    education_width: int
    minimum_cell: int
    covariate_sd: float

    @property
    def action(self) -> tuple:
        return (self.puma_width, self.education_width, self.minimum_cell, self.covariate_sd)


def adversary_view(release: Release, findings: list[Finding]) -> Release:
    """The release as seen by an adversary who exploits every listed finding.

    Relaxations only ever widen the view: a smaller floor, a finer geography or
    education code, a sharper reading. Applying a capability twice changes
    nothing, and the order of findings does not matter.
    """
    view = release
    for finding in findings:
        for capability in finding.grants:
            change = CAPABILITIES[capability]
            if 'minimum_cell' in change:
                view = replace(view, minimum_cell=min(view.minimum_cell, change['minimum_cell']))
            if 'puma_width' in change:
                view = replace(view, puma_width=min(view.puma_width, change['puma_width']))
            if 'education_width' in change:
                view = replace(view, education_width=min(view.education_width,
                                                         change['education_width']))
            if 'covariate_sd' in change:
                view = replace(view, covariate_sd=min(view.covariate_sd, change['covariate_sd']))
            if 'covariate_sd_factor' in change:
                sharper = max(min(LEVELS[3]), view.covariate_sd*change['covariate_sd_factor'])
                view = replace(view, covariate_sd=min(view.covariate_sd, sharper))
    return view


def upper_bound(successes: int, trials: int, alpha: float) -> float:
    """Exact one-sided upper confidence bound for a binomial proportion."""
    if successes >= trials:
        return 1.
    return float(beta.ppf(1-alpha, successes+1, trials-successes))


@dataclass
class PenTestPlan:
    """A frozen plan: which adversaries are audited, at what budget, with how many trials."""
    programme: str
    release: Release
    findings: list[Finding]
    tolerance: float
    alpha: float
    trials: int
    seed: int
    coordinates: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if not 0 < self.tolerance < 1 or not 0 < self.alpha < 1 or self.trials < 1:
            raise ValueError('tolerance and alpha in (0,1), positive trials')
        confirmed = [f for f in self.findings if f.status == 'confirmed']
        ids = [f.identifier for f in self.findings]
        if len(set(ids)) != len(ids):
            raise ValueError('finding identifiers must be distinct')
        coordinates = [dict(name='published', findings=[], view=self.release.action)]
        for finding in confirmed:
            coordinates.append(dict(name=finding.identifier, findings=[finding.identifier],
                                    view=adversary_view(self.release, [finding]).action))
        if len(confirmed) > 1:
            coordinates.append(dict(name='pooled', findings=[f.identifier for f in confirmed],
                                    view=adversary_view(self.release, confirmed).action))
        share = self.alpha/len(coordinates)
        for c in coordinates:
            c['alpha_share'] = share
            c['trials'] = self.trials
        self.coordinates = coordinates

    def as_dict(self) -> dict:
        return dict(programme=self.programme, release=self.release.action,
                    findings=[f.__dict__ for f in self.findings], tolerance=self.tolerance,
                    alpha=self.alpha, trials=self.trials, seed=self.seed,
                    coordinates=self.coordinates)


def run(plan: PenTestPlan, build_view, ledger: AuditLedger | None = None) -> dict:
    """Audit every coordinate of the plan with fresh targets.

    ``build_view(action) -> Scenario`` constructs the attackable release for a
    view; it is supplied by the caller because it depends on the data product.
    The plan is logged before any outcome, each outcome after it is measured.
    """
    ledger = ledger if ledger is not None else AuditLedger()
    ledger.append('pentest_plan', plan.as_dict())
    rows = []
    for index, coordinate in enumerate(plan.coordinates):
        scenario: Scenario = build_view(tuple(coordinate['view']))
        rng = np.random.default_rng(np.random.SeedSequence([plan.seed, index]))
        successes = execute(scenario, coordinate['trials'], rng)
        bound = upper_bound(successes, coordinate['trials'], coordinate['alpha_share'])
        row = dict(coordinate, successes=int(successes), rate=successes/coordinate['trials'],
                   upper_bound=bound, within_tolerance=bool(bound <= plan.tolerance))
        ledger.append('pentest_outcome', dict(name=row['name'], successes=row['successes'],
                                              trials=row['trials'], upper_bound=bound,
                                              within_tolerance=row['within_tolerance']))
        rows.append(row)
    published = rows[0]
    verdict = dict(
        release_within_tolerance_as_published=published['within_tolerance'],
        findings_that_breach=[r['name'] for r in rows[1:]
                              if not r['within_tolerance'] and r['name'] != 'pooled'],
        pooled_breaches=any(r['name'] == 'pooled' and not r['within_tolerance'] for r in rows),
        scope='the declared attack on the relaxed view; exploitability is the tester\'s finding')
    report = dict(plan=plan.as_dict(), coordinates=rows, verdict=verdict,
                  ledger_head=ledger.head, ledger_valid=AuditLedger.verify(ledger.entries))
    ledger.append('pentest_verdict', verdict)
    report['ledger_head'] = ledger.head
    return report


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


def latex_table(report: dict, label: str = 'tab:pentest') -> str:
    """One row per audited adversary, for the paper."""
    plan = report['plan']
    lines = [r'\begin{table}[htbp]', r'\centering\small',
             r'\caption{Penetration-test findings audited as adversary knowledge. Each row is '
             'the declared linkage attack run with fresh targets on the release as the named '
             f"adversary sees it; {plan['trials']} trials per row, tolerance {plan['tolerance']}, "
             f"error budget {plan['alpha']} split equally over the rows. The upper bound is the "
             'exact one-sided binomial bound.}',
             f'\\label{{{label}}}', r'\begin{tabular}{llrrrl}', r'\toprule',
             r'Adversary & View (PUMA, educ., floor, sd) & Successes & Rate & Upper bound & Within \\',
             r'\midrule']
    for row in report['coordinates']:
        p, e, m, s = row['view']
        lines.append(f"{row['name'].replace('_', chr(92)+'_')} & ({p}, {e}, {m}, {s:g}) & "
                     f"{row['successes']} & {row['rate']:.3f} & {row['upper_bound']:.3f} & "
                     f"{'yes' if row['within_tolerance'] else 'no'} " + chr(92)*2)
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    return '\n'.join(lines)+'\n'
