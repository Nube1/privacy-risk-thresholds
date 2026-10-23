"""Red-team harness: findings compose monotonically, the plan is frozen first, bounds are exact."""
import unittest

import numpy as np
from scipy.stats import beta

from privacy_risk.adaptive_audit import AuditLedger
from privacy_risk.injector_search import Scenario
from privacy_risk.redteam import (
    CAPABILITIES, Finding, PenTestPlan, Release, adversary_view, latex_table, run, upper_bound,
)


def finding(identifier, grants, status='confirmed'):
    return Finding(identifier, 'title', 'component', 'high', status, tuple(grants))


def synthetic_view(action, records=400, seed=3):
    """A toy release: coarser views have fewer distinct keys and noisier readings."""
    puma, education, minimum, sd = action
    rng = np.random.default_rng(seed)
    distinct = max(2, int(records/(puma*education*minimum)))
    key = rng.integers(0, distinct, records)
    income = rng.normal(0, 1, records)
    return Scenario(tuple(action), {}, key, income)


class RedTeamTests(unittest.TestCase):
    def setUp(self):
        self.release = Release(100, 8, 10, 2.)

    def test_capabilities_only_widen_the_view_and_compose_in_any_order(self):
        a = finding('A', ['read_suppressed_cells'])
        b = finding('B', ['sharpen_auxiliary'])
        self.assertEqual(adversary_view(self.release, []).action, self.release.action)
        self.assertEqual(adversary_view(self.release, [a]).action, (100, 8, 1, 2.))
        self.assertEqual(adversary_view(self.release, [b]).action, (100, 8, 10, 1.))
        self.assertEqual(adversary_view(self.release, [a, b]).action,
                         adversary_view(self.release, [b, a]).action)
        self.assertEqual(adversary_view(self.release, [a, a]).action,
                         adversary_view(self.release, [a]).action)
        # A sharper reading never goes below the finest declared level.
        fine = Release(1, 1, 1, .1)
        self.assertEqual(adversary_view(fine, [b]).action, fine.action)
        micro = finding('M', ['read_microdata'])
        self.assertEqual(adversary_view(self.release, [micro]).action, (1, 1, 1, .1))
        for name in CAPABILITIES:
            view = adversary_view(self.release, [finding('X', [name])])
            self.assertLessEqual(view.minimum_cell, self.release.minimum_cell)
            self.assertLessEqual(view.covariate_sd, self.release.covariate_sd)

    def test_finding_validation(self):
        with self.assertRaisesRegex(ValueError, 'unknown capabilities'):
            finding('A', ['read_everything'])
        with self.assertRaisesRegex(ValueError, 'severity'):
            Finding('A', 't', 'c', 'severe', 'confirmed', ())
        with self.assertRaisesRegex(ValueError, 'status'):
            Finding('A', 't', 'c', 'high', 'maybe', ())
        with self.assertRaisesRegex(ValueError, 'distinct'):
            PenTestPlan('p', self.release, [finding('A', []), finding('A', [])], .2, .05, 100, 1)

    def test_plan_audits_only_confirmed_findings_and_adds_the_pooled_adversary(self):
        findings = [finding('A', ['read_suppressed_cells']),
                    finding('T', ['read_microdata'], status='theoretical'),
                    finding('B', ['sharpen_auxiliary']),
                    finding('F', ['read_microdata'], status='fixed')]
        plan = PenTestPlan('p', self.release, findings, .2, .05, 100, 1)
        names = [c['name'] for c in plan.coordinates]
        self.assertEqual(names, ['published', 'A', 'B', 'pooled'])
        self.assertTrue(all(abs(c['alpha_share'] - .05/4) < 1e-12 for c in plan.coordinates))
        self.assertEqual(plan.coordinates[-1]['view'], (100, 8, 1, 1.))
        single = PenTestPlan('p', self.release, [findings[0]], .2, .05, 100, 1)
        self.assertEqual([c['name'] for c in single.coordinates], ['published', 'A'])

    def test_upper_bound_is_the_exact_binomial_bound(self):
        self.assertAlmostEqual(upper_bound(3, 100, .05), float(beta.ppf(.95, 4, 97)))
        self.assertEqual(upper_bound(100, 100, .05), 1.)
        self.assertGreater(upper_bound(0, 100, .05), 0.)
        self.assertLess(upper_bound(0, 100, .05), .04)

    def test_run_logs_plan_before_outcomes_and_reports_a_verdict(self):
        findings = [finding('A', ['read_suppressed_cells']), finding('B', ['sharpen_auxiliary'])]
        plan = PenTestPlan('p', self.release, findings, .2, .05, 300, 7)
        ledger = AuditLedger()
        report = run(plan, synthetic_view, ledger)
        kinds = [e['kind'] for e in ledger.entries]
        self.assertEqual(kinds[0], 'pentest_plan')
        self.assertEqual(kinds[1:-1], ['pentest_outcome']*4)
        self.assertEqual(kinds[-1], 'pentest_verdict')
        self.assertTrue(AuditLedger.verify(ledger.entries))
        self.assertEqual(report['ledger_head'], ledger.head)
        rows = {r['name']: r for r in report['coordinates']}
        for row in rows.values():
            self.assertEqual(row['within_tolerance'], row['upper_bound'] <= .2)
            self.assertAlmostEqual(row['rate'], row['successes']/300)
        # The suppression bypass yields more distinct keys, so more successes than published.
        self.assertGreater(rows['A']['successes'], rows['published']['successes'])
        self.assertGreaterEqual(rows['pooled']['successes'], rows['published']['successes'])
        verdict = report['verdict']
        self.assertEqual(set(verdict), {'release_within_tolerance_as_published',
                                        'findings_that_breach', 'pooled_breaches', 'scope'})
        # Same plan, same seed: the outcome is reproducible.
        again = run(plan, synthetic_view, AuditLedger())
        self.assertEqual([r['successes'] for r in again['coordinates']],
                         [r['successes'] for r in report['coordinates']])
        table = latex_table(report)
        self.assertIn('pooled', table)
        self.assertEqual(table.count(chr(92)*2), 1 + 4)  # header plus four rows


if __name__ == '__main__':
    unittest.main()
