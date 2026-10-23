"""Regression probes for content substitution and renamed holdout copies."""
import copy
import tempfile
import unittest
from pathlib import Path

from privacy_risk.provenance import LedgerError, ProvenanceLedger
from privacy_risk.sentry_r import (
    MAX_RESPONSE_BYTES, ScenarioSchema, join_evidence, review_proposals,
    stage_llm, train_model,
)


class InjectorIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.schema = ScenarioSchema(('a',), (-2.,), (2.,), ('link',), (.2,))
        self.chain = ProvenanceLedger(Path(self.tmp.name)/'chain.jsonl', b'x'*32)

    def row(self, name, value):
        return dict(record_id=name, group=name, evidence_id=name,
                    evidence_kind='observed_development_attack', features={'a': value},
                    successes=[1], trials=[20])

    def measured(self):
        staged = stage_llm([dict(proposal_id='p', scenario='test', features={'a': .3},
                                llm_model='fixture', prompt_hash='fixture')],
                           self.schema, provenance=self.chain)
        approved = review_proposals(staged, [dict(
            proposal_id='p', proposal_digest=staged[0]['proposal_digest'],
            approved=True, reviewer='test fixture', reason='synthetic unit test')],
            provenance=self.chain)
        return join_evidence(approved, [dict(
            proposal_id='p', proposal_digest=staged[0]['proposal_digest'], group='injected',
            evidence_id='measured', evidence_kind='observed_development_attack',
            successes=[1], trials=[20])], self.schema, provenance=self.chain)[0]

    def test_measured_counts_cannot_change_before_training(self):
        row = self.measured()
        row['successes'] = [0]
        with self.assertRaisesRegex(LedgerError, 'content'):
            train_model([self.row('base', -.4), row], [self.row('valid', .8)],
                        self.schema, depth=0, epochs=2, provenance=self.chain)

    def test_measured_features_cannot_change_before_training(self):
        row = self.measured()
        row['features']['a'] = .6
        with self.assertRaisesRegex(LedgerError, 'content'):
            train_model([self.row('base', -.4), row], [self.row('valid', .8)],
                        self.schema, depth=0, epochs=2, provenance=self.chain)

    def test_renaming_validation_copy_does_not_hide_it(self):
        valid = self.row('valid', .8)
        renamed = copy.deepcopy(valid)
        renamed.update(record_id='renamed', group='different', evidence_id='renamed')
        renamed['proposal_digest'] = 'renamed-injected-proposal'
        with self.assertRaisesRegex(ValueError, 'feature overlap'):
            train_model([self.row('base', -.4), renamed], [valid],
                        self.schema, depth=0, epochs=2)

    def test_file_intake_metadata_cannot_bypass_size_limit(self):
        row = dict(proposal_id='p', scenario='test', features={'a': .3},
                   llm_model='x'*(MAX_RESPONSE_BYTES+1), prompt_hash='fixture')
        with self.assertRaisesRegex(ValueError, 'bytes'):
            stage_llm([row], self.schema)

