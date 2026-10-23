"""Cost and retry safety tests. Provider replies here are explicit test fixtures."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from privacy_risk.injector_search import ACTION_SCHEMA, Scenario, replay_prompt
from privacy_risk.llm_pilot import MODEL, collect


class LLMPilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prompts = self.root/'prompts'
        self.prompts.mkdir()
        self.output = self.root/'output'
        self.pool = [Scenario((p, e, k, sd), {}, np.array([0, 1]), np.array([0., 1.]))
                     for p in (1, 10) for e in (1, 2) for k in (1, 3)
                     for sd in (.1, .25, .5, 1.)]
        for state in (0, 1):
            (self.prompts/f'state-{state}.txt').write_text(replay_prompt(self.pool, state))

    def reply(self, payload, key):
        self.assertEqual(payload['model'], MODEL)
        self.assertFalse(payload['store'])
        self.assertEqual(payload['reasoning']['effort'], 'none')
        proposals = [dict(proposal_id=f'test-{i}', scenario='mock fixture, not research evidence',
                          features=dict(zip(ACTION_SCHEMA.features, s.action)))
                     for i, s in enumerate(self.pool)]
        return dict(status='completed', model=MODEL, id='unit-test-only',
                    output=[dict(type='message', content=[dict(type='output_text',
                                                              text=json.dumps(proposals))])],
                    usage=dict(input_tokens=1000, output_tokens=2000)), 'test-request-only'

    def test_missing_key_prepares_without_a_call_or_cost_reservation(self):
        with patch.dict(os.environ, {}, clear=True):
            result = collect(self.prompts, self.output,
                             call=lambda *_: self.fail('must not call without a key'))
        self.assertEqual(result['status'], 'needs_api_key')
        self.assertFalse(list(self.output.glob('*.reservation.json')))

    def test_two_requests_are_bounded_validated_and_never_silently_repeated(self):
        secret = 'unit-test-secret-never-log'
        with patch.dict(os.environ, {'OPENAI_API_KEY': secret}):
            result = collect(self.prompts, self.output, call=self.reply)
            self.assertEqual(result['paid_requests'], 2)
            self.assertLess(result['total_reserved_usd'], .25)
            with self.assertRaisesRegex(ValueError, 'attempted request'):
                collect(self.prompts, self.output, call=lambda *_: self.fail('no retry allowed'))
        manifest = json.loads((self.output/'manifest.json').read_text())
        self.assertEqual(set(manifest), {'0', '1'})
        self.assertTrue(all(secret not in p.read_text() for p in self.output.rglob('*.json')))

    def test_unknown_transport_outcome_remains_reserved(self):
        def interrupted(*_):
            raise RuntimeError('simulated transport interruption')
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'fixture'}):
            result = collect(self.prompts, self.output, call=interrupted)
            self.assertEqual(result['status'], 'failed_or_invalid')
            self.assertTrue((self.output/'state-0.reservation.json').exists())
            self.assertFalse((self.output/'state-1.reservation.json').exists())
            with self.assertRaisesRegex(ValueError, 'attempted request'):
                collect(self.prompts, self.output, call=self.reply)

    def test_plan_is_rejected_before_spending_if_budget_is_insufficient(self):
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            collect(self.prompts, self.output, budget_usd=.001, call=self.reply)
        self.assertFalse(list(self.output.glob('*.reservation.json')))
