"""Attack-law equivalence, selection leakage and simultaneous confirmation checks."""
import unittest
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from privacy_risk.injector_search import (
    ACTION_SCHEMA, Scenario, confirm, execute, paired_bootstrap, replay_prompt, replay_selection,
)


class PublicSearchTests(unittest.TestCase):
    def test_batched_attack_matches_scalar_log_weighted_attack(self):
        s = Scenario((1, 1, 1, .5), {}, np.array([0, 0, 1, 2]),
                     np.array([1., 1.3, 2., 5.]))
        rng = np.random.default_rng(92)
        targets = rng.integers(0, 4, 401)
        readings = s.income[targets]+rng.normal(0, .5, 401)
        success = 0
        for target, reading in zip(targets, readings):
            prior = np.where(s.key == s.key[target], .9, .1/3)
            score = np.log(prior)-.5*((reading-s.income)/.5)**2
            success += int(np.argmax(score) == target)
        self.assertEqual(execute(s, 401, np.random.default_rng(92)), success)

    def test_confirmation_counts_cells_once_and_uses_fresh_stream(self):
        a = Scenario((1, 1, 1, .1), {}, np.arange(4), np.array([0., 10., 20., 30.]))
        b = Scenario((1, 2, 1, .1), {}, np.arange(4), np.array([0., 10., 20., 30.]))
        rows, cells = confirm([a, b], [0, 1], 88, .05/128)
        self.assertEqual(cells, 1)
        self.assertEqual(sum(r['confirmed_high_risk'] for r in rows), 2)
        self.assertTrue(all(r['trials'] == 300 and r['lower_bound'] > .2 for r in rows))

    def test_bootstrap_pairs_states_rather_than_individual_attack_trials(self):
        raw = [dict(state=s, arm=arm, confirmed_cells=base+offset)
               for s, base in [(1, 2), (2, 5), (3, 3)]
               for arm, offset in [('sobol', 0), ('neural_search', 1)]]
        result = paired_bootstrap(raw, replicates=100)
        neural = next(r for r in result if r['arm'] == 'neural_search')
        self.assertEqual(neural['states'], 3)
        self.assertEqual(neural['difference_from_sobol'], 1.)
        self.assertEqual(neural['paired_state_bootstrap_ci95'], [1., 1.])

    def test_replay_binds_prompt_and_requires_unique_executable_actions(self):
        pool = [Scenario((1, 1, i+1, .1), {}, np.array([0, 1]), np.array([0., 2.]))
                for i in range(10)]
        entry = dict(model='fixture, not a live model', prompt_sha256='wrong', response_file='x')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(ValueError, 'prompt differs'):
                replay_selection(pool, 1, entry, path)
            entry['prompt_sha256'] = hashlib.sha256(replay_prompt(pool, 1).encode()).hexdigest()
            row = dict(proposal_id='fixture', scenario='unit test only',
                       features=dict(zip(ACTION_SCHEMA.features, pool[0].action)))
            (path/'x').write_text(json.dumps([row]))
            with self.assertRaisesRegex(ValueError, '32 distinct'):
                replay_selection(pool, 1, entry, path)

    def test_replay_accepts_32_catalogue_actions_without_generated_labels(self):
        pool = [Scenario((p, e, k, sd), {}, np.array([0, 1]), np.array([0., 2.]))
                for p in (1, 10) for e in (1, 2) for k in (1, 3) for sd in (.1, .25, .5, 1.)]
        response = [dict(proposal_id=f'fixture-{i}', scenario='unit test, not a live model',
                         features=dict(zip(ACTION_SCHEMA.features, s.action))) for i, s in enumerate(pool)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'responses.json').write_text(json.dumps(response))
            entry = dict(model='fixture', response_file='responses.json',
                         prompt_sha256=hashlib.sha256(replay_prompt(pool, 1).encode()).hexdigest())
            indices, metadata = replay_selection(pool, 1, entry, path)
        self.assertEqual(indices, list(range(32)))
        self.assertEqual(metadata['model'], 'fixture')
