"""Local Ollama pilot: valid lists are kept, malformed ones are retried and counted."""
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from privacy_risk.injector_search import LEVELS, Scenario, replay_prompt
from privacy_risk.llm_pilot_local import collect, output_text, request_payload

POOL = [Scenario(a, {}, np.array([], dtype=int), np.array([]))
        for a in itertools.product(*LEVELS)]
RECORD = dict(model='qwen3:4b', digest='sha256:abc', size_bytes=1, modified_at='x',
              ollama_version='0.12', family='qwen3', parameter_size='4B', quantization='Q4')
NAMES = ('puma_width', 'education_width', 'minimum_cell', 'covariate_sd')


def proposals(actions):
    return json.dumps([dict(proposal_id=f'p{i}', scenario='catalogue tuple',
                            features=dict(zip(NAMES, a))) for i, a in enumerate(actions)])


def reply(content, done=True):
    return dict(model='qwen3:4b', done=done, done_reason='stop', eval_count=10,
                prompt_eval_count=5, total_duration=2_000_000_000,
                message=dict(role='assistant', content=content))


class LocalPilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        prompts = self.root/'prompts'
        prompts.mkdir()
        for state in (0, 1):
            (prompts/f'state-{state}.txt').write_text(replay_prompt(POOL, state), encoding='utf-8')
        self.prompts = prompts

    def tearDown(self):
        self.tmp.cleanup()

    def test_fenced_valid_list_is_accepted_and_manifest_matches_replay_contract(self):
        good = proposals([s.action for s in POOL[:32]])
        calls = []

        def call(payload, host):
            calls.append(payload)
            return reply('```json\n' + good + '\n```')
        status = collect(self.prompts, self.root/'out', call=call, record=lambda m, h: RECORD)
        self.assertEqual(status['status'], 'collected_and_validated')
        self.assertEqual(status['valid_states'], [0, 1])
        self.assertEqual(status['rejected_attempts'], 0)
        self.assertEqual(status['cost_usd'], 0.)
        manifest = json.loads((self.root/'out'/'manifest.json').read_text())
        self.assertEqual(set(manifest), {'0', '1'})
        self.assertTrue(manifest['0']['code_fence_removed'])
        self.assertEqual(manifest['0']['digest'], 'sha256:abc')
        self.assertEqual((self.root/'out'/'responses'/'state-0.json').read_text(), good)
        # Seeds differ by state, thinking is off, the prompt is sent verbatim.
        self.assertNotEqual(calls[0]['options']['seed'], calls[1]['options']['seed'])
        self.assertFalse(calls[0]['think'])
        self.assertEqual(calls[0]['messages'][0]['content'],
                         (self.prompts/'state-0.txt').read_text(encoding='utf-8'))

    def test_malformed_lists_are_retried_with_new_seeds_and_every_attempt_is_kept(self):
        good = proposals([s.action for s in POOL[:32]])
        short = proposals([s.action for s in POOL[:31]])
        duplicate = proposals([POOL[0].action]*32)
        off_catalogue = proposals([*(s.action for s in POOL[:31]), (7, 7, 7, 7.)])
        with_counts = json.dumps([dict(proposal_id='p', scenario='x', successes=[3],
                                       features=dict(zip(NAMES, POOL[0].action)))])
        answers = iter([reply(short), reply(duplicate), reply(good),
                        reply(off_catalogue), reply(with_counts), reply('not json')])
        seeds = []

        def call(payload, host):
            seeds.append(payload['options']['seed'])
            return next(answers)
        status = collect(self.prompts, self.root/'out', call=call, attempts=3,
                         record=lambda m, h: RECORD)
        self.assertEqual(status['status'], 'incomplete')
        self.assertEqual(status['valid_states'], [0])
        self.assertEqual(status['attempts'], 6)
        self.assertEqual(status['rejected_attempts'], 5)
        self.assertEqual(len(set(seeds)), 6)
        log = json.loads((self.root/'out'/'attempts.json').read_text())
        self.assertEqual([a['status'] for a in log],
                         ['rejected', 'rejected', 'valid', 'rejected', 'rejected', 'rejected'])
        self.assertEqual(len(list((self.root/'out'/'attempts').glob('*.provider.json'))), 6)
        manifest = json.loads((self.root/'out'/'manifest.json').read_text())
        self.assertEqual(manifest['0']['attempt'], 2)

    def test_server_failure_stops_without_spinning_and_plan_cannot_change(self):
        def down(payload, host):
            raise RuntimeError('Ollama is not reachable')
        status = collect(self.prompts, self.root/'out', call=down, record=lambda m, h: RECORD)
        self.assertEqual(status['status'], 'incomplete')
        self.assertEqual(status['attempts'], 2)  # one per state, no retry on a dead server
        with self.assertRaisesRegex(ValueError, 'plan differs'):
            collect(self.prompts, self.root/'out', call=down, attempts=2,
                    record=lambda m, h: RECORD)
        with self.assertRaisesRegex(ValueError, 'did not finish'):
            output_text(reply('[]', done=False))
        payload = request_payload('p', 'qwen3:4b', 7)
        self.assertEqual(payload['options']['seed'], 7)
        self.assertFalse(payload['stream'])
        schema = payload['format']
        self.assertEqual((schema['minItems'], schema['maxItems']), (32, 32))
        self.assertEqual(schema['items']['properties']['features']['properties']
                         ['covariate_sd']['enum'], [.1, .25, .5, 1., 2.])
        self.assertEqual(schema['items']['required'], ['proposal_id', 'scenario', 'features'])


if __name__ == '__main__':
    unittest.main()
