"""Budget-matched search and independent confirmation on public empirical files.

This estimates attack success conditional on an already public empirical file.
It does not estimate undisclosed-population identification risk or grant a release
certificate. Only aggregate scenario measurements are exported.
"""
from __future__ import annotations

import itertools
import hashlib
import json
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta, qmc

from .realdata import RealFile, build_release, key_structure
from .sentry_r import DeepRiskNet, ScenarioSchema, llm_prompt, propose_from_llm, train_model

LEVELS = ((1, 10, 100), (1, 2, 4, 8), (1, 3, 5, 10), (.1, .25, .5, 1., 2.))
ARMS = ('random', 'heuristic', 'sobol', 'neural_search')
FEATURES = ('retained', 'reciprocal', 'log_records', 'puma_width',
            'education_width', 'minimum_cell', 'covariate_sd')
SCHEMA = ScenarioSchema(FEATURES, (0, 0, 0, 0, 1, 1, .1),
                        (1, 1, 12, 3, 8, 10, 2), ('linkage',), (.2,))
ACTION_SCHEMA = ScenarioSchema(('puma_width', 'education_width', 'minimum_cell', 'covariate_sd'),
                              (1, 1, 1, .1), (100, 8, 10, 2), ('linkage',), (.2,))


@dataclass
class Scenario:
    action: tuple
    features: dict
    key: np.ndarray
    income: np.ndarray

    @property
    def cell(self):
        puma, education, minimum, precision = self.action
        return (int(np.log10(puma)), int(education > 2), int(minimum > 1),
                int(precision > .5))


def candidates(file: RealFile) -> list[Scenario]:
    result = []
    for puma, education, minimum in itertools.product(*LEVELS[:3]):
        release = build_release(file, {'puma': puma, 'education': education,
                                       'experience': 10}, minimum)
        if release.size < 30:
            continue
        structure = key_structure(file, release)
        income = file.frame[file.covariate].to_numpy()[release.index].astype(float)
        for precision in LEVELS[3]:
            features = dict(retained=release.size/file.size,
                            reciprocal=structure['mean_reciprocal_frequency'],
                            log_records=float(np.log1p(file.size)),
                            puma_width=float(np.log10(puma)), education_width=float(education),
                            minimum_cell=float(minimum), covariate_sd=float(precision))
            result.append(Scenario((puma, education, minimum, precision), features,
                                   release.key, income))
    return result


def execute(scenario: Scenario, trials: int, rng: np.random.Generator) -> int:
    """MAP linkage: noisy income and an exact-key prior, evaluated in log space.

    Conditional on this fixed public file, targets and independent Gaussian
    auxiliary readings are drawn with replacement on every trial. No invented
    success label or fitted risk is used by this evaluator.
    """
    if trials < 1 or len(scenario.key) < 2:
        raise ValueError('positive trials and at least two candidate records required')
    targets = rng.integers(0, len(scenario.key), trials)
    sd = scenario.action[3]
    readings = scenario.income[targets] + rng.normal(0, sd, trials)
    counts = 0
    for start in range(0, trials, 128):
        stop = min(trials, start+128)
        matched = scenario.key[targets[start:stop], None] == scenario.key[None, :]
        prior = np.where(matched, .9, .1/(len(scenario.key)-1))
        score = np.log(prior) - .5*((readings[start:stop, None]-scenario.income)/sd)**2
        guessed = np.argmax(score, axis=1)
        counts += int(np.sum(guessed == targets[start:stop]))
    return counts


def measure(scenario, trials, rng, group, identity):
    return dict(record_id=f'{group}-{identity}', group=group,
                evidence_id=f'{group}-{identity}', features=scenario.features,
                evidence_kind='observed_development_attack',
                successes=[execute(scenario, trials, rng)], trials=[trials])


def sobol_order(pool, seed):
    lookup = {s.action: i for i, s in enumerate(pool)}
    order = []
    for point in qmc.Sobol(4, scramble=True, seed=seed).random_base2(10):
        action = tuple(level[min(int(v*len(level)), len(level)-1)]
                       for v, level in zip(point, LEVELS))
        index = lookup.get(action)
        if index is not None and index not in order:
            order.append(index)
    return order + [i for i in range(len(pool)) if i not in order]


def replay_question(pool, state):
    return (f'Public Census empirical state code {state}. Select exactly 32 distinct '
            'scenarios to discover diverse high-risk linkage configurations. '
            'Each features object must use puma_width, education_width, minimum_cell, '
            'covariate_sd. Select only complete tuples from this executable catalogue: '
            + json.dumps([list(s.action) for s in pool], separators=(',', ':'))
            + '. Supply no counts, predicted risks, or executable code.')


def replay_prompt(pool, state):
    return llm_prompt(ACTION_SCHEMA, replay_question(pool, state))


def replay_selection(pool, state, entry, directory):
    expected = hashlib.sha256(replay_prompt(pool, state).encode()).hexdigest()
    if entry['prompt_sha256'] != expected:
        raise ValueError('replay prompt differs from this state and executable catalogue')
    response = (directory/entry['response_file']).read_text(encoding='utf-8')
    staged = propose_from_llm(lambda _: response, ACTION_SCHEMA, replay_question(pool, state),
                             model_name=entry['model'])
    lookup = {s.action: i for i, s in enumerate(pool)}
    actions = [tuple(row['features'][name] for name in ACTION_SCHEMA.features) for row in staged]
    if len(actions) != 32 or len(set(actions)) != 32 or any(a not in lookup for a in actions):
        raise ValueError('replay must contain 32 distinct executable catalogue actions')
    return [lookup[a] for a in actions], dict(
        model=entry['model'], prompt_sha256=expected,
        response_sha256=hashlib.sha256(response.encode()).hexdigest(),
        identity='caller assertion, not provider attestation')


def select_and_measure(arm, pool, base, valid, seed, budget=32, trials=64, epochs=600):
    rng = np.random.default_rng(seed)
    measured, selected, fits = [], [], []
    if arm == 'random':
        fixed = rng.permutation(len(pool)).tolist()
    elif arm == 'heuristic':
        fixed = sorted(range(len(pool)), key=lambda i: (
            -(pool[i].features['reciprocal'] + .2/(.1+pool[i].action[3])), pool[i].action))
    elif arm == 'sobol':
        fixed = sobol_order(pool, seed)
    elif arm != 'neural_search':
        raise ValueError('unknown search arm')
    for round_index in range(0, budget, 8):
        remaining = [i for i in range(len(pool)) if i not in selected]
        if arm == 'neural_search':
            model = train_model([*base, *measured], valid, SCHEMA, width=16, depth=2,
                                epochs=epochs, patience=60, seed=seed)
            fits.append(dict(best_epoch=model['metrics']['best_epoch'], epoch_cap=epochs))
            x = np.array([SCHEMA.vector(pool[i].features) for i in remaining])
            pred = DeepRiskNet.from_dict(model['network']).predict(
                (x-np.asarray(model['mean']))/np.asarray(model['scale']))[:, 0]
            # Fixed exploration bonus, not a posterior uncertainty estimate.
            previous = np.array([SCHEMA.vector(row['features']) for row in [*base, *measured]])
            scale = np.asarray(SCHEMA.upper)-np.asarray(SCHEMA.lower)
            distance = np.sqrt(np.sum(((x[:, None, :]-previous[None, :, :])/scale)**2,
                                      axis=2)).min(axis=1)
            score = pred + .15*distance
            order = [remaining[i] for i in np.argsort(-score, kind='stable')]
        else:
            order = [i for i in fixed if i not in selected]
        batch = order[:min(8, budget-round_index)]
        selected.extend(batch)
        for i in batch:
            measured.append(measure(pool[i], trials, rng, base[0]['group'], f'search-{arm}-{i}'))
    return selected, measured, fits


def confirm(pool, selected, seed, alpha_per_candidate, tolerance=.2, trials=300):
    rows, cells = [], set()
    for position, index in enumerate(selected):
        scenario = pool[index]
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        successes = execute(scenario, trials, rng)
        lower = float(beta.ppf(alpha_per_candidate, successes, trials-successes+1)) \
            if successes else 0.
        found = lower > tolerance
        if found:
            cells.add(scenario.cell)
        rows.append(dict(query=position+1, action=list(scenario.action), cell=list(scenario.cell),
                         successes=successes, trials=trials, lower_bound=lower,
                         confirmed_high_risk=bool(found)))
    return rows, len(cells)


def paired_bootstrap(raw, reference='sobol', seed=771, replicates=10000):
    states = sorted({r['state'] for r in raw})
    lookup = {(r['state'], r['arm']): r for r in raw}
    picks = np.random.default_rng(seed).integers(0, len(states), (replicates, len(states)))
    result = []
    for arm in sorted({r['arm'] for r in raw}):
        values = np.array([lookup[(s, arm)]['confirmed_cells'] for s in states], dtype=float)
        differences = values-np.array([lookup[(s, reference)]['confirmed_cells'] for s in states])
        result.append(dict(arm=arm, states=len(states), mean_confirmed_cells=float(values.mean()),
                           difference_from_sobol=float(differences.mean()),
                           paired_state_bootstrap_ci95=np.quantile(differences[picks].mean(axis=1),
                                                                  [.025, .975]).tolist()))
    return result
