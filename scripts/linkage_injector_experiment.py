"""Execute nearest-record linkage attacks on synthetic populations, then fit SENTRY-R.

No real persons, no external data, no invented Bernoulli risk law. Every success
is an actual nearest-neighbour match to a randomly selected target in a fresh
64-person synthetic population. This is an empirical synthetic experiment, not
a real-data privacy assessment or an operational release certificate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from injector_experiment import FEATURES, SCHEMA, interval
from privacy_risk.sentry_r import DeepRiskNet, digest, train_model


def attack_count(features, trials, rng):
    """Three quasi-identifiers, quantized keys, noisy auxiliary data and releases.

    Higher sparsity means finer key resolution. Higher auxiliary strength means
    less auxiliary noise. Overlap controls 1--4 releases of the same population;
    their independent Gaussian perturbations are averaged. Each independent
    trial regenerates the population, target and noises. Candidate records are
    compared by squared Euclidean distance; success means correct target index.
    """
    sparsity, auxiliary, overlap, noise = features
    resolution = 2 + int(np.floor(30*sparsity))
    releases = 1 + int(np.floor(3*overlap))
    population = rng.integers(0, resolution, (trials, 64, 3))/(resolution-1)
    target = rng.integers(0, 64, trials)
    aux = population[np.arange(trials), target] + rng.normal(
        0, .01+.6*(1-auxiliary), (trials, 3))
    # Averaging iid Gaussian releases is exactly equivalent to this draw.
    released = population + rng.normal(
        0, (.01+.6*noise)/np.sqrt(releases), population.shape)
    distances = np.sum((released-aux[:, None, :])**2, axis=2)
    guessed = np.argmin(distances, axis=1)
    return int(np.sum(guessed == target))


def measured_rows(x, trials, rng, group):
    return [dict(record_id=f'{group}-{i}', group=group, evidence_id=f'{group}-{i}',
                 features=dict(zip(FEATURES, map(float, v))),
                 evidence_kind='observed_development_attack',
                 successes=[attack_count(v, trials, rng)], trials=[trials])
            for i, v in enumerate(x)]


def run(output, seeds=5):
    output.mkdir(parents=True, exist_ok=True)
    config = dict(seeds=list(range(201, 201+seeds)), base_rows=120, stopping_rows=80,
                  test_rows=250, proposals=32, development_trials=100, test_trials=1000,
                  population_size=64, quasi_identifiers=3, epochs=1200, patience=70,
                  schema=SCHEMA.as_dict(), evaluator='synthetic-nearest-record-linkage-v1')
    (output/'design.json').write_text(json.dumps(config, indent=2)+'\n')
    result = []
    for seed in config['seeds']:
        rng = np.random.default_rng(seed)
        base_x = rng.uniform(0, .7, (120, 4))
        stop_x = rng.uniform(0, 1, (80, 4))
        test_x = rng.uniform(.5, 1, (250, 4))
        random_x = rng.uniform(0, 1, (32, 4))
        heuristic_x = rng.uniform(0, 1, (32, 4))
        heuristic_x[:16, :3] = rng.uniform(.7, 1, (16, 3))
        heuristic_x[:16, 3] *= .3
        base = measured_rows(base_x, 100, np.random.default_rng(seed+10), 'base')
        valid = measured_rows(stop_x, 100, np.random.default_rng(seed+11), 'valid')
        test = measured_rows(test_x, 1000, np.random.default_rng(seed+12), 'test')
        (output/f'evidence-{seed}.json').write_text(json.dumps(
            dict(base=base, stopping=valid, test=test), indent=2)+'\n')
        empirical = np.array([r['successes'][0]/r['trials'][0] for r in test])[:, None]
        for arm, x in dict(base=np.empty((0, 4)), random=random_x,
                           heuristic=heuristic_x).items():
            extra = measured_rows(x, 100, np.random.default_rng(seed+13), 'injected')
            (output/f'augmentation-{seed}-{arm}.json').write_text(
                json.dumps(extra, indent=2)+'\n')
            for depth in (0, 2):
                model = train_model([*base, *extra], valid, SCHEMA, depth=depth,
                                    width=16, epochs=1200, patience=70, seed=seed)
                net = DeepRiskNet.from_dict(model['network'])
                pred = np.clip(net.predict((test_x-np.asarray(model['mean']))/
                                           np.asarray(model['scale'])), 1e-12, 1-1e-12)
                loss = float(np.mean(-empirical*np.log(pred)-(1-empirical)*np.log(1-pred)))
                result.append(dict(seed=seed, arm=arm, depth=depth, test_log_loss=loss,
                                   model_sha256=digest(model), test_sha256=digest(test),
                                   training_sha256=digest([*base, *extra]),
                                   mean_observed_test_attack_success=float(empirical.mean()),
                                   best_epoch=model['metrics']['best_epoch'],
                                   epoch_cap_reached=model['metrics']['best_epoch'] >= 1200))
        print(f'completed executable linkage seed {seed}', flush=True)
    summary = []
    for depth in (0, 2):
        base = {r['seed']: r for r in result if r['depth'] == depth and r['arm'] == 'base'}
        for arm in ('base', 'random', 'heuristic'):
            subset = [r for r in result if r['depth'] == depth and r['arm'] == arm]
            summary.append(dict(depth=depth, arm=arm,
                                log_loss=interval([r['test_log_loss'] for r in subset]),
                                paired_minus_base=interval([r['test_log_loss']-
                                    base[r['seed']]['test_log_loss'] for r in subset])))
    final = dict(status='Executed synthetic linkage attacks; no genuine LLM arm or real data',
                 config=config, config_sha256=digest(config), summary=summary, raw=result)
    (output/'results.json').write_text(json.dumps(final, indent=2)+'\n')
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts-linkage-injector'))
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error('at least two independent seeds are required')
    run(args.output, args.seeds)
