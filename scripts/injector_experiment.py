"""Paired synthetic injector ablation. Genuine LLM replays are optional, never simulated."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom, t

from privacy_risk.sentry_r import (
    DeepRiskNet, ScenarioSchema, digest, propose_from_llm, train_model,
)

FEATURES = ('sparsity', 'auxiliary', 'overlap', 'noise')
SCHEMA = ScenarioSchema(FEATURES, (0.,)*4, (1.,)*4, ('attack_success',), (.2,))
QUESTION = ('Return exactly 32 diverse scenarios. Probe combinations of high sparsity, '
            'strong auxiliary data, release overlap and low noise. Do not supply labels.')


def law(x, kind):
    z = -3 + 1.3*x[:, 0] + x[:, 1] + .7*x[:, 2] - x[:, 3]
    if kind == 'interaction':
        z += 4*(x[:, 0] > .7)*(x[:, 1] > .7)*x[:, 2]
    return (1/(1+np.exp(-z)))[:, None]


def rows(x, rng, group, kind):
    k = rng.binomial(100, law(x, kind))
    return [dict(record_id=f'{group}-{i}', group=group, evidence_id=f'{group}-{i}',
                 features=dict(zip(FEATURES, map(float, v))),
                 evidence_kind='observed_development_attack',
                 successes=k[i].tolist(), trials=[100]) for i, v in enumerate(x)]


def interval(values):
    a = np.asarray(values, dtype=float)
    radius = float(t.ppf(.975, len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)))
    return dict(mean=float(a.mean()), ci95=[float(a.mean()-radius), float(a.mean()+radius)])


def audit_restarts(replicates=20000):
    """Null: true attack risk .201 exceeds tolerance .2; every certificate is false.

    Fresh independent draws on each attempt. This is a statistical stress test,
    not a claim that the current file ledger implements an external audit service.
    """
    rng = np.random.default_rng(817)
    results = []
    for attempts in (1, 5, 20):
        counts = rng.binomial(500, .201, size=(replicates, attempts))
        for rule, alpha in (('reset_alpha', .05), ('preallocated_budget', .05/attempts)):
            upper = beta.ppf(1-alpha, np.arange(501)+1, 500-np.arange(501))
            upper[-1] = 1.
            flags = (upper[counts] <= .2).any(axis=1)
            critical = np.flatnonzero(upper <= .2)
            one = float(binom.cdf(critical[-1], 500, .201)) if len(critical) else 0.
            successes = int(flags.sum())
            lo = float(beta.ppf(.025, successes, replicates-successes+1)) if successes else 0.
            hi = float(beta.ppf(.975, successes+1, replicates-successes))
            results.append(dict(attempts=attempts, rule=rule, simulations=replicates,
                                false_certificates=successes, rate=float(flags.mean()),
                                ci95=[lo, hi], exact_rate=1-(1-one)**attempts))
    return results


def run(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    config = dict(seeds=list(range(101, 101+args.seeds)), base_rows=120,
                  validation_rows=80, test_rows=1500, proposals=32, trials_per_row=100,
                  epochs=args.epochs, patience=70, depths=[0, 2], width=16,
                  laws=['additive', 'interaction'], schema=SCHEMA.as_dict(), question=QUESTION,
                  llm_manifest_sha256=(hashlib.sha256(args.llm_replays.read_bytes()).hexdigest()
                                       if args.llm_replays else None))
    # Persist the design before drawing experimental outcomes.
    (root/'design.json').write_text(json.dumps(config, indent=2)+'\n')
    manifest = json.loads(args.llm_replays.read_text()) if args.llm_replays else {}
    raw = []
    for seed in config['seeds']:
        common = np.random.default_rng(seed)
        base_x = common.uniform(0, .7, (120, 4))
        stop_x = common.uniform(0, 1, (80, 4))
        test_sets = dict(iid=common.uniform(0, 1, (1500, 4)),
                         shifted=common.uniform(.5, 1, (1500, 4)))
        rand = np.random.default_rng(seed+500).uniform(0, 1, (32, 4))
        heuristic = np.random.default_rng(seed+600).uniform(0, 1, (32, 4))
        heuristic[:16, :3] = np.random.default_rng(seed+601).uniform(.7, 1, (16, 3))
        heuristic[:16, 3] *= .3
        proposals = dict(base=np.empty((0, 4)), random=rand, heuristic=heuristic)
        replay_metadata = None
        if manifest:
            entry = manifest[str(seed)]
            response_path = args.llm_replays.parent/entry['response_file']
            response = response_path.read_text(encoding='utf-8')
            def replay(prompt):
                if hashlib.sha256(prompt.encode()).hexdigest() != entry['prompt_sha256']:
                    raise ValueError('Replay prompt differs from registered experiment prompt')
                return response
            staged = propose_from_llm(replay, SCHEMA, QUESTION, model_name=entry['model'])
            if len(staged) != 32:
                raise ValueError('Matched budget requires exactly 32 proposals per seed')
            proposals['llm_replay'] = np.stack([SCHEMA.vector(r['features']) for r in staged])
            replay_metadata = dict(model=entry['model'], prompt_sha256=entry['prompt_sha256'],
                                   response_sha256=hashlib.sha256(response.encode()).hexdigest(),
                                   status='caller-supplied replay; provider identity not attested')
        for kind in config['laws']:
            base = rows(base_x, np.random.default_rng(seed+1), 'base', kind)
            valid = rows(stop_x, np.random.default_rng(seed+2), 'validation', kind)
            for arm, x in proposals.items():
                extra = rows(x, np.random.default_rng(seed+3), 'injected', kind)
                for depth in config['depths']:
                    start = time.perf_counter()
                    model = train_model([*base, *extra], valid, SCHEMA, depth=depth,
                                        width=16, epochs=args.epochs, patience=70, seed=seed)
                    fit_seconds = time.perf_counter()-start
                    net = DeepRiskNet.from_dict(model['network'])
                    for shift, test_x in test_sets.items():
                        p = law(test_x, kind)
                        pred = net.predict((test_x-np.asarray(model['mean']))/
                                           np.asarray(model['scale']))
                        clipped = np.clip(pred, 1e-12, 1-1e-12)
                        unsafe = p[:, 0] > .2
                        metrics = dict(
                            log_loss=float(np.mean(-p*np.log(clipped)-(1-p)*np.log(1-clipped))),
                            probability_mse=float(np.mean((pred-p)**2)),
                            unsafe_recall=float(np.mean(pred[unsafe, 0] > .2)),
                            unsafe_count=int(unsafe.sum()),
                            dangerous_underprediction=float(np.mean(unsafe & (pred[:, 0] <= .2))))
                        raw.append(dict(seed=seed, law=kind, shift=shift, arm=arm,
                                        depth=depth, metrics=metrics, fit_seconds=fit_seconds,
                                        best_epoch=model['metrics']['best_epoch'],
                                        epoch_cap_reached=model['metrics']['best_epoch'] >= args.epochs,
                                        model_sha256=digest(model),
                                        train_sha256=digest([*base, *extra]),
                                        test_features_sha256=digest(test_x.tolist()),
                                        injected_truth_mean=float(law(x, kind).mean()) if len(x) else None,
                                        replay=replay_metadata if arm == 'llm_replay' else None))
        print(f'completed paired seed {seed}', flush=True)
    summaries = []
    for kind in config['laws']:
        for shift in ('iid', 'shifted'):
            for depth in config['depths']:
                for arm in proposals:
                    subset = [r for r in raw if (r['law'], r['shift'], r['depth'], r['arm']) ==
                              (kind, shift, depth, arm)]
                    base = {r['seed']: r for r in raw if
                            (r['law'], r['shift'], r['depth'], r['arm']) ==
                            (kind, shift, depth, 'base')}
                    summaries.append(dict(law=kind, shift=shift, depth=depth, arm=arm,
                                          log_loss=interval([r['metrics']['log_loss'] for r in subset]),
                                          paired_log_loss_minus_base=interval([
                                              r['metrics']['log_loss']-
                                              base[r['seed']]['metrics']['log_loss'] for r in subset]),
                                          dangerous_underprediction=interval([
                                              r['metrics']['dangerous_underprediction'] for r in subset])))
    result = dict(status='SYNTHETIC PILOT; not empirical privacy evidence or a release certificate',
                  llm_status='caller-supplied replays' if manifest else 'NOT RUN: no genuine LLM replays supplied',
                  config=config, config_sha256=digest(config), python=platform.python_version(),
                  numpy=np.__version__, summaries=summaries, raw=raw, audit_restarts=audit_restarts())
    (root/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    plot(result, root)
    return result


def plot(result, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    data = [r for r in result['summaries'] if r['law'] == 'interaction' and
            r['shift'] == 'shifted' and r['depth'] == 2]
    for i, r in enumerate(data):
        v = r['paired_log_loss_minus_base']
        axes[0].errorbar(i, v['mean'], yerr=[[v['mean']-v['ci95'][0]],
                                           [v['ci95'][1]-v['mean']]], fmt='o', capsize=5)
    axes[0].set_xticks(range(len(data)), [r['arm'] for r in data])
    axes[0].axhline(0, color='grey', linestyle='--')
    axes[0].set(title='Interaction law, shifted test, two hidden layers',
                ylabel='Paired log loss minus base (95% t interval)')
    for rule, label in [('reset_alpha', 'Reset alpha'),
                        ('preallocated_budget', 'Preallocated total alpha=.05')]:
        d = [r for r in result['audit_restarts'] if r['rule'] == rule]
        axes[1].plot([r['attempts'] for r in d], [r['rate'] for r in d], 'o-', label=label)
    axes[1].axhline(.05, color='grey', linestyle='--')
    axes[1].set(xlabel='Independent audit attempts', ylabel='False certification probability',
                title='Synthetic unsafe boundary stress test')
    axes[1].legend(fontsize=8)
    fig.suptitle('SENTRY-R synthetic pilot — LLM arm pending' if
                 result['llm_status'].startswith('NOT RUN') else 'SENTRY-R replay pilot')
    fig.savefig(root/'pilot.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts-injector'))
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1200)
    parser.add_argument('--llm-replays', type=Path)
    args = parser.parse_args()
    if args.seeds < 2 or args.epochs < 1:
        parser.error('at least two seeds and a positive epoch cap are required')
    run(args)
