"""Run a declared search budget on public Census states; confirm on fresh draws."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy

from privacy_risk.injector_search import (
    ARMS, LEVELS, candidates, confirm, measure, paired_bootstrap, replay_prompt,
    replay_selection, select_and_measure,
)
from privacy_risk.realdata import RealFile, load_census2000
from privacy_risk.sentry_r import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts-public-search'))
    parser.add_argument('--states', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--llm-replays', type=Path,
                        help='state-keyed manifest of genuine responses and exact prompt hashes')
    args = parser.parse_args()
    if args.states < 2 or args.epochs < 1:
        parser.error('at least two states and positive epochs required')
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    data = load_census2000()
    manifest = json.loads(args.llm_replays.read_text()) if args.llm_replays else None
    arms = (*ARMS, 'llm_replay') if manifest is not None else ARMS
    groups = [(int(state), frame.reset_index(drop=True)) for state, frame in
              data.frame.groupby('state_code', sort=True) if len(frame) >= 100]
    selected_groups = groups[:args.states]
    config = dict(status='exploratory public empirical-file search; not population risk',
                  selected_states=[s for s, _ in selected_groups], selection='first state codes with >=100 rows',
                  seed=20261023, arms=list(arms), levels=LEVELS,
                  warmstart=16, stopping=8, search_queries=32, search_trials=64,
                  confirmation_trials=300, tolerance=.2, family_alpha=.05,
                  primary_contrast='neural_search minus sobol, confirmed scenario cells',
                  cell_definition='puma level x (education>2) x (minimum>1) x (sd>.5)',
                  epochs=args.epochs, patience=60, exploration_weight=.15,
                  public_data_sha256=hashlib.sha256(data.frame.to_csv(index=False).encode()).hexdigest(),
                  versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  llm_status='caller-supplied genuine responses' if manifest is not None else
                  'not run: no genuine provider responses')
    (root/'design.json').write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
    raw, actual_trials = [], 0
    for block_index, (state, frame) in enumerate(selected_groups):
        partner_state, partner_frame = groups[(block_index+1) % len(groups)]
        file = RealFile(f'state-{state}', frame, data.block, data.key_columns, data.covariate)
        partner = RealFile(f'state-{partner_state}', partner_frame, data.block,
                           data.key_columns, data.covariate)
        pool, stopping_pool = candidates(file), candidates(partner)
        seed = config['seed']+1000*state
        rng = np.random.default_rng(seed)
        start_indices = rng.choice(len(pool), 16, replace=False)
        base = [measure(pool[i], 64, rng, f'state-{state}', f'warm-{i}') for i in start_indices]
        valid_indices = rng.choice(len(stopping_pool), 8, replace=False)
        valid = [measure(stopping_pool[i], 64, rng, f'state-{partner_state}', f'stop-{i}')
                 for i in valid_indices]
        actual_trials += 24*64
        pool = [s for i, s in enumerate(pool) if i not in set(start_indices)]
        if len(pool) < 32:
            raise ValueError('insufficient distinct scenarios for matched query budget')
        (root/f'prompt-state-{state}.txt').write_text(replay_prompt(pool, state), encoding='utf-8')
        for arm in arms:
            replay_metadata = None
            if arm == 'llm_replay':
                ids, replay_metadata = replay_selection(pool, state, manifest[str(state)],
                                                       args.llm_replays.parent)
                measure_rng = np.random.default_rng(seed+1)
                measured = [measure(pool[i], 64, measure_rng, f'state-{state}',
                                    f'search-{arm}-{i}') for i in ids]
                fits = []
            else:
                ids, measured, fits = select_and_measure(arm, pool, base, valid,
                    seed+1, epochs=args.epochs)
            confirmed, cell_count = confirm(pool, ids, seed+2, .05/(len(arms)*32))
            actual_trials += 32*(64+300)
            raw.append(dict(state=state, arm=arm, source_rows=len(frame),
                            pool_size=len(pool), confirmed_cells=cell_count,
                            confirmed_candidates=sum(r['confirmed_high_risk'] for r in confirmed),
                            search_successes=sum(r['successes'][0] for r in measured),
                            search_trials=32*64, confirmation=confirmed, fits=fits,
                            measured_development=measured,
                            llm_replay=replay_metadata,
                            warmstart_sha256=digest(base), stopping_sha256=digest(valid)))
        (root/f'state-{state}.json').write_text(json.dumps(
            dict(warmstart=base, stopping=valid, arms=[r for r in raw if r['state'] == state]),
            indent=2)+'\n', encoding='utf-8')
        print(f'completed public state {state}: {block_index+1}/{len(selected_groups)}', flush=True)
    result = dict(config=config, config_sha256=digest(config), attack_trials=actual_trials,
                  neural_fits=sum(len(r['fits']) for r in raw), summary=paired_bootstrap(raw), raw=raw)
    (root/'results.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(attack_trials=actual_trials, summary=result['summary']), indent=2), flush=True)


if __name__ == '__main__':
    main()
