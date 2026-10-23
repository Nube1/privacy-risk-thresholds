"""Table and numbers for the local open-weight LLM arm of the public-file search."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

NAMES = dict(random='Random', heuristic='Domain heuristic', sobol='Scrambled Sobol',
             neural_search='Neural search', llm_replay='Qwen3-4B proposals')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search', type=Path, default=Path('artifacts-public-search-llm-local-30'))
    parser.add_argument('--pilot', type=Path, default=Path('artifacts-llm-local-30'))
    parser.add_argument('--table', type=Path, default=Path('paper/generated/table_public_search_llm.tex'))
    parser.add_argument('--numbers', type=Path, default=Path('paper/generated/numbers_llm_local.json'))
    args = parser.parse_args()
    result = json.loads((args.search/'results.json').read_text(encoding='utf-8'))
    status = json.loads((args.pilot/'status.json').read_text(encoding='utf-8'))
    attempts = json.loads((args.pilot/'attempts.json').read_text(encoding='utf-8'))
    manifest = json.loads((args.pilot/'manifest.json').read_text(encoding='utf-8'))
    states = result['config']['selected_states']
    by_arm = {r['arm']: r for r in result['summary']}
    raw = result['raw']
    confirmed = {arm: float(np.mean([r['confirmed_candidates'] for r in raw if r['arm'] == arm]))
                 for arm in NAMES}
    # How the model chose: share of its 32 tuples at the finest geography, by state.
    enumerated = 0
    for state, entry in manifest.items():
        rows = json.loads((args.pilot/entry['response_file']).read_text(encoding='utf-8'))
        if all(row['features']['puma_width'] == 1 for row in rows):
            enumerated += 1
    rejections = Counter(a['error'] for a in attempts if a['status'] == 'rejected')
    provider = status['provider']
    lines = [r'\begin{table}[htbp]', r'\centering\small',
             r'\caption{Public-file search with a local open-weight language model as a fifth arm. '
             f'The {len(states)} state samples are those for which the model returned a valid list of 32 '
             'catalogue tuples within three attempts; every arm is rerun on exactly these states. '
             'Intervals are paired-state bootstrap differences from Sobol.}',
             r'\label{tab:public-search-llm}', r'\begin{tabular}{lrrr}', r'\toprule',
             r'Search & Confirmed cells & Confirmed settings & Difference [95\% interval] \\',
             r'\midrule']
    for arm, name in NAMES.items():
        r = by_arm[arm]
        lo, hi = r['paired_state_bootstrap_ci95']
        lines.append(f"{name} & {r['mean_confirmed_cells']:.2f} & {confirmed[arm]:.2f} & "
                     f"{r['difference_from_sobol']:+.2f} [{lo:+.2f}, {hi:+.2f}] " + chr(92)*2)
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    llm = by_arm['llm_replay']
    numbers = dict(
        states_valid=len(states), states_requested=len(status['requested_states']),
        attempts=status['attempts'], rejected_attempts=status['rejected_attempts'],
        rejection_reasons=dict(rejections),
        valid_at_first_attempt=sum(a['status'] == 'valid' and a['attempt'] == 0 for a in attempts),
        states_all_finest_geography=enumerated,
        attack_trials=result['attack_trials'],
        model=provider['model'], digest=provider['digest'], quantization=provider['quantization'],
        parameters=provider['parameter_size'], ollama_version=provider['ollama_version'],
        llm_cells=llm['mean_confirmed_cells'], llm_confirmed=confirmed['llm_replay'],
        llm_difference_from_sobol=llm['difference_from_sobol'],
        llm_interval=llm['paired_state_bootstrap_ci95'],
        sobol_cells=by_arm['sobol']['mean_confirmed_cells'],
        neural_cells=by_arm['neural_search']['mean_confirmed_cells'],
        heuristic_confirmed=confirmed['heuristic'], sobol_confirmed=confirmed['sobol'],
        mean_generation_seconds=float(np.mean([e['usage']['total_seconds']
                                               for e in manifest.values()])))
    args.numbers.write_text(json.dumps(numbers, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(numbers, indent=2))


if __name__ == '__main__':
    main()
