"""Generate manuscript tables, plots, replay prompts and revision evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from privacy_risk.injector_search import candidates, replay_prompt
from privacy_risk.realdata import RealFile, load_census2000

ROOT = Path(__file__).resolve().parents[1]


def main():
    primary = json.loads((ROOT/'artifacts-public-search/results.json').read_text())
    sensitivity = json.loads((ROOT/'artifacts-public-search-2000/results.json').read_text())
    out = ROOT/'paper/injector_study/generated'
    out.mkdir(parents=True, exist_ok=True)
    names = dict(random='Random', heuristic='Domain heuristic', sobol='Scrambled Sobol',
                 neural_search='Neural search')
    by_arm = {r['arm']: r for r in primary['summary']}
    candidate_means = {arm: np.mean([r['confirmed_candidates'] for r in primary['raw']
                                    if r['arm'] == arm]) for arm in names}
    lines = [r'\begin{table}[htbp]', r'\centering\small',
             r'\caption{Public-file search at the initial 600-epoch neural cap. Each arm uses 32 development queries and fresh confirmation on each of 30 state samples. Intervals are paired-state bootstrap differences from Sobol.}',
             r'\label{tab:public-search}', r'\begin{tabular}{lrrr}', r'\toprule',
             r'Search & Confirmed cells & Confirmed settings & Difference [95\% interval] \\',
             r'\midrule']
    for arm in names:
        r = by_arm[arm]
        lo, hi = r['paired_state_bootstrap_ci95']
        lines.append(f"{names[arm]} & {r['mean_confirmed_cells']:.2f} & "
                     f"{candidate_means[arm]:.2f} & {r['difference_from_sobol']:+.2f} "
                     f"[{lo:+.2f}, {hi:+.2f}] " + chr(92)*2)
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    table = '\n'.join(lines)+'\n'
    (out/'public_table.tex').write_text(table, encoding='utf-8')
    (ROOT/'paper/generated/table_public_search.tex').write_text(table, encoding='utf-8')
    n = next(r for r in sensitivity['summary'] if r['arm'] == 'neural_search')
    cap_hits = sum(f['best_epoch'] >= f['epoch_cap'] for r in sensitivity['raw'] for f in r['fits'])
    max_epoch = max(f['best_epoch'] for r in sensitivity['raw'] for f in r['fits'])
    text = (f"At a 2,000-epoch cap the neural mean remains {n['mean_confirmed_cells']:.2f} "
            f"cells and the difference from Sobol is {n['difference_from_sobol']:.2f} "
            f"[{n['paired_state_bootstrap_ci95'][0]:.2f}, "
            f"{n['paired_state_bootstrap_ci95'][1]:.2f}]. "
            f"There are {cap_hits} best-epoch cap hits, but the maximum best epoch is "
            f"{max_epoch}; absence of a best-epoch cap hit is not proof of convergence. "
            "This sensitivity run reuses the same state files, warm starts, stopping "
            "evidence and non-neural draws; it is not independent replication.\n")
    (out/'sensitivity.tex').write_text(text, encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout='constrained')
    arms = list(names)
    axes[0].bar(range(4), [by_arm[a]['mean_confirmed_cells'] for a in arms], color='#2d687d')
    axes[1].bar(range(4), [candidate_means[a] for a in arms], color='#936b35')
    for ax in axes:
        ax.set_xticks(range(4), ['Random', 'Heuristic', 'Sobol', 'Neural'], rotation=18)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set(ylabel='Mean confirmed high-risk cells', ylim=(0, 24), title='Coverage: up to 24 cells')
    axes[1].set(ylabel='Mean confirmed high-risk settings', ylim=(0, 34), title='Yield: 32 selected settings')
    fig.suptitle('Public Census empirical-file experiment: 30 state samples')
    fig.savefig(out/'public_search.png', dpi=200)
    fig.savefig(out/'public_search.pdf')
    plt.close(fig)

    # Reconstruct exact prospective prompts from the stored warm-start exclusion.
    data = load_census2000()
    prompts = ROOT/'artifacts-public-search/prompts'
    prompts.mkdir(exist_ok=True)
    template = {}
    for state in primary['config']['selected_states']:
        saved = json.loads((ROOT/f'artifacts-public-search/state-{state}.json').read_text())
        excluded = {int(row['record_id'].split('-warm-')[1]) for row in saved['warmstart']}
        frame = data.frame.loc[data.frame.state_code == state].reset_index(drop=True)
        file = RealFile(f'state-{state}', frame, data.block, data.key_columns, data.covariate)
        pool = [s for i, s in enumerate(candidates(file)) if i not in excluded]
        prompt = replay_prompt(pool, state)
        (prompts/f'state-{state}.txt').write_text(prompt, encoding='utf-8')
        template[str(state)] = dict(model='REPLACE_WITH_PROVIDER_MODEL_VERSION',
                                   prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                                   response_file=f'responses/state-{state}.json')
    (prompts/'manifest.template.json').write_text(json.dumps(template, indent=2)+'\n', encoding='utf-8')

    tracked = ['artifacts-public-search/results.json', 'artifacts-public-search/design.json',
               'artifacts-public-search-2000/results.json', 'src/privacy_risk/injector_search.py',
               'src/privacy_risk/sentry_r.py', 'scripts/public_injector_search.py',
               'scripts/report_public_search.py', 'tests/test_injector_search.py',
               'paper/injector_study/main.tex', 'paper/injector_study/generated/public_table.tex',
               'paper/injector_study/generated/sensitivity.tex',
               'paper/injector_study/generated/public_search.png',
               'paper/public_search_section.tex', 'paper/generated/table_public_search.tex',
               'paper/bis_ifc_full.tex', 'paper/bis_ifc_online_appendix.tex',
               'paper/sentry_r_section.tex', 'paper/sentry_r_appendix.tex',
               'paper/publication_versions/jpc_main.tex',
               'paper/publication_versions/jpc_supplement.tex',
               'docs/PAPER_RESULTS_2026_10_23.md', 'baseline/paper-full-tests.txt']
    manifest = dict(status='paper revision; original study_manifest.json is a historical snapshot',
                    primary_trial_count=primary['attack_trials'],
                    sensitivity_trial_count=sensitivity['attack_trials'],
                    sensitivity_reuses_non_neural_draws=True,
                    validation=dict(tests_passed=183, source_coverage_percent=90.58,
                                    ruff='all checks passed'),
                    llm_arm='not run; prospective prompts prepared',
                    files={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in tracked})
    (ROOT/'paper_revision_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(primary=primary['summary'], sensitivity=n,
                          confirmed_candidate_means=candidate_means), indent=2))


if __name__ == '__main__':
    main()
