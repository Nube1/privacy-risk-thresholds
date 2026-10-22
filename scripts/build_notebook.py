"""Build the Colab notebook that walks through the paper's argument."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("""# Quantitative privacy risk: proof, test and governance

This Colab accompanies Ronald Rethabile Malamule Mahomane's paper. It uses synthetic
data only. A green check establishes that the code implements the stated mathematics. It
does **not** establish that the adversary specification is realistic or that the policy
tolerance is right. Those are empirical and policy claims, and the last section lists what
would refute them."""),
    code("""# In Colab: clone or upload the repository, then run this cell from its root.
import subprocess
import sys

if 'google.colab' in sys.modules:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.'], check=True)
print(sys.version)"""),

    md("""## 1. A finite population, a sample and a key

The register carries common key classes and rare ones. The confidential variable is used
to measure utility, and it is also what gives the adversary something to link on beyond
the key."""),
    code("""import json
from pathlib import Path

import numpy as np

from privacy_risk.experiments import (
    conventional_measures, error_model, k_anonymity, key_structure, mechanism_grid,
    programme, programme_budget, certification, formal_privacy,
)
from privacy_risk.linkage import ErrorModel, attack
from privacy_risk.risk import prosecutor_risk, marketer_sample_risk
from privacy_risk.synthetic import identity_release, make_register
from privacy_risk.uncertainty import binomial_upper, certification_replicates

config = json.loads(Path('configs/base.json').read_text())
register = make_register(config)
release = identity_release(register)
key_structure(register)"""),

    md("""## 2. The reciprocal-frequency identity is a theorem, not a definition

For a key class of size $f_k$, equal posterior weights give $1/f_k$. Each occupied class
contributes one to the sum of record risks, so the mean equals the number of occupied
classes divided by $n$. Both statements are algebra, and the test below is an
implementation check."""),
    code("""risks = prosecutor_risk(release.key)
assert np.isclose(risks.mean(), marketer_sample_risk(release.key))
{'mean reciprocal frequency': risks.mean(),
 'distinct keys / n': marketer_sample_risk(release.key)}"""),

    md("""## 3. The conventional measures answer different experiments

The sample-frequency measure, the population-frequency plug-in and the unconditional
measure differ by orders of magnitude on the same release. Any one of them is only as
meaningful as the attack population and conditioning event printed beside it."""),
    code("""measures = conventional_measures(register, config)
{k: measures[k] for k in ('prosecutor_mean', 'plugin_population_frequency',
                          'marketer_unconditional', 'pi_times_plugin')} | {
 'ordering': measures['plugin_ordering']}"""),

    md("""## 4. Measurement error does not simply reduce risk

The declared adversary misreads the age band and the occupation group, which spreads
posterior mass, and holds one noisy reading of the confidential variable, which
concentrates it. Reporting only the first would understate the risk."""),
    code("""model = error_model(config)
key_only = attack(register, release, model, 400, np.random.default_rng(1),
                  use_covariate=False)
informed = attack(register, release, model, 800, np.random.default_rng(2))
{'exact matching': float(prosecutor_risk(release.key).mean()),
 'key error only': key_only.rate,
 'with one covariate': informed.rate,
 'upper bound': binomial_upper(informed.successes, informed.trials, config['alpha'])}"""),

    md("""## 5. Zero observed attacks do not prove zero risk

With no success in $B$ independent replicates the one-sided bound is $1-\\alpha^{1/B}$,
about $3/B$ at 95 per cent. Certifying a small tolerance is therefore a simulation budget,
and choosing the tolerance and the evidence standard is one decision."""),
    code("""[{'tolerance': t, 'replicates at 95%': certification_replicates(t, 0.05)}
 for t in (0.05, 0.01, 1e-3, 1e-4, 1e-5)]"""),

    md("""## 6. $k$-anonymity does not deliver $1/k$

Two controls are reported: the average risk for a target drawn uniformly from the release,
and the risk of the worst target among those in the smallest cells. The second is the
supremum the governance rule is written with, and it exceeds $1/k$."""),
    code("""rows = k_anonymity(register, config)
[{k: row[k] for k in ('k', 'bound', 'average', 'average_upper', 'worst', 'worst_lower',
                      'share_lower_above_bound')} for row in rows]"""),

    md("""## 7. The frontier, and the tolerance a relative harm weight implies

Every setting is plotted at its one-sided bound, not its point estimate. Once the
institution says how it values public benefit against harm, the tolerance follows: it is
the risk at the setting that maximises value."""),
    code("""import matplotlib.pyplot as plt

grid = mechanism_grid(register, config)
risk = np.array([s['risk_upper'] for s in grid['settings']])
utility = np.array([s['utility'] for s in grid['settings']])
front = np.array(grid['frontier'], dtype=int)
plt.scatter(risk, utility, s=10, facecolor='none', edgecolor='#B8BEC6')
plt.plot(risk[front], utility[front], 'o-', color='#1A1A1A', ms=3)
plt.axvline(config['risk_tolerance'], ls='--', color='#7B2D26')
plt.xlabel('upper bound on scenario risk'); plt.ylabel('utility')
plt.title('Risk--utility feasible set'); plt.show()
grid['implied_tolerances']"""),

    md("""## 8. Marginal risks do not bound a pooled attack

Four releases are built from one sample, each with a minimum cell count of three, so none
contains a unique record. An adversary that intersects their candidate sets does far
better than the bound the marginals imply. This is the result that makes a programme, not
a release, the object that needs governing."""),
    code("""result = programme(register, config)
lower, upper = result['marginal_lower'], result['union_upper']
{'marginal risks': result['marginal'],
 'bounds implied by the marginals': (lower, upper),
 'independence counterfactual': result['independence'],
 'pooled attack by m': result['pooled'],
 'pooled uniqueness by m': result['unique_share'],
 'pooled beats the union bound': result['pooled'][-1] > upper}"""),

    md("""## 9. A programme tolerance is allocated by value

Under a concave value curve the first-order conditions equalise marginal value per unit of
risk. For the fitted power family the shares are the value weights raised to
$1/(1-\\gamma)$, which is flatter than proportional allocation."""),
    code("""programme_budget(config, grid['fit'])"""),

    md("""## 10. A privacy budget is a different regime

An $\\varepsilon$ bounds how far posterior odds can move. It is not a probability that a
person was identified, and the probability it permits depends on the prior the adversary
brings."""),
    code("""formal_privacy(config)['posterior_bounds']"""),

    md("""## 11. The governance gate

The gate applies the prespecified tolerance to the bound, records the owner, the expiry
date and the adversary, and refuses to convert a failed validation into a release
decision. `PASS`, `FAIL` and `INVALID` are three different answers."""),
    code("""from privacy_risk.pipeline import run

results = run(config, Path('artifacts'))
print(json.dumps(results['decision'], indent=2, default=float)[:1200])
results['validation']['checks']"""),

    md("""## 12. What would falsify the argument?

- Held-out linkage outcomes are not calibrated to the posterior match probabilities.
- The population-frequency model fails external or held-out frequency prediction.
- A stronger plausible adversary crosses the tolerance.
- Claimed stronger protection raises measured risk beyond the prespecified allowance.
- A joint attack crosses the programme tolerance even though every marginal release passes.
- New auxiliary registers or linkage methods move attack performance outside the range the
  assessment declared, which is what the expiry date exists to catch.

Each of these is an empirical test capable of failure. Reproducing an equation in code is
only an implementation check, and this notebook labels the two apart."""),
]

nb.cells = cells
path = Path("notebooks/privacy_risk_thresholds_colab.ipynb")
path.parent.mkdir(exist_ok=True)
nbf.write(nb, path)
print(f"wrote {path} with {len(cells)} cells")
