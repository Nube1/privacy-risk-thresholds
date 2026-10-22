# Multiscale neural proposals and shifted audit bounds

The complete derivations are in `paper/multiscale_extension.tex`, integrated into the full manuscript and the journal supplement. `privacy_risk.multiscale` adds the following computational contracts:

| Function | Mathematical object | What it establishes |
|---|---|---|
| `MultiscaleFeatures.transform` | finite coordinatewise Mexican-hat dictionary | features on a declared input box |
| `MultiscaleFeatures.jacobian` | raw-unit derivative of the dictionary | analytic feature sensitivity |
| `proposal_sensitivity` | chain-rule Jacobian and spectral norm bound | sensitivity of the fitted predictor only |
| `shifted_audit` | exact binomial upper bounds plus external TV radii | simultaneous conditional bounds for a fixed family |
| `zero_success_plan` | continuous cost minimizer and upward-rounded counts | planning budget if zero attacks succeed |

The existing `DeepRiskNet` supplies depth-configurable counted-likelihood training; the existing SENTRY-R pipeline retains review and provenance for LLM scenario proposals. No model output, Lipschitz bound or LLM assertion can substitute for measured final audit outcomes. The new code does not claim orthogonal wavelet interpolation or a bound on true attack risk from neural derivatives.

Install and run:

```bash
python -m pip install -e '.[dev]'
python scripts/multiscale_demo.py --output artifacts-multiscale/demo.json
pytest
```

The frozen synthetic benchmark uses seed 30926, five raw features, two harm coordinates, 400 training rows, 150 validation rows and 150 test rows. Each coordinate has fifty development trials per row. Network depth and representation are chosen before test results are inspected. The three held-out counted log losses are 0.24522193 (linear), 0.22520334 (deep raw inputs), and 0.22473617 (deep multiscale). They are one-seed results under an intentionally nonlinear synthetic law; they do not establish real-world superiority or statistical significance.

The example's zero-success audit is an algebraic illustration, not sampled evidence. TV radii are hypothetical externally supplied bounds. A full deployment claim would need a justified threat law and shift radius, independent held-out attack sampling and a predeclared fixed sample size. For optional stopping, use the existing sequential audit rather than repeatedly inspecting `shifted_audit`.

Sensitivity output has shape observations × risk coordinates × raw features. The all-coordinate global bound is conservative and can grow rapidly with finer scales and deeper networks. No coordinate monotonicity is imposed, since the existing schema does not declare which raw features must monotonically increase risk.
