"""Reproducible synthetic training, chain-rule and shifted-audit demonstration."""
import argparse
import json
from pathlib import Path
import numpy as np
from privacy_risk.multiscale import MultiscaleFeatures, proposal_sensitivity, zero_success_plan, shifted_audit
from privacy_risk.sentry_r import DeepRiskNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts-multiscale/demo.json'))
    args = parser.parse_args()
    rng = np.random.default_rng(30926)
    x = rng.uniform(-1, 1, (700, 5))
    z = -3 + 1.2*x[:, 0]*x[:, 1] + .7*np.sin(4*x[:, 2])
    p = 1/(1+np.exp(-z))
    n = np.full((700, 2), 50)
    k = rng.binomial(n, np.column_stack([p, np.minimum(1, 1.3*p)]))
    f = MultiscaleFeatures([-1]*5, [1]*5, 2)
    phi = f.transform(x)
    reports = {}
    for name, xx, depth in [('linear', x, 0), ('deep_raw', x, 2), ('deep_multiscale', phi, 2)]:
        net = DeepRiskNet(xx.shape[1], 2, width=16, depth=depth, seed=30926)
        train = net.fit((xx[:400], k[:400], n[:400]), (xx[400:550], k[400:550], n[400:550]), epochs=120, patience=20)
        reports[name] = {**train, 'held_out_counted_log_loss': net.loss_grads(xx[550:], k[550:], n[550:])[0]}
        if name == 'deep_multiscale':
            result = proposal_sensitivity(net, f, x[550:553])
            reports[name]['raw_input_jacobian'] = result['jacobian'].tolist()
            reports[name]['model_lipschitz_bound'] = result['model_lipschitz_bound']
    plan = zero_success_plan([.12, .15], [.02, .03], costs=[1, 2])
    audit = shifted_audit([0, 0], plan['trials_if_zero_successes'], [.12, .15], [.02, .03], weights=plan['weights'])
    # Zero-success counts are an algebraic illustration, not sampled final evidence.
    def clean(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o).__name__)
    output = {'seed': 30926, 'status': 'synthetic development benchmark; illustrative zero-success audit', 'models': reports, 'plan': plan, 'audit': audit}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, default=clean, indent=2)+'\n')
    print(json.dumps({name: row['held_out_counted_log_loss'] for name, row in reports.items()}))


if __name__ == '__main__':
    main()
