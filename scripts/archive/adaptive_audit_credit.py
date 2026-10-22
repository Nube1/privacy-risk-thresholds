"""Run the registered sequential gate on the synthetic credit register.

Each protection policy gets two coordinates: identification in one release, and
identification pooled over the vintages. The pooled coordinate is registered only
after the single-release coordinate of the same policy has been audited, and only
if that audit certified it, which is the adaptive registration the gate allows. Each
coordinate draws its own fresh targets from its own random stream, and the bet on
each draw is a clipped running estimate of the success rate computed from earlier
draws of that coordinate only.

For comparison the script also applies a fixed-sample rule to the same streams: a
family of every coordinate frozen in advance, a fixed number of attacks each, and a
Bonferroni-corrected Clopper-Pearson upper bound.

    python scripts/adaptive_audit_credit.py --output paper/publication_versions/credit_audit.json

The register is synthetic. The output measures the gate on this register, not on
confidential data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from privacy_risk.adaptive_audit import AdaptiveAudit
from privacy_risk.credit import make_credit_register
from privacy_risk.evaluation import Policy, identification_attack, released_rows
from privacy_risk.uncertainty import binomial_upper

POLICIES = [("none", "no suppression", 1), ("k_anonymity_5", "$k$-anonymity, $k=5$", 5),
            ("k_anonymity_10", "$k$-anonymity, $k=10$", 10),
            ("k_anonymity_40", "$k$-anonymity, $k=40$", 40),
            ("k_anonymity_50", "$k$-anonymity, $k=50$", 50)]


def bet(successes: int, trials: int, tolerance: float) -> float:
    """A predictable alternative: the shrunk running rate, kept below the tolerance."""
    estimate = (successes + 0.5) / (trials + 1.0)
    return float(min(estimate, 0.8 * tolerance))


def run(config: dict, tolerance: float, alpha: float, cap: int, fixed: int,
        seed: int) -> dict:
    register = make_credit_register(config)
    covariate_sd = float(config.get("credit", {}).get("covariate_sd", 0.6))
    slots = 2 * len(POLICIES)
    weight = 1.0 / slots
    gate = AdaptiveAudit(alpha)
    rows = []
    streams: dict[str, list[int]] = {}
    for index, (name, label, k) in enumerate(POLICIES):
        policy = Policy(name, "microdata", k)
        entry = {"policy": name, "label": label,
                 "released": int(released_rows(register, policy, 0).size)}
        for offset, pooled in enumerate((False, True)):
            coordinate_name = f"{name}:{'pooled' if pooled else 'single'}"
            stream = np.random.default_rng(seed + 100 * index + offset)
            outcome = identification_attack(register, policy, covariate_sd, cap, stream,
                                            pooled=pooled, return_hits=True)
            hits = outcome.detail["hits"]
            streams[coordinate_name] = hits
            key = "pooled" if pooled else "single"
            if pooled and not entry["single"]["certified"]:
                entry[key] = {"registered": False}
                continue
            coord = gate.register(coordinate_name, tolerance, weight)
            for hit in hits:
                if coord.observe(int(hit), bet(coord.successes, coord.trials, tolerance)):
                    break
            entry[key] = {"registered": True, "certified": coord.certified,
                          "trials": coord.trials, "successes": coord.successes,
                          "rate_at_stop": coord.successes / max(coord.trials, 1),
                          "rate_full_stream": float(np.mean(hits))}
        rows.append(entry)

    fixed_rows = []
    for name, label, _ in POLICIES:
        record = {"policy": name, "label": label}
        for key in ("single", "pooled"):
            hits = streams[f"{name}:{key}"][:fixed]
            upper = float(binomial_upper(int(sum(hits)), len(hits), alpha / slots))
            record[key] = {"trials": len(hits), "successes": int(sum(hits)),
                           "upper": upper, "certified": upper <= tolerance}
        record["eligible"] = record["single"]["certified"] and record["pooled"]["certified"]
        fixed_rows.append(record)

    for entry in rows:
        entry["eligible"] = bool(entry["single"].get("certified")
                                 and entry["pooled"].get("certified"))
    return {"tolerance": tolerance, "alpha": alpha, "weight_per_slot": weight,
            "slots": slots, "cap": cap, "fixed_trials": fixed, "seed": seed,
            "threshold_log_e": float(-np.log(alpha * weight)),
            "registered": len(gate.coordinates), "weight_spent": gate.spent,
            "sequential": rows, "fixed_sample": fixed_rows,
            "note": "Synthetic register; each coordinate uses its own fresh target stream."}


def summarise(runs: list[dict]) -> dict:
    """Share of seeds in which each coordinate certifies, and its median attacks."""
    summary = {}
    for index, (name, _, _) in enumerate(POLICIES):
        entry = {"seeds": len(runs)}
        for key in ("single", "pooled"):
            cells = [r["sequential"][index][key] for r in runs]
            done = [c for c in cells if c.get("registered")]
            entry[key] = {
                "registered_share": len(done) / len(runs),
                "certified_share": sum(bool(c.get("certified")) for c in done) / len(runs),
                "median_trials": float(np.median([c["trials"] for c in done])) if done
                else None}
        fixed = [r["fixed_sample"][index]["eligible"] for r in runs]
        entry["fixed_eligible_share"] = sum(fixed) / len(runs)
        entry["sequential_eligible_share"] = sum(
            bool(r["sequential"][index]["eligible"]) for r in runs) / len(runs)
        summary[name] = entry
    return summary


def _th(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def table(result: dict) -> str:
    """Decisions over the replicated seeds, with the attacks each coordinate used."""
    summary = result["replications"]
    seeds = next(iter(summary.values()))["seeds"]
    body = ""
    for seq in result["sequential"]:
        entry = summary[seq["policy"]]
        single, pooled = entry["single"], entry["pooled"]
        body += (f"{seq['label']} & {_th(seq['released'])} & "
                 f"{single['certified_share']:.2f} & {_th(int(single['median_trials']))} & "
                 f"{pooled['certified_share']:.2f} & {_th(int(pooled['median_trials']))} & "
                 f"{entry['fixed_eligible_share']:.2f} \\\\\n")
    caption = (
        "The registered gate on the synthetic credit register at tolerance "
        f"{result['tolerance']:.2f} and $\\alpha={result['alpha']:g}$, over {seeds} "
        "seeds. Each policy has a single-release and a pooled coordinate, each with "
        f"weight $1/{result['slots']}$; the pooled coordinate is registered only after "
        "the single-release coordinate certifies. Each coordinate draws its own fresh "
        f"targets, at most {_th(result['cap'])}. Share is the fraction of seeds in which "
        "the coordinate certifies; attacks is the median number used. The last column "
        "is the fraction of seeds in which a family frozen in advance, with "
        f"{_th(result['fixed_trials'])} attacks per coordinate and Bonferroni-corrected "
        "Clopper--Pearson bounds, certifies both coordinates.")
    return ("\\begin{table}[t]\n\\centering\n\\caption{" + caption + "}\n"
            "\\label{tab:credit-audit}\n\\small\n\\begin{tabular}{lrrrrrr}\n\\toprule\n"
            " & & \\multicolumn{2}{c}{Single release} & \\multicolumn{2}{c}{Pooled} & "
            "Fixed \\\\\n"
            "Policy & Kept & Share & Attacks & Share & Attacks & Share \\\\\n"
            "\\midrule\n" + body + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--output", default="paper/publication_versions/credit_audit.json")
    parser.add_argument("--table", default="paper/publication_versions/table_credit_audit.tex")
    parser.add_argument("--tolerance", type=float, default=0.12)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--cap", type=int, default=3000)
    parser.add_argument("--fixed", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--replications", type=int, default=20,
                        help="further seeds for the summary of decisions and trials")
    arguments = parser.parse_args()
    config = json.loads(Path(arguments.config).read_text())
    result = run(config, arguments.tolerance, arguments.alpha, arguments.cap,
                 arguments.fixed, arguments.seed)
    result["replications"] = summarise([
        run(config, arguments.tolerance, arguments.alpha, arguments.cap, arguments.fixed,
            arguments.seed + 10_000 * (r + 1)) for r in range(arguments.replications)])
    Path(arguments.output).write_text(json.dumps(result, indent=2) + "\n")
    Path(arguments.table).write_text(table(result))
    print(json.dumps({row["policy"]: {"single": row["single"], "pooled": row["pooled"]}
                      for row in result["sequential"]}, indent=1))


if __name__ == "__main__":
    main()
