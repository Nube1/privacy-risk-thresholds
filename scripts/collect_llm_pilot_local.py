"""Collect genuine proposals from a local open-weight model through Ollama.

Example:
    ollama pull qwen3:4b
    python scripts/collect_llm_pilot_local.py --states 0 1
    python scripts/public_injector_search.py --states 2 \
        --output artifacts-public-search-llm-local \
        --llm-replays artifacts-llm-pilot-local/manifest.json
"""
import argparse
import json
from pathlib import Path

from privacy_risk.llm_pilot_local import DEFAULT_HOST, DEFAULT_MODEL, collect


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prompts', type=Path, default=Path('artifacts-public-search/prompts'))
    parser.add_argument('--output', type=Path, default=Path('artifacts-llm-pilot-local'))
    parser.add_argument('--states', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--host', default=DEFAULT_HOST)
    parser.add_argument('--attempts', type=int, default=3,
                        help='regenerations allowed per state when the list is malformed')
    args = parser.parse_args()
    result = collect(args.prompts, args.output, states=args.states, model=args.model,
                     host=args.host, attempts=args.attempts)
    print(json.dumps(result, indent=2))
    if result['status'] != 'collected_and_validated':
        raise SystemExit(2)
