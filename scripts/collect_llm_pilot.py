"""Collect one genuine response per state for a two-state, $0.25-capped pilot."""
import argparse
import json
import os
from pathlib import Path

from privacy_risk.llm_pilot import collect


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompts', type=Path, default=Path('artifacts-public-search/prompts'))
    parser.add_argument('--output', type=Path, default=Path('artifacts-llm-pilot'))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--env-file', type=Path, default=Path('.env.local'))
    args = parser.parse_args()
    if not args.prepare_only and not os.environ.get('OPENAI_API_KEY') and args.env_file.exists():
        for line in args.env_file.read_text(encoding='utf-8-sig').splitlines():
            name, separator, value = line.partition('=')
            if separator and name.strip() == 'OPENAI_API_KEY':
                os.environ['OPENAI_API_KEY'] = value.strip().strip('"').strip("'")
                break
    result = collect(args.prompts, args.output, prepare_only=args.prepare_only)
    print(json.dumps(result, indent=2))
    if result['status'] not in ('prepared_only', 'collected_and_validated'):
        raise SystemExit(2)
