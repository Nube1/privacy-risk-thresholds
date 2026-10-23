"""Small, bounded OpenAI proposal collection; never invent a model response.

The default plan makes at most one request for each of two states. A reservation
is saved before each request; interrupted and failed requests are never silently
retried or refunded. This is a serial research runner, not a billing service.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from .injector_search import Scenario, replay_selection

MODEL = 'gpt-5.4-mini-2026-03-17'
MAX_OUTPUT_TOKENS = 8192
INPUT_USD_PER_MILLION = .75
OUTPUT_USD_PER_MILLION = 4.5


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def request_payload(prompt):
    return dict(model=MODEL, input=prompt, max_output_tokens=MAX_OUTPUT_TOKENS,
                reasoning=dict(effort='none'), store=False)


def maximum_request_cost(payload):
    # Conservative byte-based input allowance includes serialized request and
    # 4096 extra tokens for protocol overhead. No tools or image charges apply.
    input_allowance = len(json.dumps(payload, ensure_ascii=False).encode('utf-8'))+4096
    return (input_allowance*INPUT_USD_PER_MILLION +
            MAX_OUTPUT_TOKENS*OUTPUT_USD_PER_MILLION)/1_000_000


def output_text(response):
    if response.get('status') != 'completed':
        raise ValueError('provider response was incomplete or unsuccessful')
    pieces = []
    for item in response.get('output', []):
        for content in item.get('content', []):
            if content.get('type') == 'refusal':
                raise ValueError('provider refused the proposal request')
            if content.get('type') == 'output_text':
                pieces.append(content['text'])
    if not pieces:
        raise ValueError('provider returned no proposal text')
    return ''.join(pieces)


def invoke(payload, key):
    data = json.dumps(payload).encode('utf-8')
    request = Request('https://api.openai.com/v1/responses', data=data,
                      headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=120) as handle:
            raw = handle.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError('provider envelope exceeded the pilot size limit')
            return json.loads(raw), handle.headers.get('x-request-id')
    except HTTPError as error:
        raise RuntimeError(f'OpenAI request failed with HTTP {error.code}; no automatic retry') from None
    except URLError:
        raise RuntimeError('OpenAI connection failed; no automatic retry') from None


def catalogue_from_prompt(prompt):
    encoded = prompt.split('executable catalogue: ', 1)[1].split(
        '. Supply no counts, predicted risks, or executable code.', 1)[0]
    return [Scenario(tuple(action), {}, np.array([], dtype=int), np.array([]))
            for action in json.loads(encoded)]


def collect(prompts: Path, output: Path, *, budget_usd=.25, prepare_only=False, call=invoke):
    if not 0 < budget_usd <= .25:
        raise ValueError('this small pilot permits a maximum total reservation of $0.25')
    states = (0, 1)
    output.mkdir(parents=True, exist_ok=True)
    (output/'responses').mkdir(exist_ok=True)
    plan = dict(model=MODEL, states=list(states), requests_per_state=1, automatic_retries=0,
                proposals_per_request=32, maximum_output_tokens=MAX_OUTPUT_TOKENS,
                reasoning_effort='none', budget_usd=budget_usd,
                input_usd_per_million=INPUT_USD_PER_MILLION,
                output_usd_per_million=OUTPUT_USD_PER_MILLION,
                price_source='https://developers.openai.com/api/docs/models/gpt-5.4-mini',
                status='prepared; not an experimental result')
    prompt_texts = {state: (prompts/f'state-{state}.txt').read_text(encoding='utf-8') for state in states}
    plan['prompt_sha256'] = {str(state): sha(prompt) for state, prompt in prompt_texts.items()}
    plan['maximum_planned_cost_usd'] = sum(maximum_request_cost(request_payload(p))
                                           for p in prompt_texts.values())
    if plan['maximum_planned_cost_usd'] > budget_usd:
        raise ValueError('the complete two-request plan exceeds the pilot budget')
    plan_path = output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('existing pilot plan differs; preserve this directory and review the plan')
    plan_path.write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
    if prepare_only:
        return dict(status='prepared_only', paid_requests=0, plan=plan)
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not key:
        return dict(status='needs_api_key', paid_requests=0, plan=plan)

    manifest = {}
    for state in states:
        reservation_path = output/f'state-{state}.reservation.json'
        if reservation_path.exists():
            raise ValueError('pilot already contains an attempted request; no automatic retry')
    for state, prompt in prompt_texts.items():
        payload = request_payload(prompt)
        cost = maximum_request_cost(payload)
        spent = sum(json.loads(p.read_text())['reserved_usd']
                    for p in output.glob('*.reservation.json'))
        if spent+cost > budget_usd:
            raise ValueError('remaining pilot budget is insufficient')
        reservation = dict(state=state, reserved_usd=cost, model=MODEL,
                           prompt_sha256=sha(prompt),
                           recorded_utc=datetime.now(timezone.utc).isoformat())
        # Exclusive creation prevents an accidental duplicate of this state.
        with (output/f'state-{state}.reservation.json').open('x', encoding='utf-8') as handle:
            json.dump(reservation, handle, indent=2)
        (output/f'state-{state}.request.json').write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
        try:
            response, request_id = call(payload, key)
            (output/f'state-{state}.provider.json').write_text(json.dumps(response, indent=2)+'\n', encoding='utf-8')
            text = output_text(response)
            path = output/'responses'/f'state-{state}.json'
            path.write_text(text, encoding='utf-8')
            entry = dict(model=MODEL, prompt_sha256=sha(prompt),
                         response_file=f'responses/state-{state}.json',
                         provider_request_id=request_id, provider_response_id=response.get('id'),
                         returned_model=response.get('model'), usage=response.get('usage'))
            replay_selection(catalogue_from_prompt(prompt), state, entry, output)
            manifest[str(state)] = entry
            (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
        except (RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            # Error descriptions are controlled here; never serialize a key or headers.
            status = dict(state=state, status='failed_or_invalid',
                          error_type=type(error).__name__, error=str(error), automatic_retry=False,
                          reserved_usd=cost, prior_valid_states=list(manifest))
            (output/'status.json').write_text(json.dumps(status, indent=2)+'\n', encoding='utf-8')
            return status
    status = dict(status='collected_and_validated', paid_requests=2,
                  valid_states=list(manifest), total_reserved_usd=sum(
                      json.loads(p.read_text())['reserved_usd'] for p in output.glob('*.reservation.json')))
    (output/'status.json').write_text(json.dumps(status, indent=2)+'\n', encoding='utf-8')
    return status
