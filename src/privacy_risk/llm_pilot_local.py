"""Local open-weight proposal collection through Ollama; never invent a response.

The model runs on the operator's own machine, so no account, key or charge is
involved and nothing leaves the machine. The plan is the same as the hosted
pilot: one prompt per state, 32 proposals, the executable attacker supplies the
outcomes. A small model often returns a malformed list, so each state is
allowed a fixed number of regeneration attempts with distinct seeds; every
attempt, valid or not, is kept on disk and counted in the result. The manifest
written here is what ``scripts/public_injector_search.py --llm-replays`` reads.

The prompt is the one the hosted pilot saved, byte for byte. What differs is
decoding: Ollama's structured output constrains the model to the JSON shape the
prompt already asks for (an array of objects with ``proposal_id``, ``scenario``
and a ``features`` object holding the four named numbers, each drawn from its
catalogue levels). The grammar fixes the shape, not the choice: which tuples the
model picks, whether they are distinct and whether they exist in the executable
catalogue are still checked afterwards by ``replay_selection``.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .injector_search import ACTION_SCHEMA, LEVELS, replay_selection
from .llm_pilot import catalogue_from_prompt

DEFAULT_MODEL = 'qwen3:4b'
DEFAULT_HOST = 'http://127.0.0.1:11434'
MAX_OUTPUT_TOKENS = 8192
CONTEXT_TOKENS = 16384
BASE_SEED = 20261023
PROPOSALS = 32


def response_schema():
    """JSON schema for constrained decoding: the shape the prompt asks for."""
    features = {name: {"type": "number", "enum": list(levels)}
                for name, levels in zip(ACTION_SCHEMA.features, LEVELS)}
    item = {"type": "object",
            "properties": {"proposal_id": {"type": "string", "maxLength": 24},
                           "scenario": {"type": "string", "maxLength": 160},
                           "features": {"type": "object", "properties": features,
                                        "required": list(ACTION_SCHEMA.features),
                                        "additionalProperties": False}},
            "required": ["proposal_id", "scenario", "features"],
            "additionalProperties": False}
    return {"type": "array", "items": item, "minItems": PROPOSALS, "maxItems": PROPOSALS}


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _request(host, path, payload=None, timeout=900):
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    request = Request(host.rstrip('/')+path, data=data,
                      headers={'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=timeout) as handle:
            raw = handle.read(4_000_001)
            if len(raw) > 4_000_000:
                raise ValueError('local provider envelope exceeded the pilot size limit')
            return json.loads(raw)
    except HTTPError as error:
        raise RuntimeError(f'Ollama request failed with HTTP {error.code}') from None
    except URLError:
        raise RuntimeError(f'Ollama is not reachable at {host}; start it and pull the model') from None


def model_record(model, host=DEFAULT_HOST):
    """Version record for the paper: server version, model digest and parameters."""
    version = _request(host, '/api/version').get('version')
    tags = {m['name']: m for m in _request(host, '/api/tags').get('models', [])}
    found = tags.get(model) or tags.get(model+':latest')
    if found is None:
        raise RuntimeError(f'model {model!r} is not pulled; run: ollama pull {model}')
    show = _request(host, '/api/show', dict(model=model))
    details = show.get('details', {})
    return dict(model=model, digest=found.get('digest'), size_bytes=found.get('size'),
                modified_at=found.get('modified_at'), ollama_version=version,
                family=details.get('family'), parameter_size=details.get('parameter_size'),
                quantization=details.get('quantization_level'))


def request_payload(prompt, model, seed):
    return dict(model=model, stream=False, think=False, format=response_schema(),
                messages=[dict(role='user', content=prompt)],
                options=dict(temperature=.3, seed=int(seed), num_predict=MAX_OUTPUT_TOKENS,
                             num_ctx=CONTEXT_TOKENS))


def invoke(payload, host=DEFAULT_HOST):
    return _request(host, '/api/chat', payload)


def output_text(response):
    if response.get('done') is not True:
        raise ValueError('local provider response did not finish')
    text = response.get('message', {}).get('content', '')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('local provider returned no proposal text')
    stripped = text.strip()
    fenced = stripped.startswith('```')
    if fenced:
        # Remove a Markdown code fence only; the content inside is left as returned.
        stripped = stripped.split('\n', 1)[1] if '\n' in stripped else ''
        if stripped.rstrip().endswith('```'):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip(), fenced


def collect(prompts: Path, output: Path, *, states=(0, 1), model=DEFAULT_MODEL,
            host=DEFAULT_HOST, attempts=3, call=invoke, record=model_record):
    if not 1 <= attempts <= 10:
        raise ValueError('between one and ten attempts per state')
    output.mkdir(parents=True, exist_ok=True)
    (output/'responses').mkdir(exist_ok=True)
    (output/'attempts').mkdir(exist_ok=True)
    provider = record(model, host)
    prompt_texts = {int(s): (prompts/f'state-{s}.txt').read_text(encoding='utf-8')
                    for s in states}
    plan = dict(provider=provider, host=host, states=sorted(prompt_texts),
                attempts_per_state=attempts, proposals_per_request=32,
                maximum_output_tokens=MAX_OUTPUT_TOKENS, context_tokens=CONTEXT_TOKENS,
                temperature=.3, thinking=False, base_seed=BASE_SEED, cost_usd=0.,
                decoding='constrained to response_schema(); prompt unchanged',
                response_schema=response_schema(),
                prompt_sha256={str(s): sha(p) for s, p in prompt_texts.items()},
                status='local open-weight pilot; proposals only, outcomes measured separately')
    plan_path = output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('existing pilot plan differs; use a new output directory')
    plan_path.write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
    manifest, attempts_log = {}, []
    for state, prompt in prompt_texts.items():
        pool = catalogue_from_prompt(prompt)
        for attempt in range(attempts):
            seed = BASE_SEED + 100*state + attempt
            payload = request_payload(prompt, model, seed)
            started = datetime.now(timezone.utc).isoformat(timespec='seconds')
            stem = output/'attempts'/f'state-{state}-attempt-{attempt}'
            stem.with_suffix('.request.json').write_text(json.dumps(payload, indent=2)+'\n',
                                                         encoding='utf-8')
            record_row = dict(state=state, attempt=attempt, seed=seed, started_utc=started)
            try:
                response = call(payload, host)
                stem.with_suffix('.provider.json').write_text(json.dumps(response, indent=2)+'\n',
                                                              encoding='utf-8')
                text, fenced = output_text(response)
                path = output/'responses'/f'state-{state}.json'
                path.write_text(text, encoding='utf-8')
                entry = dict(model=model, prompt_sha256=sha(prompt),
                             response_file=f'responses/state-{state}.json',
                             returned_model=response.get('model'), seed=seed, attempt=attempt,
                             code_fence_removed=fenced, digest=provider['digest'],
                             usage=dict(prompt_tokens=response.get('prompt_eval_count'),
                                        output_tokens=response.get('eval_count'),
                                        total_seconds=(response.get('total_duration') or 0)/1e9),
                             done_reason=response.get('done_reason'))
                replay_selection(pool, state, entry, output)
                manifest[str(state)] = entry
                record_row.update(status='valid', response_sha256=sha(text))
                attempts_log.append(record_row)
                break
            except (RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                record_row.update(status='rejected', error_type=type(error).__name__,
                                  error=str(error))
                attempts_log.append(record_row)
                if isinstance(error, RuntimeError):
                    break  # the server is down or the model is missing; do not spin
        (output/'attempts.json').write_text(json.dumps(attempts_log, indent=2)+'\n',
                                            encoding='utf-8')
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n',
                                            encoding='utf-8')
    status = dict(status='collected_and_validated' if len(manifest) == len(prompt_texts)
                  else 'incomplete', valid_states=sorted(int(s) for s in manifest),
                  requested_states=sorted(prompt_texts),
                  attempts=len(attempts_log),
                  rejected_attempts=sum(a['status'] == 'rejected' for a in attempts_log),
                  provider=provider, cost_usd=0.)
    (output/'status.json').write_text(json.dumps(status, indent=2)+'\n', encoding='utf-8')
    return status
