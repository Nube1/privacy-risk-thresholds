# Small LLM pilot: blocked before proposal collection

The prospective pilot requests 32 proposals in each of two state blocks from
`gpt-5.4-mini-2026-03-17`. Prompts were saved before collection and match the
public-file benchmark catalogues. The model supplies scenarios only; the
executable linkage attacker supplies measured outcomes.

The first live request returned HTTP 429, with no proposal response. Collection
stopped before the second request and made no automatic retry. The OpenAI
Developers dashboard subsequently displayed US$0.00 credit remaining. The HTTP
error body was not retained, so its precise provider error subtype is unknown.
Funding is required before another attempt; a different rate or access limit
may also need attention afterward.

`artifacts-llm-pilot/` preserves the plan, prompt hashes, request, reservation,
and failed-attempt status. The $0.0424035 reservation is a conservative budget
allowance, **not a measured charge**. The complete two-request plan reserves at
most $0.0851745 against the authorized $0.25 cap at the recorded prices.

The API credential is local in the ignored `.env.local` file and must never be
committed. No credential or authorization header is included in these artifacts.

No genuine LLM comparison result exists yet. Do not substitute fixture replies
or treat this infrastructure failure as evidence for an experimental hypothesis.
The completed public-file baseline results remain the paper's empirical evidence.

The runner refuses to repeat a directory containing an attempted request.
Preserve this failed attempt. A deliberate follow-up must record its relationship
to this attempt and account for the existing reservation within the total budget.
