"""Strict intake for model-proposed scenarios and plan freezing (Chapter 22).

A proposal batch is bytes. It is refused unless it is a JSON array of 1 to 32 objects with
exactly the keys id, description and features, every feature declared, finite and in range.
Text is inert. A model can propose; it cannot hand over an outcome.
"""
from __future__ import annotations

import hashlib
import json
import math

MAX_BYTES = 256_000
MAX_ITEMS = 32
KEYS = {"id", "description", "features"}


def _no_dupes(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate JSON key")
    return dict(pairs)


def _refuse_const(name):
    raise ValueError(f"non-finite constant {name}")


def validate_proposals(raw: bytes, schema: dict) -> list[dict]:
    """``schema`` maps feature name to (type, low, high) with type int or float."""
    if len(raw) > MAX_BYTES:
        raise ValueError("batch is too large")
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_dupes, parse_constant=_refuse_const)
    except UnicodeDecodeError as e:
        raise ValueError("not valid UTF-8") from e
    if not isinstance(data, list) or not 1 <= len(data) <= MAX_ITEMS:
        raise ValueError("need a JSON array of 1 to 32 objects")
    seen = set()
    for item in data:
        if not isinstance(item, dict) or set(item) != KEYS:
            raise ValueError("each proposal must have exactly id, description and features")
        if not isinstance(item["id"], str) or not isinstance(item["description"], str):
            raise ValueError("id and description must be strings")
        if item["id"] in seen:
            raise ValueError("duplicate identifier")
        seen.add(item["id"])
        feats = item["features"]
        if not isinstance(feats, dict) or set(feats) != set(schema):
            raise ValueError("features must match the declared list exactly")
        for name, (typ, lo, hi) in schema.items():
            v = feats[name]
            if isinstance(v, bool) or not isinstance(v, typ if typ is not float else (int, float)):
                raise ValueError(f"feature {name} has the wrong type")
            if typ is int and not isinstance(v, int):
                raise ValueError(f"feature {name} must be an integer")
            if not math.isfinite(v) or not lo <= v <= hi:
                raise ValueError(f"feature {name} is outside its range")
    return data


def freeze_plan(plan: dict) -> str:
    """SHA-256 of a canonical serialisation. Detects later change; proves neither author nor time."""
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
