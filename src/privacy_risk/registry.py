"""The programme error budget and frozen plans of Chapter 18.

A programme opens with a total budget. A plan is registered by the hash of its
canonical JSON. Each attempt reserves its share inside a write transaction before
the audit draws a target, and an attempt that fails or is abandoned keeps its spend.
The registry detects a later edit to a stored plan by recomputing the hash. It does
not prove who wrote a row or when.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3


class BudgetExhausted(RuntimeError):
    pass


def plan_hash(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Registry:
    def __init__(self, path: str = ":memory:", total_budget: float | None = None):
        self.db = sqlite3.connect(path, isolation_level=None)
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS programme(id INTEGER PRIMARY KEY CHECK(id=1), total REAL);"
            "CREATE TABLE IF NOT EXISTS plan(hash TEXT PRIMARY KEY, body TEXT);"
            "CREATE TABLE IF NOT EXISTS attempt(id INTEGER PRIMARY KEY AUTOINCREMENT, plan TEXT,"
            " share REAL, status TEXT, seed INTEGER);")
        if total_budget is not None:
            self.db.execute("INSERT OR IGNORE INTO programme VALUES (1, ?)", (total_budget,))

    def total(self) -> float:
        return self.db.execute("SELECT total FROM programme").fetchone()[0]

    def spent(self) -> float:
        return self.db.execute("SELECT COALESCE(SUM(share),0) FROM attempt").fetchone()[0]

    def remaining(self) -> float:
        return round(self.total() - self.spent(), 12)

    def register_plan(self, plan: dict) -> str:
        h = plan_hash(plan)
        self.db.execute("INSERT OR IGNORE INTO plan VALUES (?, ?)",
                        (h, json.dumps(plan, sort_keys=True, separators=(",", ":"))))
        return h

    def reserve(self, plan: str, share: float) -> int:
        if share <= 0:
            raise ValueError("a share must be positive")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if self.db.execute("SELECT 1 FROM plan WHERE hash=?", (plan,)).fetchone() is None:
                raise KeyError("plan was not registered before the audit")
            if self.spent() + share > self.total() + 1e-12:
                raise BudgetExhausted("the programme has no budget left for this share")
            cur = self.db.execute("INSERT INTO attempt(plan,share,status,seed) VALUES (?,?,?,?)",
                                  (plan, share, "open", secrets.randbits(63)))
            self.db.execute("COMMIT")
            return cur.lastrowid
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def close(self, attempt: int, status: str) -> None:
        if status not in ("passed", "failed", "abandoned"):
            raise ValueError("status must be passed, failed or abandoned")
        self.db.execute("UPDATE attempt SET status=? WHERE id=?", (status, attempt))

    def verify_hashes(self) -> list[str]:
        """Hashes whose stored body no longer matches. An empty list means no edits."""
        return [h for h, body in self.db.execute("SELECT hash, body FROM plan")
                if plan_hash(json.loads(body)) != h]
