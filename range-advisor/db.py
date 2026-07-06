"""SQLite persistence for analyst verdicts.

One table. Scores are computed on demand by scorer.py, never stored, so the
schema stays trivial and there is no cache to invalidate.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

DB_PATH = os.getenv("ADVISOR_DB", "advisor.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS verdicts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,   -- ISO-8601 UTC
    model         TEXT    NOT NULL,
    price_at_call REAL,
    position_json TEXT    NOT NULL,
    market_json   TEXT    NOT NULL,
    verdict_json  TEXT    NOT NULL
);
"""


def _connect(path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: str | None = None) -> None:
    with _connect(path) as conn:
        conn.executescript(_SCHEMA)


def insert_verdict(
    model: str,
    price_at_call: float | None,
    position: dict[str, Any],
    market: dict[str, Any],
    verdict: dict[str, Any],
    path: str | None = None,
    created_at: str | None = None,
) -> int:
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    with _connect(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO verdicts
                (created_at, model, price_at_call,
                 position_json, market_json, verdict_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                model,
                price_at_call,
                json.dumps(position),
                json.dumps(market),
                json.dumps(verdict),
            ),
        )
        return int(cur.lastrowid)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "model": row["model"],
        "price_at_call": row["price_at_call"],
        "position": json.loads(row["position_json"]),
        "market": json.loads(row["market_json"]),
        "verdict": json.loads(row["verdict_json"]),
    }


def get_verdicts(limit: int = 50, path: str | None = None) -> list[dict[str, Any]]:
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM verdicts ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_all_verdicts(path: str | None = None) -> list[dict[str, Any]]:
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM verdicts ORDER BY id DESC"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]
