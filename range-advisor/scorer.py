"""Grade past verdicts against what price actually did.

The pure scoring math (`score_closes`) takes a verdict row plus a list of
realized 1h closes so it can be unit-tested without any network. The network
piece (`fetch_closes_since`) is a thin Binance wrapper.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import db
import market

# Only grade verdicts that have had at least this long to play out.
MIN_AGE_HOURS = 24
# ADX threshold for the "dumb baseline": ADX > 25 at call time == trending.
ADX_TREND_THRESHOLD = 25.0


def _parse_iso(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def fetch_closes_since(
    symbol: str, start_ms: int, interval: str = "1h", limit: int = 1000
) -> list[float]:
    """Fetch 1h close prices from Binance at/after start_ms (epoch millis)."""
    rows = market.fetch_raw_klines(
        {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "limit": limit,
        }
    )
    return [float(row[4]) for row in rows]


def score_closes(
    price_at_call: float | None,
    suggested_low: float,
    suggested_high: float,
    verdict_regime: str,
    adx_at_call: float | None,
    closes: list[float],
) -> dict[str, Any]:
    """Pure scoring math over a realized close series. No network."""
    if not closes:
        return {
            "in_suggested_range_pct": None,
            "still_in_range_now": None,
            "max_adverse_move_pct": None,
            "regime_matched_baseline": None,
            "n_closes": 0,
        }

    inside = [1 for c in closes if suggested_low <= c <= suggested_high]
    in_range_pct = len(inside) / len(closes) * 100.0

    still_in_range_now = suggested_low <= closes[-1] <= suggested_high

    # Max adverse move: furthest excursion *outside* the suggested range,
    # expressed as % of price at call time (0 if it never left the range).
    ref = price_at_call if price_at_call else closes[0]
    worst = 0.0
    for c in closes:
        if c > suggested_high:
            excursion = (c - suggested_high) / ref * 100.0
        elif c < suggested_low:
            excursion = (suggested_low - c) / ref * 100.0
        else:
            excursion = 0.0
        worst = max(worst, excursion)

    # Dumb baseline: was the market actually trending (ADX>25) at call time?
    baseline_matched = None
    if adx_at_call is not None:
        baseline_trending = adx_at_call > ADX_TREND_THRESHOLD
        verdict_trending = verdict_regime in ("trending_up", "trending_down")
        baseline_matched = baseline_trending == verdict_trending

    return {
        "in_suggested_range_pct": round(in_range_pct, 2),
        "still_in_range_now": still_in_range_now,
        "max_adverse_move_pct": round(worst, 2),
        "regime_matched_baseline": baseline_matched,
        "n_closes": len(closes),
    }


def score_verdict_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Score one stored verdict row. Returns None if too young to grade."""
    created = _parse_iso(row["created_at"])
    age_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600.0
    if age_hours < MIN_AGE_HOURS:
        return None

    verdict = row["verdict"]
    market = row.get("market", {}) or {}
    low = float(verdict["suggested_range_low"])
    high = float(verdict["suggested_range_high"])

    symbol = market.get("symbol", "ETHUSDT")
    start_ms = int(created.timestamp() * 1000)
    try:
        closes = fetch_closes_since(symbol, start_ms)
    except Exception as exc:  # keep scoring resilient to a bad fetch
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "model": row["model"],
            "error": f"could not fetch closes: {exc}",
        }

    scores = score_closes(
        price_at_call=row.get("price_at_call"),
        suggested_low=low,
        suggested_high=high,
        verdict_regime=verdict["regime"],
        adx_at_call=market.get("adx_14"),
        closes=closes,
    )
    scores.update(
        {
            "id": row["id"],
            "created_at": row["created_at"],
            "model": row["model"],
            "regime": verdict["regime"],
            "action": verdict["action"],
            "age_hours": round(age_hours, 1),
        }
    )
    return scores


def _aggregate(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-model aggregates so model slugs can be compared over time."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in scored:
        if "error" in s or s.get("in_suggested_range_pct") is None:
            continue
        buckets[s["model"]].append(s)

    agg: dict[str, Any] = {}
    for model, items in buckets.items():
        n = len(items)
        if n == 0:
            continue
        held_now = sum(1 for i in items if i["still_in_range_now"])
        baseline_hits = [
            i["regime_matched_baseline"]
            for i in items
            if i["regime_matched_baseline"] is not None
        ]
        agg[model] = {
            "n_scored": n,
            "avg_in_suggested_range_pct": round(
                sum(i["in_suggested_range_pct"] for i in items) / n, 2
            ),
            "pct_still_in_range_now": round(held_now / n * 100.0, 2),
            "avg_max_adverse_move_pct": round(
                sum(i["max_adverse_move_pct"] for i in items) / n, 2
            ),
            "baseline_match_rate_pct": (
                round(sum(baseline_hits) / len(baseline_hits) * 100.0, 2)
                if baseline_hits
                else None
            ),
        }
    return agg


def score_all(path: str | None = None) -> dict[str, Any]:
    """Score every gradable verdict and return per-verdict + per-model views."""
    rows = db.get_all_verdicts(path)
    scored: list[dict[str, Any]] = []
    for row in rows:
        result = score_verdict_row(row)
        if result is not None:
            scored.append(result)
    return {"verdicts": scored, "by_model": _aggregate(scored)}
