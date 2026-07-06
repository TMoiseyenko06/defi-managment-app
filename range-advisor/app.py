"""FastAPI app: serves the dashboard and the read-only JSON API.

Design rule: a failing external API (RPC, Binance, GeckoTerminal, OpenRouter)
must never take down the dashboard. Every endpoint catches its own errors and
returns {"error": ...} with a sensible status code; the static page always
loads and each section degrades on its own.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

import analyst
import chain
import db
import market
import scorer
from config import ConfigError, load_config, require_api_key

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="range-advisor", docs_url=None, redoc_url=None)


@app.on_event("startup")
def _startup() -> None:
    db.init_db()


def _err(message: str, status: int = 502) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/position")
def api_position() -> Any:
    try:
        cfg = load_config()
        if not cfg.has_position:
            msg = "POSITION_TOKEN_ID is not set in .env."
            if cfg.has_wallet:
                msg += " Run `python -m chain` to list your wallet's position ids."
            return _err(msg, status=400)
        return chain.build_position_snapshot(cfg)
    except ConfigError as exc:
        return _err(str(exc), status=400)
    except chain.ChainError as exc:
        return _err(f"On-chain read failed: {exc}", status=502)
    except Exception as exc:  # never 500 the dashboard
        return _err(f"Unexpected error reading position: {exc}", status=502)


@app.get("/api/market")
def api_market() -> Any:
    try:
        cfg = load_config()
        return market.build_market_snapshot(cfg)
    except market.MarketError as exc:
        return _err(f"Market data failed: {exc}", status=502)
    except Exception as exc:
        return _err(f"Unexpected error building market snapshot: {exc}", status=502)


@app.get("/api/candles")
def api_candles() -> Any:
    """1h closes for the last 30d, for the dashboard price chart."""
    try:
        cfg = load_config()
        df = market.fetch_klines(cfg.market_symbol, "1h", 720)
        return {
            "symbol": cfg.market_symbol,
            "times": [int(t) for t in df["open_time"].tolist()],
            "closes": [float(c) for c in df["close"].tolist()],
        }
    except Exception as exc:
        return _err(f"Could not fetch candles: {exc}", status=502)


@app.post("/api/analyze")
def api_analyze() -> Any:
    try:
        cfg = load_config()
        api_key = require_api_key(cfg)
    except ConfigError as exc:
        return _err(str(exc), status=400)

    try:
        position = chain.build_position_snapshot(cfg)
    except Exception as exc:
        return _err(f"Could not build position snapshot: {exc}", status=502)

    try:
        market_snap = market.build_market_snapshot(cfg)
    except Exception as exc:
        return _err(f"Could not build market snapshot: {exc}", status=502)

    try:
        result = analyst.run_analysis(position, market_snap, api_key, cfg.model)
    except analyst.AnalystError as exc:
        return _err(f"Analyst error: {exc}", status=502)
    except Exception as exc:
        return _err(f"Unexpected analyst error: {exc}", status=502)

    price_at_call = market_snap.get("price")
    try:
        verdict_id = db.insert_verdict(
            model=result["model"],
            price_at_call=price_at_call,
            position=position,
            market=market_snap,
            verdict=result["verdict"],
        )
    except Exception as exc:
        return _err(f"Verdict computed but could not be stored: {exc}", status=500)

    return {
        "id": verdict_id,
        "model": result["model"],
        "usage": result.get("usage", {}),
        "price_at_call": price_at_call,
        "verdict": result["verdict"],
        "position": position,
        "market": market_snap,
    }


@app.get("/api/history")
def api_history(limit: int = 50) -> Any:
    try:
        limit = max(1, min(int(limit), 500))
        return {"verdicts": db.get_verdicts(limit)}
    except Exception as exc:
        return _err(f"Could not read history: {exc}", status=500)


@app.get("/api/scores")
def api_scores() -> Any:
    try:
        return scorer.score_all()
    except Exception as exc:
        return _err(f"Could not compute scores: {exc}", status=502)
