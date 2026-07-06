"""Binance candle fetch + deterministic technical-indicator engine.

All indicators are computed by hand with pandas -- no pandas-ta / TA-Lib. The
LLM never sees candle data; it only receives the small `market_snapshot` dict
returned by build_market_snapshot().
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import requests

from config import Config, load_config

# Binance market-data hosts, tried in order. The primary public API is
# geo-restricted in some regions (HTTP 451); data-api.binance.vision is the
# public market-data mirror and serves identical klines without that block.
KLINES_HOSTS = [
    "https://api.binance.com",
    "https://data-api.binance.vision",
]

# Hours per year, used to annualize hourly realized volatility.
HOURS_PER_YEAR = 24 * 365

# Reference asset for the decision-rule "BTC moving the same direction" signal.
BTC_REF_SYMBOL = "BTCUSDT"


class MarketError(Exception):
    """Raised when candle data cannot be fetched."""


# --------------------------------------------------------------------------- #
# Data fetch
# --------------------------------------------------------------------------- #

def fetch_raw_klines(params: dict) -> list:
    """GET /api/v3/klines with host fallback. Returns the raw JSON rows."""
    last_exc: Exception | None = None
    for host in KLINES_HOSTS:
        try:
            resp = requests.get(
                f"{host}/api/v3/klines", params=params, timeout=20
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # try the next host on any failure
            last_exc = exc
            continue
    raise MarketError(f"Binance klines request failed on all hosts: {last_exc}")


def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Fetch OHLCV klines from Binance into a DataFrame (oldest first)."""
    rows = fetch_raw_klines(
        {"symbol": symbol, "interval": interval, "limit": limit}
    )
    if not rows:
        raise MarketError(f"No klines returned for {symbol} {interval}.")
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_base", "taker_quote", "ignore",
        ],
    )
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# Indicator primitives (Wilder smoothing where noted)
# --------------------------------------------------------------------------- #

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing == EMA with alpha = 1/period."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = _wilder(gain, period)
    avg_loss = _wilder(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    out = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss == 0 (pure uptrend) RSI is 100.
    out = out.where(avg_loss != 0.0, 100.0)
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _wilder(true_range(df), period)


def adx(df: pd.DataFrame, period: int = 14) -> dict[str, pd.Series]:
    """ADX with +DI / -DI using Wilder smoothing."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move.clip(lower=0)
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move.clip(lower=0)

    atr_series = atr(df, period)
    plus_di = 100.0 * _wilder(plus_dm, period) / atr_series.replace(0.0, float("nan"))
    minus_di = 100.0 * _wilder(minus_dm, period) / atr_series.replace(0.0, float("nan"))

    di_sum = (plus_di + minus_di).replace(0.0, float("nan"))
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx_series = _wilder(dx.fillna(0.0), period)
    return {"adx": adx_series, "plus_di": plus_di, "minus_di": minus_di}


def realized_vol(close: pd.Series, window: int) -> float | None:
    """Annualized realized vol (fraction) from hourly log returns."""
    log_ret = (close / close.shift(1)).apply(
        lambda x: math.log(x) if x and x > 0 else float("nan")
    )
    tail = log_ret.dropna().tail(window)
    if len(tail) < 2:
        return None
    return float(tail.std(ddof=1) * math.sqrt(HOURS_PER_YEAR))


def bollinger_bandwidth_pct(close: pd.Series, period: int = 20, mult: float = 2.0) -> float | None:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = mid + mult * std
    lower = mid - mult * std
    bw = (upper - lower) / mid * 100.0
    last = bw.iloc[-1]
    return None if pd.isna(last) else float(last)


def _pos(price: float, level: float) -> str:
    return "above" if price >= level else "below"


def _last(series: pd.Series) -> float | None:
    val = series.iloc[-1]
    return None if pd.isna(val) else float(val)


# --------------------------------------------------------------------------- #
# Snapshot assembly
# --------------------------------------------------------------------------- #

def _signal_facts(
    df1d: pd.DataFrame | None, btc_1d: pd.DataFrame | None
) -> dict[str, Any]:
    """Deterministic SIGNAL flags for the decision rules (no LLM math).

    Computes the three confirmations the recenter rule needs:
      (i)   3+ of the last 5 daily closes in the move's direction,
      (ii)  BTC moving the same direction over the last 7 days,
      (iii) volume larger on trend days than counter-trend days.
    """
    facts: dict[str, Any] = {
        "move_direction": None,
        "daily_trend_days_last5": None,
        "signal_daily_trend": None,
        "btc_symbol": BTC_REF_SYMBOL,
        "btc_change_24h_pct": None,
        "btc_change_7d_pct": None,
        "btc_direction_7d": None,
        "signal_btc_aligned": None,
        "volume_trend_vs_counter_ratio": None,
        "signal_volume_expansion": None,
        "signals_confirmed": 0,
    }
    if df1d is None or len(df1d) < 9:
        return facts

    closes = df1d["close"].astype(float)
    move_chg = (closes.iloc[-1] - closes.iloc[-8]) / closes.iloc[-8] * 100.0
    move_up = move_chg > 0
    facts["move_direction"] = (
        "up" if move_chg > 0 else "down" if move_chg < 0 else "flat"
    )

    # (i) daily-close trend over the last 5 daily changes.
    daily_chg = closes.diff().dropna().tail(5)
    cnt = int((daily_chg > 0).sum()) if move_up else int((daily_chg < 0).sum())
    facts["daily_trend_days_last5"] = cnt
    facts["signal_daily_trend"] = cnt >= 3

    # (ii) BTC alignment over 7 days.
    if btc_1d is not None and len(btc_1d) >= 9:
        bc = btc_1d["close"].astype(float)
        b7 = (bc.iloc[-1] - bc.iloc[-8]) / bc.iloc[-8] * 100.0
        b24 = (bc.iloc[-1] - bc.iloc[-2]) / bc.iloc[-2] * 100.0
        facts["btc_change_7d_pct"] = round(b7, 2)
        facts["btc_change_24h_pct"] = round(b24, 2)
        facts["btc_direction_7d"] = (
            "up" if b7 > 0 else "down" if b7 < 0 else "flat"
        )
        facts["signal_btc_aligned"] = (
            bool((b7 > 0) == move_up) if move_chg != 0 else False
        )

    # (iii) volume expansion on trend vs counter-trend days (last ~14 days).
    d = df1d.tail(15).copy()
    d["chg"] = d["close"].astype(float).diff()
    d = d.dropna()
    if move_up:
        trend_v, counter_v = d.loc[d["chg"] > 0, "volume"], d.loc[d["chg"] < 0, "volume"]
    else:
        trend_v, counter_v = d.loc[d["chg"] < 0, "volume"], d.loc[d["chg"] > 0, "volume"]
    if len(trend_v) and len(counter_v) and counter_v.mean() > 0:
        ratio = float(trend_v.mean() / counter_v.mean())
        facts["volume_trend_vs_counter_ratio"] = round(ratio, 2)
        facts["signal_volume_expansion"] = ratio > 1.0

    facts["signals_confirmed"] = int(
        bool(facts["signal_daily_trend"])
        + bool(facts["signal_btc_aligned"])
        + bool(facts["signal_volume_expansion"])
    )
    return facts


def build_market_snapshot(cfg: Config | None = None) -> dict[str, Any]:
    """Fetch 4h + 1h + 1d candles and return one deterministic snapshot dict."""
    cfg = cfg or load_config()
    symbol = cfg.market_symbol

    df4 = fetch_klines(symbol, "4h", 500)   # ~83 days for trend
    df1 = fetch_klines(symbol, "1h", 720)   # 30 days for vol
    df1d = fetch_klines(symbol, "1d", 120)  # daily trend / volume signals
    try:
        btc_1d = fetch_klines(BTC_REF_SYMBOL, "1d", 120)  # BTC alignment signal
    except Exception:
        btc_1d = None  # BTC signal degrades to null; app keeps working

    return compute_market_snapshot(symbol, df4, df1, df1d, btc_1d)


def compute_market_snapshot(
    symbol: str,
    df4: pd.DataFrame,
    df1: pd.DataFrame,
    df1d: pd.DataFrame | None = None,
    btc_1d: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Pure indicator computation over the candle frames (no network)."""
    price = float(df1["close"].iloc[-1])

    # --- 4h trend indicators ---
    ema20 = _last(ema(df4["close"], 20))
    ema50 = _last(ema(df4["close"], 50))
    ema200 = _last(ema(df4["close"], 200))
    rsi14 = _last(rsi(df4["close"], 14))
    adx_out = adx(df4, 14)
    adx14 = _last(adx_out["adx"])
    plus_di = _last(adx_out["plus_di"])
    minus_di = _last(adx_out["minus_di"])
    atr14 = _last(atr(df4, 14))
    atr_pct = (atr14 / price * 100.0) if (atr14 is not None and price) else None

    # --- 1h volatility indicators ---
    vol_7d = realized_vol(df1["close"], 7 * 24)
    vol_30d = realized_vol(df1["close"], 30 * 24)
    bb_bw_pct = bollinger_bandwidth_pct(df1["close"], 20, 2.0)

    # --- expected move (price * vol * sqrt(days/365)) using 30d annualized vol ---
    def expected_move(days: int) -> float | None:
        if vol_30d is None:
            return None
        return price * vol_30d * math.sqrt(days / 365.0)

    exp_3d = expected_move(3)
    exp_7d = expected_move(7)

    # --- rolling 7d high/low + % changes on 1h ---
    tail_7d = df1.tail(7 * 24)
    high_7d = float(tail_7d["high"].max())
    low_7d = float(tail_7d["low"].min())

    def pct_change(bars: int) -> float | None:
        if len(df1) <= bars:
            return None
        past = float(df1["close"].iloc[-1 - bars])
        return (price - past) / past * 100.0 if past else None

    change_24h = pct_change(24)
    change_7d = pct_change(7 * 24)

    def r(x: float | None, n: int = 2) -> float | None:
        return None if x is None else round(x, n)

    # Deterministic SIGNAL facts for the decision rules.
    signals = _signal_facts(df1d, btc_1d)

    return {
        "symbol": symbol,
        "price": r(price, 2),
        # 4h trend
        "ema_20": r(ema20, 2),
        "ema_50": r(ema50, 2),
        "ema_200": r(ema200, 2),
        "price_vs_ema20": _pos(price, ema20) if ema20 else None,
        "price_vs_ema50": _pos(price, ema50) if ema50 else None,
        "price_vs_ema200": _pos(price, ema200) if ema200 else None,
        "rsi_14": r(rsi14, 1),
        "adx_14": r(adx14, 1),
        "plus_di_14": r(plus_di, 1),
        "minus_di_14": r(minus_di, 1),
        "atr_14": r(atr14, 2),
        "atr_pct": r(atr_pct, 2),
        "trend_strength": (
            "strong" if (adx14 or 0) > 25 else "weak"
        ),
        # 1h volatility (annualized, %)
        "realized_vol_7d_pct": r(vol_7d * 100.0, 1) if vol_7d is not None else None,
        "realized_vol_30d_pct": r(vol_30d * 100.0, 1) if vol_30d is not None else None,
        "bollinger_bandwidth_pct": r(bb_bw_pct, 2),
        # expected move (absolute price + % of price)
        "expected_move_3d": r(exp_3d, 2),
        "expected_move_3d_pct": r(exp_3d / price * 100.0, 2) if exp_3d else None,
        "expected_move_7d": r(exp_7d, 2),
        "expected_move_7d_pct": r(exp_7d / price * 100.0, 2) if exp_7d else None,
        # ranges / momentum
        "high_7d": r(high_7d, 2),
        "low_7d": r(low_7d, 2),
        "change_24h_pct": r(change_24h, 2),
        "change_7d_pct": r(change_7d, 2),
        # decision-rule SIGNAL facts (deterministic)
        **signals,
    }


def _smoke_test() -> None:
    import json

    snap = build_market_snapshot()
    print(json.dumps(snap, indent=2))
    print("\n--- sanity ---")
    print(f"RSI in 0-100: {snap['rsi_14']}")
    print(f"ADX in 0-100: {snap['adx_14']}")
    print(f"realized_vol_30d (%): {snap['realized_vol_30d_pct']}")


if __name__ == "__main__":
    _smoke_test()
