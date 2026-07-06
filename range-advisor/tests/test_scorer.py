"""Unit tests for the pure scoring math with synthetic candles."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scorer  # noqa: E402


def test_in_suggested_range_pct_exact():
    # 10 closes; 6 sit inside [2900, 3100] -> 60%.
    closes = [2800, 2950, 3000, 3050, 3100, 2900, 3200, 3300, 2850, 3000]
    inside = [c for c in closes if 2900 <= c <= 3100]
    assert len(inside) == 6

    result = scorer.score_closes(
        price_at_call=3000.0,
        suggested_low=2900.0,
        suggested_high=3100.0,
        verdict_regime="ranging",
        adx_at_call=18.0,
        closes=closes,
    )
    assert result["in_suggested_range_pct"] == 60.0
    assert result["n_closes"] == 10
    # last close 3000 is inside the range
    assert result["still_in_range_now"] is True


def test_max_adverse_move():
    # Highest excursion above 3100 is 3300 -> (3300-3100)/3000 = 6.67%.
    closes = [3000, 3100, 3300, 3050]
    result = scorer.score_closes(
        price_at_call=3000.0,
        suggested_low=2900.0,
        suggested_high=3100.0,
        verdict_regime="ranging",
        adx_at_call=18.0,
        closes=closes,
    )
    assert result["max_adverse_move_pct"] == 6.67
    # last close 3050 is inside range
    assert result["still_in_range_now"] is True


def test_baseline_trending_match():
    # ADX 30 (>25) == trending baseline; verdict trending_up -> match.
    result = scorer.score_closes(
        price_at_call=3000.0, suggested_low=2900, suggested_high=3100,
        verdict_regime="trending_up", adx_at_call=30.0, closes=[3000],
    )
    assert result["regime_matched_baseline"] is True


def test_baseline_ranging_mismatch():
    # ADX 30 (>25) == trending baseline; verdict ranging -> mismatch.
    result = scorer.score_closes(
        price_at_call=3000.0, suggested_low=2900, suggested_high=3100,
        verdict_regime="ranging", adx_at_call=30.0, closes=[3000],
    )
    assert result["regime_matched_baseline"] is False


def test_empty_closes_is_safe():
    result = scorer.score_closes(
        price_at_call=3000.0, suggested_low=2900, suggested_high=3100,
        verdict_regime="ranging", adx_at_call=None, closes=[],
    )
    assert result["in_suggested_range_pct"] is None
    assert result["n_closes"] == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
