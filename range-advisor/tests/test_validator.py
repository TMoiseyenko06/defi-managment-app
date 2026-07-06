"""Unit tests for the analyst JSON validator against canned responses."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyst  # noqa: E402


VALID = """```json
{
  "regime": "ranging",
  "confidence": 0.72,
  "action": "hold",
  "suggested_range_low": 2800.0,
  "suggested_range_high": 3400.0,
  "reasoning": "ADX 18 and price between EMA20 and EMA50 suggest chop.",
  "invalidation": "A 4h close above 3400 with ADX rising over 25."
}
```"""


def test_valid_with_code_fence():
    parsed = analyst.extract_json(VALID)
    verdict = analyst.validate_verdict(parsed)
    assert verdict["regime"] == "ranging"
    assert verdict["action"] == "hold"
    assert verdict["confidence"] == 0.72
    assert verdict["suggested_range_low"] < verdict["suggested_range_high"]


def test_prose_wrapped_json_is_extracted():
    text = 'Here is my call:\n{"regime":"trending_up","confidence":0.5,' \
           '"action":"recenter","suggested_range_low":1,"suggested_range_high":2,' \
           '"reasoning":"x","invalidation":"y"} — hope that helps!'
    verdict = analyst.validate_verdict(analyst.extract_json(text))
    assert verdict["action"] == "recenter"


def _expect_error(text_or_obj, is_obj=False):
    try:
        obj = text_or_obj if is_obj else analyst.extract_json(text_or_obj)
        analyst.validate_verdict(obj)
    except analyst.AnalystError:
        return True
    return False


def test_invalid_enum_regime():
    bad = dict(regime="sideways", confidence=0.5, action="hold",
               suggested_range_low=1, suggested_range_high=2,
               reasoning="x", invalidation="y")
    assert _expect_error(bad, is_obj=True)


def test_invalid_action():
    bad = dict(regime="ranging", confidence=0.5, action="ape_in",
               suggested_range_low=1, suggested_range_high=2,
               reasoning="x", invalidation="y")
    assert _expect_error(bad, is_obj=True)


def test_confidence_out_of_range():
    bad = dict(regime="ranging", confidence=1.4, action="hold",
               suggested_range_low=1, suggested_range_high=2,
               reasoning="x", invalidation="y")
    assert _expect_error(bad, is_obj=True)


def test_range_low_not_less_than_high():
    bad = dict(regime="ranging", confidence=0.5, action="hold",
               suggested_range_low=3000, suggested_range_high=2900,
               reasoning="x", invalidation="y")
    assert _expect_error(bad, is_obj=True)


def test_missing_keys():
    bad = dict(regime="ranging", confidence=0.5)
    assert _expect_error(bad, is_obj=True)


def test_not_json_at_all():
    assert _expect_error("the model refused to answer, sorry")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
