"""OpenRouter call: turn two deterministic snapshots into one JSON verdict.

The LLM is a classifier/synthesizer only. It receives pre-computed snapshots
and must return a strict JSON object. It never computes indicators and never
sees raw candle data. It cannot and does not execute anything -- the user acts
manually.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

VALID_REGIMES = {"trending_up", "trending_down", "ranging", "volatile_expansion"}
VALID_ACTIONS = {"hold", "recenter", "tighten", "widen", "exit_to_stables"}

REQUIRED_KEYS = (
    "regime",
    "confidence",
    "action",
    "suggested_range_low",
    "suggested_range_high",
    "reasoning",
    "invalidation",
)

SYSTEM_PROMPT = """You are a risk-aware analyst for a Uniswap V3 concentrated \
liquidity position. You do NOT trade and you do NOT execute anything; a human \
reads your output and acts manually.

You are given two pre-computed JSON snapshots (position state and market \
technical indicators). You did not compute these numbers and you must not \
recompute them -- treat them as ground truth. Your job is to classify the \
market regime and recommend a range action.

Respond with ONLY a single JSON object and nothing else. No markdown, no code \
fences, no commentary before or after. The object must match this schema \
exactly:

{
  "regime": "trending_up | trending_down | ranging | volatile_expansion",
  "confidence": 0.0,
  "action": "hold | recenter | tighten | widen | exit_to_stables",
  "suggested_range_low": 0.0,
  "suggested_range_high": 0.0,
  "reasoning": "3-5 sentences referencing specific snapshot numbers",
  "invalidation": "one concrete observable condition that would flip this call"
}

Rules:
- "regime" must be exactly one of the four enum values.
- "action" must be exactly one of the five enum values.
- "confidence" is a float between 0.0 and 1.0.
- suggested_range_low < suggested_range_high, in the same price units as the \
position snapshot (stable per volatile, e.g. USDC per WETH).
- Rebalancing realizes impermanent loss and costs gas + fees, so "hold" is the \
default unless the data clearly argues otherwise.
- The suggested range must account for the expected move over roughly a week \
(see expected_move_7d in the market snapshot).
- reasoning must cite specific numbers from the snapshots."""

USER_TEMPLATE = """This is a WETH/USDC-style concentrated-liquidity position on \
Uniswap V3 (Arbitrum). Rebalancing realizes impermanent loss and costs fees, so \
prefer "hold" unless the data argues otherwise. Any suggested range should cover \
roughly a week of expected movement.

POSITION_SNAPSHOT:
{position}

MARKET_SNAPSHOT:
{market}

Return only the JSON verdict object."""


class AnalystError(Exception):
    """Raised when the LLM response cannot be turned into a valid verdict."""


def build_messages(
    position: dict[str, Any], market: dict[str, Any]
) -> list[dict[str, str]]:
    user = USER_TEMPLATE.format(
        position=json.dumps(position, indent=2),
        market=json.dumps(market, indent=2),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def extract_json(text: str) -> dict[str, Any]:
    """Strip code fences / prose and parse the first JSON object found."""
    if text is None:
        raise AnalystError("Empty response from model.")
    cleaned = text.strip()
    # Remove ```json ... ``` or ``` ... ``` fences if present.
    fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back: grab the outermost {...} span.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise AnalystError(f"Could not parse JSON: {exc}") from exc
    raise AnalystError("No JSON object found in model response.")


def validate_verdict(obj: Any) -> dict[str, Any]:
    """Validate + normalize a parsed verdict. Raises AnalystError if invalid."""
    if not isinstance(obj, dict):
        raise AnalystError("Verdict is not a JSON object.")

    missing = [k for k in REQUIRED_KEYS if k not in obj]
    if missing:
        raise AnalystError(f"Verdict missing keys: {', '.join(missing)}")

    regime = obj["regime"]
    if regime not in VALID_REGIMES:
        raise AnalystError(f"Invalid regime: {regime!r}")

    action = obj["action"]
    if action not in VALID_ACTIONS:
        raise AnalystError(f"Invalid action: {action!r}")

    try:
        confidence = float(obj["confidence"])
    except (TypeError, ValueError) as exc:
        raise AnalystError(f"confidence not a number: {obj['confidence']!r}") from exc
    if not 0.0 <= confidence <= 1.0:
        raise AnalystError(f"confidence out of range [0,1]: {confidence}")

    try:
        low = float(obj["suggested_range_low"])
        high = float(obj["suggested_range_high"])
    except (TypeError, ValueError) as exc:
        raise AnalystError("suggested_range bounds are not numbers.") from exc
    if not (low < high):
        raise AnalystError(
            f"suggested_range_low ({low}) must be < suggested_range_high ({high})."
        )

    reasoning = obj["reasoning"]
    invalidation = obj["invalidation"]
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise AnalystError("reasoning must be a non-empty string.")
    if not isinstance(invalidation, str) or not invalidation.strip():
        raise AnalystError("invalidation must be a non-empty string.")

    return {
        "regime": regime,
        "confidence": round(confidence, 3),
        "action": action,
        "suggested_range_low": low,
        "suggested_range_high": high,
        "reasoning": reasoning.strip(),
        "invalidation": invalidation.strip(),
    }


def _call_openrouter(
    messages: list[dict[str, str]],
    api_key: str,
    model: str,
    network_retries: int = 2,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(network_retries + 1):
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    # Optional attribution headers accepted by OpenRouter.
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "range-advisor",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0,
                },
                timeout=90,
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            # Transient DNS / connection / timeout: back off and retry.
            last_exc = exc
            if attempt < network_retries:
                time.sleep(2 ** attempt)
                continue
            raise AnalystError(
                "Could not reach openrouter.ai after retries (network/DNS/timeout). "
                "Check your internet connection, VPN, or firewall, then try again."
            ) from exc

        if resp.status_code != 200:
            raise AnalystError(
                f"OpenRouter returned HTTP {resp.status_code}: {resp.text[:400]}"
            )
        return resp.json()

    # Unreachable, but keeps type-checkers happy.
    raise AnalystError(f"OpenRouter request failed: {last_exc}")


def run_analysis(
    position: dict[str, Any],
    market: dict[str, Any],
    api_key: str,
    model: str,
) -> dict[str, Any]:
    """Call the model, parse+validate, retry once on invalid JSON.

    Returns {"verdict": {...}, "model": str, "usage": {...}}.
    """
    messages = build_messages(position, market)
    last_error: Exception | None = None
    usage: dict[str, Any] = {}

    for attempt in range(2):  # one retry on invalid JSON
        data = _call_openrouter(messages, api_key, model)
        usage = data.get("usage", {}) or {}
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise AnalystError(f"Unexpected OpenRouter schema: {exc}") from exc

        # Log token usage (never the key or the content body).
        print(
            f"[analyst] model={model} attempt={attempt + 1} "
            f"prompt_tokens={usage.get('prompt_tokens')} "
            f"completion_tokens={usage.get('completion_tokens')} "
            f"total_tokens={usage.get('total_tokens')}"
        )

        try:
            parsed = extract_json(content)
            verdict = validate_verdict(parsed)
            return {"verdict": verdict, "model": model, "usage": usage}
        except AnalystError as exc:
            last_error = exc
            # Nudge the model to fix its output on the retry.
            messages = messages + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        f"That was invalid ({exc}). Respond again with ONLY the "
                        "JSON object matching the schema, no other text."
                    ),
                },
            ]

    raise AnalystError(f"Model did not return valid JSON after retry: {last_error}")
