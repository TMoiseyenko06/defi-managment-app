"""Loads and validates configuration from the environment / .env file.

Nothing here ever loads a signing key. This app is read-only by design; the
only secret it touches is the OpenRouter API key used to call the LLM.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env once at import time. Real environment variables win over .env.
load_dotenv()

# Defaults. The model slug was verified against https://openrouter.ai/models
# (anthropic/claude-sonnet-4.5 is a valid slug). Swap MODEL in .env to compare
# Opus-class models, e.g. anthropic/claude-opus-4.8 or anthropic/claude-sonnet-5.
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_RPC = "https://arb1.arbitrum.io/rpc"
DEFAULT_SYMBOL = "ETHUSDT"


class ConfigError(Exception):
    """Raised when required configuration is missing or malformed."""


@dataclass
class Config:
    openrouter_api_key: str | None
    position_token_id: str | None
    wallet_address: str | None
    model: str
    arbitrum_rpc: str
    market_symbol: str

    @property
    def has_position(self) -> bool:
        return bool(self.position_token_id)

    @property
    def has_wallet(self) -> bool:
        return bool(self.wallet_address)


def _clean(value: str | None) -> str | None:
    """Return a stripped value, or None if empty/whitespace."""
    if value is None:
        return None
    value = value.strip()
    return value or None


def load_config() -> Config:
    """Read config from the environment without hard-failing.

    Validation of *required* fields is deferred to require_* helpers so that
    unit tests and the dashboard can import cleanly without a fully populated
    .env, and so a missing key surfaces as a clean API error rather than a
    crash on import.
    """
    return Config(
        openrouter_api_key=_clean(os.getenv("OPENROUTER_API_KEY")),
        position_token_id=_clean(os.getenv("POSITION_TOKEN_ID")),
        wallet_address=_clean(os.getenv("WALLET_ADDRESS")),
        model=_clean(os.getenv("MODEL")) or DEFAULT_MODEL,
        arbitrum_rpc=_clean(os.getenv("ARBITRUM_RPC")) or DEFAULT_RPC,
        market_symbol=_clean(os.getenv("MARKET_SYMBOL")) or DEFAULT_SYMBOL,
    )


def require_api_key(cfg: Config) -> str:
    if not cfg.openrouter_api_key:
        raise ConfigError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file."
        )
    return cfg.openrouter_api_key


def require_position_or_wallet(cfg: Config) -> None:
    if not cfg.has_position and not cfg.has_wallet:
        raise ConfigError(
            "Set POSITION_TOKEN_ID (your Uniswap V3 position NFT id) in .env, "
            "or set WALLET_ADDRESS to list your position ids."
        )
