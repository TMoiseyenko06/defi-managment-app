"""Read-only Uniswap V3 position + pool reader for Arbitrum One.

Everything here is an eth_call. No transactions are ever sent, no key is ever
loaded. The one write-shaped function on the contract (`collect`) is invoked
via a static call purely to *read* the uncollected fees owed.

All position/fee/price math lives here in plain Python. The LLM never sees any
of this code or the raw chain data -- only the finished snapshot dict.
"""

from __future__ import annotations

from typing import Any

import requests
from web3 import Web3

from config import Config, load_config

# --- Canonical Uniswap V3 addresses (identical across most chains). Verified
# against Arbiscan / Uniswap docs for Arbitrum One. ---
POSITION_MANAGER = Web3.to_checksum_address(
    "0xC36442b4a4522E871399CD717aBDD847Ab11FE88"
)
FACTORY = Web3.to_checksum_address("0x1F98431c8aD98523631AE4a59f267346ea31F984")

# Max uint128, used as amount0Max/amount1Max in the collect static call.
MAX_UINT128 = 2**128 - 1

# Symbols we treat as USD stables so pricing renders as "USDC per WETH".
STABLE_SYMBOLS = {
    "USDC",
    "USDC.E",
    "USDCE",
    "USDT",
    "DAI",
    "USDBC",
    "FRAX",
    "LUSD",
    "MIM",
    "TUSD",
}

GECKO_POOL_URL = (
    "https://api.geckoterminal.com/api/v2/networks/arbitrum/pools/{pool}"
)

# --- Minimal hand-written ABIs: only the functions we actually call. ---

POSITION_MANAGER_ABI = [
    {
        "name": "positions",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "tokenId", "type": "uint256"}],
        "outputs": [
            {"name": "nonce", "type": "uint96"},
            {"name": "operator", "type": "address"},
            {"name": "token0", "type": "address"},
            {"name": "token1", "type": "address"},
            {"name": "fee", "type": "uint24"},
            {"name": "tickLower", "type": "int24"},
            {"name": "tickUpper", "type": "int24"},
            {"name": "liquidity", "type": "uint128"},
            {"name": "feeGrowthInside0LastX128", "type": "uint256"},
            {"name": "feeGrowthInside1LastX128", "type": "uint256"},
            {"name": "tokensOwed0", "type": "uint128"},
            {"name": "tokensOwed1", "type": "uint128"},
        ],
    },
    {
        "name": "ownerOf",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "tokenId", "type": "uint256"}],
        "outputs": [{"name": "owner", "type": "address"}],
    },
    {
        "name": "balanceOf",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "owner", "type": "address"}],
        "outputs": [{"name": "balance", "type": "uint256"}],
    },
    {
        "name": "tokenOfOwnerByIndex",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "index", "type": "uint256"},
        ],
        "outputs": [{"name": "tokenId", "type": "uint256"}],
    },
    {
        # collect((tokenId, recipient, amount0Max, amount1Max))
        # Called via eth_call (from=owner) to read fees owed. Never sent.
        "name": "collect",
        "type": "function",
        "stateMutability": "payable",
        "inputs": [
            {
                "name": "params",
                "type": "tuple",
                "components": [
                    {"name": "tokenId", "type": "uint256"},
                    {"name": "recipient", "type": "address"},
                    {"name": "amount0Max", "type": "uint128"},
                    {"name": "amount1Max", "type": "uint128"},
                ],
            }
        ],
        "outputs": [
            {"name": "amount0", "type": "uint256"},
            {"name": "amount1", "type": "uint256"},
        ],
    },
]

ERC20_ABI = [
    {
        "name": "symbol",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "string"}],
    },
    {
        "name": "decimals",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint8"}],
    },
]

FACTORY_ABI = [
    {
        "name": "getPool",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "tokenA", "type": "address"},
            {"name": "tokenB", "type": "address"},
            {"name": "fee", "type": "uint24"},
        ],
        "outputs": [{"name": "pool", "type": "address"}],
    }
]

POOL_ABI = [
    {
        "name": "slot0",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [
            {"name": "sqrtPriceX96", "type": "uint160"},
            {"name": "tick", "type": "int24"},
            {"name": "observationIndex", "type": "uint16"},
            {"name": "observationCardinality", "type": "uint16"},
            {"name": "observationCardinalityNext", "type": "uint16"},
            {"name": "feeProtocol", "type": "uint8"},
            {"name": "unlocked", "type": "bool"},
        ],
    },
    {
        "name": "liquidity",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint128"}],
    },
]

Q96 = 2**96


class ChainError(Exception):
    """Raised for unrecoverable on-chain read problems."""


def get_web3(rpc: str) -> Web3:
    w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 20}))
    return w3


# --------------------------------------------------------------------------- #
# Pure math helpers (unit-testable, no network)
# --------------------------------------------------------------------------- #

def _raw_price_token1_per_token0(sqrt_price_x96: int) -> float:
    """token1-per-token0 in *raw* (smallest-unit) terms."""
    return (sqrt_price_x96 / Q96) ** 2


def _human_price_token1_per_token0(
    raw_price: float, decimals0: int, decimals1: int
) -> float:
    """Convert raw token1/token0 price to human units (token1 per token0)."""
    return raw_price * (10 ** (decimals0 - decimals1))


def _orient_price(price_t1_per_t0: float, stable_is_token0: bool) -> float:
    """Return price in 'stable per volatile' (USDC-per-WETH) orientation.

    If token1 is the stable, token1/token0 already == stable/volatile.
    If token0 is the stable, invert so we show stable-per-volatile.
    """
    if stable_is_token0:
        if price_t1_per_t0 == 0:
            return 0.0
        return 1.0 / price_t1_per_t0
    return price_t1_per_t0


def compute_amounts(
    liquidity: int,
    sqrt_price_x96: int,
    tick_lower: int,
    tick_upper: int,
) -> tuple[float, float]:
    """Token amounts (raw units) held by the position at the current price.

    Standard Uniswap V3 liquidity math. Float precision is fine for display.
    """
    sqrt_p = sqrt_price_x96 / Q96
    sqrt_lower = 1.0001 ** (tick_lower / 2)
    sqrt_upper = 1.0001 ** (tick_upper / 2)
    L = float(liquidity)

    if sqrt_p <= sqrt_lower:
        # Entirely below range -> all token0.
        amount0 = L * (sqrt_upper - sqrt_lower) / (sqrt_lower * sqrt_upper)
        amount1 = 0.0
    elif sqrt_p >= sqrt_upper:
        # Entirely above range -> all token1.
        amount0 = 0.0
        amount1 = L * (sqrt_upper - sqrt_lower)
    else:
        amount0 = L * (sqrt_upper - sqrt_p) / (sqrt_p * sqrt_upper)
        amount1 = L * (sqrt_p - sqrt_lower)
    return amount0, amount1


def _signed_pct(from_price: float, to_price: float) -> float | None:
    if not from_price:
        return None
    return (to_price - from_price) / from_price * 100.0


# --------------------------------------------------------------------------- #
# Chain reads
# --------------------------------------------------------------------------- #

def _erc20(w3: Web3, address: str):
    return w3.eth.contract(
        address=Web3.to_checksum_address(address), abi=ERC20_ABI
    )


def _read_token_meta(w3: Web3, address: str) -> dict[str, Any]:
    token = _erc20(w3, address)
    return {
        "address": Web3.to_checksum_address(address),
        "symbol": token.functions.symbol().call(),
        "decimals": token.functions.decimals().call(),
    }


def _fetch_gecko_stats(pool_address: str) -> dict[str, Any]:
    """Pool TVL + 24h volume from GeckoTerminal. Fails soft to nulls."""
    out = {"tvl_usd": None, "volume_24h_usd": None}
    try:
        resp = requests.get(
            GECKO_POOL_URL.format(pool=pool_address), timeout=15
        )
        resp.raise_for_status()
        attrs = resp.json()["data"]["attributes"]
        tvl = attrs.get("reserve_in_usd")
        out["tvl_usd"] = float(tvl) if tvl is not None else None
        vol = (attrs.get("volume_usd") or {}).get("h24")
        out["volume_24h_usd"] = float(vol) if vol is not None else None
    except Exception:
        # Non-fatal: the rest of the snapshot is still useful.
        pass
    return out


def list_wallet_positions(w3: Web3, wallet: str) -> list[dict[str, Any]]:
    """Return non-empty position ids owned by `wallet` (for setup help)."""
    pm = w3.eth.contract(address=POSITION_MANAGER, abi=POSITION_MANAGER_ABI)
    owner = Web3.to_checksum_address(wallet)
    count = pm.functions.balanceOf(owner).call()
    out: list[dict[str, Any]] = []
    for i in range(count):
        token_id = pm.functions.tokenOfOwnerByIndex(owner, i).call()
        pos = pm.functions.positions(token_id).call()
        liquidity = pos[7]
        if liquidity == 0:
            continue
        out.append({"token_id": token_id, "liquidity": liquidity})
    return out


def build_position_snapshot(cfg: Config | None = None) -> dict[str, Any]:
    """Read the full position + pool state and return one snapshot dict."""
    cfg = cfg or load_config()
    if not cfg.position_token_id:
        raise ChainError("POSITION_TOKEN_ID is not set.")

    token_id = int(cfg.position_token_id)
    w3 = get_web3(cfg.arbitrum_rpc)
    if not w3.is_connected():
        raise ChainError(f"Could not connect to RPC: {cfg.arbitrum_rpc}")

    pm = w3.eth.contract(address=POSITION_MANAGER, abi=POSITION_MANAGER_ABI)
    pos = pm.functions.positions(token_id).call()
    (
        _nonce,
        _operator,
        token0_addr,
        token1_addr,
        fee,
        tick_lower,
        tick_upper,
        liquidity,
        _fg0,
        _fg1,
        _owed0,
        _owed1,
    ) = pos

    owner = pm.functions.ownerOf(token_id).call()

    token0 = _read_token_meta(w3, token0_addr)
    token1 = _read_token_meta(w3, token1_addr)
    dec0, dec1 = token0["decimals"], token1["decimals"]

    # Pool discovery + live state.
    factory = w3.eth.contract(address=FACTORY, abi=FACTORY_ABI)
    pool_addr = factory.functions.getPool(token0_addr, token1_addr, fee).call()
    if int(pool_addr, 16) == 0:
        raise ChainError("Factory returned zero pool address.")
    pool = w3.eth.contract(
        address=Web3.to_checksum_address(pool_addr), abi=POOL_ABI
    )
    slot0 = pool.functions.slot0().call()
    sqrt_price_x96, current_tick = slot0[0], slot0[1]
    pool_liquidity = pool.functions.liquidity().call()

    # --- Orientation: which token is the USD stable? ---
    sym0 = (token0["symbol"] or "").upper()
    sym1 = (token1["symbol"] or "").upper()
    stable_is_token0 = sym0 in STABLE_SYMBOLS
    stable_is_token1 = sym1 in STABLE_SYMBOLS
    has_stable = stable_is_token0 or stable_is_token1

    # --- Prices (display orientation = stable per volatile). ---
    raw_current = _raw_price_token1_per_token0(sqrt_price_x96)
    price_t1_t0_current = _human_price_token1_per_token0(raw_current, dec0, dec1)
    price_current = _orient_price(price_t1_t0_current, stable_is_token0)

    def tick_to_display_price(tick: int) -> float:
        raw = 1.0001 ** tick
        p_t1_t0 = _human_price_token1_per_token0(raw, dec0, dec1)
        return _orient_price(p_t1_t0, stable_is_token0)

    p_lower_disp = tick_to_display_price(tick_lower)
    p_upper_disp = tick_to_display_price(tick_upper)
    # Inversion flips ordering; keep low < high for display.
    price_lower = min(p_lower_disp, p_upper_disp)
    price_upper = max(p_lower_disp, p_upper_disp)

    # --- Token amounts held now. ---
    amount0_raw, amount1_raw = compute_amounts(
        liquidity, sqrt_price_x96, tick_lower, tick_upper
    )
    amount0 = amount0_raw / (10 ** dec0)
    amount1 = amount1_raw / (10 ** dec1)

    # --- Uncollected fees via static collect call (read-only). ---
    fees0 = fees1 = None
    try:
        fee_owed = pm.functions.collect(
            (token_id, owner, MAX_UINT128, MAX_UINT128)
        ).call({"from": owner})
        fees0 = fee_owed[0] / (10 ** dec0)
        fees1 = fee_owed[1] / (10 ** dec1)
    except Exception:
        pass  # Fees display becomes null; rest of snapshot stands.

    # --- USD valuation (price both legs off the current pool price). ---
    if has_stable:
        if stable_is_token1:
            volatile_amount, stable_amount = amount0, amount1
            volatile_symbol, stable_symbol = sym0, sym1
        else:
            volatile_amount, stable_amount = amount1, amount0
            volatile_symbol, stable_symbol = sym1, sym0
        volatile_value_usd = volatile_amount * price_current
        stable_value_usd = stable_amount
        value_usd: float | None = volatile_value_usd + stable_value_usd
        fees_value_usd = None
        if fees0 is not None and fees1 is not None:
            if stable_is_token1:
                fees_value_usd = fees0 * price_current + fees1
            else:
                fees_value_usd = fees1 * price_current + fees0
    else:
        # No stable leg -> can't price in USD; report split in raw terms only.
        volatile_symbol = sym0
        stable_symbol = sym1
        volatile_value_usd = stable_value_usd = None
        value_usd = None
        fees_value_usd = None

    # --- Token split (%). By USD value when we have a stable, else by nothing. ---
    if value_usd and value_usd > 0:
        volatile_split_pct = volatile_value_usd / value_usd * 100.0
        stable_split_pct = stable_value_usd / value_usd * 100.0
    else:
        volatile_split_pct = stable_split_pct = None

    # --- Range / distance. ---
    in_range = tick_lower <= current_tick < tick_upper
    pct_to_lower = _signed_pct(price_current, price_lower)
    pct_to_upper = _signed_pct(price_current, price_upper)

    # --- Pool stats + naive pool-wide fee APR. ---
    gecko = _fetch_gecko_stats(pool_addr)
    fee_apr_pct = None
    if gecko["tvl_usd"] and gecko["volume_24h_usd"] is not None:
        tvl = gecko["tvl_usd"]
        if tvl > 0:
            fee_fraction = fee / 1_000_000.0
            fee_apr_pct = (
                gecko["volume_24h_usd"] * fee_fraction / tvl * 365 * 100.0
            )

    return {
        "token_id": token_id,
        "owner": owner,
        "pool_address": Web3.to_checksum_address(pool_addr),
        "fee_tier": fee,
        "fee_tier_pct": round(fee / 10_000.0, 4),  # e.g. 500 -> 0.05%
        "pair": f"{volatile_symbol}/{stable_symbol}",
        "token0": {"symbol": sym0, "decimals": dec0, "address": token0["address"]},
        "token1": {"symbol": sym1, "decimals": dec1, "address": token1["address"]},
        "tick_lower": tick_lower,
        "tick_upper": tick_upper,
        "current_tick": current_tick,
        "liquidity": str(liquidity),
        "pool_liquidity": str(pool_liquidity),
        "price_current": round(price_current, 6),
        "price_lower": round(price_lower, 6),
        "price_upper": round(price_upper, 6),
        "price_orientation": f"{stable_symbol} per {volatile_symbol}"
        if has_stable
        else f"{sym1} per {sym0}",
        "in_range": in_range,
        "pct_to_lower": round(pct_to_lower, 3) if pct_to_lower is not None else None,
        "pct_to_upper": round(pct_to_upper, 3) if pct_to_upper is not None else None,
        "amount0": round(amount0, 8),
        "amount1": round(amount1, 8),
        "value_usd": round(value_usd, 2) if value_usd is not None else None,
        "volatile_split_pct": round(volatile_split_pct, 2)
        if volatile_split_pct is not None
        else None,
        "stable_split_pct": round(stable_split_pct, 2)
        if stable_split_pct is not None
        else None,
        "uncollected_fees0": round(fees0, 8) if fees0 is not None else None,
        "uncollected_fees1": round(fees1, 8) if fees1 is not None else None,
        "uncollected_fees_usd": round(fees_value_usd, 2)
        if fees_value_usd is not None
        else None,
        "pool_tvl_usd": round(gecko["tvl_usd"], 2)
        if gecko["tvl_usd"] is not None
        else None,
        "pool_volume_24h_usd": round(gecko["volume_24h_usd"], 2)
        if gecko["volume_24h_usd"] is not None
        else None,
        "pool_fee_apr_pct": round(fee_apr_pct, 2) if fee_apr_pct is not None else None,
    }


def _smoke_test() -> None:
    """`python -m chain` -> print the position snapshot (or wallet ids)."""
    import json

    cfg = load_config()
    if not cfg.position_token_id and cfg.wallet_address:
        w3 = get_web3(cfg.arbitrum_rpc)
        print(f"Listing positions for wallet {cfg.wallet_address} ...\n")
        positions = list_wallet_positions(w3, cfg.wallet_address)
        if not positions:
            print("No non-empty Uniswap V3 positions found for this wallet.")
            return
        print("Found these active position ids:")
        for p in positions:
            print(f"  token_id={p['token_id']}  liquidity={p['liquidity']}")
        print("\nPut one of these into POSITION_TOKEN_ID in your .env file.")
        return

    snapshot = build_position_snapshot(cfg)
    print(json.dumps(snapshot, indent=2))

    # Sanity hints for eyeballing the smoke test.
    split = None
    if snapshot["volatile_split_pct"] is not None:
        split = round(
            snapshot["volatile_split_pct"] + snapshot["stable_split_pct"], 2
        )
    print("\n--- sanity ---", flush=True)
    print(f"in_range: {snapshot['in_range']}")
    print(
        f"bounds ordered: {snapshot['price_lower']} < "
        f"{snapshot['price_current']} < {snapshot['price_upper']}: "
        f"{snapshot['price_lower'] <= snapshot['price_upper']}"
    )
    if split is not None:
        print(f"split sums to ~100%: {split}")


if __name__ == "__main__":
    _smoke_test()
