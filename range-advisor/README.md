# range-advisor

A local, **read-only** analyst dashboard for a Uniswap V3 concentrated-liquidity
position on Arbitrum.

It reads your position and pool state from chain, computes market technical
indicators deterministically in Python, sends a single pre-computed snapshot to
an LLM (via OpenRouter) for regime classification and a range recommendation,
logs every verdict to SQLite, and scores past verdicts against what price
actually did — all rendered in a simple browser dashboard.

## What it is — and is not

- **Read-only, always.** It never holds a private key, never signs, never sends
  a transaction. It observes and recommends; **you act manually**.
- **The LLM does no math.** All indicators, position math, and fee math are
  deterministic Python. The model receives one JSON snapshot and returns one
  JSON verdict — it is a classifier/synthesizer, not a calculator and not a
  trader.
- Not financial advice.

## Layout

```
range-advisor/
  config.py        # loads/validates env
  chain.py         # position + pool reads (web3) + GeckoTerminal stats
  market.py        # Binance candles + indicator engine -> market snapshot
  analyst.py       # prompt build + OpenRouter call + JSON validation
  db.py            # SQLite: verdicts
  scorer.py        # grades past verdicts against realized price
  app.py           # FastAPI app + endpoints, serves static/index.html
  static/index.html
  tests/           # validator + scorer unit tests
```

## Setup

```bash
cd range-advisor
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then edit .env (see below)
```

### Configure `.env`

- `OPENROUTER_API_KEY` — **required.** Get one at
  <https://openrouter.ai/keys>. This is the only secret the app uses.
- `POSITION_TOKEN_ID` — **required** (unless you use `WALLET_ADDRESS`). Your
  Uniswap V3 position NFT id on Arbitrum.

  **How to find your position token id:**
  1. Go to <https://app.uniswap.org> and open (or paste) your wallet.
  2. Click the concentrated-liquidity position you want to monitor.
  3. The id is the number at the end of the URL:
     `.../positions/v3/arbitrum/<ID>`. It's the same as the token id of your
     "Uniswap V3 Positions NFT" on <https://arbiscan.io>.

  Or, leave `POSITION_TOKEN_ID` blank, set `WALLET_ADDRESS`, and run
  `python -m chain` — it lists your active position ids and exits.
- `MODEL` — OpenRouter slug. Default `anthropic/claude-sonnet-4.5`. Swap in an
  Opus-class model (e.g. `anthropic/claude-opus-4.8`) to compare; the scorer
  tracks results per model.
- `ARBITRUM_RPC`, `MARKET_SYMBOL` — sensible defaults; override if you like.

## Run

```bash
# Optional smoke tests (each prints a snapshot):
python -m chain        # your position snapshot (or, with WALLET_ADDRESS, your ids)
python -m market       # market indicator snapshot

# Unit tests (no network, no key needed):
python tests/test_validator.py
python tests/test_scorer.py
# or, if pytest is installed:  pytest -q

# Start the dashboard:
python -m uvicorn app:app --reload
# then open http://127.0.0.1:8000
```

`python -m uvicorn` is used instead of the bare `uvicorn` command so it works
even when the venv's scripts folder isn't on your PATH.

**Windows (PowerShell).** If `uvicorn` "is not recognized", the venv isn't on
PATH. The most reliable form calls the venv's Python directly — no activation
needed:

```powershell
cd range-advisor
.venv\Scripts\python.exe -m pip install -r requirements.txt   # first time only
.venv\Scripts\python.exe -m uvicorn app:app --reload
```

(If `Activate.ps1` fails on execution policy, either use the direct
`.venv\Scripts\python.exe ...` form above, or run once in that shell:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.)

Click **Run Analysis** to send the current snapshots to the model, store the
verdict, and draw the suggested range on the chart.

## Endpoints

| Method | Path            | Returns                                            |
|--------|-----------------|----------------------------------------------------|
| GET    | `/`             | dashboard                                          |
| GET    | `/api/position` | position snapshot                                  |
| GET    | `/api/market`   | market indicator snapshot                          |
| GET    | `/api/candles`  | 1h closes (30d) for the chart                      |
| POST   | `/api/analyze`  | builds snapshots, calls the LLM, stores + returns  |
| GET    | `/api/history`  | recent verdicts (`?limit=50`)                      |
| GET    | `/api/scores`   | per-verdict + per-model scores                     |

Any failing external API returns `{"error": ...}` with a proper status code —
it never takes down the dashboard.

## How to read the verdict

- **regime** — the model's read of the market: `trending_up`, `trending_down`,
  `ranging`, or `volatile_expansion`.
- **action** — what to consider doing:
  - `hold` — do nothing (the default; rebalancing realizes impermanent loss and
    costs fees, so the bar to move is high).
  - `recenter` — move the range so it's centered on the current price.
  - `tighten` — narrow the range for more fee density (higher out-of-range risk).
  - `widen` — broaden the range to reduce out-of-range risk (lower fee density).
  - `exit_to_stables` — step out of the position entirely.
- **confidence** — the model's self-reported confidence (0–1).
- **suggested_range_low / high** — a proposed range, sized to cover roughly a
  week of expected movement.
- **reasoning** — 3–5 sentences citing specific snapshot numbers.
- **invalidation** — one observable condition that would flip the call.

### Decision policy (hard constraints)

The model is bound by a strict `hold`/`recenter` policy. The gating facts are
computed **deterministically in Python** and passed in the snapshots — the LLM
only applies the rules, it never computes them:

- **LOCATION** (position snapshot): `in_upper_decile` (price in the outer 10% of
  the range near the upper bound, i.e. `>= upper_recenter_trigger` =
  `price_lower` + 90% of range width) and `in_lower_decile` (bottom 10%).
- **SIGNAL** (market snapshot, from **hourly** candles): `signals_confirmed` =
  how many of `signal_hourly_trend` (a majority of the last `trend_window_hours`
  hourly closes in the move's direction), `signal_btc_aligned` (BTC same
  direction over the same hourly window), and `signal_volume_expansion` (volume
  larger on trend hours than counter-trend hours over `volume_window_hours`) are
  true. Windows are tunable constants at the top of `market.py`
  (`SIGNAL_TREND_HOURS`, `SIGNAL_VOLUME_HOURS`).

Rules: `recenter` is allowed **only** when `in_upper_decile` **and**
`signals_confirmed >= 2`; otherwise `hold`. Near the lower bound recenter is
forbidden — converting to ETH there is an accepted accumulation outcome, not a
failure. An upper-bound break to USDC is acceptable; never act only to preserve
fee continuity. Confidence < 0.70 forces `hold`. Never add capital to repair
drawdown. The `reasoning` leads with the action and states which triggers fired
and the distance to the next decision point in % and $.

The **History & Scores** table then grades each past verdict once it is >24h
old: what fraction of 1h closes stayed inside the suggested range, whether it's
still in range now, the worst adverse excursion, and whether the regime call
agreed with a dumb ADX>25 baseline. Aggregates are broken out per model so you
can compare slugs over time.

> Read-only analytics. Not financial advice. All actions are manual.
