"""
DeepVibe AI Hedge Fund — configuration.

┌──────────────────────────────────────────────────────────────────────────┐
│ Adjust the CONTROL PANEL section below to change strategy behavior.      │
│ Everything past the control panel is implementation detail rarely        │
│ touched after the project is set up.                                     │
└──────────────────────────────────────────────────────────────────────────┘

Typical pipeline (run once after fresh clone, then re-run step 3+ to iterate):

    1. pip install -e ".[dev]"                            # once
    2. cp .env.example .env                               # paste Alpaca keys
    3. python -m deepvibe_hedge.alpaca_fetcher            # download OHLCV
    4. python -m deepvibe_hedge.data_splitter             # splits + SMA columns
    5. python -m deepvibe_hedge.mad.backtester            # research + dashboard
    6. python -m deepvibe_hedge.mad.live_bot              # paper/live trading

The control panel covers the four decisions users actually tune:

  * Which indexes run (SPY / QQQ / IWM)         → MAD_INDEX_ENABLED_ETFS
  * Risk-on / index / risk-off schemes            → section 3
  * Cartesian grid axes (any combo, 7×7×7 max)    → ``MAD_GRID_SEARCH_*`` (section 5)
  * Paper vs live brokerage                      → BOT_MODE

Inspect OHLCV: ``python -m deepvibe_hedge.db_utils``.
"""
from __future__ import annotations

from datetime import datetime, timezone

from deepvibe_hedge.hedge_assets import hedge_assets
from deepvibe_hedge.nasdaq100 import nasdaq100
from deepvibe_hedge.russell2000 import russell2000
from deepvibe_hedge.sp500 import sp500

# =============================================================================
# STRATEGY CONTROL PANEL
#
# Only these ~10 knobs change strategy behavior for day-to-day research.
# Everything below this section is plumbing.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Execution mode
# -----------------------------------------------------------------------------
# Multi-index allocator: runs the stock picker per enabled index slot
# (``MAD_INDEX_ENABLED_ETFS``) and blends the books by each ETF's MRAT + 200D
# trend via ``MAD_INDEX_WEIGHTING_SCHEME``. Progressive risk-off: each slot
# below trend routes its 1/N share to the risk-off sleeve; the rest of the
# book stays fully invested in the passing slots.
#
# Both the backtester and the live bot support this mode. When False, both
# fall back to single-universe mode driven by ``MAD_UNIVERSE_TICKERS`` +
# ``MAD_REGIME_TICKER`` (section 8 below).
MAD_INDEX_ALLOCATOR_ENABLED = True

# -----------------------------------------------------------------------------
# 2. Multi-index selection (backtester only)
# -----------------------------------------------------------------------------
# Pick which index sleeves to run. The backtester runs the stock picker
# independently on each enabled sleeve's universe, then allocates across
# them using the index ETF's own MRAT. Each sleeve has its own 200D trend
# filter; failing slots route their weight to the risk-off sleeve (section 9).
#
#   None                 → run all three (IWM + QQQ + SPY, ~2449 names)
#   ("QQQ", "SPY")       → drop small-caps, keep NASDAQ-100 + S&P 500 (~519)
#   ("SPY",)             → S&P 500 only
#   ("QQQ",)             → NASDAQ-100 only (equivalent to legacy single-universe)
#
# Why drop IWM: small-caps have been a standalone drag (-25% over 5 yrs) and
# their trend filter breaks ~50% of years, forcing excess risk-off.
MAD_INDEX_ENABLED_ETFS: tuple[str, ...] | None = ("QQQ", "SPY")

# -----------------------------------------------------------------------------
# 3. Weighting schemes
# -----------------------------------------------------------------------------
# Both the stock picker and the top-level allocator use the same 7-scheme
# taxonomy (see ``mad/weighting.py``):
#
#   "equal"                    — 1/N across names (legacy default)
#   "mrat_distance"            — weight ∝ (MRAT - 1), momentum tilt
#   "mrat_zscore"              — standardized MRAT tilt
#   "softmax"                  — soft-argmax over MRATs; see tau below
#   "rank"                     — decile-ranked weights (stable, less spiky)
#   "inv_vol"                  — 1/σ weighting (stable, vol-normalized)
#   "mrat_distance_inv_vol"    — hybrid: MRAT tilt × vol normalization

# Stock picker: applied to names within each index slot. ``mrat_distance``
# weights each name ∝ (MRAT - 1) — a direct expression of the 21/200 MRAT
# thesis. Names stretched further above their long SMA get more weight,
# weakening names get less; no separate hyperparameter to tune (unlike
# softmax τ) so it doesn't overfit to a specific eval window.
MAD_WEIGHTING_SCHEME = "mrat_distance"

# Top-level allocator: applied across the enabled index slots. When ``None``,
# falls back to equal weight across slots that pass trend.
# ``mrat_distance_inv_vol`` = MRAT-distance tilt × 1/σ normalization. At the
# index level we only have 2-3 ETFs with very different vol profiles (IWM ≈
# 1.5-2× SPY's daily vol), so pure MRAT tilt can over-allocate to the
# highest-MRAT index even when its realized vol is much higher; the
# inv-vol term keeps the mix balanced. Locked in from the 7×7 grid search
# (QQQ+SPY, 2021-06 → 2026-04, winner by Sharpe = 1.11 @ +239%).
MAD_INDEX_WEIGHTING_SCHEME: str | None = "mrat_distance_inv_vol"

# Risk-off sleeve: applied to the hedge-asset basket (bonds + metals +
# defensive ETFs) when an index fails its trend filter. When ``None``, the
# sleeve uses fixed equal weights (1/N across trend-passing legs + safe
# harbor fallback). Setting this to ``"mrat_distance"`` makes the hedge
# leg consistent with the stock and index layers — whichever hedge asset
# has stretched furthest above its 200D gets the biggest weight (e.g.
# GLD vs TLT during a 2022-style bear). Uses the same MRAT pair
# (``MAD_SLEEVE_MRAT_SHORT/LONG``, default 21/200). When every sleeve leg
# fails its own trend filter, the safe harbor (``MAD_REGIME_OFF_SAFE_HARBOR``,
# default BIL) gets 100%.
MAD_SLEEVE_WEIGHTING_SCHEME: str | None = "inv_vol"

# -----------------------------------------------------------------------------
# 4. Softmax temperature (only used when weighting scheme is "softmax")
# -----------------------------------------------------------------------------
# Lower τ = sharper concentration (bets on top-ranked names). Not active
# with the current default (mrat_distance); keep documented for when users
# experiment with ``MAD_WEIGHTING_SCHEME = "softmax"``.
#
# τ sweep on QQQ+SPY, eval window 2021-06 → 2026-04:
#   τ=0.02 / 0.05 → Sharpe ~1.2, but 2024 = +414% (one-stock overfit)
#   τ=0.10        → Sharpe ~1.12, 2024 = +337%
#   τ=0.20        → Sharpe ~0.98, 2024 = +219%
#   τ=0.50        → Sharpe 0.97,  2024 =  +98% (balanced momentum tilt)
#   τ=1.0 / 2.0   → effectively equal weight
MAD_WEIGHT_SOFTMAX_TAU = 0.5

# -----------------------------------------------------------------------------
# 5. Grid search (backtest optimization)
# -----------------------------------------------------------------------------
# **Cartesian product** over three independent axes (each axis = 7 schemes or
# "fixed to section 3"). Toggle any combination:
#
#   * ``MAD_GRID_SEARCH_STOCK``  — vary ``MAD_WEIGHTING_SCHEME`` (risk-on, per slot)
#   * ``MAD_GRID_SEARCH_INDEX`` — vary ``MAD_INDEX_WEIGHTING_SCHEME`` (allocator)
#   * ``MAD_GRID_SEARCH_SLEEVE`` — vary ``MAD_SLEEVE_WEIGHTING_SCHEME`` (hedge basket)
#
# Examples:
#   Stock only     : ``STOCK=True``, ``INDEX=False``, ``SLEEVE=False`` → 7 runs
#   Index only     : ``STOCK=False``, ``INDEX=True``, ``SLEEVE=False`` → 7 runs
#   Sleeve only    : ``STOCK=False``, ``INDEX=False``, ``SLEEVE=True`` → 7 runs
#   Stock × index  : ``STOCK=True``, ``INDEX=True``, ``SLEEVE=False`` → 49 runs
#   Full 7×7×7     : all ``True`` → 343 runs (slow; per-stock slot cache speeds inner loops)
#
# A dimension set to ``False`` uses the **single** scheme from section 3 for that
# layer (no multiplication on that axis).
#
# **Legacy aliases** (OR'd in the backtester — you can keep using them):
#   ``MAD_STOCK_WEIGHTING_GRID_SEARCH=True``  → forces stock axis on
#   ``MAD_SLEEVE_WEIGHTING_GRID_SEARCH=True`` → forces sleeve axis on
#   ``MAD_INDEX_WEIGHTING_GRID_SEARCH=True``  → forces **stock + index** axes (49-combo)

MAD_GRID_SEARCH_STOCK: bool = False
MAD_GRID_SEARCH_INDEX: bool = False
MAD_GRID_SEARCH_SLEEVE: bool = True

MAD_STOCK_WEIGHTING_GRID_SEARCH: bool = False
MAD_SLEEVE_WEIGHTING_GRID_SEARCH: bool = False
MAD_INDEX_WEIGHTING_GRID_SEARCH: bool = False

# Single metric for all grid modes (leaderboard sort).
MAD_GRID_OPTIMIZE_METRIC: str = "sharpe_ratio"

# Optional: restrict an axis to a subset of scheme names. ``None`` = all 7.
MAD_STOCK_WEIGHTING_GRID: tuple[str, ...] | None = None
MAD_INDEX_WEIGHTING_GRID: tuple[str, ...] | None = None
MAD_SLEEVE_WEIGHTING_GRID: tuple[str, ...] | None = None

# Deprecated aliases for ``MAD_*_WEIGHTING_GRID`` (stock / index subsets for old 49-grid).
MAD_WEIGHTING_GRID: tuple[str, ...] | None = None

# Metric to maximize during grid search. Must be a key in metrics dict:
#   "sharpe_ratio"    (default; smooths out lucky high-vol runs)
#   "sortino_ratio"   (downside-only Sharpe; more forgiving of upside spikes)
#   "net_total_return" (raw CAGR; concentrates into biggest historical winner)
#   "profit_factor"   (gross gains / gross losses)
# Kept for back-compat; grid sort uses ``MAD_GRID_OPTIMIZE_METRIC`` when set.
MAD_INDEX_OPTIMIZE_METRIC: str = "sharpe_ratio"

# -----------------------------------------------------------------------------
# 6. Backtest evaluation window
# -----------------------------------------------------------------------------
# Dashboard + reported metrics restrict to ``[start, end]`` (ISO ``YYYY-MM-DD``
# or ``None`` for no bound). OHLCV is still loaded for the full history so
# SMAs warm up properly; this just trims the eval window.
#
# Why 2021-06-01: current OHLCV for SPY/QQQ/IWM effectively starts 2020-07-27
# (one orphan 2018-11-01 bar, then nothing until mid-2020). The 200D SMA
# doesn't warm until ~2021-05-10, so starting eval earlier would show forced
# risk-off from NaN SMAs for the first 5 months. Re-fetch OHLCV back to
# 2019-01-01 (re-run ``alpaca_fetcher``) to move this back to "2021-01-01".
MAD_BACKTEST_START_DATE: str | None = "2021-06-01"
MAD_BACKTEST_END_DATE: str | None = None

# -----------------------------------------------------------------------------
# 7. Alpaca account (live bot)
# -----------------------------------------------------------------------------
# Selects which Alpaca API the live bot hits:
#   "paper"         → paper API (fake money; recommended for development)
#   "cash" / "live" → live brokerage (real money)
BOT_MODE = "cash"

# Whether the live bot can open short positions (also affects the backtester's
# default long/short handling — see ``MAD_DIRECTION_MODE``).
LIVE_BOT_ALLOW_SHORT = False

# Fraction of NAV the strategy targets (keeps a small cash buffer for fees).
MAD_LIVE_EQUITY_FRACTION = 0.98

# -----------------------------------------------------------------------------
# 8. Single-universe fallback (live bot + allocator-off mode)
# -----------------------------------------------------------------------------
# Used when ``MAD_INDEX_ALLOCATOR_ENABLED = False`` — the legacy single-panel
# flow. Pick a universe (``nasdaq100``, ``sp500``, ``russell2000``, or a
# custom tuple) and a regime ETF for the trend filter.
MAD_UNIVERSE_TICKERS = nasdaq100
MAD_REGIME_TICKER = "QQQ"

# -----------------------------------------------------------------------------
# 9. Risk-off hedging universe
# -----------------------------------------------------------------------------
# The universe the strategy rotates into when the market regime fails. See
# ``hedge_assets.py`` for the curated list (bonds + metals + defensive ETFs).
# Used by both backtester and live bot via ``mad/regime_sleeve.py``.
HEDGE_ASSETS = hedge_assets

# =============================================================================
# DATA PIPELINE (``alpaca_fetcher`` + ``data_splitter``)
# =============================================================================

# Bar granularity. Most users leave this at "1d" (daily bars).
TARGET_CANDLE_GRANULARITY = "1d"

# OHLCV date range. End date can be a fixed ISO string ("2026-04-17") when
# ``OHLCV_DOWNLOAD_END_MODE = 'fixed'``, or dynamic via "utc_now".
TARGET_START_DATE = "2010-01-01"
TARGET_END_DATE = "now"
OHLCV_DOWNLOAD_END_MODE = "utc_now"

# When True, union HEDGE_ASSETS into the fetcher universe so the risk-off
# sleeve's OHLCV is always available.
OHLCV_PIPELINE_INCLUDE_HEDGE_ASSETS = True

# Alpaca bar adjustment: "split" is the standard default for backtests.
ALPACA_BAR_ADJUSTMENT = "split"

# Alpaca historical-data feed (``iex`` is free tier; ``sip`` requires paid sub).
LIVE_BOT_DATA_FEED = "iex"

# Fee rate applied per side in the backtester (0.1% = commission + spread proxy).
BACKTEST_FEE_RATE = 0.001

# -----------------------------------------------------------------------------
# Data splitter (walk-forward splits + SMA precompute)
# -----------------------------------------------------------------------------
# Split 0 = warmup so the longest SMA in ``splitter_ma_periods()`` is valid
# on the first bar of split 1. Split assignments persist across runs.
SPLITTER_NUM_SPLITS = 10
SPLITTER_ENABLE_SPLIT_ASSIGNMENT = True
SPLITTER_ENABLE_MA_PRECOMPUTE = True
SPLITTER_DB_WRITE_RETRIES = 6
SPLITTER_DB_WRITE_RETRY_SEC = 5

# In-sample / out-of-sample split IDs (used by walkforward research tools).
SPLIT_PLAN_IN_SAMPLE = (1, 3, 5, 7, 9)
SPLIT_PLAN_OUT_OF_SAMPLE = (2, 4, 6, 8, 10)

# =============================================================================
# MRAT CORE (stock-level backtester knobs)
# =============================================================================

# "long_only", "short_only", or "long_short" — direction mode for entry sig.
MAD_DIRECTION_MODE = "long_only"

# MRAT = SMA(short) / SMA(long). Changing these requires a fresh
# ``data_splitter`` run so the new SMA columns get precomputed.
MAD_SMA_SHORT = 21
MAD_SMA_LONG = 200

# Cross-sectional σ multipliers applied to MRAT when computing entry bands.
# Leave at 1.0 unless you specifically want to widen/narrow bands.
MAD_LONG_SIGMA_MULT = 1.0
MAD_SHORT_SIGMA_MULT = 1.0
MAD_SYMMETRIC_SHORT_SIGMA = False

# Decile cutoffs (1-10 scale). Default: top decile longs, bottom decile shorts.
MAD_LONG_DECILE_MIN = 10
MAD_SHORT_DECILE_MAX = 1

# Minimum number of daily bars a ticker needs before it's eligible. Filters
# fresh IPOs that lack warmup data.
MAD_MIN_HISTORY_BARS = 252

# Aggregate intraday bars to daily bars during backtest (safe default).
MAD_AGGREGATE_TO_DAILY = True

# Per-stock data-completeness filter: drop any symbol whose bar coverage over
# the MRAT long window is below this fraction. Mutes thin-trading names that
# dominate the Russell 2000 long tail. Set to 0.0 to disable.
MAD_MIN_DATA_COMPLETENESS = 0.8

# Rolling window (daily bars) for realized vol used by ``inv_vol`` /
# ``mrat_distance_inv_vol`` weighting. Shifted by one bar to avoid look-ahead.
MAD_WEIGHT_REALIZED_VOL_LOOKBACK = 20

# Per-name cap/floor applied after the weighting scheme produces raw weights.
# ``None`` disables. Use with concentrating schemes like ``mrat_distance``.
MAD_WEIGHT_MAX_PER_NAME = None
MAD_WEIGHT_MIN_PER_NAME = None

# Blend with equal-weight baseline: final = (1-b) * scheme + b * equal.
# 0.0 = pure scheme, 1.0 = pure equal. Safety lever when the book shrinks.
MAD_WEIGHT_EQUAL_BLEND = 0.0

# Evaluate across ALL non-warmup splits (True) or just in-sample splits (False).
# Typically True for final backtest runs; False during walkforward research.
MAD_EVAL_ALL_SPLITS = True

# Walkforward split counts (used by ``mad/walkforward_oos.py``).
MAD_IS_SPLITS = 6
MAD_OOS_SPLITS = 4
MAD_WF_OPTIM_SPLIT = "avg"
MAD_WF_OOS_SPLIT = "all"

# Dashboard ports.
MAD_DASHBOARD_PORT = 8063
MAD_WF_DASHBOARD_PORT = 8064

# -----------------------------------------------------------------------------
# Legacy single-ticker regime (allocator-off mode only)
# -----------------------------------------------------------------------------
# When the multi-index allocator is OFF, ``MAD_REGIME_TICKER`` is the ETF whose
# 200D SMA gates the whole book. When the allocator is ON, each slot uses its
# own trend filter and these settings are ignored.
MAD_REGIME_MA_ENABLED = True
MAD_REGIME_MA = 200

# Exit SMA (optional): when enabled, positions exit when the ticker closes
# below its exit SMA. Currently disabled — strategy uses decile-only exits.
MAD_EXIT_MA_ENABLED = False
MAD_EXIT_MA_PERIOD = 0

# Grid variants (advanced research — used by walkforward for parameter sweeps).
MAD_SMA_SHORT_GRID = (21,)
MAD_SMA_LONG_GRID = (200,)
MAD_EXIT_MA_GRID = (0, 50, 100, 150, 200)
MAD_REGIME_MA_GRID = (0, 50, 100, 150, 200)

# =============================================================================
# MULTI-INDEX ALLOCATOR (``mad/index_allocator.py``)
#
# Index slot definitions + per-slot trend filter. Used when
# ``MAD_INDEX_ALLOCATOR_ENABLED = True``. Enabled subset is controlled by
# ``MAD_INDEX_ENABLED_ETFS`` (control panel section 2).
# =============================================================================

# (ETF, universe tuple, display name) — tuple order encodes exclusive-
# assignment priority. A ticker present in multiple slots' universes is
# claimed by the FIRST matching slot here.
#
# Current priority IWM → QQQ → SPY:
#   * IWM claims ~1,930 small-caps first (Russell excludes mega-caps anyway)
#   * QQQ claims all 102 NASDAQ-100 names untouched (Russell doesn't overlap)
#   * SPY keeps every S&P name not already in NASDAQ-100 (~400 mid/large caps)
#
# Flipping to SPY → QQQ → IWM makes SPY claim every mega-cap first, which
# shrinks QQQ's exclusive universe to ~15 ADRs and cripples its standalone
# contribution. Keep the current order unless you have a specific reason.
MAD_INDEX_SLOTS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("IWM", russell2000, "Russell 2000"),
    ("QQQ", nasdaq100,   "NASDAQ-100"),
    ("SPY", sp500,       "S&P 500"),
)

# Per-slot trend filter: each slot's ETF must be above its SMA(regime_ma)
# for its slot to be active. A failing slot's weight routes to the risk-off
# sleeve (progressive blending — 1/N per failing slot).
MAD_INDEX_REGIME_MA = 200

# MRAT pair used to score each index ETF for the top-level allocation.
MAD_INDEX_MRAT_SHORT = 21
MAD_INDEX_MRAT_LONG = 200

# =============================================================================
# RISK-OFF SLEEVE (``mad/regime_sleeve.py``)
#
# Hedging allocation when index trend filters fail. Bonds + metals + defensive
# ETFs from ``HEDGE_ASSETS`` (control panel section 9). Each leg has its own
# trend filter so failing legs route to the safe-harbor ticker.
# =============================================================================

# Ensemble regime tickers: in allocator-off mode, risk-off fires when
# ``MAD_REGIME_MODE`` condition is met across these.
#   "all_below"  → all tickers must fail their SMA trend → risk-off
#   "any_below"  → any ticker failing its SMA → risk-off
MAD_REGIME_TICKERS = ("QQQ", "SPY")
MAD_REGIME_MODE = "all_below"

# Safe harbor: when every hedge-asset leg fails its trend filter, 100% of the
# risk-off allocation goes to this single ticker (typically ``BIL`` — 1-3mo
# T-bills). Also the fallback leg in fixed-weight sleeve mode.
MAD_REGIME_OFF_SAFE_HARBOR = "BIL"

# Default trend MA for every hedge-asset leg. Override per-leg via
# ``MAD_REGIME_OFF_SLEEVE_TREND_OVERRIDES`` (e.g. shorter MAs for short-duration
# bonds). Tickers not in the override dict fall back to the default.
MAD_REGIME_OFF_SLEEVE_TREND_MA = 200
MAD_REGIME_OFF_SLEEVE_TREND_OVERRIDES: dict[str, int] = {
    # "SHY": 50,   # example: shorter trend filter for low-duration legs
    # "IEI": 100,
    # "IEF": 150,
}

# Dynamic sleeve weighting scheme is defined in control panel section 3
# (``MAD_SLEEVE_WEIGHTING_SCHEME``) alongside the stock and index layers.
# The sleeve-specific MRAT pair below is used to score hedge assets when
# that scheme is active; defaults to the same 21/200 pair as the rest.
MAD_SLEEVE_MRAT_SHORT = 21
MAD_SLEEVE_MRAT_LONG = 200


def _build_sleeve_from_hedge_assets() -> tuple[tuple[str, float, int], ...]:
    """Build ``MAD_REGIME_OFF_SLEEVE`` = [(ticker, weight, trend_ma), ...].

    * Safe-harbor ticker is excluded — it is a first-class leg via the dynamic
      weighting path and a fallback in the fixed-weight path, so it should
      not appear twice in the sleeve tuple.
    * Equal fixed weights (1.0 each); normalized at runtime. Dynamic schemes
      ignore these and redistribute per-day.
    * Per-leg trend MA resolved from ``MAD_REGIME_OFF_SLEEVE_TREND_OVERRIDES``
      with ``MAD_REGIME_OFF_SLEEVE_TREND_MA`` as the default.
    """
    safe = (MAD_REGIME_OFF_SAFE_HARBOR or "").strip().upper()
    out: list[tuple[str, float, int]] = []
    for t in _hedge_asset_tickers():
        if t == safe:
            continue
        ma = int(
            MAD_REGIME_OFF_SLEEVE_TREND_OVERRIDES.get(t, MAD_REGIME_OFF_SLEEVE_TREND_MA)
        )
        out.append((t, 1.0, ma))
    return tuple(out)


def _hedge_asset_tickers() -> tuple[str, ...]:
    """Normalized list of hedge-asset symbols (flat tuple, uppercase, de-duped)."""
    raw = HEDGE_ASSETS
    if isinstance(raw, str):
        return (raw.strip().upper(),)
    seen: list[str] = []
    for x in raw:
        t = str(x).strip().upper()
        if t and t not in seen:
            seen.append(t)
    return tuple(seen)


# Public alias (some older call sites use this name).
hedge_asset_tickers = _hedge_asset_tickers

MAD_REGIME_OFF_SLEEVE = _build_sleeve_from_hedge_assets()

# =============================================================================
# LIVE BOT (``mad/live_bot.py``)
#
# Alpaca execution plumbing. Most users only touch BOT_MODE and the equity
# fraction in the control panel; the rest are runtime tuning.
# =============================================================================

# Cycle timing
MAD_LIVE_POLL_SECONDS = 300
# After Alpaca session close, only submit orders if now is within this many
# minutes of ``close`` (US/Eastern). Prevents a bot that starts hours late
# (e.g. 10 p.m.) from placing that session's rebalance. ``0`` = off (any
# time after close). Early closes use the calendar ``close`` as the anchor.
MAD_LIVE_REBALANCE_WINDOW_MINUTES = 90
MAD_LIVE_TRADE_ONLY_WHEN_MARKET_OPEN = False
MAD_LIVE_LOAD_PARAMS_FROM_DB = True

# Live parameter overrides (None = use main MAD_* values).
MAD_LIVE_SMA_SHORT = None
MAD_LIVE_SMA_LONG = None
MAD_LIVE_EXIT_MA = None
MAD_LIVE_REGIME_MA = 200
MAD_LIVE_REGIME_TICKER = None

# Sizing and order placement
MAD_LIVE_MAX_GROSS_USD = None
MAD_LIVE_MIN_ORDER_USD = 1.0
# Alpaca supports fractional qty on market/limit DAY orders. True = equal-dollar
# MRAT targets as share floats (matches backtest); False = whole-share ``floor``.
MAD_LIVE_FRACTIONAL_SHARES = True
# Cancel open orders for a symbol before submitting a new reconcile delta
# (avoids stacking duplicate DAY limits across poll passes). Set False if you
# place manual working orders on the same symbols as the bot.
MAD_LIVE_CANCEL_OPEN_BEFORE_RECONCILE = True

# OHLCV health check at cycle start
MAD_LIVE_OHLCV_HEALTH_CHECK = True
MAD_LIVE_HEALTH_REFERENCE_TICKER = None
MAD_LIVE_OHLCV_RECENT_REF_BARS = 60
MAD_LIVE_OHLCV_MAX_STALE_CALENDAR_DAYS = 1
MAD_LIVE_ABORT_ON_OHLCV_ISSUES = False

# Splitter refresh behavior in live mode
MAD_LIVE_REFRESH_SPLITTER_DB = False
MAD_LIVE_REFRESH_SPLITTER_ONCE_PER_UTC_DAY = True
MAD_LIVE_REFRESH_SPLITTER_ON_STARTUP = True

# When True, ``compute_mad_live_snapshot`` uses precomputed ``sma_<n>``
# columns from SQLite (fast). Falls back to rolling close if missing.
MAD_LIVE_USE_PRECOMPUTED_SMA = True

# After each live cycle, pull missing daily bars into ``data/ohlcv/*.db``,
# recompute ``split``, and set SMA columns. Requires existing DBs (run fetcher
# first). Only active when ``TARGET_CANDLE_GRANULARITY = '1d'``.
MAD_LIVE_APPEND_DAILY_OHLCV = True
MAD_LIVE_APPEND_SLEEP_SEC = 0.05

# Extended-hours orders
MAD_LIVE_EXTENDED_HOURS_ORDERS = True
# When False (default), extended-hours limit price is anchored to live
# ask (buys) / bid (sells) with a ±1% cushion — not the daily bar close.
MAD_LIVE_EXT_HRS_LIMIT_FROM_DAILY_CLOSE = False

# Risk-off live behavior (used only when allocator is off)
MAD_LIVE_REGIME_OFF_PROXY_TICKER = "BIL"
MAD_LIVE_REGIME_OFF_CLOSE_ALL_NON_PROXY = True
MAD_LIVE_REGIME_OFF_EQUITY_FRACTION = 0.995

# Alpaca connection retries
MAD_LIVE_ALPACA_CONNECT_RETRIES = 5
MAD_LIVE_ALPACA_CONNECT_RETRY_SEC = 2.0

# =============================================================================
# PERMUTATION TEST (``mad/permutation_test.py``)
#
# Research tool for significance testing. Rarely touched in production.
# =============================================================================

MAD_PERM_N = 10_000
MAD_PERM_ALPHA = 0.05
MAD_PERM_BLOCK_SIZE = 5
MAD_PERM_PORT = 8065
MAD_PERM_OPTIM_SPLIT = "avg"
MAD_PERM_IS_SPLITS = MAD_IS_SPLITS


# =============================================================================
# HELPER FUNCTIONS (consumed by the fetcher, splitter, and bot)
# =============================================================================


def bot_mode_is_paper() -> bool:
    """True if ``BOT_MODE`` selects Alpaca paper trading; False for live."""
    m = str(globals().get("BOT_MODE", "paper")).strip().lower()
    if m == "paper":
        return True
    if m in ("cash", "live"):
        return False
    raise ValueError(
        f"Invalid BOT_MODE={m!r}. Use 'paper' or 'cash' (alias 'live' for live account)."
    )


def ohlcv_download_start_utc() -> datetime:
    s = str(globals().get("TARGET_START_DATE", "2010-01-01")).strip()
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def ohlcv_download_end_utc() -> datetime:
    mode = str(globals().get("OHLCV_DOWNLOAD_END_MODE", "fixed")).strip().lower()
    if mode in ("utc_now", "now", "live"):
        return datetime.now(timezone.utc)
    if mode == "fixed":
        e = str(globals().get("TARGET_END_DATE", "2099-12-31")).strip()
        return datetime.fromisoformat(e).replace(tzinfo=timezone.utc)
    raise ValueError(
        f"Invalid OHLCV_DOWNLOAD_END_MODE={mode!r}. Use 'fixed' or 'utc_now'."
    )


def ohlcv_pipeline_tickers() -> tuple[str, ...]:
    """Symbols fetched by ``alpaca_fetcher`` / ``data_splitter`` (one DB each).

    Always starts from ``MAD_UNIVERSE_TICKERS`` (risk-on), then unions in:
      * regime gate ticker (``MAD_REGIME_TICKER`` if ``MAD_REGIME_MA_ENABLED``)
      * hedge assets (if ``OHLCV_PIPELINE_INCLUDE_HEDGE_ASSETS``)
      * multi-index ETFs + all slot constituents (if allocator is ON —
        fetched for ALL slots even if some are filtered out by
        ``MAD_INDEX_ENABLED_ETFS``, so toggling a slot back on is instant)
      * regime-sleeve + multi-ticker-regime symbols
    """
    raw = MAD_UNIVERSE_TICKERS
    base: tuple[str, ...] = (
        (raw.strip().upper(),)
        if isinstance(raw, str)
        else tuple(str(x).strip().upper() for x in raw if str(x).strip())
    )
    if MAD_REGIME_MA_ENABLED:
        rt = (MAD_REGIME_TICKER or "").strip().upper()
        if rt and rt not in base:
            base = (*base, rt)
    if OHLCV_PIPELINE_INCLUDE_HEDGE_ASSETS:
        for t in _hedge_asset_tickers():
            if t and t not in base:
                base = (*base, t)
    if MAD_INDEX_ALLOCATOR_ENABLED:
        # Fetch EVERY slot's tickers, ignoring ``MAD_INDEX_ENABLED_ETFS``,
        # so the user can flip an index on/off without a re-fetch.
        for etf, universe, _label in MAD_INDEX_SLOTS:
            etf_sym = (etf or "").strip().upper()
            if etf_sym and etf_sym not in base:
                base = (*base, etf_sym)
            raw_universe = universe if not isinstance(universe, str) else (universe,)
            for t in raw_universe:
                tt = str(t).strip().upper()
                if tt and tt not in base:
                    base = (*base, tt)
    try:
        from deepvibe_hedge.mad.regime_sleeve import (  # noqa: PLC0415
            sleeve_and_regime_symbols,
        )
        import sys as _sys  # noqa: PLC0415

        extras = sleeve_and_regime_symbols(_sys.modules[__name__])
        for t in extras:
            if t and t not in base:
                base = (*base, t)
    except Exception:
        pass
    return base


def splitter_ma_periods() -> tuple[int, ...]:
    """SMA lookbacks written to each OHLCV DB (``sma_<n>`` columns only).

    Union of: MRAT short/long grids, positive exit-MA grid, positive regime-MA
    grid, live regime MA, the multi-ticker regime MA, and every sleeve
    member's trend MA — matches what the backtest grid and live snapshot
    may use.
    """
    periods: set[int] = set()
    for grid in (MAD_SMA_SHORT_GRID, MAD_SMA_LONG_GRID):
        periods.update(int(x) for x in grid)
    periods.update(int(x) for x in MAD_EXIT_MA_GRID if int(x) > 0)
    periods.update(int(x) for x in MAD_REGIME_MA_GRID if int(x) > 0)
    live_r = MAD_LIVE_REGIME_MA
    if live_r is not None and int(live_r) > 0:
        periods.add(int(live_r))
    rm = MAD_REGIME_MA
    if rm is not None and int(rm) > 0:
        periods.add(int(rm))
    for _sym, _w, tma in MAD_REGIME_OFF_SLEEVE or ():
        if int(tma) > 0:
            periods.add(int(tma))
    if MAD_INDEX_ALLOCATOR_ENABLED:
        if int(MAD_INDEX_REGIME_MA) > 0:
            periods.add(int(MAD_INDEX_REGIME_MA))
        if int(MAD_INDEX_MRAT_SHORT) > 0:
            periods.add(int(MAD_INDEX_MRAT_SHORT))
        if int(MAD_INDEX_MRAT_LONG) > 0:
            periods.add(int(MAD_INDEX_MRAT_LONG))
    ordered = sorted(periods)
    if not ordered:
        raise ValueError("splitter_ma_periods(): empty — set MAD_SMA_*_GRID and related grids.")
    return tuple(ordered)


def splitter_warmup_min_calendar_days() -> int:
    """Distinct calendar days kept in split 0 — equals longest SMA period."""
    return max(splitter_ma_periods())
