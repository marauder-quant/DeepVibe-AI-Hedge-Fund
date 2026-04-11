"""
MAD / MRAT standalone configuration (data pipeline, splitter, strategy, live).

Typical flow
------------
1. Set universe, bar size, and date range below.
2. ``PYTHONPATH=src python -m deepvibe_hedge.alpaca_fetcher`` → writes
   ``<project>/data/ohlcv/{TICKER}_{gran}.db`` and ``.csv`` (``data/`` is gitignored but present on disk).
3. ``PYTHONPATH=src python -m deepvibe_hedge.data_splitter`` → updates those DBs/CSVs in place (splits + SMAs).
4. ``PYTHONPATH=src python -m deepvibe_hedge.mad.backtester``
5. Live: ``PYTHONPATH=src python -m deepvibe_hedge.mad.live_bot`` (after OHLCV is current).

Inspect OHLCV: ``PYTHONPATH=src python -m deepvibe_hedge.db_utils`` (overview of all DBs under ``data/ohlcv/``).
"""
from __future__ import annotations

from datetime import datetime, timezone

from deepvibe_hedge.nasdaq100 import nasdaq100

# -----------------------------------------------------------------------------
# Data pipeline
# -----------------------------------------------------------------------------

MAD_UNIVERSE_TICKERS = nasdaq100

TARGET_TICKER = "QQQ"

OHLCV_PIPELINE_MODE = "mad_universe"  # "mad_universe" | "target_only"


def ohlcv_pipeline_tickers() -> tuple[str, ...]:
    """Symbols for ``alpaca_fetcher`` / ``data_splitter`` (one ``data/ohlcv/{SYM}_*.db`` each)."""
    mode = OHLCV_PIPELINE_MODE.strip().lower()
    if mode == "target_only":
        base = (TARGET_TICKER.strip().upper(),)
    elif mode in ("mad_universe", "universe", "mad"):
        raw = MAD_UNIVERSE_TICKERS
        base = (
            (raw.strip().upper(),)
            if isinstance(raw, str)
            else tuple(str(x).strip().upper() for x in raw if str(x).strip())
        )
    else:
        raise ValueError(
            f"Invalid OHLCV_PIPELINE_MODE={mode!r}. Use 'mad_universe' or 'target_only'."
        )
    if MAD_REGIME_MA_ENABLED:
        rt = (MAD_REGIME_TICKER or "").strip().upper()
        if rt and rt not in base:
            base = (*base, rt)
    return base


TARGET_CANDLE_GRANULARITY = "1d"

TARGET_START_DATE = "2010-01-01"
TARGET_END_DATE = "now"
OHLCV_DOWNLOAD_END_MODE = "utc_now"


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


ALPACA_BAR_ADJUSTMENT = "split"
BACKTEST_FEE_RATE = 0.001

# Alpaca market data feed for historical bars (``alpaca_fetcher``).
LIVE_BOT_DATA_FEED = "iex"
# Alpaca account: ``paper`` = paper API; ``cash`` = live brokerage (real money). Optional alias: ``live``.
BOT_MODE = "cash"


def bot_mode_is_paper() -> bool:
    """True if ``BOT_MODE`` selects Alpaca paper trading; False for live (``cash`` / ``live``)."""
    m = str(globals().get("BOT_MODE", "paper")).strip().lower()
    if m == "paper":
        return True
    if m in ("cash", "live"):
        return False
    raise ValueError(
        f"Invalid BOT_MODE={m!r}. Use 'paper' or 'cash' (alias 'live' for live account)."
    )


LIVE_BOT_ALLOW_SHORT = True

# -----------------------------------------------------------------------------
# Data splitter (walk-forward splits + SMA precompute on OHLCV SQLite)
#
# Split 0 = warmup so the longest SMA in ``splitter_ma_periods()`` is valid on the first bar of split 1.
# SMA periods are derived from MAD grids (see ``splitter_ma_periods`` at end of this file).
# -----------------------------------------------------------------------------

SPLITTER_NUM_SPLITS = 10

SPLITTER_ENABLE_SPLIT_ASSIGNMENT = True
SPLITTER_ENABLE_MA_PRECOMPUTE = True

SPLITTER_DB_WRITE_RETRIES = 6
SPLITTER_DB_WRITE_RETRY_SEC = 5

SPLIT_PLAN_IN_SAMPLE = (1, 3, 5, 7, 9)
SPLIT_PLAN_OUT_OF_SAMPLE = (2, 4, 6, 8, 10)

# -----------------------------------------------------------------------------
# MAD / MRAT
# -----------------------------------------------------------------------------

MAD_DIRECTION_MODE = "long_only"

MAD_SMA_SHORT = 21
MAD_SMA_LONG = 200
MAD_SMA_SHORT_GRID = (21,)
MAD_SMA_LONG_GRID = (200,)

MAD_EXIT_MA_ENABLED = False
MAD_EXIT_MA_PERIOD = 0
MAD_EXIT_MA_GRID = (0, 50, 100, 150, 200)

MAD_REGIME_MA_ENABLED = True
MAD_REGIME_TICKER = "QQQ"
MAD_REGIME_MA_GRID = (0, 50, 100, 150, 200)

MAD_LONG_SIGMA_MULT = 1.0
MAD_SHORT_SIGMA_MULT = 1.0
MAD_SYMMETRIC_SHORT_SIGMA = False

MAD_LONG_DECILE_MIN = 10
MAD_SHORT_DECILE_MAX = 1
MAD_MIN_HISTORY_BARS = 252
MAD_AGGREGATE_TO_DAILY = True

MAD_EVAL_ALL_SPLITS = True

MAD_IS_SPLITS = 6
MAD_OOS_SPLITS = 4

MAD_DASHBOARD_PORT = 8063
MAD_WF_DASHBOARD_PORT = 8064
MAD_WF_OPTIM_SPLIT = "avg"
MAD_WF_OOS_SPLIT = "all"

MAD_PERM_N = 10_000
MAD_PERM_ALPHA = 0.05
MAD_PERM_BLOCK_SIZE = 5
MAD_PERM_PORT = 8065
MAD_PERM_OPTIM_SPLIT = "avg"
MAD_PERM_IS_SPLITS = MAD_IS_SPLITS

MAD_LIVE_POLL_SECONDS = 300  # How often live_bot wakes to check for EOD rebalance time.
# After Alpaca session close, only submit orders if now is within this many minutes of ``close``
# (US/Eastern). Stops a bot that starts hours later (e.g. 10 p.m.) from placing that session's
# rebalance. Extended-hours limits still apply; this only gates *when* the cycle runs. ``0`` = off
# (legacy: any time after close). Early closes use the calendar ``close`` time as the anchor.
MAD_LIVE_REBALANCE_WINDOW_MINUTES = 90
# Unused by live_bot: reconcile runs once per session after Alpaca calendar close (US/Eastern).
MAD_LIVE_TRADE_ONLY_WHEN_MARKET_OPEN = False
MAD_LIVE_LOAD_PARAMS_FROM_DB = True

MAD_LIVE_SMA_SHORT = None
MAD_LIVE_SMA_LONG = None
MAD_LIVE_EXIT_MA = None
MAD_LIVE_REGIME_MA = 200
MAD_LIVE_REGIME_TICKER = None

MAD_LIVE_EQUITY_FRACTION = 0.98
MAD_LIVE_MAX_GROSS_USD = None
MAD_LIVE_MIN_ORDER_USD = 1.0
# Alpaca supports fractional ``qty`` on market/limit DAY orders (incl. many extended-hours limits).
# True → equal-dollar MRAT targets as share floats (matches backtest); False → whole-share ``floor``.
MAD_LIVE_FRACTIONAL_SHARES = True

# Before submitting a reconcile delta for a symbol, cancel all **open** Alpaca orders for that symbol
# (avoids stacking duplicate DAY limits across poll passes). Filled positions are unchanged; next
# cycle still places a new order if desired ≠ filled qty. Set False if you place manual working orders
# on the same symbols as the bot.
MAD_LIVE_CANCEL_OPEN_BEFORE_RECONCILE = True

MAD_LIVE_OHLCV_HEALTH_CHECK = True
MAD_LIVE_HEALTH_REFERENCE_TICKER = None
MAD_LIVE_OHLCV_RECENT_REF_BARS = 60
MAD_LIVE_OHLCV_MAX_STALE_CALENDAR_DAYS = 1
MAD_LIVE_ABORT_ON_OHLCV_ISSUES = False

MAD_LIVE_REFRESH_SPLITTER_DB = False
MAD_LIVE_REFRESH_SPLITTER_ONCE_PER_UTC_DAY = True
MAD_LIVE_REFRESH_SPLITTER_ON_STARTUP = True

# When True, ``compute_mad_live_snapshot`` uses ``sma_<n>`` columns from OHLCV SQLite (from
# ``data_splitter``) for MRAT ma_s/ma_l (and exit SMA when present). Falls back to rolling close
# if columns are missing. Regime uses precomputed SMA only for ``TARGET_CANDLE_GRANULARITY`` 1d
# (intraday DB SMAs are not the same as daily regime MA).
MAD_LIVE_USE_PRECOMPUTED_SMA = True

# After each live cycle (non-dry-run), pull missing daily bars from Alpaca into ``data/ohlcv/*.db``,
# recompute ``split``, and set ``sma_21`` + ``sma_200`` on universe names; regime ETF (``MAD_REGIME_TICKER``
# / ``QQQ``) gets ``sma_200`` only (``sma_21`` cleared). Requires existing DBs (run fetcher first).
# Only active when ``TARGET_CANDLE_GRANULARITY`` is ``1d``.
MAD_LIVE_APPEND_DAILY_OHLCV = True
MAD_LIVE_APPEND_SLEEP_SEC = 0.05

MAD_LIVE_EXTENDED_HOURS_ORDERS = True
# When False (default), extended-hours **limit** price is anchored to live **ask** (buys) / **bid**
# (sells), then ``alpaca_live._extended_hours_limit_price`` applies the ±1% cushion — not MRAT's
# daily bar close (which is stale vs after-hours prints). Set True only to restore old behavior.
MAD_LIVE_EXT_HRS_LIMIT_FROM_DAILY_CLOSE = False
MAD_LIVE_REGIME_OFF_PROXY_TICKER = "BIL"
MAD_LIVE_REGIME_OFF_CLOSE_ALL_NON_PROXY = True
MAD_LIVE_REGIME_OFF_EQUITY_FRACTION = 0.995

MAD_LIVE_ALPACA_CONNECT_RETRIES = 5
MAD_LIVE_ALPACA_CONNECT_RETRY_SEC = 2.0


def splitter_ma_periods() -> tuple[int, ...]:
    """
    SMA lookbacks written to each OHLCV DB (``sma_<n>`` only).

    Union of MRAT short/long grids, positive exit-MA grid, positive regime-MA grid, and live regime MA
    when set — matches what the backtest grid and live snapshot may use.
    """
    periods: set[int] = set()
    for grid in (MAD_SMA_SHORT_GRID, MAD_SMA_LONG_GRID):
        periods.update(int(x) for x in grid)
    periods.update(int(x) for x in MAD_EXIT_MA_GRID if int(x) > 0)
    periods.update(int(x) for x in MAD_REGIME_MA_GRID if int(x) > 0)
    live_r = MAD_LIVE_REGIME_MA
    if live_r is not None and int(live_r) > 0:
        periods.add(int(live_r))
    ordered = sorted(periods)
    if not ordered:
        raise ValueError("splitter_ma_periods(): empty — set MAD_SMA_*_GRID and related grids.")
    return tuple(ordered)


def splitter_warmup_min_calendar_days() -> int:
    """Distinct calendar days kept in split 0 (with MA precompute) — equals longest SMA period."""
    return max(splitter_ma_periods())
