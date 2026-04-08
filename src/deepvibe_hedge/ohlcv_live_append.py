"""
Append the latest daily OHLCV bar from Alpaca into each local SQLite DB (``data/ohlcv``), then
recompute walk-forward ``split`` and live SMA columns.

- **Universe names** (``mad_universe_tickers()``): ``sma_21`` and ``sma_200`` from ``close``
  (splitter-style ``rolling(..., min_periods=...).mean().round(4)``).
- **Regime ETF** (``MAD_REGIME_TICKER`` or ``QQQ``): ``sma_200`` only; ``sma_21`` is cleared (NaN).

Used by ``mad.live_bot`` so databases stay current without re-running the full fetcher. Intended for
``TARGET_CANDLE_GRANULARITY`` ``1d`` only.
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from alpaca.data.historical.stock import StockHistoricalDataClient

from deepvibe_hedge import config
from deepvibe_hedge.alpaca_fetcher import _make_client, fetch_ohlcv_between
from deepvibe_hedge.data_splitter import assign_splits, save_back, _filename
from deepvibe_hedge.mad.backtester import mad_regime_ticker_symbol, mad_universe_tickers
from deepvibe_hedge.paths import OHLCV_DIR, ensure_data_dirs


def _regime_etf_symbol() -> str:
    t = mad_regime_ticker_symbol()
    if t:
        return str(t).strip().upper()
    return (getattr(config, "MAD_REGIME_TICKER", None) or "QQQ").strip().upper()


def live_ohlcv_append_symbols() -> tuple[str, ...]:
    syms = set(mad_universe_tickers())
    syms.add(_regime_etf_symbol())
    return tuple(sorted(syms))


def _load_ohlcv_table_all_cols(db_path) -> pd.DataFrame | None:
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql("SELECT * FROM ohlcv", con, parse_dates=["timestamp"])
    if df.empty:
        return None
    df = df.set_index("timestamp").sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def _merge_new_bars(existing: pd.DataFrame, new_bars: pd.DataFrame) -> pd.DataFrame:
    if new_bars.empty:
        return existing
    new_bars = new_bars.copy()
    extra_cols = [c for c in existing.columns if c not in new_bars.columns]
    for c in extra_cols:
        new_bars[c] = np.nan
    all_cols = sorted(set(existing.columns) | set(new_bars.columns))
    new_bars = new_bars.reindex(columns=all_cols)
    existing = existing.reindex(columns=all_cols)
    combined = pd.concat([existing, new_bars])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


def _apply_live_sma_columns(df: pd.DataFrame, symbol_upper: str) -> pd.DataFrame:
    close = df["close"].astype(float)
    sma200 = close.rolling(200, min_periods=200).mean().round(4)
    sma21 = close.rolling(21, min_periods=21).mean().round(4)
    regime = _regime_etf_symbol()
    out = df.copy()
    if symbol_upper == regime:
        out["sma_200"] = sma200
        out["sma_21"] = np.nan
    else:
        out["sma_21"] = sma21
        out["sma_200"] = sma200
    return out


def _apply_splits_best_effort(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return assign_splits(df.copy())
    except ValueError:
        return df


def append_latest_daily_for_symbol(
    symbol: str,
    *,
    client: StockHistoricalDataClient,
    quiet: bool = True,
) -> str:
    """
    Returns ``"ok"`` | ``"skip"`` | ``"no_db"`` | ``"empty_fetch"`` | ``"error"``.
    """
    sym = str(symbol).strip().upper()
    gran = str(getattr(config, "TARGET_CANDLE_GRANULARITY", "1d")).strip().lower()
    if gran != "1d":
        return "skip"

    db_path = OHLCV_DIR / f"{_filename(sym)}.db"
    existing = _load_ohlcv_table_all_cols(db_path)
    if existing is None:
        return "no_db"

    last_ts = existing.index.max()
    start = last_ts + timedelta(microseconds=1)
    end = datetime.now(timezone.utc)
    if start >= end:
        return "skip"

    try:
        new_bars = fetch_ohlcv_between(sym, start, end, client=client)
    except Exception:
        return "error"

    if new_bars.empty:
        return "empty_fetch"

    for c in ("open", "high", "low", "close", "volume"):
        if c not in new_bars.columns:
            if c == "volume":
                new_bars["volume"] = 0.0
            elif c == "open":
                new_bars["open"] = new_bars["close"]
            else:
                return "error"

    merged = _merge_new_bars(existing, new_bars)
    merged = _apply_splits_best_effort(merged)
    merged = _apply_live_sma_columns(merged, sym)

    save_back(merged, sym, verbose=not quiet)
    if not quiet:
        print(f"[ohlcv_live_append] {sym}: +{len(new_bars)} row(s) → saved {db_path.name}")
    return "ok"


def append_latest_daily_for_universe(
    *,
    client: StockHistoricalDataClient | None = None,
    quiet: bool = True,
) -> dict[str, str]:
    """
    Update every universe + regime ETF DB. Creates client if None.
    Returns per-symbol status strings.
    """
    if str(getattr(config, "TARGET_CANDLE_GRANULARITY", "1d")).strip().lower() != "1d":
        if not quiet:
            print("[ohlcv_live_append] skip: TARGET_CANDLE_GRANULARITY is not 1d")
        return {}

    ensure_data_dirs()
    c = client or _make_client()
    out: dict[str, str] = {}
    for sym in live_ohlcv_append_symbols():
        out[sym] = append_latest_daily_for_symbol(sym, client=c, quiet=quiet)
        time.sleep(float(getattr(config, "MAD_LIVE_APPEND_SLEEP_SEC", 0.05)))
    return out


def summarize_append_status(status: dict[str, str]) -> str:
    if not status:
        return ""
    ok = sum(1 for v in status.values() if v == "ok")
    no_db = [k for k, v in status.items() if v == "no_db"]
    errs = [k for k, v in status.items() if v == "error"]
    parts = [f"updated={ok}/{len(status)}"]
    if no_db:
        parts.append(f"missing_db={len(no_db)}")
    if errs:
        parts.append(f"errors={len(errs)}")
    return " | ".join(parts)
