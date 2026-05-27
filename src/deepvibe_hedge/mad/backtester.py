"""
Moving Average Distance (MRAT) cross-sectional panel backtest.

MRAT = SMA(short) / SMA(long) per ticker. Each date: cross-sectional σ of MRAT, deciles
via rank percentiles; long when top decile and MRAT > 1 + k_long·σ; short when bottom decile
and MRAT < 1 − k_short·σ (k from config MAD_*_SIGMA_MULT, default 1). Equal-weight portfolio;
optional daily aggregation
from intraday OHLCV (see config MAD_AGGREGATE_TO_DAILY).

Interpretation
--------------
- Buy & hold in the dashboard is the equal-weight 1/N daily return of the full universe
  (nan-mean of each name's close-to-close return), not a single ticker.
- trades / rebalance_days counts calendar days where portfolio weights changed (turnover),
  not stock fills or round-trips. Rare values are normal when entry rules are strict.
- Long-only (``MAD_DIRECTION_MODE``) disables the short leg; ``both`` enables shorts when names hit
  the short band (decile ≤ ``MAD_SHORT_DECILE_MAX``, MRAT < 1 − k·σ), equal-weight −1/n per short.
  Turn regime off (e.g. ``MAD_REGIME_MA_GRID`` includes 0) to allow shorts without the ETF filter.
  ``MAD_SYMMETRIC_SHORT_SIGMA`` uses the same k as longs on the short margin.
- The +sigma term uses **cross-sectional** σ(MRAT) each day. Broad simultaneous ramps widen σ, so
  MRAT > 1+σ can fail for every name even when many are above their long MA (see MAD_LONG_SIGMA_MULT).
- days_with_position counts scored days with non-zero gross exposure (sum of |w| > 0).
- Optional **market regime** (e.g. QQQ): if MAD_REGIME_MA_ENABLED and regime SMA length > 0, the whole
  MRAT book is flat when regime ETF close was not above its SMA on the prior bar (same timing as entry).

Run:
    PYTHONPATH=src python -m deepvibe_hedge.mad.backtester
    PYTHONPATH=src python -m deepvibe_hedge.mad.backtester --no-dashboard

Live (Alpaca): refresh OHLCV, then ``PYTHONPATH=src python -m deepvibe_hedge.mad.live_bot`` (see ``MAD_LIVE_*`` in config).
"""
from __future__ import annotations

import argparse
import itertools
import re
import sqlite3
from dataclasses import dataclass, field, replace as dc_replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, dash_table, dcc, html
import plotly.graph_objects as go

from deepvibe_hedge import config
from deepvibe_hedge.breakout_plotting import (
    AVG_KEY,
    AVG_SLIDER_VAL,
    bars_per_year,
    comparison_stats_df,
    fig_equity,
    format_stats,
)
from deepvibe_hedge.paths import MAD_DATA_DIR, OHLCV_DIR
from deepvibe_hedge.mad.weighting import (
    WeightConfig,
    WeightInputs,
    compute_weights,
    realized_vol_pivot,
    resolve_weight_config,
)
from deepvibe_hedge.mad.regime_sleeve import (
    RegimeBacktestSeries,
    build_regime_backtest_series,
    describe_sleeve_config,
    evaluate_regime_live,
    evaluate_sleeve_allocation_live,
    format_regime_state_line,
    resolve_regime_ma,
    resolve_regime_mode,
    resolve_regime_tickers,
    resolve_safe_harbor,
    resolve_sleeve,
    resolve_sleeve_mrat_pair,
    resolve_sleeve_weight_config,
    resolve_sleeve_weighting_scheme,
)
from deepvibe_hedge.mad.index_allocator import (
    RISK_OFF_KEY,
    build_index_allocation_series,
    evaluate_index_allocation_live,
    filter_universe_by_data_completeness,
    format_index_allocation_banner,
    index_allocator_enabled,
    resolve_index_mrat_pair,
    resolve_index_regime_ma,
    resolve_index_slots,
    resolve_index_weight_config,
    resolve_min_data_completeness,
)
from deepvibe_hedge.walkforward_oos_common import read_explicit_split_plan_from_config, resolve_split_plan

# Panel bar index and ``split`` ids: all universe names reindexed onto this symbol (``build_panel_long``).
# Resolution order: first MAD_INDEX_SLOTS ETF (when allocator is on), else MAD_REGIME_TICKER.


def mad_reference_ticker() -> str:
    """Panel-calendar + ``RESULTS_DB`` reference ticker.

    Under the multi-index allocator, there is no single "target" symbol — we
    pick the first *enabled* slot's ETF (honoring ``MAD_INDEX_ENABLED_ETFS``
    whitelist if set) as the calendar anchor. When the allocator is off, fall
    back to ``MAD_REGIME_TICKER``.
    """
    if bool(getattr(config, "MAD_INDEX_ALLOCATOR_ENABLED", False)):
        slots = getattr(config, "MAD_INDEX_SLOTS", ())
        enabled_raw = getattr(config, "MAD_INDEX_ENABLED_ETFS", None)
        enabled = (
            {str(t).strip().upper() for t in enabled_raw if str(t).strip()}
            if enabled_raw
            else None
        )
        for entry in slots:
            if not entry:
                continue
            etf = str(entry[0]).strip().upper()
            if not etf:
                continue
            if enabled is not None and etf not in enabled:
                continue
            return etf
    raw = getattr(config, "MAD_REGIME_TICKER", None)
    if raw is not None and str(raw).strip():
        return str(raw).strip().upper()
    raise RuntimeError(
        "mad_reference_ticker: need MAD_INDEX_SLOTS (allocator on) or "
        "MAD_REGIME_TICKER (allocator off) to resolve a reference ticker."
    )

# Cross-sectional MRAT eligibility (not user config).
MAD_DEFAULT_MIN_PRICE = 5.0

RESULTS_DB = MAD_DATA_DIR / f"{mad_reference_ticker()}_{config.TARGET_CANDLE_GRANULARITY}_mad_optim.db"
DATASETS_DIR = MAD_DATA_DIR
PORT = int(getattr(config, "MAD_DASHBOARD_PORT", 8063))


def mad_universe_tickers() -> tuple[str, ...]:
    """Union of every ticker the MAD backtest / live bot can hold.

    * Allocator ON  → union of every ``MAD_INDEX_SLOTS`` constituent universe
      (deduped, preserving first-seen slot order). This is what
      ``build_panel_long`` needs so ``evaluate_mad_multi_index`` can filter
      ``daily_long`` to each slot's exclusive universe and actually find rows.
    * Allocator OFF → legacy ``MAD_UNIVERSE_TICKERS`` only.

    Raises if the resulting universe is empty.
    """
    if bool(getattr(config, "MAD_INDEX_ALLOCATOR_ENABLED", False)):
        seen: list[str] = []
        slots = getattr(config, "MAD_INDEX_SLOTS", ())
        enabled_raw = getattr(config, "MAD_INDEX_ENABLED_ETFS", None)
        enabled_set = (
            {str(t).strip().upper() for t in enabled_raw if str(t).strip()}
            if enabled_raw
            else None
        )
        for entry in slots:
            if not entry:
                continue
            etf_sym = str(entry[0]).strip().upper()
            if enabled_set is not None and etf_sym not in enabled_set:
                continue
            univ = entry[1]
            raw_univ = univ if not isinstance(univ, str) else (univ,)
            for t in raw_univ:
                tt = str(t).strip().upper()
                if tt and tt not in seen:
                    seen.append(tt)
        if not seen:
            raise RuntimeError(
                "mad_universe_tickers: MAD_INDEX_ALLOCATOR_ENABLED=True but "
                "MAD_INDEX_SLOTS produced an empty union of constituents."
            )
        return tuple(seen)

    raw = getattr(config, "MAD_UNIVERSE_TICKERS", None)
    if raw is None:
        raise RuntimeError(
            "mad_universe_tickers: MAD_UNIVERSE_TICKERS is unset in config."
        )
    if isinstance(raw, str):
        return (raw.strip().upper(),)
    return tuple(str(x).strip().upper() for x in raw if str(x).strip())


def _normalize_direction_mode(raw: str | None) -> str:
    mode = str(raw or "both").strip().lower()
    aliases = {
        "both": "both",
        "long": "long_only",
        "long_only": "long_only",
        "short": "short_only",
        "short_only": "short_only",
    }
    if mode not in aliases:
        raise ValueError(f"Invalid MAD_DIRECTION_MODE={raw!r}. Use: both, long_only, short_only")
    return aliases[mode]


def _load_one_ohlcv(
    db_path: Path,
    *,
    sma_periods: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"OHLCV DB not found: {db_path}")
    with sqlite3.connect(db_path) as con:
        cols = [row[1] for row in con.execute("PRAGMA table_info(ohlcv)").fetchall()]
        colset = set(cols)
        wanted = [c for c in ("timestamp", "open", "close", "split") if c in colset]
        if "timestamp" not in wanted or "close" not in wanted:
            raise RuntimeError(f"{db_path} missing timestamp/close on ohlcv table.")
        extra: list[str] = []
        if sma_periods:
            for p in sma_periods:
                c = f"sma_{int(p)}"
                if c in colset:
                    extra.append(c)
        extra = sorted(set(extra), key=lambda x: int(x.split("_")[1]))
        select_cols = wanted + extra
        df = pd.read_sql(
            f"SELECT {', '.join(select_cols)} FROM ohlcv",
            con,
            parse_dates=["timestamp"],
        )
    df = df.set_index("timestamp").sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    if "open" not in df.columns:
        df["open"] = df["close"]
    if "split" not in df.columns:
        df["split"] = 0
    return df


def build_panel_long(
    universe: tuple[str, ...],
    granularity: str,
    reference_ticker: str,
    ohlcv_dir: Path,
    *,
    include_sma_periods: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    ref_path = ohlcv_dir / f"{reference_ticker}_{granularity}.db"
    ref = _load_one_ohlcv(ref_path, sma_periods=None)
    ref_idx = ref.index
    rows: list[pd.DataFrame] = []
    missing: list[str] = []
    for t in universe:
        p = ohlcv_dir / f"{t}_{granularity}.db"
        if not p.exists():
            missing.append(t)
            continue
        df = _load_one_ohlcv(p, sma_periods=include_sma_periods)
        aligned = df.reindex(ref_idx)
        sub = pd.DataFrame(
            {
                "timestamp": ref_idx,
                "ticker": t,
                "open": aligned["open"].to_numpy(),
                "close": aligned["close"].to_numpy(),
                "split": ref["split"].to_numpy(),
            }
        )
        for c in aligned.columns:
            if c.startswith("sma_") and c not in sub.columns:
                sub[c] = aligned[c].to_numpy()
        sub = sub.dropna(subset=["close"])
        rows.append(sub)
    if missing:
        print(f"[MAD] Warning: missing OHLCV DB for {len(missing)} ticker(s): {missing[:20]}{'...' if len(missing) > 20 else ''}")
    if not rows:
        raise RuntimeError("No universe tickers loaded; check MAD_UNIVERSE_TICKERS and OHLCV files.")
    out = pd.concat(rows, ignore_index=True)
    return out


def _load_regime_daily_close(
    ticker: str,
    granularity: str,
    ohlcv_dir: Path,
    *,
    aggregate_to_daily: bool,
) -> pd.Series:
    """One row per UTC calendar day; last close. Index tz-aware UTC normalized."""
    sym = str(ticker).strip().upper()
    path = ohlcv_dir / f"{sym}_{granularity}.db"
    if not path.exists():
        raise FileNotFoundError(
            f"Regime OHLCV missing: {path}. Fetch {sym} at {granularity} "
            "(``ohlcv_pipeline_tickers()`` includes MAD_REGIME_TICKER when MAD_REGIME_MA_ENABLED)."
        )
    df = _load_one_ohlcv(path)
    if aggregate_to_daily and str(granularity).lower() != "1d":
        tmp = df.reset_index()
        tmp["day"] = pd.to_datetime(tmp["timestamp"], utc=True).dt.normalize()
        daily = tmp.groupby("day", sort=True)["close"].last()
        daily.index = pd.DatetimeIndex(daily.index, tz="UTC")
        s = daily.astype(float)
    else:
        s = pd.Series(
            df["close"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(df.index, tz="UTC").normalize(),
            dtype=float,
        )
        s = s.groupby(level=0).last()
    if s.index.duplicated().any():
        s = s.groupby(level=0).last()
    return s


def _load_sleeve_ret_piv(
    tickers: tuple[str, ...],
    granularity: str,
    *,
    daily_agg: bool,
    ohlcv_dir: Path,
) -> pd.DataFrame:
    """Daily pct-change pivot for sleeve / safe-harbor tickers on the panel calendar.

    Columns match ``tickers`` exactly (missing DBs emit NaN so the caller falls back
    to zero weights for that symbol). Returns are ``close.pct_change()`` — same convention
    as ``daily_ret`` in the MRAT panel.
    """
    if not tickers:
        return pd.DataFrame()
    frames: dict[str, pd.Series] = {}
    for sym in tickers:
        path = ohlcv_dir / f"{sym}_{granularity}.db"
        if not path.exists():
            continue
        try:
            close = _load_regime_daily_close(
                sym, granularity, ohlcv_dir, aggregate_to_daily=daily_agg
            )
        except Exception:
            continue
        frames[sym] = close.pct_change()
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1, sort=False).sort_index()
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    return df


def _regime_entry_allow_series(close: pd.Series, ma_period: int) -> pd.Series:
    """Same bar timing as MRAT: risk-on today uses prior close vs SMA (shift(1))."""
    if int(ma_period) <= 0:
        return pd.Series(True, index=close.index, dtype=bool)
    sma = _sma(close, int(ma_period))
    above = close > sma
    return above.shift(1).fillna(False).astype(bool)


def resolve_backtest_window() -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Resolve ``MAD_BACKTEST_START_DATE`` / ``MAD_BACKTEST_END_DATE`` from config.

    Returns a ``(start_ts, end_ts)`` tuple of UTC-normalized ``pd.Timestamp`` (or
    ``None`` when the bound is unset). Used to trim both the strategy and the
    buy-and-hold benchmark to the same eval window, so the dashboard doesn't
    show a long flat warmup stretch while B&H is already fully invested.
    """
    def _parse(val: object) -> pd.Timestamp | None:
        if val is None:
            return None
        s = str(val).strip()
        if not s:
            return None
        ts = pd.Timestamp(s)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.normalize()

    start = _parse(getattr(config, "MAD_BACKTEST_START_DATE", None))
    end = _parse(getattr(config, "MAD_BACKTEST_END_DATE", None))
    if start is not None and end is not None and end < start:
        raise RuntimeError(
            f"MAD_BACKTEST_END_DATE ({end.date()}) is before MAD_BACKTEST_START_DATE ({start.date()})."
        )
    return start, end


def _eval_dates_from_window(
    calendar_like,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> set | None:
    """Build an ``eval_dates`` set (calendar keys) from a window + a candidate
    index / iterable of timestamps. Returns ``None`` when both bounds are unset
    so downstream code keeps its "no restriction" fast path.
    """
    if start is None and end is None:
        return None
    idx = pd.DatetimeIndex(list(calendar_like))
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    idx = idx.normalize()
    mask = pd.Series(True, index=idx, dtype=bool)
    if start is not None:
        mask &= idx >= start
    if end is not None:
        mask &= idx <= end
    kept = idx[mask.to_numpy()]
    return {mad_calendar_key(d) for d in kept}


def mad_regime_ticker_symbol() -> str | None:
    if not bool(getattr(config, "MAD_REGIME_MA_ENABLED", False)):
        return None
    t = getattr(config, "MAD_REGIME_TICKER", None)
    s = str(t).strip().upper() if t is not None else ""
    return s or None


def _regime_ma_grid() -> tuple[int, ...]:
    if mad_regime_ticker_symbol() is None:
        return (0,)
    g = getattr(config, "MAD_REGIME_MA_GRID", (0,))
    if isinstance(g, int):
        return (max(0, int(g)),)
    return tuple(max(0, int(x)) for x in g)


def _build_regime_allow(
    regime_ma_period: int,
    regime_ticker: str | None,
    granularity: str,
    aggregate_to_daily: bool,
    ohlcv_dir: Path,
) -> pd.Series | None:
    if int(regime_ma_period or 0) <= 0:
        return None
    sym = (regime_ticker or "").strip().upper() or (mad_regime_ticker_symbol() or "QQQ")
    close = _load_regime_daily_close(sym, granularity, ohlcv_dir, aggregate_to_daily=aggregate_to_daily)
    close.index = pd.DatetimeIndex([mad_calendar_key(i) for i in close.index], tz="UTC")
    if close.index.duplicated().any():
        close = close.groupby(level=0).last()
    allow = _regime_entry_allow_series(close, int(regime_ma_period))
    allow.index = close.index
    return allow


def aggregate_panel_to_daily(panel_long: pd.DataFrame) -> pd.DataFrame:
    """Last close / first open per calendar day (UTC); split from last bar of day; last precomputed SMA."""
    df = panel_long.copy()
    df["day"] = df["timestamp"].dt.normalize()
    g = df.groupby(["day", "ticker"], sort=True)
    agg_kw: dict[str, tuple[str, str]] = {
        "open": ("open", "first"),
        "close": ("close", "last"),
        "split": ("split", "last"),
    }
    for c in df.columns:
        if c.startswith("sma_"):
            agg_kw[c] = (c, "last")
    daily = g.agg(**agg_kw).reset_index()
    daily = daily.rename(columns={"day": "date"})
    return daily


def _sma(series: pd.Series, window: int) -> pd.Series:
    w = int(window)
    return series.rolling(window=w, min_periods=w).mean()


def _decile_rank_pct(x: pd.Series) -> pd.Series:
    r = x.rank(pct=True, method="average")
    d = np.ceil(r.to_numpy(dtype=float) * 10.0)
    return pd.Series(np.clip(d, 1.0, 10.0), index=x.index, dtype=float).astype(int)


def mad_calendar_key(ts: pd.Timestamp | np.datetime64 | object) -> pd.Timestamp:
    """UTC midnight key for matching daily panel dates to portfolio path index (tz-safe)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.normalize()


def compute_mrat_panel(
    daily_long: pd.DataFrame,
    *,
    short_w: int,
    long_w: int,
    min_price: float,
    min_history: int,
    direction_mode: str,
    long_sigma_mult: float | None = None,
    short_sigma_mult: float | None = None,
    exit_ma_period: int = 0,
    long_decile_min: int | None = None,
    short_decile_max: int | None = None,
    symmetric_short_sigma: bool | None = None,
    prefer_precomputed_sma: bool = False,
) -> pd.DataFrame:
    """Adds mrat, sigma, decile, signal, entry_signal, daily_ret per row.

    exit_ma_period: per ticker, long only held if close > SMA(exit_ma_period); shorts (if enabled)
    only if close < that SMA. 0 disables. Applied after MRAT signal merge (natural re-entry gate).

    Long: decile >= long_decile_min and MRAT > 1 + lsm*σ. Short: decile <= short_decile_max and
    MRAT < 1 - ssm*σ (ssm = lsm when symmetric_short_sigma). Overlap → flat (0) for that name.
    """
    lsm = float(
        long_sigma_mult
        if long_sigma_mult is not None
        else float(getattr(config, "MAD_LONG_SIGMA_MULT", 1.0))
    )
    ssm = float(
        short_sigma_mult
        if short_sigma_mult is not None
        else float(getattr(config, "MAD_SHORT_SIGMA_MULT", 1.0))
    )
    if bool(
        symmetric_short_sigma
        if symmetric_short_sigma is not None
        else getattr(config, "MAD_SYMMETRIC_SHORT_SIGMA", False)
    ):
        ssm = lsm
    ld_min = int(
        long_decile_min
        if long_decile_min is not None
        else int(getattr(config, "MAD_LONG_DECILE_MIN", 10))
    )
    sd_max = int(
        short_decile_max
        if short_decile_max is not None
        else int(getattr(config, "MAD_SHORT_DECILE_MAX", 1))
    )
    ld_min = max(1, min(10, ld_min))
    sd_max = max(1, min(10, sd_max))
    dm = _normalize_direction_mode(direction_mode)
    df = daily_long.sort_values(["ticker", "date"]).copy()
    cs, cl = f"sma_{int(short_w)}", f"sma_{int(long_w)}"
    if (
        prefer_precomputed_sma
        and cs in df.columns
        and cl in df.columns
        and df[cs].notna().any()
        and df[cl].notna().any()
    ):
        df["ma_s"] = df[cs].astype(float)
        df["ma_l"] = df[cl].astype(float)
    else:
        df["ma_s"] = df.groupby("ticker", sort=False)["close"].transform(lambda s: _sma(s, short_w))
        df["ma_l"] = df.groupby("ticker", sort=False)["close"].transform(lambda s: _sma(s, long_w))
    df["mrat"] = df["ma_s"] / df["ma_l"]
    hist_n = df.groupby("ticker", sort=False).cumcount() + 1
    ok = (
        (df["close"] >= float(min_price))
        & df["mrat"].notna()
        & (hist_n >= int(min_history))
    )
    work = df.loc[ok].copy()
    if work.empty:
        df["sigma"] = np.nan
        df["decile"] = np.nan
        df["signal"] = 0
        df["mad_exit_ma_level"] = np.nan
        df["entry_signal"] = 0
        df["daily_ret"] = df.groupby("ticker", sort=False)["close"].pct_change()
        return df

    sig = work.groupby("date", sort=True)["mrat"].transform("std")
    work["sigma"] = sig
    work["decile"] = work.groupby("date", sort=True)["mrat"].transform(_decile_rank_pct)

    work["signal"] = 0
    long_ok = (work["decile"] >= ld_min) & (work["mrat"] > 1.0 + lsm * work["sigma"])
    short_ok = (work["decile"] <= sd_max) & (work["mrat"] < 1.0 - ssm * work["sigma"])
    both_legs = long_ok & short_ok
    work.loc[long_ok & ~both_legs, "signal"] = 1
    work.loc[short_ok & ~both_legs, "signal"] = -1
    if dm == "long_only":
        work.loc[work["signal"] < 0, "signal"] = 0
    elif dm == "short_only":
        work.loc[work["signal"] > 0, "signal"] = 0

    df = df.merge(
        work[["date", "ticker", "sigma", "decile", "signal"]],
        on=["date", "ticker"],
        how="left",
    )
    df["signal"] = df["signal"].fillna(0).astype(int)
    ex_n = int(exit_ma_period or 0)
    if ex_n > 0:
        cex = f"sma_{ex_n}"
        if prefer_precomputed_sma and cex in df.columns and df[cex].notna().any():
            df["mad_exit_ma_level"] = df[cex].astype(float)
        else:
            df["mad_exit_ma_level"] = df.groupby("ticker", sort=False)["close"].transform(
                lambda s: _sma(s, ex_n)
            )
        long_m = df["signal"].to_numpy(dtype=int) == 1
        c = df["close"].to_numpy(dtype=float)
        mx = df["mad_exit_ma_level"].to_numpy(dtype=float)
        block_long = long_m & (~np.isfinite(mx) | (c <= mx))
        df.loc[block_long, "signal"] = 0
        if dm in ("both", "short_only"):
            short_m = df["signal"].to_numpy(dtype=int) == -1
            block_short = short_m & (~np.isfinite(mx) | (c >= mx))
            df.loc[block_short, "signal"] = 0
    else:
        df["mad_exit_ma_level"] = np.nan
    df["entry_signal"] = df.groupby("ticker", sort=False)["signal"].shift(1).fillna(0).astype(int)
    df["daily_ret"] = df.groupby("ticker", sort=False)["close"].pct_change()
    return df


def _weights_from_entries(entry_row: pd.Series) -> pd.Series:
    """Long +1/nL, short −1/nS, else 0 (same index as entry_row)."""
    s = entry_row.fillna(0).astype(float)
    L = s[s == 1].index
    S = s[s == -1].index
    nL, nS = len(L), len(S)
    w = pd.Series(0.0, index=s.index, dtype=float)
    if nL:
        w.loc[L] = 1.0 / nL
    if nS:
        w.loc[S] = -1.0 / nS
    return w


@dataclass(frozen=True)
class MadLiveSnapshot:
    """Last completed calendar row MRAT targets for the next session (matches backtest signal → next-day hold).

    ``weight_by_ticker`` always spans the MAD universe; on risk-off sessions the MRAT
    columns are zero and ``sleeve_weight_by_ticker`` carries the risk-off allocation
    (see ``mad.regime_sleeve``). The live bot should trade the union of both maps.

    ``regime_info`` is a human-readable summary for logs / dashboards (``format_regime_state_line``
    output). ``weighting_scheme`` is the resolved scheme name (``MAD_WEIGHTING_SCHEME``).
    """

    as_of: pd.Timestamp
    tickers: tuple[str, ...]
    weight_by_ticker: dict[str, float]
    close_by_ticker: dict[str, float]
    regime_ok: bool
    mad_sma_short: int
    mad_sma_long: int
    mad_exit_ma: int
    mad_regime_ma: int
    n_long: int
    n_short: int
    sleeve_weight_by_ticker: dict[str, float] = field(default_factory=dict)
    sleeve_close_by_ticker: dict[str, float] = field(default_factory=dict)
    regime_info: str = ""
    weighting_scheme: str = "equal"


def _regime_risk_on_from_db_precomputed(
    sym: str,
    granularity: str,
    ohlcv_dir: Path,
    w: int,
) -> bool | None:
    """
    Last row: close vs splitter ``sma_<w>``. Only valid when OHLCV granularity is daily (see caller).
    Returns None if column missing or values non-finite.
    """
    path = ohlcv_dir / f"{str(sym).strip().upper()}_{granularity}.db"
    if not path.exists():
        return None
    col = f"sma_{int(w)}"
    if not re.fullmatch(r"sma_[0-9]+", col):
        return None
    with sqlite3.connect(path) as con:
        names = [r[1] for r in con.execute("PRAGMA table_info(ohlcv)").fetchall()]
        if col not in names:
            return None
        row = con.execute(
            f"SELECT close, {col} AS sx FROM ohlcv ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    c, s = float(row[0]), float(row[1])
    if not (np.isfinite(c) and np.isfinite(s)):
        return None
    return bool(c > s)


def _build_mad_live_mrat_panel(
    *,
    short_w: int,
    long_w: int,
    exit_ma_period: int,
    regime_ma_period: int,
    regime_ticker: str | None = None,
    ohlcv_dir: Path | None = None,
    direction_mode: str | None = None,
    universe: tuple[str, ...] | None = None,
    reference_ticker: str | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.DataFrame, bool, pd.Series, dict[str, float], dict[str, float], str]:
    """
    Shared OHLCV → MRAT panel for live bot and dashboard.

    Returns ``panel`` (all dates), ``last_date`` (UTC), ``sub`` (last-day rows per ticker),
    ``regime_ok``, ``sig_series`` aligned to the resolved universe, and the risk-off sleeve
    ``(weights_by_ticker, closes_by_ticker, info_line)`` derived from ``mad.regime_sleeve``.

    ``universe`` / ``reference_ticker`` default to ``mad_universe_tickers()`` /
    ``mad_reference_ticker()`` (legacy single-panel behavior). Callers wiring the
    multi-index live allocator pass a per-slot universe + ETF override so each
    slot is picked independently.
    """
    odir = ohlcv_dir or OHLCV_DIR
    ref = reference_ticker if reference_ticker else mad_reference_ticker()
    universe = universe if universe else mad_universe_tickers()
    gran = str(config.TARGET_CANDLE_GRANULARITY)
    daily_agg = bool(getattr(config, "MAD_AGGREGATE_TO_DAILY", True)) and gran.lower() != "1d"
    dm = direction_mode if direction_mode is not None else getattr(config, "MAD_DIRECTION_MODE", "both")
    use_pc = bool(getattr(config, "MAD_LIVE_USE_PRECOMPUTED_SMA", True))
    sma_periods: tuple[int, ...] | None = None
    if use_pc:
        ps = {int(short_w), int(long_w)}
        ex = int(exit_ma_period or 0)
        if ex > 0:
            ps.add(ex)
        sma_periods = tuple(sorted(ps))

    panel_long = build_panel_long(
        universe, gran, ref, odir, include_sma_periods=sma_periods
    )
    if daily_agg:
        daily_long = aggregate_panel_to_daily(panel_long)
    else:
        dl = panel_long.copy()
        dl["date"] = pd.to_datetime(dl["timestamp"], utc=True).dt.normalize()
        daily_long = dl.drop(columns=["timestamp"], errors="ignore")

    panel = compute_mrat_panel(
        daily_long,
        short_w=int(short_w),
        long_w=int(long_w),
        min_price=float(MAD_DEFAULT_MIN_PRICE),
        min_history=int(getattr(config, "MAD_MIN_HISTORY_BARS", 252)),
        direction_mode=str(dm),
        exit_ma_period=int(exit_ma_period or 0),
        prefer_precomputed_sma=use_pc,
    )
    if panel.empty or panel["date"].isna().all():
        raise RuntimeError("MAD panel is empty; check OHLCV DBs and universe.")

    dcol = pd.to_datetime(panel["date"], utc=True).dt.normalize()
    last_date = pd.Timestamp(dcol.max())
    if last_date.tzinfo is None:
        last_date = last_date.tz_localize("UTC")
    else:
        last_date = last_date.tz_convert("UTC")

    sub = panel.loc[dcol == last_date].drop_duplicates(subset=["ticker"], keep="last")
    sig_series = sub.set_index("ticker")["signal"].reindex(list(universe)).fillna(0).astype(int)

    # Multi-ticker regime + sleeve (``mad.regime_sleeve``). An explicit ``regime_ticker`` override
    # collapses to a single-ticker gate (live bot preserves its override semantics); otherwise
    # we honor the full ``MAD_REGIME_TICKERS`` list plus the trend-gated sleeve composition.
    reg_ma = int(regime_ma_period or 0)
    mode = resolve_regime_mode(config)
    sleeve = resolve_sleeve(config)
    safe_harbor = resolve_safe_harbor(config)
    if regime_ticker is not None and str(regime_ticker).strip():
        regime_tickers = (str(regime_ticker).strip().upper(),)
    else:
        regime_tickers = resolve_regime_tickers(config) or (
            (mad_regime_ticker_symbol() or "QQQ"),
        )

    if reg_ma <= 0 or not regime_tickers:
        regime_ok = True
        sleeve_weights: dict[str, float] = {}
        sleeve_closes: dict[str, float] = {}
        info_line = "regime off (no filter)"
    else:
        try:
            state = evaluate_regime_live(
                ohlcv_dir=odir,
                granularity=gran,
                aggregate_to_daily=daily_agg,
                regime_tickers=regime_tickers,
                regime_ma=reg_ma,
                mode=mode,
                sleeve=sleeve,
                safe_harbor=safe_harbor,
                prefer_precomputed_sma=use_pc,
                sleeve_weight_cfg=resolve_sleeve_weight_config(config),
                sleeve_mrat_pair=resolve_sleeve_mrat_pair(config),
            )
        except FileNotFoundError as exc:
            print(f"[MAD] regime_sleeve fallback — {exc}")
            regime_ok = _regime_risk_on_for_next_session(
                reg_ma,
                regime_tickers[0],
                gran,
                daily_agg,
                odir,
                prefer_precomputed_sma=use_pc,
            )
            sleeve_weights = {}
            sleeve_closes = {}
            info_line = (
                f"regime[{mode} {reg_ma}D] (legacy fallback: {regime_tickers[0]}) "
                f"{'RISK-ON' if regime_ok else 'RISK-OFF'}"
            )
        else:
            regime_ok = bool(state.risk_on)
            sleeve_weights = dict(state.sleeve_weights)
            sleeve_closes = {}
            for sym in sleeve_weights:
                try:
                    path = odir / f"{sym}_{gran}.db"
                    with sqlite3.connect(path) as con:
                        row = con.execute(
                            "SELECT close FROM ohlcv ORDER BY timestamp DESC LIMIT 1"
                        ).fetchone()
                    sleeve_closes[sym] = float(row[0]) if row and row[0] is not None else float("nan")
                except Exception:
                    sleeve_closes[sym] = float("nan")
            info_line = format_regime_state_line(state, regime_ma=reg_ma, mode=mode)
    return panel, last_date, sub, regime_ok, sig_series, sleeve_weights, sleeve_closes, info_line


def mad_live_watchlist_reason(
    row: pd.Series | None,
    *,
    regime_ok: bool,
    direction_mode: str,
    exit_ma_period: int,
) -> str:
    """Human-readable why a name is in/out of the book (last panel row)."""
    dm = _normalize_direction_mode(direction_mode)
    if not regime_ok:
        return "Regime risk-off (flat targets)"
    if row is None or row.empty:
        return "Missing OHLCV / not in panel"
    mrat = float(row.get("mrat", float("nan")))
    if not np.isfinite(mrat):
        return "MRAT unavailable (history, min price, or MA)"

    sig = int(row.get("signal", 0) or 0)
    if sig == 1:
        return "Long (in MRAT book)"
    if sig == -1:
        return "Short (in MRAT book)"

    d = float(row.get("decile", float("nan")))
    sma_cs = float(row.get("sigma", float("nan")))
    c = float(row.get("close", float("nan")))
    ld_min = int(getattr(config, "MAD_LONG_DECILE_MIN", 10))
    lsm = float(getattr(config, "MAD_LONG_SIGMA_MULT", 1.0))
    sd_max = int(getattr(config, "MAD_SHORT_DECILE_MAX", 1))
    ssm = float(getattr(config, "MAD_SHORT_SIGMA_MULT", 1.0))
    if bool(getattr(config, "MAD_SYMMETRIC_SHORT_SIGMA", False)):
        ssm = lsm

    if not np.isfinite(sma_cs) or not np.isfinite(d):
        return "Thin cross-section (σ or decile missing)"

    gate_long = 1.0 + lsm * sma_cs
    gate_short = 1.0 - ssm * sma_cs
    ex_n = int(exit_ma_period or 0)
    exv = row.get("mad_exit_ma_level")
    exf = float(exv) if exv is not None and np.isfinite(exv) else float("nan")

    if dm in ("long_only", "both") and ex_n > 0:
        if (
            d >= ld_min
            and mrat > gate_long
            and np.isfinite(exf)
            and np.isfinite(c)
            and c <= exf
        ):
            return f"Exit MA (close ≤ {ex_n}d SMA)"
    if dm in ("short_only", "both") and ex_n > 0:
        if (
            d <= sd_max
            and mrat < gate_short
            and np.isfinite(exf)
            and np.isfinite(c)
            and c >= exf
        ):
            return f"Exit MA (close ≥ {ex_n}d SMA)"

    parts: list[str] = []
    if dm in ("long_only", "both"):
        if d < ld_min:
            parts.append(f"decile {d:.0f}% < {ld_min}")
        elif mrat <= gate_long:
            parts.append(f"MRAT {mrat:.3f} ≤ 1+kσ ({gate_long:.3f})")
        else:
            parts.append("long leg not elected")
    if dm in ("short_only", "both"):
        if d > sd_max:
            parts.append(f"decile {d:.0f}% > {sd_max}")
        elif mrat >= gate_short:
            parts.append(f"MRAT {mrat:.3f} ≥ 1−kσ ({gate_short:.3f})")
        else:
            parts.append("short leg not elected")
    if parts:
        return "; ".join(parts)
    return "Flat"


def _live_weight_inputs_from_sub(
    sub: pd.DataFrame,
    sig_series: pd.Series,
    *,
    realized_vol_lookback: int,
    panel: pd.DataFrame,
) -> WeightInputs:
    """Per-ticker MRAT/σ/decile + realized vol from the last panel date for live sizing."""
    idxed = sub.drop_duplicates(subset=["ticker"], keep="last").set_index("ticker")

    def _col(name: str) -> pd.Series:
        if name in idxed.columns:
            s = idxed[name].astype(float).reindex(sig_series.index)
        else:
            s = pd.Series(np.nan, index=sig_series.index, dtype=float)
        return s

    mrat = _col("mrat")
    decile = _col("decile")
    sig_vals = idxed["sigma"].astype(float) if "sigma" in idxed.columns else pd.Series(dtype=float)
    sigma_val = float(sig_vals.dropna().iloc[0]) if not sig_vals.dropna().empty else float("nan")

    rvol: pd.Series | None = None
    if realized_vol_lookback and realized_vol_lookback >= 2 and "daily_ret" in panel.columns:
        piv = realized_vol_pivot(panel, lookback=int(realized_vol_lookback))
        if not piv.empty:
            last_row = piv.iloc[-1]
            rvol = last_row.reindex(sig_series.index).astype(float)

    return WeightInputs(
        entry_signal=sig_series,
        mrat=mrat,
        sigma=sigma_val,
        decile=decile,
        realized_vol=rvol,
    )


def _mad_live_pack_snapshot(
    panel: pd.DataFrame,
    last_date: pd.Timestamp,
    sub: pd.DataFrame,
    regime_ok: bool,
    sig_series: pd.Series,
    *,
    short_w: int,
    long_w: int,
    exit_ma_period: int,
    regime_ma_period: int,
    sleeve_weights: dict[str, float] | None = None,
    sleeve_close_by_ticker: dict[str, float] | None = None,
    regime_info: str = "",
    weight_cfg: WeightConfig | None = None,
    universe: tuple[str, ...] | None = None,
) -> MadLiveSnapshot:
    """Pack a per-slot (or single-panel) ``MadLiveSnapshot``.

    ``universe`` lets the multi-index live path scope a slot's MRAT book to its
    own constituent list; defaults to ``mad_universe_tickers()`` for the legacy
    single-panel call sites.
    """
    universe = universe if universe else mad_universe_tickers()
    cfg = weight_cfg or resolve_weight_config(config)

    if regime_ok:
        sig_aligned = sig_series.reindex(universe).fillna(0).astype(int)
        inp = _live_weight_inputs_from_sub(
            sub,
            sig_aligned,
            realized_vol_lookback=cfg.realized_vol_lookback,
            panel=panel,
        )
        w = compute_weights(inp, cfg)
    else:
        w = pd.Series(0.0, index=universe, dtype=float)

    idxed = sub.set_index("ticker")
    weight_by_ticker: dict[str, float] = {}
    close_by_ticker: dict[str, float] = {}
    for t in universe:
        weight_by_ticker[t] = float(w.reindex([t]).fillna(0.0).iloc[0])
        if t in idxed.index:
            close_by_ticker[t] = float(idxed.loc[t, "close"])
        else:
            tail = panel.loc[panel["ticker"] == t, "close"]
            close_by_ticker[t] = float(tail.iloc[-1]) if len(tail) else float("nan")

    n_long = int((sig_series == 1).sum())
    n_short = int((sig_series == -1).sum())
    return MadLiveSnapshot(
        as_of=last_date,
        tickers=universe,
        weight_by_ticker=weight_by_ticker,
        close_by_ticker=close_by_ticker,
        regime_ok=regime_ok,
        mad_sma_short=int(short_w),
        mad_sma_long=int(long_w),
        mad_exit_ma=int(exit_ma_period or 0),
        mad_regime_ma=int(regime_ma_period or 0),
        n_long=n_long,
        n_short=n_short,
        sleeve_weight_by_ticker=dict(sleeve_weights or {}),
        sleeve_close_by_ticker=dict(sleeve_close_by_ticker or {}),
        regime_info=regime_info,
        weighting_scheme=cfg.scheme,
    )


def compute_mad_live_panel_and_snapshot(
    *,
    short_w: int,
    long_w: int,
    exit_ma_period: int,
    regime_ma_period: int,
    regime_ticker: str | None = None,
    ohlcv_dir: Path | None = None,
    direction_mode: str | None = None,
) -> tuple[pd.DataFrame, MadLiveSnapshot, pd.DataFrame]:
    """Full MRAT panel, live snapshot, and last-day panel rows (one DB pass)."""
    (
        panel,
        last_date,
        sub,
        regime_ok,
        sig_series,
        sleeve_w,
        sleeve_c,
        info_line,
    ) = _build_mad_live_mrat_panel(
        short_w=short_w,
        long_w=long_w,
        exit_ma_period=exit_ma_period,
        regime_ma_period=regime_ma_period,
        regime_ticker=regime_ticker,
        ohlcv_dir=ohlcv_dir,
        direction_mode=direction_mode,
    )
    snap = _mad_live_pack_snapshot(
        panel,
        last_date,
        sub,
        regime_ok,
        sig_series,
        short_w=short_w,
        long_w=long_w,
        exit_ma_period=exit_ma_period,
        regime_ma_period=regime_ma_period,
        sleeve_weights=sleeve_w,
        sleeve_close_by_ticker=sleeve_c,
        regime_info=info_line,
    )
    return panel, snap, sub


def mad_live_watchlist_table(
    sub: pd.DataFrame,
    *,
    regime_ok: bool,
    weight_by_ticker: dict[str, float],
    universe: tuple[str, ...],
    direction_mode: str,
    exit_ma_period: int,
) -> list[dict[str, Any]]:
    """Rows for Dash ``DataTable`` (sorted by target weight magnitude, then ticker)."""
    idxed = sub.set_index("ticker")
    rows: list[dict[str, Any]] = []
    for t in universe:
        w = float(weight_by_ticker.get(t, 0.0))
        row: pd.Series | None
        if t in idxed.index:
            raw = idxed.loc[t]
            row = raw.iloc[-1] if isinstance(raw, pd.DataFrame) else raw
        else:
            row = None
        reason = mad_live_watchlist_reason(
            row,
            regime_ok=regime_ok,
            direction_mode=direction_mode,
            exit_ma_period=exit_ma_period,
        )
        sig = int(row["signal"]) if row is not None and np.isfinite(row.get("signal", np.nan)) else 0
        mrat = float(row["mrat"]) if row is not None and np.isfinite(row.get("mrat", np.nan)) else float("nan")
        dec = float(row["decile"]) if row is not None and np.isfinite(row.get("decile", np.nan)) else float("nan")
        sig_cs = float(row["sigma"]) if row is not None and np.isfinite(row.get("sigma", np.nan)) else float("nan")
        lsm = float(getattr(config, "MAD_LONG_SIGMA_MULT", 1.0))
        gate = 1.0 + lsm * sig_cs if np.isfinite(sig_cs) else float("nan")
        rows.append(
            {
                "ticker": t,
                "weight_pct": round(w * 100.0, 4),
                "signal": sig,
                "decile": round(dec, 2) if np.isfinite(dec) else None,
                "mrat": round(mrat, 4) if np.isfinite(mrat) else None,
                "sigma": round(sig_cs, 4) if np.isfinite(sig_cs) else None,
                "one_plus_k_sigma": round(gate, 4) if np.isfinite(gate) else None,
                "reason": reason,
            }
        )
    rows.sort(key=lambda r: (-abs(float(r["weight_pct"])), r["ticker"]))
    return rows


def _regime_risk_on_for_next_session(
    regime_ma_period: int,
    regime_ticker: str | None,
    granularity: str,
    aggregate_to_daily: bool,
    ohlcv_dir: Path,
    *,
    prefer_precomputed_sma: bool = False,
) -> bool:
    """
    After the last regime close in SQLite, is the next session risk-on?

    Aligns with ``portfolio_path_from_panel``: for calendar day *t*, regime gates weights using
    ``close[t-1] > SMA[t-1]`` on the regime ETF. After the final close *D* in the DB, the next session
    uses ``close[D] > SMA[D]`` (same bar included in SMA).
    """
    if int(regime_ma_period or 0) <= 0:
        return True
    sym = (regime_ticker or "").strip().upper() or (mad_regime_ticker_symbol() or "QQQ")
    w = int(regime_ma_period)
    gran_lc = str(granularity).strip().lower()
    if prefer_precomputed_sma and gran_lc == "1d":
        pc = _regime_risk_on_from_db_precomputed(sym, granularity, ohlcv_dir, w)
        if pc is not None:
            return pc
    close = _load_regime_daily_close(sym, granularity, ohlcv_dir, aggregate_to_daily=aggregate_to_daily)
    close = close.sort_index()
    if len(close) < w:
        return False
    sma = _sma(close, w)
    c = float(close.iloc[-1])
    s = float(sma.iloc[-1])
    return bool(np.isfinite(c) and np.isfinite(s) and c > s)


def compute_mad_live_snapshot(
    *,
    short_w: int,
    long_w: int,
    exit_ma_period: int,
    regime_ma_period: int,
    regime_ticker: str | None = None,
    ohlcv_dir: Path | None = None,
    direction_mode: str | None = None,
) -> MadLiveSnapshot:
    """
    Build equal-weight targets from local OHLCV DBs (same pipeline as the MAD backtester).

    Uses the **last** panel date's ``signal`` (not ``entry_signal``): that is the book to hold for the
    session after that close, consistent with ``entry_signal`` shifting by one bar in the backtest.

    When ``config.MAD_LIVE_USE_PRECOMPUTED_SMA`` is True, loads ``sma_<short>``, ``sma_<long>``, and
    optional ``sma_<exit>`` from each symbol DB (run ``data_splitter`` after fetch). MRAT then matches
    the splitter-rounded SMAs. Missing columns fall back to rolling ``close``. Regime precomputed path
    applies only for ``TARGET_CANDLE_GRANULARITY`` ``1d``.
    """
    (
        panel,
        last_date,
        sub,
        regime_ok,
        sig_series,
        sleeve_w,
        sleeve_c,
        info_line,
    ) = _build_mad_live_mrat_panel(
        short_w=short_w,
        long_w=long_w,
        exit_ma_period=exit_ma_period,
        regime_ma_period=regime_ma_period,
        regime_ticker=regime_ticker,
        ohlcv_dir=ohlcv_dir,
        direction_mode=direction_mode,
    )
    return _mad_live_pack_snapshot(
        panel,
        last_date,
        sub,
        regime_ok,
        sig_series,
        short_w=short_w,
        long_w=long_w,
        exit_ma_period=exit_ma_period,
        regime_ma_period=regime_ma_period,
        sleeve_weights=sleeve_w,
        sleeve_close_by_ticker=sleeve_c,
        regime_info=info_line,
    )


def compute_mad_multi_index_live_snapshot(
    *,
    short_w: int,
    long_w: int,
    exit_ma_period: int,
    ohlcv_dir: Path | None = None,
    direction_mode: str | None = None,
) -> MadLiveSnapshot:
    """Live snapshot for the multi-index allocator (mirror of ``evaluate_mad_multi_index``).

    Flow (last-bar, matches MRAT entry timing via ``shift(1)`` inside helpers):

      1. ``resolve_index_slots(config)`` returns the enabled ``IndexSlot``s
         (respects ``MAD_INDEX_ENABLED_ETFS``).
      2. ``evaluate_index_allocation_live`` computes per-ETF allocation weights
         + ``risk_off_share`` from each ETF's latest MRAT + 200D trend gate.
      3. For each slot with a non-zero allocation weight, run the stock picker
         scoped to that slot's exclusive universe + ETF (via
         ``_build_mad_live_mrat_panel`` with ``universe`` / ``reference_ticker``
         overrides). The picker's ``regime_ma_period=0`` here — the allocator's
         top-level trend gate already decided the slot gets equity exposure.
      4. Risk-off sleeve composition (hedge-asset basket with dynamic weighting
         when ``MAD_SLEEVE_WEIGHTING_SCHEME`` is set) is loaded via
         ``evaluate_sleeve_allocation_live`` — unconditional on top-level regime
         because the allocator may be mixing equity + sleeve on the same day.
      5. Final ticker weights:
            w_equity[t]  = slot_weight[slot.etf] * stock_pick_weight[t]
            w_sleeve[t]  = risk_off_share * sleeve_weight[t]
         Both maps are merged into ``weight_by_ticker`` (a single flat dict)
         so the live bot's reconcile loop can size them uniformly against one
         gross-notional value (no separate sleeve-notional path, unlike the
         legacy single-universe risk-off branch).

    The returned snapshot's ``sleeve_weight_by_ticker`` is always empty in
    multi-index mode — the ``close_all_non_proxy`` / sleeve-gross branch is
    bypassed, since the book holds equity and sleeve simultaneously.
    """
    odir = ohlcv_dir or OHLCV_DIR
    gran = str(config.TARGET_CANDLE_GRANULARITY)
    daily_agg = bool(getattr(config, "MAD_AGGREGATE_TO_DAILY", True)) and gran.lower() != "1d"
    use_pc = bool(getattr(config, "MAD_LIVE_USE_PRECOMPUTED_SMA", True))

    slots = resolve_index_slots(config)
    if not slots:
        raise RuntimeError(
            "compute_mad_multi_index_live_snapshot: no enabled index slots "
            "(check MAD_INDEX_SLOTS / MAD_INDEX_ENABLED_ETFS in config)."
        )

    # --- 1. Top-level allocation (per-ETF + risk-off share) from latest bar. --------
    allocation = evaluate_index_allocation_live(
        slots=slots,
        regime_ma=resolve_index_regime_ma(config),
        mrat_pair=resolve_index_mrat_pair(config),
        weight_cfg=resolve_index_weight_config(config),
        granularity=gran,
        ohlcv_dir=odir,
        aggregate_to_daily=daily_agg,
        prefer_precomputed_sma=use_pc,
    )
    idx_weights = dict(allocation.index_weights)
    risk_off_share = float(allocation.risk_off_weight)

    # --- 2. Per-slot stock picker. Only run for slots with non-zero allocation. -----
    per_slot_weights: dict[str, dict[str, float]] = {}
    per_slot_closes: dict[str, dict[str, float]] = {}
    per_slot_as_of: dict[str, pd.Timestamp] = {}
    per_slot_n_long: dict[str, int] = {}
    per_slot_n_short: dict[str, int] = {}
    weight_cfg = resolve_weight_config(config)

    for slot in slots:
        slot_w = float(idx_weights.get(slot.etf, 0.0))
        if slot_w <= 1e-12:
            continue  # slot failed trend gate; no stock-picker work needed
        if not slot.universe:
            continue
        slot_univ = tuple(slot.universe)
        try:
            (panel, last_date, sub, _regime_ok_unused, sig_series,
             _sleeve_unused, _closes_unused, _info_unused) = _build_mad_live_mrat_panel(
                short_w=int(short_w),
                long_w=int(long_w),
                exit_ma_period=int(exit_ma_period or 0),
                # The allocator's own per-ETF trend gate already decided this
                # slot gets equity. Disable the per-slot regime filter so the
                # picker doesn't zero out weights via a redundant second gate.
                regime_ma_period=0,
                regime_ticker=None,
                ohlcv_dir=odir,
                direction_mode=direction_mode,
                universe=slot_univ,
                reference_ticker=slot.etf,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{slot.etf}] SKIP live snapshot: {exc}")
            continue

        slot_snap = _mad_live_pack_snapshot(
            panel,
            last_date,
            sub,
            True,  # regime_ok — decided at the allocator layer
            sig_series,
            short_w=int(short_w),
            long_w=int(long_w),
            exit_ma_period=int(exit_ma_period or 0),
            regime_ma_period=0,
            universe=slot_univ,
        )
        per_slot_weights[slot.etf] = dict(slot_snap.weight_by_ticker)
        per_slot_closes[slot.etf] = dict(slot_snap.close_by_ticker)
        per_slot_as_of[slot.etf] = slot_snap.as_of
        per_slot_n_long[slot.etf] = slot_snap.n_long
        per_slot_n_short[slot.etf] = slot_snap.n_short

    # --- 3. Risk-off sleeve composition (always build, scale by risk_off_share). ----
    sleeve_w_raw: dict[str, float] = {}
    sleeve_closes: dict[str, float] = {}
    if risk_off_share > 1e-12:
        try:
            sleeve_w_raw, _per_sleeve = evaluate_sleeve_allocation_live(
                ohlcv_dir=odir,
                granularity=gran,
                aggregate_to_daily=daily_agg,
                sleeve=resolve_sleeve(config),
                safe_harbor=resolve_safe_harbor(config),
                prefer_precomputed_sma=use_pc,
                sleeve_weight_cfg=resolve_sleeve_weight_config(config),
                sleeve_mrat_pair=resolve_sleeve_mrat_pair(config),
            )
        except FileNotFoundError as exc:
            print(f"[multi-index live] sleeve load failed, safe-harbor fallback: {exc}")
            sh = (resolve_safe_harbor(config) or "").strip().upper()
            if sh:
                sleeve_w_raw = {sh: 1.0}
        for sym in sleeve_w_raw:
            try:
                path = odir / f"{sym}_{gran}.db"
                with sqlite3.connect(path) as con:
                    row = con.execute(
                        "SELECT close FROM ohlcv ORDER BY timestamp DESC LIMIT 1"
                    ).fetchone()
                sleeve_closes[sym] = float(row[0]) if row and row[0] is not None else float("nan")
            except Exception:
                sleeve_closes[sym] = float("nan")

    # --- 4. Merge per-slot equity + sleeve into one flat ticker-weight map. ---------
    merged_weights: dict[str, float] = {}
    merged_closes: dict[str, float] = {}
    for etf, slot_w in idx_weights.items():
        if slot_w <= 1e-12 or etf not in per_slot_weights:
            continue
        for t, stock_w in per_slot_weights[etf].items():
            merged_weights[t] = merged_weights.get(t, 0.0) + float(slot_w) * float(stock_w)
            if t not in merged_closes:
                merged_closes[t] = per_slot_closes[etf].get(t, float("nan"))
    for sym, s_w in sleeve_w_raw.items():
        merged_weights[sym] = merged_weights.get(sym, 0.0) + risk_off_share * float(s_w)
        if sym not in merged_closes:
            merged_closes[sym] = sleeve_closes.get(sym, float("nan"))

    # Stable ordering: slots in priority order → their constituents → sleeve tickers.
    ordered: list[str] = []
    seen: set[str] = set()
    for slot in slots:
        for t in per_slot_weights.get(slot.etf, {}):
            if t not in seen:
                ordered.append(t)
                seen.add(t)
    for sym in sleeve_w_raw:
        if sym not in seen:
            ordered.append(sym)
            seen.add(sym)
    tickers_tuple = tuple(ordered)

    # --- 5. Build diagnostics + final snapshot. -------------------------------------
    if per_slot_as_of:
        as_of = max(per_slot_as_of.values())
    else:
        as_of = pd.Timestamp.now(tz="UTC").normalize()

    idx_parts = [
        f"{etf}={idx_weights.get(etf, 0.0):.0%}"
        for etf in (s.etf for s in slots)
        if idx_weights.get(etf, 0.0) > 1e-12
    ]
    info_line = (
        f"multi-index | risk_off={risk_off_share:.1%}"
        + (f" | {' / '.join(idx_parts)}" if idx_parts else "")
    )

    return MadLiveSnapshot(
        as_of=as_of,
        tickers=tickers_tuple,
        weight_by_ticker={t: merged_weights.get(t, 0.0) for t in tickers_tuple},
        close_by_ticker={t: merged_closes.get(t, float("nan")) for t in tickers_tuple},
        # ``regime_ok`` is True whenever any equity allocation exists — the live
        # bot uses this flag to decide whether to route through the legacy
        # sleeve-flatten branch. In multi-index mode we always want the unified
        # sizing loop (equity and sleeve coexist), so only flag risk-off when
        # the allocator is 100% risk-off.
        regime_ok=(risk_off_share < 1.0 - 1e-9),
        mad_sma_short=int(short_w),
        mad_sma_long=int(long_w),
        mad_exit_ma=int(exit_ma_period or 0),
        mad_regime_ma=int(resolve_index_regime_ma(config) or 0),
        n_long=int(sum(per_slot_n_long.values())),
        n_short=int(sum(per_slot_n_short.values())),
        # Intentionally empty in multi-index mode: sleeve weights are merged
        # into ``weight_by_ticker`` above so the live bot sizes them uniformly
        # against the equity gross notional (no separate sleeve-gross / flatten
        # branch). Kept as a dict (not None) to preserve the dataclass type.
        sleeve_weight_by_ticker={},
        sleeve_close_by_ticker={},
        regime_info=info_line,
        weighting_scheme=weight_cfg.scheme,
    )


def _format_regime_sleeve_banner(
    regime_sym: str | None,
    regime_grid: tuple[int, ...],
) -> list[str]:
    """Multi-line banner for the risk-on/off regime + sleeve.

    Falls back to the single-ticker / cash-only description when ``MAD_REGIME_TICKERS`` or
    ``MAD_REGIME_OFF_SLEEVE`` aren't configured (legacy behavior).

    When ``MAD_INDEX_ALLOCATOR_ENABLED`` is True, the legacy global regime line is
    replaced with a per-slot trend-filter description (each index slot is gated
    against its own ETF's SMA inside the allocator, so the global regime doesn't
    apply).
    """
    allocator_on = index_allocator_enabled(config)
    sleeve = resolve_sleeve(config)
    safe_harbor = resolve_safe_harbor(config)
    sleeve_scheme = resolve_sleeve_weighting_scheme(config)

    lines: list[str] = []

    if allocator_on:
        slots = resolve_index_slots(config)
        index_ma = int(resolve_index_regime_ma(config) or 200)
        if slots:
            gated = " | ".join(f"{s.etf} vs {index_ma}D SMA" for s in slots)
            lines.append(
                f"  Per-slot trend   : {gated} (applied at BOTH the allocator "
                f"and each slot's stock picker; legacy global regime bypassed)"
            )
        else:
            lines.append(
                "  Per-slot trend   : allocator ON but MAD_INDEX_SLOTS empty"
            )
    else:
        if regime_sym is None:
            return ["  Regime filter    : off (MAD_REGIME_MA_ENABLED=False)"]
        reg_tickers = resolve_regime_tickers(config)
        if not reg_tickers:
            reg_tickers = (regime_sym,)
        mode = resolve_regime_mode(config)
        mode_text = "ALL below SMA" if mode == "all_below" else "ANY below SMA"
        lines.append(
            f"  Regime filter    : {'+'.join(reg_tickers)} ({mode_text}) | SMA grid {regime_grid} (0 = off)",
        )

    if sleeve:
        sleeve_bits: list[str] = []
        for sym, wt, tma in sleeve:
            if sleeve_scheme is None:
                piece = f"{sym} {wt:.0%}"
            else:
                piece = sym
            if int(tma) > 0:
                piece += f" (trend {tma}D)"
            sleeve_bits.append(piece)
        lines.append(
            f"  Risk-off sleeve  : {', '.join(sleeve_bits)} → safe harbor {safe_harbor or 'cash'}"
        )
    else:
        lines.append(f"  Risk-off sleeve  : full cash (safe harbor {safe_harbor or 'cash'})")
    if sleeve_scheme is not None:
        sh, lg = resolve_sleeve_mrat_pair(config)
        pool = [sym for sym, _w, _ in sleeve]
        if safe_harbor and safe_harbor not in pool:
            pool.append(safe_harbor)
        lines.append(
            f"  Sleeve weighting : scheme={sleeve_scheme} | pool={'+'.join(pool)} | "
            f"MRAT {sh}/{lg} (fixed weights in config ignored)"
        )
    else:
        lines.append("  Sleeve weighting : scheme=fixed (uses per-leg weights from MAD_REGIME_OFF_SLEEVE)")
    return lines


def _format_sleeve_desc(pairs: list[tuple[str, float]]) -> str:
    """Format risk-off sleeve allocation for equity-curve hover.

    ``pairs`` is a list of ``(ticker, weight)`` tuples whose magnitudes sum to at most 1.0.
    Zero-weight entries are kept so you can see the trend filter dropped a sleeve leg.
    """
    if not pairs:
        return "cash (0%)"
    bits = [f"{w * 100:.0f}% {sym}" for sym, w in pairs if abs(w) > 1e-9]
    if not bits:
        return "cash (sleeve fully dropped by trend filter)"
    return ", ".join(bits)


def _format_equity_sleeve_desc(long_w: float, short_w: float) -> str:
    """Format risk-on (equities) allocation for equity-curve hover."""
    if long_w <= 1e-9 and short_w <= 1e-9:
        return "flat (no MRAT entries)"
    parts: list[str] = []
    if long_w > 1e-9:
        parts.append(f"{long_w * 100:.0f}% MRAT long")
    if short_w > 1e-9:
        parts.append(f"{short_w * 100:.0f}% MRAT short")
    return ", ".join(parts)


def _format_weight_config_banner(cfg: WeightConfig) -> str:
    """One-line human summary of ``MAD_WEIGHTING_*`` — for backtester stdout banner + dashboard."""
    bits: list[str] = [f"scheme={cfg.scheme}"]
    if cfg.scheme == "softmax":
        bits.append(f"τ={cfg.softmax_tau:g}")
    if cfg.scheme in ("inv_vol", "mrat_distance_inv_vol"):
        bits.append(f"rvol_n={cfg.realized_vol_lookback}")
    if cfg.max_per_name is not None:
        bits.append(f"max={cfg.max_per_name:.2%}")
    if cfg.min_per_name is not None:
        bits.append(f"min={cfg.min_per_name:.2%}")
    if cfg.equal_blend and cfg.equal_blend > 0:
        bits.append(f"equal_blend={cfg.equal_blend:.0%}")
    return " | ".join(bits)


def _gross_simple_portfolio(w: pd.Series, r: pd.Series) -> float:
    """
    Dot(w, r) with NaN returns dropped and each side (long / short) renormalized over finite names.

    Raw (w * r) with NaNs zeroes out whole terms via nansum and understates exposure; this keeps
    intended full long (or short) notional on the names that have a valid close-to-close return.
    """
    w = w.reindex(r.index).fillna(0.0).astype(float)
    ri = r.to_numpy(dtype=float)
    wi = w.to_numpy(dtype=float)
    fin = np.isfinite(ri)
    g = 0.0
    pos = wi > 1e-15
    neg = wi < -1e-15
    if pos.any():
        m = pos & fin
        if m.any():
            wp, rp = wi[m], ri[m]
            g += float(np.dot(wp / wp.sum(), rp))
    if neg.any():
        m = neg & fin
        if m.any():
            wn, rn = wi[m], ri[m]
            abs_sum = float(np.sum(-wn))
            if abs_sum > 1e-15:
                g += float(np.dot(wn / abs_sum, rn))
    return g


def portfolio_path_from_panel(
    df: pd.DataFrame,
    *,
    fee_rate: float,
    regime_allow: pd.Series | None = None,
    weight_cfg: WeightConfig | None = None,
    sleeve_weights_piv: pd.DataFrame | None = None,
    sleeve_ret_piv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build date-level gross/net log returns, BH equal-weight universe, turnover fees.
    Expects columns: date, ticker, entry_signal, daily_ret. For schemes that depend on
    MRAT / σ / decile (``weight_cfg`` != equal), additional columns ``mrat``, ``sigma``,
    ``decile`` are required.

    regime_allow: optional bool Series indexed by UTC calendar date (``mad_calendar_key``).
    False → zero MRAT weights that day. When ``sleeve_weights_piv`` + ``sleeve_ret_piv``
    are provided, the risk-off book is replaced by the sleeve allocation (and sleeve
    returns contribute to portfolio equity / turnover) instead of going to cash.

    weight_cfg: ``None`` or ``WeightConfig(scheme="equal", ...)`` uses the legacy equal-weight
    sizing (preserves current behavior). Any other scheme is dispatched through
    ``mad.weighting.compute_weights`` per date.
    """
    dates = sorted(df["date"].dropna().unique())
    if len(dates) < 2:
        return pd.DataFrame()

    cfg = weight_cfg or WeightConfig()
    use_scheme = cfg.scheme != "equal"

    entry_piv = df.pivot_table(index="date", columns="ticker", values="entry_signal", aggfunc="last")
    ret_piv = df.pivot_table(index="date", columns="ticker", values="daily_ret", aggfunc="last")
    entry_piv = entry_piv.reindex(dates)
    ret_piv = ret_piv.reindex(dates)

    mrat_piv = decile_piv = rvol_piv = None
    sigma_per_date: pd.Series | None = None
    if use_scheme:
        if {"mrat", "sigma", "decile"}.issubset(df.columns):
            mrat_piv = df.pivot_table(index="date", columns="ticker", values="mrat", aggfunc="last").reindex(dates)
            decile_piv = df.pivot_table(index="date", columns="ticker", values="decile", aggfunc="last").reindex(dates)
            sigma_per_date = df.groupby("date", sort=True)["sigma"].first().reindex(dates)
            # Same-bar timing: entry_signal[t] already equals signal[t-1]; shift the
            # weight inputs to match so scheme sizing doesn't peek into date t.
            mrat_piv = mrat_piv.shift(1)
            decile_piv = decile_piv.shift(1)
            sigma_per_date = sigma_per_date.shift(1)
        if cfg.realized_vol_lookback and cfg.realized_vol_lookback >= 2:
            rvol_piv = realized_vol_pivot(df, lookback=int(cfg.realized_vol_lookback)).reindex(dates)

    # ck_index maps panel ``dates`` onto UTC-normalized calendar keys so sleeve / regime
    # pivots (always tz-aware UTC) align even when ``dates`` are naive numpy datetimes.
    ck_index = pd.DatetimeIndex([mad_calendar_key(d) for d in dates], tz="UTC")

    # Sleeve integration: extend the returns pivot with sleeve tickers so their P&L is
    # reflected on risk-off days. Columns unique to the sleeve start as zero before the
    # per-date merge; MRAT universe columns keep their original returns.
    all_cols = pd.Index(ret_piv.columns)
    if sleeve_weights_piv is not None and not sleeve_weights_piv.empty:
        new_cols = [c for c in sleeve_weights_piv.columns if c not in all_cols]
        if new_cols:
            add = pd.DataFrame(np.nan, index=ret_piv.index, columns=new_cols)
            ret_piv = pd.concat([ret_piv, add], axis=1)
            all_cols = pd.Index(ret_piv.columns)
    if sleeve_ret_piv is not None and not sleeve_ret_piv.empty:
        slr = sleeve_ret_piv.reindex(ck_index)
        slr.index = ret_piv.index  # re-label tz-aware → panel index
        for c in slr.columns:
            if c not in ret_piv.columns:
                ret_piv[c] = np.nan
            ret_piv[c] = ret_piv[c].where(~slr[c].notna(), slr[c])
        all_cols = pd.Index(ret_piv.columns)

    if regime_allow is not None:
        allow_arr = regime_allow.reindex(ck_index, fill_value=False).to_numpy(dtype=bool)
    else:
        allow_arr = np.ones(len(dates), dtype=bool)

    sleeve_arr: np.ndarray | None = None
    sleeve_cols: list[str] = []
    if sleeve_weights_piv is not None and not sleeve_weights_piv.empty:
        sleeve_cols = list(sleeve_weights_piv.columns)
        sleeve_arr = (
            sleeve_weights_piv.reindex(ck_index, fill_value=0.0).to_numpy(dtype=float)
        )

    gross_list: list[float] = []
    net_log_list: list[float] = []
    bh_log_list: list[float] = []
    flip_list: list[int] = []
    abs_w_list: list[float] = []
    regime_state_list: list[str] = []
    sleeve_desc_list: list[str] = []
    w_prev = pd.Series(0.0, index=ret_piv.columns, dtype=float)

    for j, d in enumerate(dates):
        er = entry_piv.loc[d]
        if use_scheme and mrat_piv is not None and decile_piv is not None:
            inp = WeightInputs(
                entry_signal=er,
                mrat=mrat_piv.loc[d] if d in mrat_piv.index else pd.Series(np.nan, index=er.index),
                sigma=(
                    float(sigma_per_date.loc[d])
                    if (sigma_per_date is not None and d in sigma_per_date.index)
                    else float("nan")
                ),
                decile=decile_piv.loc[d] if d in decile_piv.index else pd.Series(np.nan, index=er.index),
                realized_vol=(
                    rvol_piv.loc[d] if rvol_piv is not None and d in rvol_piv.index else None
                ),
            )
            w_core = compute_weights(inp, cfg)
        else:
            w_core = _weights_from_entries(er)
        w = pd.Series(0.0, index=all_cols, dtype=float)
        w.loc[w_core.index] = w_core.to_numpy()
        if allow_arr[j]:
            regime_state_list.append("risk-on")
            long_w = float(w[w > 0].sum())
            short_w = float(-w[w < 0].sum())
            sleeve_desc_list.append(
                _format_equity_sleeve_desc(long_w, short_w)
            )
        else:
            w = pd.Series(0.0, index=all_cols, dtype=float)
            sleeve_pairs: list[tuple[str, float]] = []
            if sleeve_arr is not None and sleeve_cols:
                sw = sleeve_arr[j]
                for k, col in enumerate(sleeve_cols):
                    if col in w.index and sw[k] != 0.0:
                        w.loc[col] = w.loc[col] + float(sw[k])
                    if sw[k] != 0.0:
                        sleeve_pairs.append((col, float(sw[k])))
            regime_state_list.append("risk-off")
            sleeve_desc_list.append(_format_sleeve_desc(sleeve_pairs))
        r = ret_piv.loc[d]
        arr = r.to_numpy(dtype=float)
        if arr.size == 0 or not np.any(np.isfinite(arr)):
            bh_simple = 0.0
        else:
            bh_simple = float(np.nanmean(arr))
        if j == 0 or not np.isfinite(bh_simple):
            bh_log_list.append(np.nan)
        else:
            bh_log_list.append(float(np.log1p(bh_simple)) if bh_simple > -1.0 else np.nan)

        if j == 0:
            gross_list.append(np.nan)
            net_log_list.append(np.nan)
            flip_list.append(0)
            abs_w_list.append(float(np.sum(np.abs(w.reindex(ret_piv.columns).fillna(0.0).to_numpy(dtype=float)))))
            w_prev = w.reindex(ret_piv.columns).fillna(0.0)
            continue

        turn = 0.5 * float(np.nansum(np.abs(w.to_numpy(dtype=float) - w_prev.to_numpy(dtype=float))))
        fee = float(fee_rate) * turn
        g_simple = _gross_simple_portfolio(w, r)
        n_simple = g_simple - fee
        gross_list.append(g_simple)
        net_log_list.append(float(np.log1p(n_simple)) if n_simple > -1.0 else np.nan)
        flip_list.append(1 if turn > 1e-12 else 0)
        abs_w_list.append(float(np.sum(np.abs(w.to_numpy(dtype=float)))))
        w_prev = w.reindex(ret_piv.columns).fillna(0.0)

    out = pd.DataFrame(
        {
            "date": dates,
            "gross_simple": gross_list,
            "next_log_return": bh_log_list,
            "net_log_return": net_log_list,
            "flip": flip_list,
            "abs_weight_sum": abs_w_list,
            "regime_state": regime_state_list,
            "sleeve_desc": sleeve_desc_list,
        }
    )
    out = out.set_index("date")
    out.index = pd.to_datetime(out.index, utc=True)
    return out


def _split_per_date(df: pd.DataFrame) -> pd.Series:
    m = df.groupby("date", sort=True)["split"].last()
    return m


def _pf(vals: np.ndarray) -> float:
    wins = float(vals[vals > 0].sum())
    losses = float(abs(vals[vals < 0].sum()))
    if losses <= 0.0:
        return np.inf if wins > 0.0 else np.nan
    return wins / losses


def _sharpe(vals: np.ndarray, bpy: float) -> float:
    if len(vals) < 2:
        return np.nan
    std = float(np.std(vals, ddof=1))
    if std <= 0.0:
        return np.nan
    return float(np.mean(vals) / std * np.sqrt(bpy))


def _sortino(vals: np.ndarray, bpy: float) -> float:
    if len(vals) < 2:
        return np.nan
    down = vals[vals < 0]
    if len(down) < 2:
        return np.nan
    std = float(np.std(down, ddof=1))
    if std <= 0.0:
        return np.nan
    return float(np.mean(vals) / std * np.sqrt(bpy))


def mad_cross_section_diagnostics(
    panel: pd.DataFrame,
    eval_date_keys: set,
    *,
    long_sigma_mult: float,
    short_sigma_mult: float,
    long_decile_min: int = 10,
    short_decile_max: int = 1,
) -> dict[str, float]:
    """
    Per eval day: cross-section validity, long-band vs MRAT gate, short-band vs MRAT gate.
    ``mad_diag_pct_days_any_top_decile`` = any name in the long decile band (≥ long_decile_min).
    ``mad_diag_pct_days_any_short_band`` = any name in the short decile band (≤ short_decile_max).
    """
    nan_block: dict[str, float] = {
        "mad_diag_eval_days": 0.0,
        "mad_diag_pct_days_valid_cross_section": float("nan"),
        "mad_diag_pct_days_any_top_decile": float("nan"),
        "mad_diag_pct_days_pass_long_gate": float("nan"),
        "mad_diag_pct_days_any_long": float("nan"),
        "mad_diag_pct_top_decile_days_no_long": float("nan"),
        "mad_diag_mean_long_names_when_long": float("nan"),
        "mad_diag_pct_days_any_short_band": float("nan"),
        "mad_diag_pct_days_pass_short_gate": float("nan"),
        "mad_diag_pct_days_any_short_signal": float("nan"),
        "mad_diag_pct_short_band_days_no_short": float("nan"),
        "mad_diag_mean_short_names_when_short": float("nan"),
    }
    need = {"date", "decile", "sigma", "mrat", "signal"}
    miss = need - set(panel.columns)
    if miss or not eval_date_keys:
        return nan_block

    lsm = float(long_sigma_mult)
    ssm = float(short_sigma_mult)
    ld_min = int(long_decile_min)
    sd_max = int(short_decile_max)
    ck_series = panel["date"].map(mad_calendar_key)
    any_valid_cs = 0
    any_long_band = 0
    any_pass_long = 0
    any_long_sig = 0
    long_band_blocked = 0
    long_counts: list[int] = []
    any_short_band = 0
    any_pass_short = 0
    any_short_sig = 0
    short_band_blocked = 0
    short_counts: list[int] = []

    for ck in eval_date_keys:
        m = ck_series == ck
        if not bool(m.any()):
            continue
        sub = panel.loc[m, ["decile", "sigma", "mrat", "signal"]]
        valid = sub["decile"].notna() & sub["sigma"].notna() & sub["mrat"].notna()
        if not bool(valid.any()):
            continue
        any_valid_cs += 1
        sv = sub.loc[valid]
        in_long_band = sv["decile"] >= ld_min
        thr_l = 1.0 + lsm * sv["sigma"]
        pass_long = in_long_band & (sv["mrat"] > thr_l)
        has_lb = bool(in_long_band.any())
        has_pl = bool(pass_long.any())
        has_long = bool(sv["signal"].eq(1).any())
        if has_lb:
            any_long_band += 1
            if not has_pl:
                long_band_blocked += 1
        if has_pl:
            any_pass_long += 1
        if has_long:
            any_long_sig += 1
            long_counts.append(int(sv["signal"].eq(1).sum()))

        in_short_band = sv["decile"] <= sd_max
        thr_s = 1.0 - ssm * sv["sigma"]
        pass_short = in_short_band & (sv["mrat"] < thr_s)
        has_sb = bool(in_short_band.any())
        has_ps = bool(pass_short.any())
        has_sh = bool(sv["signal"].eq(-1).any())
        if has_sb:
            any_short_band += 1
            if not has_ps:
                short_band_blocked += 1
        if has_ps:
            any_pass_short += 1
        if has_sh:
            any_short_sig += 1
            short_counts.append(int(sv["signal"].eq(-1).sum()))

    n = len(eval_date_keys)
    out: dict[str, float] = {
        "mad_diag_eval_days": float(n),
        "mad_diag_pct_days_valid_cross_section": 100.0 * float(any_valid_cs) / float(n),
        "mad_diag_pct_days_any_top_decile": 100.0 * float(any_long_band) / float(n),
        "mad_diag_pct_days_pass_long_gate": 100.0 * float(any_pass_long) / float(n),
        "mad_diag_pct_days_any_long": 100.0 * float(any_long_sig) / float(n),
        "mad_diag_pct_days_any_short_band": 100.0 * float(any_short_band) / float(n),
        "mad_diag_pct_days_pass_short_gate": 100.0 * float(any_pass_short) / float(n),
        "mad_diag_pct_days_any_short_signal": 100.0 * float(any_short_sig) / float(n),
    }
    out["mad_diag_pct_top_decile_days_no_long"] = (
        100.0 * float(long_band_blocked) / float(any_long_band) if any_long_band else float("nan")
    )
    out["mad_diag_mean_long_names_when_long"] = (
        float(np.mean(long_counts)) if long_counts else float("nan")
    )
    out["mad_diag_pct_short_band_days_no_short"] = (
        100.0 * float(short_band_blocked) / float(any_short_band) if any_short_band else float("nan")
    )
    out["mad_diag_mean_short_names_when_short"] = (
        float(np.mean(short_counts)) if short_counts else float("nan")
    )
    return out


def evaluate_mad(
    daily_long: pd.DataFrame,
    *,
    short_w: int,
    long_w: int,
    min_price: float,
    min_history: int,
    fee_rate: float,
    direction_mode: str,
    eval_dates: set | None,
    bars_per_year_local: float,
    exit_ma_period: int = 0,
    regime_ma_period: int = 0,
    regime_ticker: str | None = None,
    granularity: str | None = None,
    aggregate_to_daily: bool | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    panel = compute_mrat_panel(
        daily_long,
        short_w=short_w,
        long_w=long_w,
        min_price=min_price,
        min_history=min_history,
        direction_mode=direction_mode,
        exit_ma_period=exit_ma_period,
    )
    gran = str(granularity or config.TARGET_CANDLE_GRANULARITY)
    if aggregate_to_daily is None:
        daily_agg = bool(getattr(config, "MAD_AGGREGATE_TO_DAILY", True)) and gran.lower() != "1d"
    else:
        daily_agg = bool(aggregate_to_daily)

    # Multi-ticker regime + trend-following sleeve (``mad.regime_sleeve``). When a single-ticker
    # override is supplied, collapse to that ticker so existing grid searches keep their semantics;
    # otherwise honor ``MAD_REGIME_TICKERS`` + ``MAD_REGIME_OFF_SLEEVE``.
    reg_tickers_cfg = resolve_regime_tickers(config)
    reg_mode = resolve_regime_mode(config)
    sleeve_cfg = resolve_sleeve(config)
    safe_harbor = resolve_safe_harbor(config)
    if regime_ticker is not None and str(regime_ticker).strip():
        reg_tickers_use: tuple[str, ...] = (str(regime_ticker).strip().upper(),)
    else:
        reg_tickers_use = reg_tickers_cfg or ((mad_regime_ticker_symbol() or "QQQ"),)

    reg_series: RegimeBacktestSeries | None = None
    regime_allow: pd.Series | None = None
    sleeve_weights_piv: pd.DataFrame | None = None
    sleeve_ret_piv: pd.DataFrame | None = None
    if int(regime_ma_period or 0) > 0 and reg_tickers_use:
        try:
            reg_series = build_regime_backtest_series(
                ohlcv_dir=OHLCV_DIR,
                granularity=gran,
                aggregate_to_daily=daily_agg,
                regime_tickers=reg_tickers_use,
                regime_ma=int(regime_ma_period),
                mode=reg_mode,
                sleeve=sleeve_cfg,
                safe_harbor=safe_harbor,
                prefer_precomputed_sma=False,
                sleeve_weight_cfg=resolve_sleeve_weight_config(config),
                sleeve_mrat_pair=resolve_sleeve_mrat_pair(config),
            )
        except FileNotFoundError:
            # Legacy fallback: single-ticker regime with cash sleeve.
            reg_series = None
            regime_allow = _build_regime_allow(
                int(regime_ma_period or 0),
                reg_tickers_use[0],
                gran,
                daily_agg,
                OHLCV_DIR,
            )
        if reg_series is not None:
            regime_allow = reg_series.risk_on
            if not reg_series.sleeve_weights_piv.empty:
                sleeve_weights_piv = reg_series.sleeve_weights_piv
                sleeve_ret_piv = _load_sleeve_ret_piv(
                    tuple(reg_series.sleeve_tickers),
                    gran,
                    daily_agg=daily_agg,
                    ohlcv_dir=OHLCV_DIR,
                )

    weight_cfg = resolve_weight_config(config)
    path = portfolio_path_from_panel(
        panel,
        fee_rate=fee_rate,
        regime_allow=regime_allow,
        weight_cfg=weight_cfg,
        sleeve_weights_piv=sleeve_weights_piv,
        sleeve_ret_piv=sleeve_ret_piv,
    )
    split_by_d = _split_per_date(panel)
    path = path.join(split_by_d.rename("split"), how="left")

    if eval_dates is None:
        eval_date_keys = {mad_calendar_key(ix) for ix in path.index}
    else:
        eval_date_keys = {mad_calendar_key(x) for x in eval_dates}
    lsm = float(getattr(config, "MAD_LONG_SIGMA_MULT", 1.0))
    ssm_raw = float(getattr(config, "MAD_SHORT_SIGMA_MULT", 1.0))
    symm = bool(getattr(config, "MAD_SYMMETRIC_SHORT_SIGMA", False))
    ssm_eff = lsm if symm else ssm_raw
    ld_min = max(1, min(10, int(getattr(config, "MAD_LONG_DECILE_MIN", 10))))
    sd_max = max(1, min(10, int(getattr(config, "MAD_SHORT_DECILE_MAX", 1))))
    diag = mad_cross_section_diagnostics(
        panel,
        eval_date_keys,
        long_sigma_mult=lsm,
        short_sigma_mult=ssm_eff,
        long_decile_min=ld_min,
        short_decile_max=sd_max,
    )

    if eval_dates is None:
        mask_ser = pd.Series(True, index=path.index, dtype=bool)
    else:
        ed = {mad_calendar_key(x) for x in eval_dates}
        mask_ser = pd.Series([mad_calendar_key(ix) in ed for ix in path.index], index=path.index, dtype=bool)

    valid = path["net_log_return"].notna() & path["next_log_return"].notna() & mask_ser
    vals = path.loc[valid, "net_log_return"].to_numpy(dtype=float)
    gross_simple = path.loc[valid, "gross_simple"].to_numpy(dtype=float)
    gross_log = np.log1p(gross_simple[np.isfinite(gross_simple) & (gross_simple > -1.0)])

    rebalance_days = int(path.loc[valid, "flip"].sum())
    with_pos = path.loc[valid, "abs_weight_sum"].to_numpy(dtype=float) > 1e-9
    days_with_position = int(with_pos.sum())

    rma = int(regime_ma_period or 0)
    rtick = ""
    if rma > 0:
        rtick = (regime_ticker or "").strip().upper() or (mad_regime_ticker_symbol() or "")
    metrics = {
        "mad_sma_short": int(short_w),
        "mad_sma_long": int(long_w),
        "mad_exit_ma": int(exit_ma_period or 0),
        "mad_regime_ma": rma,
        "mad_regime_ticker": rtick,
        "mad_long_decile_min": int(ld_min),
        "mad_short_decile_max": int(sd_max),
        "mad_symmetric_short_sigma": float(symm),
        "mad_short_sigma_effective": float(ssm_eff),
        "bars": int(valid.sum()),
        "trades": rebalance_days,
        "rebalance_days": rebalance_days,
        "days_with_position": days_with_position,
        "profit_factor": float(_pf(vals)),
        "sharpe_ratio": float(_sharpe(vals, bars_per_year_local)),
        "sortino_ratio": float(_sortino(vals, bars_per_year_local)),
        "gross_total_log_return": float(np.sum(gross_log)) if len(gross_log) else 0.0,
        "net_total_log_return": float(np.sum(vals)) if len(vals) else 0.0,
        "net_total_return": float(np.expm1(np.sum(vals))) if len(vals) else 0.0,
    }
    metrics.update(diag)

    eval_df = path.copy()
    eval_df["next_log_return"] = eval_df["next_log_return"].where(valid, np.nan)
    eval_df["net_log_return"] = eval_df["net_log_return"].where(valid, np.nan)
    return metrics, eval_df


def evaluate_mad_multi_index(
    daily_long: pd.DataFrame,
    *,
    short_w: int,
    long_w: int,
    min_price: float,
    min_history: int,
    fee_rate: float,
    direction_mode: str,
    eval_dates: set | None,
    bars_per_year_local: float,
    exit_ma_period: int = 0,
    granularity: str | None = None,
    aggregate_to_daily: bool | None = None,
    _per_slot_cache: dict | None = None,
    _return_per_slot_cache: bool = False,
    _verbose: bool = True,
    sleeve_weight_cfg: WeightConfig | None = None,
    reuse_cached_sleeve: bool = True,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Multi-index variant of ``evaluate_mad`` — runs the stock-picker per slot and
    blends via the top-level index allocator.

    Differences from ``evaluate_mad``:
      * Stock-picker runs once per ``IndexSlot`` on the slot's exclusive universe.
      * Legacy single-ticker regime filter is DISABLED per-slot (the allocator's
        own trend filter on each index ETF takes its place at the top level).
      * Per-slot daily returns are blended by per-date index weights from
        ``build_index_allocation_series``; the ``__risk_off__`` share is composed
        with the standard risk-off sleeve (``regime_sleeve`` module).

    Returns the same ``(metrics, eval_df)`` shape as ``evaluate_mad`` so downstream
    rendering / dashboard code can consume either transparently.

    ``sleeve_weight_cfg`` overrides ``resolve_sleeve_weight_config(config)`` when
    building the risk-off sleeve time series (handy for grid searches). When
    ``_per_slot_cache`` is set, ``reuse_cached_sleeve=False`` forces a fresh
    ``build_regime_backtest_series`` with the current sleeve scheme — required
    when sweeping ``MAD_SLEEVE_WEIGHTING_SCHEME`` while reusing cached per-slot
    stock returns from the first pass.
    """
    if daily_long.empty:
        raise RuntimeError("evaluate_mad_multi_index: empty daily_long panel.")

    gran = str(granularity or config.TARGET_CANDLE_GRANULARITY)
    if aggregate_to_daily is None:
        daily_agg = bool(getattr(config, "MAD_AGGREGATE_TO_DAILY", True)) and gran.lower() != "1d"
    else:
        daily_agg = bool(aggregate_to_daily)

    slots = resolve_index_slots(config)
    if not slots:
        raise RuntimeError(
            "evaluate_mad_multi_index: MAD_INDEX_SLOTS empty — cannot run multi-index backtest."
        )

    # 1. Top-level allocation series (per-date index weights + risk-off share).
    allocation = build_index_allocation_series(
        slots=slots,
        regime_ma=resolve_index_regime_ma(config),
        mrat_pair=resolve_index_mrat_pair(config),
        weight_cfg=resolve_index_weight_config(config),
        granularity=gran,
        ohlcv_dir=OHLCV_DIR,
        aggregate_to_daily=daily_agg,
        prefer_precomputed_sma=False,
    )
    if allocation.index_weights_piv.empty:
        raise RuntimeError("evaluate_mad_multi_index: allocation series empty.")

    # Clip the allocator calendar to the user-configured eval window
    # (``MAD_BACKTEST_START_DATE`` / ``MAD_BACKTEST_END_DATE``). Data is still
    # loaded for the full range upstream so SMAs / MRATs have proper warmup,
    # but everything downstream (``cal``, ``eval_df``, B&H benchmark, per-slot
    # metrics) is restricted to the window — so the dashboard doesn't show a
    # long flat stretch while B&H captures an unfair head start.
    bt_start, bt_end = resolve_backtest_window()
    if bt_start is not None or bt_end is not None:
        full_cal = allocation.index_weights_piv.index
        win_mask = pd.Series(True, index=full_cal, dtype=bool)
        if bt_start is not None:
            win_mask &= full_cal >= bt_start
        if bt_end is not None:
            win_mask &= full_cal <= bt_end
        if not bool(win_mask.any()):
            raise RuntimeError(
                f"MAD_BACKTEST_START_DATE/END_DATE window "
                f"[{bt_start.date() if bt_start is not None else '-inf'} .. "
                f"{bt_end.date() if bt_end is not None else '+inf'}] "
                f"excludes all allocator dates ({full_cal.min().date()} → "
                f"{full_cal.max().date()})."
            )
        clip = win_mask.to_numpy()
        allocation = dc_replace(
            allocation,
            index_weights_piv=allocation.index_weights_piv.loc[clip],
            index_trend_ok=allocation.index_trend_ok.loc[clip],
            index_mrat=allocation.index_mrat.loc[clip],
        )
        clipped_cal = allocation.index_weights_piv.index
        print(
            f"  [eval window] trimmed allocator calendar: "
            f"{full_cal.min().date()} → {full_cal.max().date()} "
            f"({len(full_cal)} days) --> "
            f"{clipped_cal.min().date()} → {clipped_cal.max().date()} "
            f"({len(clipped_cal)} days)  "
            f"[config: MAD_BACKTEST_START_DATE="
            f"{getattr(config, 'MAD_BACKTEST_START_DATE', None)!r}, "
            f"MAD_BACKTEST_END_DATE="
            f"{getattr(config, 'MAD_BACKTEST_END_DATE', None)!r}]"
        )
        # Also constrain the per-slot eval_dates so standalone slot metrics
        # match the same window. ``eval_dates`` on evaluate_mad is intersected
        # with split_by_d internally, but here we override it fully since
        # multi-index runs bypass the walkforward split logic.
        window_eval_dates = {mad_calendar_key(d) for d in clipped_cal}
        eval_dates = (
            window_eval_dates
            if eval_dates is None
            else (eval_dates & window_eval_dates)
        )

    # 2. Per-slot stock-picker. Store per-slot net-simple daily return on a calendar key.
    #
    # GRID-SEARCH CACHE: when ``_per_slot_cache`` is supplied, skip the expensive
    # stock-picking + risk-off sleeve loading entirely. The per-slot picks depend
    # only on ``MAD_WEIGHTING_SCHEME`` (stock-level), not the index-level scheme,
    # so caching is safe across index-level reruns within a single stock scheme.
    completeness_frac = resolve_min_data_completeness(config)
    per_slot_returns: dict[str, pd.Series] = {}
    per_slot_metrics: dict[str, dict[str, float]] = {}
    per_slot_eval_df: dict[str, pd.DataFrame] = {}
    per_slot_status: dict[str, str] = {}
    _use_cache = _per_slot_cache is not None
    if _use_cache:
        per_slot_returns = dict(_per_slot_cache["per_slot_returns"])
        per_slot_metrics = dict(_per_slot_cache["per_slot_metrics"])
        per_slot_status = dict(_per_slot_cache["per_slot_status"])
        if _verbose:
            print("  [per-slot cache hit] reusing stock-picker + sleeve returns")

    for slot in slots:
        if _use_cache:
            break
        if not slot.universe:
            per_slot_returns[slot.etf] = pd.Series(0.0, dtype=float)
            per_slot_status[slot.etf] = "empty_universe"
            print(f"  [{slot.etf}] SKIP: exclusive universe is empty (after priority assignment).")
            continue
        slot_df_raw = daily_long[daily_long["ticker"].isin(slot.universe)].copy()
        pre_n = int(slot_df_raw["ticker"].nunique()) if not slot_df_raw.empty else 0
        if completeness_frac > 0.0:
            slot_df, dropped = filter_universe_by_data_completeness(
                slot_df_raw,
                window_days=int(long_w),
                min_completeness=completeness_frac,
            )
            if dropped:
                print(
                    f"  [{slot.etf}] dropped {len(dropped)} low-completeness tickers "
                    f"(window={long_w}d, thresh={completeness_frac:.0%})"
                )
        else:
            slot_df = slot_df_raw
        post_n = int(slot_df["ticker"].nunique()) if not slot_df.empty else 0
        if slot_df.empty:
            per_slot_returns[slot.etf] = pd.Series(0.0, dtype=float)
            per_slot_status[slot.etf] = "empty_after_filter"
            print(
                f"  [{slot.etf}] SKIP: 0 tickers survived the data-completeness filter "
                f"(configured universe size={len(slot.universe)}, panel had {pre_n} with rows)."
            )
            continue
        slot_regime_ma = int(resolve_index_regime_ma(config) or 0)
        print(
            f"  [{slot.etf}] running stock-picker on {post_n} tickers "
            f"(configured={len(slot.universe)}, present-in-panel={pre_n}, "
            f"regime gate: {slot.etf} vs {slot_regime_ma}D SMA)..."
        )
        metrics_s, eval_df_s = evaluate_mad(
            slot_df,
            short_w=int(short_w),
            long_w=int(long_w),
            min_price=float(min_price),
            min_history=int(min_history),
            fee_rate=float(fee_rate),
            direction_mode=str(direction_mode),
            eval_dates=eval_dates,
            bars_per_year_local=float(bars_per_year_local),
            exit_ma_period=int(exit_ma_period or 0),
            # Per-slot regime gate: the picker rotates into the standard
            # risk-off sleeve (``MAD_REGIME_OFF_SLEEVE``) whenever the slot's
            # own ETF is below its ``MAD_INDEX_REGIME_MA`` SMA. This is
            # *duplicative* of the top-level allocator trend filter on the
            # same ETF (same MA), but it makes the **standalone** per-slot
            # return in the summary reflect real regime-aware trading.
            # On blended dates where the allocator gives the slot non-zero
            # weight the ETF is by construction above trend, so the slot's
            # regime gate is passing too — no double-rotation drag.
            regime_ma_period=slot_regime_ma,
            regime_ticker=slot.etf,
            granularity=gran,
            aggregate_to_daily=daily_agg,
        )
        per_slot_metrics[slot.etf] = metrics_s
        per_slot_eval_df[slot.etf] = eval_df_s
        ret_series = eval_df_s["net_log_return"].dropna()
        per_slot_returns[slot.etf] = np.expm1(ret_series)  # simple return
        # Additional diagnostic: "days with position" is in metrics if evaluate_mad fills it.
        days_pos = metrics_s.get("days_with_position", None)
        bars = int(metrics_s.get("bars", 0))
        ntr = float(metrics_s.get("net_total_return", 0.0))
        if bars == 0 or (ret_series.abs().sum() == 0.0):
            per_slot_status[slot.etf] = "no_trades"
            print(
                f"  [{slot.etf}] WARNING: stock-picker returned a flat book "
                f"(bars={bars}, |returns|=0). Likely: no eval dates overlap the slot's data, "
                f"or no tickers ever hit the long decile in the eval window."
            )
        else:
            per_slot_status[slot.etf] = "ok"
            dp_txt = f", days_with_position={int(days_pos)}" if days_pos is not None else ""
            print(
                f"  [{slot.etf}] done: net_total_return={ntr:+.2%}, bars={bars}{dp_txt}"
            )

    # 3. Risk-off sleeve returns — reuse regime_sleeve with existing config (we need
    #    sleeve weights per date; top-level regime flag is ignored here since the
    #    allocator's own risk_off_share dictates when the sleeve is active).
    sw_cfg = (
        sleeve_weight_cfg
        if sleeve_weight_cfg is not None
        else resolve_sleeve_weight_config(config)
    )
    sleeve_weights_piv: pd.DataFrame | None = None
    sleeve_ret_piv: pd.DataFrame | None = None
    _load_sleeve_from_cache = (
        _use_cache
        and reuse_cached_sleeve
        and _per_slot_cache.get("sleeve_weights_piv") is not None
    )
    if _load_sleeve_from_cache:
        sleeve_weights_piv = _per_slot_cache["sleeve_weights_piv"]
        sleeve_ret_piv = _per_slot_cache["sleeve_ret_piv"]
    else:
        try:
            reg_series = build_regime_backtest_series(
                ohlcv_dir=OHLCV_DIR,
                granularity=gran,
                aggregate_to_daily=daily_agg,
                regime_tickers=resolve_regime_tickers(config) or (slots[0].etf,),
                regime_ma=max(1, int(resolve_regime_ma(config) or 200)),
                mode=resolve_regime_mode(config),
                sleeve=resolve_sleeve(config),
                safe_harbor=resolve_safe_harbor(config),
                prefer_precomputed_sma=False,
                sleeve_weight_cfg=sw_cfg,
                sleeve_mrat_pair=resolve_sleeve_mrat_pair(config),
            )
        except FileNotFoundError:
            reg_series = None
        if reg_series is not None and not reg_series.sleeve_weights_piv.empty:
            sleeve_weights_piv = reg_series.sleeve_weights_piv
            sleeve_ret_piv = _load_sleeve_ret_piv(
                tuple(reg_series.sleeve_tickers),
                gran,
                daily_agg=daily_agg,
                ohlcv_dir=OHLCV_DIR,
            )

    # 4. Blend per-date. Align every series to the allocation calendar.
    cal = allocation.index_weights_piv.index
    blended_simple = pd.Series(0.0, index=cal, dtype=float)
    for etf, ret in per_slot_returns.items():
        if ret.empty:
            continue
        ret_u = ret.copy()
        ret_u.index = pd.DatetimeIndex(ret_u.index, tz="UTC").normalize()
        ret_u = ret_u.groupby(level=0).last().reindex(cal).fillna(0.0)
        w_col = allocation.index_weights_piv[etf].reindex(cal).fillna(0.0)
        blended_simple = blended_simple.add(ret_u * w_col, fill_value=0.0)

    risk_off_ret = pd.Series(0.0, index=cal, dtype=float)
    if sleeve_weights_piv is not None and sleeve_ret_piv is not None and not sleeve_ret_piv.empty:
        sw = sleeve_weights_piv.reindex(cal).fillna(0.0)
        sr = sleeve_ret_piv.reindex(cal).fillna(0.0)
        common = sw.columns.intersection(sr.columns)
        if len(common) > 0:
            risk_off_ret = (sw[common] * sr[common]).sum(axis=1).fillna(0.0)
    w_ro = allocation.index_weights_piv[RISK_OFF_KEY].reindex(cal).fillna(0.0)
    blended_simple = blended_simple.add(risk_off_ret * w_ro, fill_value=0.0)

    # 5. Translate to the eval_df shape expected by the dashboard.
    net_log = np.log1p(blended_simple.clip(lower=-0.999999))

    # Buy-and-hold benchmark. Previously this was (incorrectly) set to the
    # strategy's own ``net_log_return.shift(-1)``, which made the dashboard
    # compare the strategy to a 1-bar-shifted copy of itself (vol, drawdown,
    # Sharpe all matched "buy & hold" by construction). We reconstruct it from
    # the panel's close series.
    #
    # We prefer SPY as the B&H benchmark whenever it's one of the configured
    # slots, regardless of slot priority order. Slot order controls exclusive
    # ticker assignment + panel calendar anchor, but everyone's mental model
    # of "buy & hold" is SPY — not IWM or QQQ — so the dashboard comparison
    # should stay stable when we re-order ``MAD_INDEX_SLOTS``. Fallback chain:
    # SPY → first slot ETF.
    slot_etfs = {s.etf for s in slots}
    ref_etf = "SPY" if "SPY" in slot_etfs else slots[0].etf
    ref_bars = daily_long[daily_long["ticker"] == ref_etf]
    if ref_bars.empty:
        next_log = pd.Series(np.nan, index=cal, dtype=float)
    else:
        ref_close = (
            ref_bars.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .set_index("date")["close"]
            .astype(float)
        )
        # Normalize index tz to match ``cal`` (``allocation.index_weights_piv.index``).
        ref_close.index = pd.DatetimeIndex(ref_close.index, tz="UTC").normalize()
        ref_close = ref_close.groupby(level=0).last().sort_index()
        ref_log_ret = np.log(ref_close / ref_close.shift(1))
        # ``next_log_return[t]`` is the BH return realized *between today's
        # close and the next bar's close* — align to the same convention as
        # the single-universe path.
        next_log = ref_log_ret.shift(-1).reindex(cal)

    # Rebalance / trade counts for the top-level allocator: count days where
    # the index weight vector (SPY/QQQ/IWM/risk-off) actually changed. The
    # allocator reweights daily in principle, but in practice weights are
    # piecewise-constant across long stretches (e.g. "all three pass trend,
    # equal weights" holds for weeks at a time), so this gives a meaningful
    # turnover signal instead of the previous hardcoded ``0``.
    w_diff = allocation.index_weights_piv.diff().abs().sum(axis=1)
    # First row has NaN diff → treat as an initial rebalance only if any weight
    # is non-zero on the first day.
    first_nonzero = bool(float(allocation.index_weights_piv.iloc[0].abs().sum()) > 1e-12)
    rebalance_mask = (w_diff > 1e-9).fillna(first_nonzero)
    rebalance_days_total = int(rebalance_mask.sum())

    eval_df = pd.DataFrame(
        {
            "net_log_return": net_log,
            "gross_simple": blended_simple,  # no separate gross/net split at top level
            "abs_weight_sum": 1.0,           # always fully invested (equity + risk-off sum to 1)
            "flip": rebalance_mask.reindex(cal).fillna(False).astype(bool),
            "next_log_return": next_log,
        },
        index=cal,
    )

    # Stash allocator diagnostics on the frame for dashboard hover.
    for col in allocation.index_weights_piv.columns:
        eval_df[f"alloc_{col}"] = allocation.index_weights_piv[col].reindex(cal).fillna(0.0)

    # Also stash per-ETF trend-filter state (True = ETF above its regime MA on
    # that day, i.e. eligible for equity allocation). Surfaces WHICH slot(s)
    # caused the allocator to rotate into risk-off, year-over-year. Missing
    # values (ETF not in trend panel) are treated as False so the diagnostic
    # reads as "not passing".
    try:
        for etf in allocation.index_trend_ok.columns:
            eval_df[f"trend_ok_{etf}"] = (
                allocation.index_trend_ok[etf].reindex(cal).fillna(False).astype(bool)
            )
    except Exception:  # noqa: BLE001
        # Defensive: if ``index_trend_ok`` is unexpectedly missing or misshaped,
        # skip silently so the main strategy output still renders.
        pass

    # Sleeve description per date (4-way split for hover): per-index weights,
    # plus — on risk-off days — a nested breakdown of the sleeve's internal
    # composition (e.g. GLD / TLT / BIL legs produced by ``regime_sleeve``).
    def _fmt_sleeve_inner(dt: pd.Timestamp) -> str:
        if sleeve_weights_piv is None or sleeve_weights_piv.empty:
            return ""
        if dt not in sleeve_weights_piv.index:
            return ""
        row = sleeve_weights_piv.loc[dt]
        parts = [
            f"{int(round(float(v) * 100))}% {sym}"
            for sym, v in row.items()
            if pd.notna(v) and float(v) >= 0.005
        ]
        return " + ".join(parts) if parts else ""

    def _fmt_alloc_row(dt: pd.Timestamp) -> str:
        row = allocation.index_weights_piv.loc[dt]
        parts: list[str] = []
        for col in allocation.index_weights_piv.columns:
            v = float(row.get(col, 0.0))
            if v >= 0.001:
                if col == RISK_OFF_KEY:
                    inner = _fmt_sleeve_inner(dt)
                    label = f"risk-off [{inner}]" if inner else "risk-off"
                else:
                    label = col
                parts.append(f"{int(round(v * 100))}% {label}")
        return " / ".join(parts) if parts else "0%"

    eval_df["sleeve_desc"] = [_fmt_alloc_row(d) for d in cal]
    eval_df["regime_state"] = np.where(
        allocation.index_weights_piv[RISK_OFF_KEY].reindex(cal).fillna(1.0) >= 0.999,
        "risk-off (all indexes failed trend)",
        np.where(
            allocation.index_weights_piv[RISK_OFF_KEY].reindex(cal).fillna(0.0) <= 0.001,
            "risk-on (all indexes passed trend)",
            "mixed (progressive risk-off blend)",
        ),
    )

    # Mask eval_dates if caller restricted.
    if eval_dates is None:
        mask_ser = pd.Series(True, index=eval_df.index, dtype=bool)
    else:
        ed = {mad_calendar_key(x) for x in eval_dates}
        mask_ser = pd.Series([mad_calendar_key(ix) in ed for ix in eval_df.index], index=eval_df.index, dtype=bool)
    valid = eval_df["net_log_return"].notna() & eval_df["next_log_return"].notna() & mask_ser
    vals = eval_df.loc[valid, "net_log_return"].to_numpy(dtype=float)
    gross_simple_arr = eval_df.loc[valid, "gross_simple"].to_numpy(dtype=float)
    gross_log = np.log1p(gross_simple_arr[np.isfinite(gross_simple_arr) & (gross_simple_arr > -1.0)])

    metrics = {
        "mad_sma_short": int(short_w),
        "mad_sma_long": int(long_w),
        "mad_exit_ma": int(exit_ma_period or 0),
        "mad_regime_ma": int(resolve_index_regime_ma(config)),
        "mad_regime_ticker": "[multi-index]",
        "mad_index_allocator": 1,
        "mad_sleeve_weighting_scheme": str(sw_cfg.scheme) if sw_cfg is not None else "fixed",
        "bars": int(valid.sum()),
        # ``trades`` / ``rebalance_days`` = # calendar days where the top-level
        # index-allocator vector actually changed (SPY/QQQ/IWM/risk-off). This
        # excludes static holding stretches (e.g. "all 3 pass trend, equal
        # weights" may hold for weeks).
        "trades": rebalance_days_total,
        "rebalance_days": rebalance_days_total,
        "days_with_position": int(valid.sum()),
        "profit_factor": float(_pf(vals)),
        "sharpe_ratio": float(_sharpe(vals, bars_per_year_local)),
        "sortino_ratio": float(_sortino(vals, bars_per_year_local)),
        "gross_total_log_return": float(np.sum(gross_log)) if len(gross_log) else 0.0,
        "net_total_log_return": float(np.sum(vals)) if len(vals) else 0.0,
        "net_total_return": float(np.expm1(np.sum(vals))) if len(vals) else 0.0,
    }
    # Surface per-slot contributions + status so main() can render a line for
    # every configured slot (including ones that short-circuited / had no trades).
    for etf, m_slot in per_slot_metrics.items():
        metrics[f"slot_{etf}_net_total_return"] = float(m_slot.get("net_total_return", float("nan")))
        metrics[f"slot_{etf}_sharpe"] = float(m_slot.get("sharpe_ratio", float("nan")))
        metrics[f"slot_{etf}_bars"] = float(m_slot.get("bars", 0))
    for etf, status in per_slot_status.items():
        metrics[f"slot_{etf}_status"] = status  # type: ignore[assignment]

    # Propagate split id for dashboard split-column support (reuse panel's inferred split).
    try:
        panel_for_splits = compute_mrat_panel(
            daily_long,
            short_w=int(short_w),
            long_w=int(long_w),
            min_price=float(min_price),
            min_history=int(min_history),
            direction_mode=str(direction_mode),
            exit_ma_period=int(exit_ma_period or 0),
        )
        split_by_d = _split_per_date(panel_for_splits)
        eval_df = eval_df.join(split_by_d.rename("split"), how="left")
    except Exception:
        eval_df["split"] = np.nan

    eval_df["next_log_return"] = eval_df["next_log_return"].where(valid, np.nan)
    eval_df["net_log_return"] = eval_df["net_log_return"].where(valid, np.nan)

    # Optionally hand back the per-slot cache so callers running a grid search
    # over index-level schemes can reuse the expensive stock-picking step.
    if _return_per_slot_cache:
        metrics["_per_slot_cache"] = {
            "per_slot_returns": per_slot_returns,
            "per_slot_metrics": per_slot_metrics,
            "per_slot_status": per_slot_status,
            "sleeve_weights_piv": sleeve_weights_piv,
            "sleeve_ret_piv": sleeve_ret_piv,
        }

    return metrics, eval_df


def _bars_per_year_for_mad(granularity: str, daily_eval: bool) -> float:
    if daily_eval or str(granularity).lower() == "1d":
        return float(bars_per_year("1d"))
    return float(bars_per_year(granularity))


def daily_split_frame(daily_long: pd.DataFrame) -> pd.DataFrame:
    """One row per trading date with walk-forward split id (from reference ticker bars)."""
    return daily_long.groupby("date", as_index=False).agg(split=("split", "last")).sort_values("date")


def resolve_mad_split_plan(daily_long: pd.DataFrame) -> tuple[list[int], list[int]]:
    ex_is, ex_oos = read_explicit_split_plan_from_config()
    plan = resolve_split_plan(
        df=daily_split_frame(daily_long),
        is_target=int(getattr(config, "MAD_IS_SPLITS", 6)),
        oos_reserved=int(getattr(config, "MAD_OOS_SPLITS", 0)),
        label="MAD",
        explicit_is=ex_is,
        explicit_oos=ex_oos,
    )
    return plan.is_splits, plan.reserved_oos_splits


def _param_grid() -> list[tuple[int, int]]:
    sg = getattr(config, "MAD_SMA_SHORT_GRID", (getattr(config, "MAD_SMA_SHORT", 21),))
    lg = getattr(config, "MAD_SMA_LONG_GRID", (getattr(config, "MAD_SMA_LONG", 200),))
    if isinstance(sg, int):
        sg = (sg,)
    if isinstance(lg, int):
        lg = (lg,)
    pairs = [(int(a), int(b)) for a, b in itertools.product(sg, lg) if int(b) > int(a)]
    if not pairs:
        raise ValueError("MAD_SMA_LONG must be > MAD_SMA_SHORT for every grid pair.")
    return pairs


def _exit_ma_grid() -> tuple[int, ...]:
    if not bool(getattr(config, "MAD_EXIT_MA_ENABLED", True)):
        return (0,)
    g = getattr(config, "MAD_EXIT_MA_GRID", (0,))
    if isinstance(g, int):
        return (max(0, int(g)),)
    return tuple(max(0, int(x)) for x in g)


def _save_outputs(
    split_metrics: pd.DataFrame,
    sweep_df: pd.DataFrame,
    split_sweep_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    ref_ticker: str,
) -> None:
    global RESULTS_DB
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{ref_ticker}_{config.TARGET_CANDLE_GRANULARITY}_mad"
    split_csv = DATASETS_DIR / f"{stem}_is_split_metrics.csv"
    sweep_csv = DATASETS_DIR / f"{stem}_robustness_sweep.csv"
    split_sweep_csv = DATASETS_DIR / f"{stem}_robustness_sweep_by_split.csv"
    summary_csv = DATASETS_DIR / f"{stem}_summary.csv"
    split_metrics.to_csv(split_csv, index=False)
    sweep_df.to_csv(sweep_csv, index=False)
    split_sweep_df.to_csv(split_sweep_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    global RESULTS_DB
    RESULTS_DB = MAD_DATA_DIR / f"{ref_ticker}_{config.TARGET_CANDLE_GRANULARITY}_mad_optim.db"
    with sqlite3.connect(RESULTS_DB) as con:
        split_metrics.to_sql("is_split_metrics", con, if_exists="replace", index=False)
        sweep_df.to_sql("robustness_sweep", con, if_exists="replace", index=False)
        split_sweep_df.to_sql("robustness_sweep_by_split", con, if_exists="replace", index=False)
        summary_df.to_sql("summary", con, if_exists="replace", index=False)
    print(
        "\nSaved MAD outputs:\n"
        f"  DB   : {RESULTS_DB}\n"
        f"  CSVs :\n"
        f"    - {split_csv}\n"
        f"    - {sweep_csv}\n"
        f"    - {split_sweep_csv}\n"
        f"    - {summary_csv}\n"
    )


def _print_mad_cross_section_diag(metrics: dict[str, float]) -> None:
    if "mad_diag_eval_days" not in metrics:
        return
    k = float(getattr(config, "MAD_LONG_SIGMA_MULT", 1.0))
    ks = float(metrics.get("mad_short_sigma_effective", getattr(config, "MAD_SHORT_SIGMA_MULT", 1.0)))
    ld = int(metrics.get("mad_long_decile_min", getattr(config, "MAD_LONG_DECILE_MIN", 10)))
    sd_ = int(metrics.get("mad_short_decile_max", getattr(config, "MAD_SHORT_DECILE_MAX", 1)))
    print(
        "  Cross-section diagnostics (all scored calendar days in eval set):\n"
        f"    Long σ mult / decile≥ : {k} / {ld}\n"
        f"    Short σ mult (eff.) / decile≤ : {ks} / {sd_}\n"
        f"    Eval days           : {int(metrics['mad_diag_eval_days'])}\n"
        f"    % days w/ valid σ+decile (≥1 name) : "
        f"{metrics.get('mad_diag_pct_days_valid_cross_section', float('nan')):.1f}\n"
        f"    % days any long decile band      : {metrics.get('mad_diag_pct_days_any_top_decile', float('nan')):.1f}\n"
        f"    % days pass long MRAT gate       : {metrics.get('mad_diag_pct_days_pass_long_gate', float('nan')):.1f}\n"
        f"    % long-band days blocked by MRAT : "
        f"{metrics.get('mad_diag_pct_top_decile_days_no_long', float('nan')):.1f}\n"
        f"    Mean long names when long        : {metrics.get('mad_diag_mean_long_names_when_long', float('nan')):.2f}\n"
        f"    % days any short decile band     : {metrics.get('mad_diag_pct_days_any_short_band', float('nan')):.1f}\n"
        f"    % days pass short MRAT gate      : {metrics.get('mad_diag_pct_days_pass_short_gate', float('nan')):.1f}\n"
        f"    % short-band days blocked by MRAT: "
        f"{metrics.get('mad_diag_pct_short_band_days_no_short', float('nan')):.1f}\n"
        f"    Mean short names when short      : {metrics.get('mad_diag_mean_short_names_when_short', float('nan')):.2f}\n"
    )


def _mad_robustness_insights(sweep_df: pd.DataFrame) -> str:
    if sweep_df.empty:
        return "No robustness sweep rows."
    pf = sweep_df["profit_factor"].replace([np.inf, -np.inf], np.nan).dropna()
    if pf.empty:
        return "No finite profit factor values in sweep."
    best_idx = int(sweep_df["profit_factor"].replace(-np.inf, np.nan).idxmax())
    br = sweep_df.loc[best_idx]
    ex_s = ""
    if "mad_exit_ma" in sweep_df.columns:
        ex_s = f", exit MA={int(br['mad_exit_ma'])}"
    rg_s = ""
    if "mad_regime_ma" in sweep_df.columns and int(br["mad_regime_ma"]) > 0:
        rg_s = f", regime MA={int(br['mad_regime_ma'])}"
    return (
        f"Best PF={float(br['profit_factor']):.4f} at SMA short={int(br['mad_sma_short'])}, "
        f"long={int(br['mad_sma_long'])}{ex_s}{rg_s} (mean across IS splits in split-level sweep).\n"
        f"Median PF across grid: {float(pf.median()):.4f}\n"
    )


def build_app(
    results: dict[int, dict[str, object]],
    sweep_df: pd.DataFrame,
    insights: str,
    ref_ticker: str,
    universe_n: int,
    *,
    combined_only: bool = False,
    weight_banner: str | None = None,
) -> Dash:
    real_splits = sorted(s for s in results if s != AVG_KEY)
    if combined_only or not real_splits:
        slider_marks = {
            AVG_SLIDER_VAL: {
                "label": "All non-warmup",
                "style": {"color": "#f9a825", "fontWeight": "bold"},
            }
        }
        slider_min = AVG_SLIDER_VAL
        slider_max = AVG_SLIDER_VAL
        first_split = AVG_SLIDER_VAL
    else:
        first_split = real_splits[0]
        slider_marks = {
            AVG_SLIDER_VAL: {"label": "Avg", "style": {"color": "#f9a825", "fontWeight": "bold"}}
        }
        slider_marks.update({s: {"label": str(s), "style": {"color": "#aaa"}} for s in real_splits})
        slider_min = AVG_SLIDER_VAL
        slider_max = real_splits[-1]

    def _robustness_fig(
        selected_short: int, selected_long: int, selected_exit: int, selected_regime: int
    ) -> go.Figure:
        if sweep_df.empty or "mad_sma_short" not in sweep_df.columns:
            fig = go.Figure()
            fig.update_layout(title="Robustness (no sweep data)", template="plotly_dark", height=320)
            return fig
        sub = sweep_df[
            (sweep_df["mad_sma_short"] == int(selected_short)) & (sweep_df["mad_sma_long"] == int(selected_long))
        ]
        if "mad_exit_ma" in sweep_df.columns:
            sub = sub[sub["mad_exit_ma"] == int(selected_exit)]
        if "mad_regime_ma" in sweep_df.columns:
            sub = sub[sub["mad_regime_ma"] == int(selected_regime)]
        if sub.empty:
            sub = sweep_df
        xlabs = sub["mad_sma_short"].astype(str) + "/" + sub["mad_sma_long"].astype(str)
        if "mad_exit_ma" in sub.columns:
            xlabs = xlabs + "/ex" + sub["mad_exit_ma"].astype(str)
        if "mad_regime_ma" in sub.columns:
            xlabs = xlabs + "/r" + sub["mad_regime_ma"].astype(str)
        fig = go.Figure(go.Bar(x=xlabs, y=sub["profit_factor"]))
        fig.update_layout(
            title="MAD parameter grid — profit factor (aggregated)",
            template="plotly_dark",
            height=320,
            yaxis_title="PF",
        )
        return fig

    app = Dash(__name__)
    app.layout = html.Div(
        style={"backgroundColor": "#111", "color": "#eee", "fontFamily": "monospace", "padding": "20px"},
        children=[
            html.H2(
                f"MAD / MRAT — {ref_ticker} {config.TARGET_CANDLE_GRANULARITY} | universe n={universe_n}",
                style={"textAlign": "center", "marginBottom": "4px"},
            ),
            html.P(
                "Cross-sectional MRAT deciles + σ thresholds | "
                f"Weighting: {weight_banner or 'scheme=equal'}",
                style={"textAlign": "center", "color": "#aaa", "marginTop": 0},
            ),
            html.Div(
                [
                    html.Label("Split:", style={"marginRight": "12px", "fontWeight": "bold"}),
                    dcc.Slider(
                        id="split-slider",
                        min=slider_min,
                        max=slider_max,
                        step=None,
                        marks=slider_marks,
                        value=first_split,
                        tooltip={"placement": "bottom"},
                    ),
                ],
                style={"padding": "20px 40px 24px"},
            ),
            html.Div(id="split-subtitle", style={"textAlign": "center", "color": "#aaa", "marginBottom": "10px"}),
            dcc.Graph(id="fig-equity"),
            dcc.Graph(id="fig-robustness"),
            html.H3(id="stats-title", style={"marginTop": "24px"}),
            dash_table.DataTable(
                id="stats-table",
                columns=[{"name": "Metric", "id": "Metric"}, {"name": "Value", "id": "Value"}],
                style_table={"maxWidth": "720px"},
                style_cell={
                    "backgroundColor": "#1a1a1a",
                    "color": "#eee",
                    "border": "1px solid #333",
                    "padding": "6px",
                },
                style_header={"backgroundColor": "#2a2a2a", "fontWeight": "bold"},
            ),
            html.H3("Robustness notes", style={"marginTop": "24px"}),
            html.Pre(
                insights,
                style={
                    "whiteSpace": "pre-wrap",
                    "lineHeight": "1.5",
                    "backgroundColor": "#1a1a1a",
                    "padding": "12px",
                    "borderRadius": "6px",
                    "border": "1px solid #333",
                },
            ),
        ],
    )

    @app.callback(
        Output("split-subtitle", "children"),
        Output("fig-equity", "figure"),
        Output("fig-robustness", "figure"),
        Output("stats-title", "children"),
        Output("stats-table", "data"),
        Input("split-slider", "value"),
    )
    def update(slider_val: int):
        key = AVG_KEY if slider_val == AVG_SLIDER_VAL else slider_val
        payload = results[key]
        eval_df = payload["eval_df"]
        metrics = payload["metrics"]
        stats_df = payload["stats_df"]
        label = str(payload["label"])
        sh = int(metrics["mad_sma_short"])
        lo = int(metrics["mad_sma_long"])
        ex = int(metrics.get("mad_exit_ma", 0))
        reg = int(metrics.get("mad_regime_ma", 0))
        rt = str(metrics.get("mad_regime_ticker", "") or "")
        rb = int(metrics.get("rebalance_days", metrics["trades"]))
        dpos = int(metrics.get("days_with_position", 0))
        ex_part = f" exit MA={ex}" if ex else " exit MA=off"
        if reg > 0 and rt:
            reg_part = f" | regime {rt} SMA={reg}"
        elif reg > 0:
            reg_part = f" | regime SMA={reg}"
        else:
            reg_part = " | regime off"
        w_part = f" | weighting {weight_banner}" if weight_banner else ""
        subtitle = (
            f"{label} | MRAT SMA {sh}/{lo} |{ex_part}{reg_part}{w_part} | "
            f"PF={float(metrics['profit_factor']):.4f} | Sharpe={float(metrics['sharpe_ratio']):.4f} | "
            f"Sortino={float(metrics['sortino_ratio']):.4f} | Rebal days={rb} | Days in mkt={dpos}"
        )
        return (
            subtitle,
            fig_equity(eval_df, label, float(metrics["profit_factor"]), strategy_curve_name="MAD portfolio"),
            _robustness_fig(sh, lo, ex, reg),
            f"Portfolio stats — {label}",
            format_stats(stats_df).to_dict("records"),
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fee-rate", type=float, default=float(getattr(config, "BACKTEST_FEE_RATE", 0.001)))
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument(
        "--single-index",
        choices=("sp500", "nasdaq100", "russell2000"),
        default=None,
        help=(
            "Temporarily run in single-universe mode for one index (disables the "
            "multi-index allocator for this run only). "
            "sp500 → (SPY, MAD_UNIVERSE=sp500), "
            "nasdaq100 → (QQQ, MAD_UNIVERSE=nasdaq100), "
            "russell2000 → (IWM, MAD_UNIVERSE=russell2000). "
            "Uses the matching ETF as regime ticker."
        ),
    )
    args = parser.parse_args()

    # --single-index: override the allocator + universe for this run only so
    # we can sanity-check each index in isolation against the old single-
    # universe path. Mutates the imported ``config`` module attributes in
    # memory (process-local); does not touch config.py on disk.
    if args.single_index is not None:
        from deepvibe_hedge.sp500 import sp500 as _sp500
        from deepvibe_hedge.nasdaq100 import nasdaq100 as _nasdaq100
        from deepvibe_hedge.russell2000 import russell2000 as _russell2000

        _SINGLE_INDEX_MAP = {
            "sp500":      ("SPY", _sp500,      "S&P 500"),
            "nasdaq100":  ("QQQ", _nasdaq100,  "NASDAQ-100"),
            "russell2000":("IWM", _russell2000,"Russell 2000"),
        }
        etf, universe_tuple, label = _SINGLE_INDEX_MAP[args.single_index]
        config.MAD_INDEX_ALLOCATOR_ENABLED = False          # turn off the top-level allocator
        config.MAD_UNIVERSE_TICKERS = tuple(universe_tuple)  # stock picker universe
        config.MAD_REGIME_TICKER = etf                       # regime gate = this index's ETF
        config.MAD_REGIME_TICKERS = (etf,)                   # multi-ticker regime collapses to this ETF
        config.MAD_REGIME_MA_ENABLED = True                  # ensure gate is on
        print(
            f"\n[--single-index] override active: {label} "
            f"(universe={len(universe_tuple)} tickers, regime={etf} vs "
            f"{int(getattr(config, 'MAD_REGIME_MA', 200))}D SMA, allocator DISABLED)\n"
        )

    ref = mad_reference_ticker()
    universe = mad_universe_tickers()
    gran = str(config.TARGET_CANDLE_GRANULARITY)
    daily_agg = bool(getattr(config, "MAD_AGGREGATE_TO_DAILY", True)) and gran.lower() != "1d"

    panel_long = build_panel_long(universe, gran, ref, OHLCV_DIR)
    if daily_agg:
        daily_long = aggregate_panel_to_daily(panel_long)
    else:
        dl = panel_long.copy()
        dl["date"] = pd.to_datetime(dl["timestamp"], utc=True).dt.normalize()
        daily_long = dl.drop(columns=["timestamp"], errors="ignore")

    is_splits, oos_splits = resolve_mad_split_plan(daily_long)
    split_by_d = daily_long.groupby("date", sort=True)["split"].last()

    n_univ = int(daily_long["ticker"].nunique())
    min_decile_names = 10
    if n_univ < min_decile_names:
        raise RuntimeError(
            f"MAD needs at least {min_decile_names} distinct tickers in MAD_UNIVERSE_TICKERS with OHLCV DBs "
            f"(decile ranks require a cross-section). Loaded n={n_univ}. "
            "Fetch and split data for more symbols, then widen MAD_UNIVERSE_TICKERS."
        )

    bpy = _bars_per_year_for_mad(gran, daily_agg or gran.lower() == "1d")
    direction = getattr(config, "MAD_DIRECTION_MODE", "both")
    min_price = float(MAD_DEFAULT_MIN_PRICE)
    min_hist = int(getattr(config, "MAD_MIN_HISTORY_BARS", 252))
    grid = _param_grid()
    exit_grid = _exit_ma_grid()
    exit_ma_enabled = bool(getattr(config, "MAD_EXIT_MA_ENABLED", True))
    regime_grid = _regime_ma_grid()
    regime_sym = mad_regime_ticker_symbol()
    eval_all = bool(getattr(config, "MAD_EVAL_ALL_SPLITS", False))
    all_research_dates = {mad_calendar_key(d) for d in split_by_d[split_by_d > 0].index}

    # Honor the user-configured eval window in the single-universe path too
    # (grid search + walkforward + final eval all flow through
    # ``all_research_dates``). We simply intersect with the window set; this
    # leaves the full-panel fast path alone when both bounds are unset.
    _bt_start_cfg, _bt_end_cfg = resolve_backtest_window()
    if _bt_start_cfg is not None or _bt_end_cfg is not None:
        window_keys = _eval_dates_from_window(
            [mad_calendar_key(d) for d in split_by_d.index],
            _bt_start_cfg,
            _bt_end_cfg,
        )
        if window_keys is not None:
            before = len(all_research_dates)
            all_research_dates = all_research_dates & window_keys
            print(
                f"  [eval window] restricted non-warmup research days: "
                f"{before:,} → {len(all_research_dates):,}  "
                f"(MAD_BACKTEST_START_DATE={getattr(config, 'MAD_BACKTEST_START_DATE', None)!r}, "
                f"MAD_BACKTEST_END_DATE={getattr(config, 'MAD_BACKTEST_END_DATE', None)!r})"
            )
            if not all_research_dates:
                raise RuntimeError(
                    "MAD_BACKTEST_START_DATE / END_DATE window excludes all "
                    "non-warmup research days. Widen the window or reduce MAD_IS/OOS splits."
                )

    if any(int(x) > 0 for x in regime_grid):
        if not regime_sym:
            raise RuntimeError(
                "MAD_REGIME_MA grid includes a positive MA but MAD_REGIME_TICKER is empty or "
                "MAD_REGIME_MA_ENABLED is False. Set MAD_REGIME_TICKER (e.g. QQQ) and enable the filter."
            )
        _load_regime_daily_close(regime_sym, gran, OHLCV_DIR, aggregate_to_daily=daily_agg)

    wcfg = resolve_weight_config(config)
    regime_lines = "\n".join(_format_regime_sleeve_banner(regime_sym, regime_grid))
    index_alloc_lines = "\n".join(format_index_allocation_banner(config))
    bt_start_cfg, bt_end_cfg = resolve_backtest_window()
    if bt_start_cfg is None and bt_end_cfg is None:
        eval_window_line = (
            "  Eval window      : full panel (no MAD_BACKTEST_START_DATE / END_DATE set)"
        )
    else:
        eval_window_line = (
            f"  Eval window      : "
            f"{bt_start_cfg.date() if bt_start_cfg is not None else '-inf'} → "
            f"{bt_end_cfg.date() if bt_end_cfg is not None else '+inf'}  "
            f"(SMAs still warm up on full data; dashboard + metrics start at window)"
        )
    print(
        f"\nMAD / MRAT panel backtest\n"
        f"  Reference ticker : {ref}\n"
        f"  Universe size    : {len(universe)} (loaded rows from DBs)\n"
        f"  Granularity      : {gran}  |  daily aggregation: {daily_agg or gran.lower() == '1d'}\n"
        f"  MAD_EVAL_ALL_SPLITS: {eval_all}\n"
        f"  In-sample splits : {is_splits}  (ignored for scoring when MAD_EVAL_ALL_SPLITS=True)\n"
        f"  Reserved OOS     : {oos_splits}\n"
        f"  MRAT grid        : {len(grid)} pair(s) {grid[:5]}{'...' if len(grid) > 5 else ''}\n"
        f"  Exit MA          : "
        f"{'disabled (MAD_EXIT_MA_ENABLED=False)' if not exit_ma_enabled else f'grid {exit_grid} (0 = off; close > SMA to hold long)'}\n"
        f"{regime_lines}\n"
        f"{eval_window_line}\n"
        f"  Weighting scheme : {_format_weight_config_banner(wcfg)}\n"
        f"  Direction mode   : {direction}\n"
        f"  Long / short decile: ≥{getattr(config, 'MAD_LONG_DECILE_MIN', 10)} / ≤{getattr(config, 'MAD_SHORT_DECILE_MAX', 1)}\n"
        f"  Symmetric short σ: {getattr(config, 'MAD_SYMMETRIC_SHORT_SIGMA', False)} "
        f"(short k = long k when True; else MAD_SHORT_SIGMA_MULT)\n"
        f"  Fee rate         : {args.fee_rate:.4%}\n"
        f"  Min history bars : {min_hist} (daily bars after aggregation; IPOs join when warm)\n"
        f"{index_alloc_lines}\n"
    )
    if eval_all:
        print(
            f"  All-splits eval    : {len(all_research_dates):,} calendar days (split > 0, excl. warmup)\n"
        )

    # ----- Multi-index allocator dispatch --------------------------------------
    # Cartesian grid over three axes (stock / index allocator / hedge sleeve),
    # each optional. Legacy bools OR into the axis flags (see config section 5).
    if index_allocator_enabled(config):
        short_w_cfg = int(getattr(config, "MAD_SMA_SHORT", 21))
        long_w_cfg = int(getattr(config, "MAD_SMA_LONG", 200))
        exit_w_cfg = int(getattr(config, "MAD_EXIT_MA_PERIOD", 0) or 0)

        _all_schemes = (
            "equal",
            "mrat_distance",
            "mrat_zscore",
            "softmax",
            "rank",
            "inv_vol",
            "mrat_distance_inv_vol",
        )

        gs = bool(getattr(config, "MAD_GRID_SEARCH_STOCK", False))
        gi = bool(getattr(config, "MAD_GRID_SEARCH_INDEX", False))
        gsv = bool(getattr(config, "MAD_GRID_SEARCH_SLEEVE", False))
        if bool(getattr(config, "MAD_STOCK_WEIGHTING_GRID_SEARCH", False)):
            gs = True
        if bool(getattr(config, "MAD_SLEEVE_WEIGHTING_GRID_SEARCH", False)):
            gsv = True
        if bool(getattr(config, "MAD_INDEX_WEIGHTING_GRID_SEARCH", False)):
            gs = True
            gi = True

        grid_any = gs or gi or gsv

        if grid_any:
            if gs:
                stock_list = tuple(
                    getattr(config, "MAD_STOCK_WEIGHTING_GRID", None)
                    or getattr(config, "MAD_WEIGHTING_GRID", None)
                    or _all_schemes
                )
            else:
                stock_list = (str(getattr(config, "MAD_WEIGHTING_SCHEME", "equal")),)

            if gi:
                index_list = tuple(
                    getattr(config, "MAD_INDEX_WEIGHTING_GRID", None) or _all_schemes
                )
            else:
                _idx = getattr(config, "MAD_INDEX_WEIGHTING_SCHEME", None)
                index_list = (_idx if _idx is not None else "equal",)

            if gsv:
                sleeve_list = tuple(
                    getattr(config, "MAD_SLEEVE_WEIGHTING_GRID", None) or _all_schemes
                )
            else:
                sleeve_list = (getattr(config, "MAD_SLEEVE_WEIGHTING_SCHEME", None),)

            combos = list(itertools.product(stock_list, index_list, sleeve_list))
            ncomb = len(combos)
            optim_metric = str(
                getattr(config, "MAD_GRID_OPTIMIZE_METRIC", None)
                or getattr(config, "MAD_INDEX_OPTIMIZE_METRIC", "sharpe_ratio")
            )

            ax_txt = (
                f"stock={'ON' if gs else 'fixed'} | "
                f"index={'ON' if gi else 'fixed'} | "
                f"sleeve={'ON' if gsv else 'fixed'}"
            )
            print(
                f"\n[Multi-index weighting grid] {ncomb} combo(s)  ({ax_txt})\n"
                f"  objective = MAX {optim_metric}\n"
                f"  Axes: {len(stock_list)} × {len(index_list)} × {len(sleeve_list)} — "
                f"full 7×7×7 = 343 when all three are ON.\n"
                f"  Per-stock scheme: first pass builds per-slot cache; later combos reuse it "
                f"(sleeve rebuilt each time)."
            )

            original_stock_scheme = getattr(config, "MAD_WEIGHTING_SCHEME", "equal")
            original_index_scheme = getattr(config, "MAD_INDEX_WEIGHTING_SCHEME", None)
            original_sleeve_scheme = getattr(config, "MAD_SLEEVE_WEIGHTING_SCHEME", None)

            combo_results: list[dict] = []
            slot_cache_by_stock: dict[str, dict] = {}

            try:
                for ci, (stock_scheme, index_scheme, sleeve_scheme) in enumerate(combos):
                    config.MAD_WEIGHTING_SCHEME = stock_scheme
                    config.MAD_INDEX_WEIGHTING_SCHEME = index_scheme
                    config.MAD_SLEEVE_WEIGHTING_SCHEME = sleeve_scheme
                    sw_try = resolve_sleeve_weight_config(config)
                    if gsv and sw_try is None:
                        print(
                            f"\n  [{ci + 1}/{ncomb}] skip sleeve={sleeve_scheme!r}: "
                            f"fixed-weight mode (not in dynamic grid)"
                        )
                        continue

                    sc = slot_cache_by_stock.get(str(stock_scheme))
                    is_first_for_stock = sc is None
                    mode = (
                        "(full per-slot)"
                        if is_first_for_stock
                        else "(cached per-slot + fresh sleeve/allocator)"
                    )
                    print(
                        f"\n  [{ci + 1}/{ncomb}] stock={stock_scheme} | index={index_scheme} | "
                        f"sleeve={sleeve_scheme}  {mode}"
                    )
                    m_try, df_try = evaluate_mad_multi_index(
                        daily_long,
                        short_w=short_w_cfg,
                        long_w=long_w_cfg,
                        min_price=min_price,
                        min_history=min_hist,
                        fee_rate=float(args.fee_rate),
                        direction_mode=direction,
                        eval_dates=None,
                        bars_per_year_local=bpy,
                        exit_ma_period=exit_w_cfg,
                        granularity=gran,
                        aggregate_to_daily=daily_agg,
                        _per_slot_cache=sc,
                        _return_per_slot_cache=is_first_for_stock,
                        # Never reuse stale sleeve rows from cache when (index|sleeve) vary.
                        reuse_cached_sleeve=(sc is None),
                        sleeve_weight_cfg=sw_try,
                        _verbose=is_first_for_stock,
                    )
                    if is_first_for_stock:
                        popped = m_try.pop("_per_slot_cache", None)
                        if popped is not None:
                            slot_cache_by_stock[str(stock_scheme)] = popped

                    combo_results.append(
                        {
                            "stock_scheme": stock_scheme,
                            "index_scheme": index_scheme,
                            "sleeve_scheme": sleeve_scheme,
                            "metrics": m_try,
                            "eval_df": df_try,
                        }
                    )
                    print(
                        f"    → return={m_try['net_total_return']:+7.2%}  "
                        f"sharpe={m_try['sharpe_ratio']:6.2f}  "
                        f"sortino={m_try['sortino_ratio']:6.2f}  "
                        f"pf={m_try['profit_factor']:5.2f}"
                    )
            finally:
                config.MAD_WEIGHTING_SCHEME = original_stock_scheme
                config.MAD_INDEX_WEIGHTING_SCHEME = original_index_scheme
                config.MAD_SLEEVE_WEIGHTING_SCHEME = original_sleeve_scheme

            if not combo_results:
                raise RuntimeError(
                    "Weighting grid: no valid combos (e.g. every sleeve candidate "
                    "resolved to fixed-weight None while sleeve axis was ON)."
                )

            combo_results.sort(
                key=lambda r: (
                    float(r["metrics"].get(optim_metric, float("-inf")))
                    if np.isfinite(r["metrics"].get(optim_metric, float("-inf")))
                    else float("-inf")
                ),
                reverse=True,
            )

            print(f"\n{'=' * 120}")
            print(f"  WEIGHTING GRID LEADERBOARD (sorted by MAX {optim_metric})")
            print(f"{'=' * 120}")
            print(
                f"  {'rank':>4} {'stock':<22} {'index':<22} {'sleeve':<22} "
                f"{'return':>10} {'sharpe':>7} {'sortino':>8} {'PF':>5}"
            )
            for r_i, r in enumerate(combo_results, 1):
                m = r["metrics"]
                marker = "  ★" if r_i == 1 else "   "
                print(
                    f"  {r_i:>4} {str(r['stock_scheme']):<22} {str(r['index_scheme']):<22} "
                    f"{str(r['sleeve_scheme']):<22} "
                    f"{m['net_total_return']:>+9.2%} {m['sharpe_ratio']:>7.2f} "
                    f"{m['sortino_ratio']:>8.2f} {m['profit_factor']:>5.2f}{marker}"
                )
            print(f"{'=' * 120}")

            winner = combo_results[0]
            config.MAD_WEIGHTING_SCHEME = winner["stock_scheme"]
            config.MAD_INDEX_WEIGHTING_SCHEME = winner["index_scheme"]
            config.MAD_SLEEVE_WEIGHTING_SCHEME = winner["sleeve_scheme"]
            metrics_mi = winner["metrics"]
            eval_df_mi = winner["eval_df"]
            print(
                f"\n  Winner (MAX {optim_metric}): "
                f"stock={winner['stock_scheme']!r} × index={winner['index_scheme']!r} × "
                f"sleeve={winner['sleeve_scheme']!r} "
                f"→ return={metrics_mi['net_total_return']:+.2%}, "
                f"sharpe={metrics_mi['sharpe_ratio']:.2f}"
            )
            print(
                "  Dashboard uses the winning triple. Persist in config.py (section 3); "
                "set MAD_GRID_SEARCH_STOCK / INDEX / SLEEVE all False to skip."
            )
        else:
            print("\n[Multi-index allocator] single-config run (grid search disabled).")
            metrics_mi, eval_df_mi = evaluate_mad_multi_index(
                daily_long,
                short_w=short_w_cfg,
                long_w=long_w_cfg,
                min_price=min_price,
                min_history=min_hist,
                fee_rate=float(args.fee_rate),
                direction_mode=direction,
                eval_dates=None,
                bars_per_year_local=bpy,
                exit_ma_period=exit_w_cfg,
                granularity=gran,
                aggregate_to_daily=daily_agg,
            )
        print("\nMulti-index backtest summary:")
        print(f"  Net total return  : {metrics_mi['net_total_return']:+.2%}")
        print(f"  Sharpe (annual)   : {metrics_mi['sharpe_ratio']:.2f}")
        print(f"  Sortino (annual)  : {metrics_mi['sortino_ratio']:.2f}")
        print(f"  Profit factor     : {metrics_mi['profit_factor']:.2f}")
        print(f"  Bars (valid)      : {metrics_mi['bars']}")
        print("  Per-slot contribution (standalone, no blend):")
        _enabled_raw = getattr(config, "MAD_INDEX_ENABLED_ETFS", None)
        _enabled_set = (
            {str(t).strip().upper() for t in _enabled_raw if str(t).strip()}
            if _enabled_raw
            else None
        )
        for etf, _u, label in getattr(config, "MAD_INDEX_SLOTS", ()):
            if _enabled_set is not None and etf.upper() not in _enabled_set:
                continue  # slot disabled via MAD_INDEX_ENABLED_ETFS whitelist
            ret_k = f"slot_{etf}_net_total_return"
            shp_k = f"slot_{etf}_sharpe"
            bars_k = f"slot_{etf}_bars"
            status_k = f"slot_{etf}_status"
            status = metrics_mi.get(status_k, None)
            if ret_k in metrics_mi and status == "ok":
                print(
                    f"    {etf:4s} ({label:15s}): "
                    f"return={metrics_mi[ret_k]:+.2%}  sharpe={metrics_mi[shp_k]:.2f}  "
                    f"bars={int(metrics_mi.get(bars_k, 0))}"
                )
            elif status == "no_trades":
                print(
                    f"    {etf:4s} ({label:15s}): NO TRADES "
                    f"(bars={int(metrics_mi.get(bars_k, 0))}, "
                    f"return={metrics_mi.get(ret_k, 0.0):+.2%})"
                )
            elif status == "empty_universe":
                print(
                    f"    {etf:4s} ({label:15s}): SKIPPED "
                    f"(exclusive universe empty after priority assignment)"
                )
            elif status == "empty_after_filter":
                print(
                    f"    {etf:4s} ({label:15s}): SKIPPED "
                    f"(0 tickers survived MAD_MIN_DATA_COMPLETENESS filter)"
                )
            else:
                # Defensive: unknown status — print whatever we have.
                ret_val = metrics_mi.get(ret_k)
                if ret_val is not None:
                    print(
                        f"    {etf:4s} ({label:15s}): "
                        f"return={ret_val:+.2%}  sharpe={metrics_mi.get(shp_k, float('nan')):.2f}"
                    )
                else:
                    print(f"    {etf:4s} ({label:15s}): (no metrics recorded)")

        # Diagnostic: average realized top-level allocator weights across the
        # full eval window. This is what the user actually sees "on average"
        # — e.g. inv_vol with SPY+QQQ+IWM typically lands near 40/40/5 + risk-off.
        try:
            alloc_cols = [
                c for c in eval_df_mi.columns
                if c.startswith("alloc_")
            ]
            if alloc_cols:
                avg = eval_df_mi[alloc_cols].mean(axis=0).sort_values(ascending=False)
                print("\n  Average allocator weights over eval window:")
                for col, v in avg.items():
                    name = col.replace("alloc_", "")
                    display = "risk-off sleeve" if name == "__risk_off__" else name
                    print(f"    {display:18s}: {float(v):6.2%}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [avg-weights diagnostic skipped: {exc}]")

        # Diagnostic: year-by-year decomposition. Reveals whether a flat stretch
        # on the equity curve is driven by (a) risk-off dominance, (b) softmax
        # whipsaw, or (c) a single leg (IWM) dragging. For each calendar year
        # we show: strategy return, B&H return, avg allocator mix, # rebalances.
        try:
            df_yr = eval_df_mi.copy()
            df_yr["year"] = df_yr.index.year
            alloc_cols_yr = [c for c in df_yr.columns if c.startswith("alloc_")]
            rows: list[str] = []
            header = (
                f"  {'Year':<6}{'Strat':>9}{'B&H':>9}{'ΔvsBH':>9}  "
                + "".join(f"{c.replace('alloc_', '').replace('__risk_off__', 'RiskOff'):>9}"
                          for c in alloc_cols_yr)
                + f"  {'Rebal':>6}  {'Bars':>5}"
            )
            rows.append(header)
            rows.append(f"  {'-' * (len(header) - 2)}")
            for yr, g in df_yr.groupby("year"):
                # Strategy realized log returns (NaN => excluded by valid mask).
                s_log = g["net_log_return"].dropna()
                bh_log = g["next_log_return"].dropna()
                s_ret = float(np.expm1(s_log.sum())) if not s_log.empty else 0.0
                bh_ret = float(np.expm1(bh_log.sum())) if not bh_log.empty else 0.0
                rebals = int(g["flip"].fillna(False).sum())
                alloc_avgs = [float(g[c].mean()) for c in alloc_cols_yr]
                rows.append(
                    f"  {int(yr):<6}{s_ret:>+8.2%} {bh_ret:>+8.2%} "
                    f"{(s_ret - bh_ret):>+8.2%}  "
                    + "".join(f"{v:>8.1%}" for v in alloc_avgs)
                    + f"  {rebals:>6d}  {len(g):>5d}"
                )
            print("\n  Year-by-year decomposition:")
            print("\n".join(rows))
            # Quick diagnosis heuristic
            flat_years = []
            for yr, g in df_yr.groupby("year"):
                s_log = g["net_log_return"].dropna()
                if s_log.empty:
                    continue
                s_ret = float(np.expm1(s_log.sum()))
                ro_col = next((c for c in alloc_cols_yr if c == "alloc___risk_off__"), None)
                ro_avg = float(g[ro_col].mean()) if ro_col else 0.0
                if abs(s_ret) < 0.10 and ro_avg < 0.5:
                    flat_years.append(
                        f"    {int(yr)}: {s_ret:+.2%} (risk-off avg only {ro_avg:.1%}; "
                        f"stock-picker whipsaw or softmax concentration miss)"
                    )
                elif abs(s_ret) < 0.10 and ro_avg >= 0.5:
                    flat_years.append(
                        f"    {int(yr)}: {s_ret:+.2%} (risk-off avg {ro_avg:.1%}; "
                        f"allocator parked in cash/bonds)"
                    )
            if flat_years:
                print("\n  Flat-year diagnosis (|return| < 10%):")
                print("\n".join(flat_years))
        except Exception as exc:  # noqa: BLE001
            print(f"  [year-by-year decomposition skipped: {exc}]")

        # Diagnostic: per-year, per-ETF count of days each index failed its
        # 200D trend filter. This is the DIRECT cause of risk-off rotation —
        # if SPY/QQQ/IWM each fail ~N days in a given year, the allocator
        # parks (N_failing / N_indexes) of the book in the risk-off sleeve
        # on average. Useful for answering "why was 2021 39% risk-off when
        # SPY/QQQ were up 28%?" → usually because IWM broke trend in Q4.
        try:
            trend_cols = [c for c in eval_df_mi.columns if c.startswith("trend_ok_")]
            if trend_cols:
                df_tr = eval_df_mi.copy()
                df_tr["year"] = df_tr.index.year
                hdr = (
                    f"  {'Year':<6}{'Bars':>6}  "
                    + "".join(
                        f"{c.replace('trend_ok_', '') + ' fail%':>14}"
                        for c in trend_cols
                    )
                )
                lines: list[str] = [hdr, f"  {'-' * (len(hdr) - 2)}"]
                for yr, g in df_tr.groupby("year"):
                    n_bars = len(g)
                    parts = []
                    for c in trend_cols:
                        fails = int((~g[c].astype(bool)).sum())
                        pct = fails / n_bars if n_bars > 0 else 0.0
                        parts.append(f"{fails:>4d} ({pct:>5.1%})")
                    lines.append(
                        f"  {int(yr):<6}{n_bars:>6d}  "
                        + "".join(f"{p:>14}" for p in parts)
                    )
                print("\n  Trend-fail days per ETF per year "
                      "(days below 200D SMA → drives risk-off rotation):")
                print("\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            print(f"  [trend-fail diagnostic skipped: {exc}]")

        # Render the multi-index eval_df on the shared dashboard using
        # ``combined_only=True`` (single "All non-warmup" view). We reuse the
        # standard equity-curve + stats components; the sleeve / per-slot
        # hover details are already embedded in ``eval_df_mi`` columns.
        if not args.no_dashboard:
            try:
                mi_stats = comparison_stats_df(metrics_mi, eval_df_mi, bpy)
            except Exception as exc:  # noqa: BLE001
                print(f"[Multi-index] comparison_stats_df failed ({exc}); continuing with empty stats.")
                mi_stats = pd.DataFrame()

            mi_insight_lines = [
                "Multi-index allocator (single-config run, grid search disabled).",
                f"  MRAT {short_w_cfg}/{long_w_cfg}, regime MA {int(resolve_index_regime_ma(config) or 200)}, "
                f"weighting scheme={getattr(config, 'MAD_INDEX_WEIGHTING_SCHEME', 'equal')!r}.",
                "",
                "Per-slot standalone contribution (no blend):",
            ]
            for etf, _u, label in getattr(config, "MAD_INDEX_SLOTS", ()):
                if _enabled_set is not None and etf.upper() not in _enabled_set:
                    continue  # slot disabled via MAD_INDEX_ENABLED_ETFS whitelist
                ret_k = f"slot_{etf}_net_total_return"
                shp_k = f"slot_{etf}_sharpe"
                status = metrics_mi.get(f"slot_{etf}_status", "unknown")
                if ret_k in metrics_mi and status == "ok":
                    mi_insight_lines.append(
                        f"  {etf} ({label}): return={metrics_mi[ret_k]:+.2%}  "
                        f"sharpe={metrics_mi.get(shp_k, float('nan')):.2f}"
                    )
                else:
                    mi_insight_lines.append(f"  {etf} ({label}): status={status}")
            mi_insights = "\n".join(mi_insight_lines)

            mi_results = {
                AVG_KEY: {
                    "eval_df": eval_df_mi,
                    "metrics": metrics_mi,
                    "stats_df": mi_stats,
                    "label": "Multi-index blend (SPY+QQQ+IWM → risk-off)",
                }
            }
            print(f"\nDashboard → http://127.0.0.1:{args.port}\n")
            build_app(
                mi_results,
                pd.DataFrame(),  # no sweep_df for single-config run
                mi_insights,
                ref,
                len(universe),
                combined_only=True,
                weight_banner=_format_weight_config_banner(wcfg),
            ).run(debug=False, port=args.port)
        return
    # ----- End multi-index allocator dispatch ----------------------------------

    if eval_all:
        context = daily_long.copy()
        split_sweep_local: list[dict[str, float | int]] = []
        best_pf = float("-inf")
        best_quad = (grid[0][0], grid[0][1], exit_grid[0], regime_grid[0])

        print("\n[All non-warmup splits] grid search (MRAT × exit MA × regime MA)")

        for sh_g, lo_g in grid:
            for ex_g in exit_grid:
                for reg_g in regime_grid:
                    metrics_try, _ = evaluate_mad(
                        context,
                        short_w=sh_g,
                        long_w=lo_g,
                        min_price=min_price,
                        min_history=min_hist,
                        fee_rate=float(args.fee_rate),
                        direction_mode=direction,
                        eval_dates=all_research_dates,
                        bars_per_year_local=bpy,
                        exit_ma_period=int(ex_g),
                        regime_ma_period=int(reg_g),
                        regime_ticker=regime_sym,
                        granularity=gran,
                        aggregate_to_daily=daily_agg,
                    )
                    pf = float(metrics_try["profit_factor"])
                    split_sweep_local.append(
                        {
                            "mad_sma_short": sh_g,
                            "mad_sma_long": lo_g,
                            "mad_exit_ma": int(ex_g),
                            "mad_regime_ma": int(reg_g),
                            "profit_factor": pf,
                            "trades": int(metrics_try["trades"]),
                            "net_total_return": float(metrics_try["net_total_return"]),
                            "split": -1,
                        }
                    )
                    if np.isfinite(pf) and pf > best_pf:
                        best_pf = pf
                        best_quad = (sh_g, lo_g, ex_g, reg_g)

        sh, lo, ex, reg = best_quad
        metrics, eval_ctx = evaluate_mad(
            context,
            short_w=sh,
            long_w=lo,
            min_price=min_price,
            min_history=min_hist,
            fee_rate=float(args.fee_rate),
            direction_mode=direction,
            eval_dates=all_research_dates,
            bars_per_year_local=bpy,
            exit_ma_period=int(ex),
            regime_ma_period=int(reg),
            regime_ticker=regime_sym,
            granularity=gran,
            aggregate_to_daily=daily_agg,
        )
        combined_eval = eval_ctx.loc[
            [i for i in eval_ctx.index if mad_calendar_key(i) in all_research_dates]
        ].copy()
        split_metrics = pd.DataFrame([{"split": -1, "fee_rate": float(args.fee_rate), **metrics}])
        split_sweep_df = pd.DataFrame(split_sweep_local)
        sweep_df = (
            split_sweep_df.groupby(
                ["mad_sma_short", "mad_sma_long", "mad_exit_ma", "mad_regime_ma"], as_index=False
            )
            .agg(
                profit_factor=("profit_factor", "mean"),
                trades=("trades", "mean"),
                net_total_return=("net_total_return", "mean"),
            )
            .sort_values(["mad_sma_short", "mad_sma_long", "mad_exit_ma", "mad_regime_ma"])
        )
        if len(sweep_df) == 1 and not np.isfinite(float(sweep_df.iloc[0]["profit_factor"])):
            sweep_df = pd.DataFrame(
                [
                    {
                        "mad_sma_short": sh,
                        "mad_sma_long": lo,
                        "mad_exit_ma": int(ex),
                        "mad_regime_ma": int(reg),
                        "profit_factor": float(metrics["profit_factor"]),
                        "trades": float(metrics["trades"]),
                        "net_total_return": float(metrics["net_total_return"]),
                    }
                ]
            )

        combined_metrics = metrics
        insights = _mad_robustness_insights(sweep_df)
        non_warmup_ids = sorted({int(x) for x in split_by_d.unique().tolist() if int(x) > 0})
        summary_df = pd.DataFrame(
            [
                {
                    "reference_ticker": ref,
                    "granularity": gran,
                    "universe": ",".join(universe),
                    "is_splits": "all_non_warmup",
                    "oos_splits": ",".join(str(s) for s in oos_splits),
                    "mad_eval_all_splits": True,
                    "non_warmup_split_ids": ",".join(str(s) for s in non_warmup_ids),
                    "mad_sma_short": int(sh),
                    "mad_sma_long": int(lo),
                    "mad_exit_ma": int(ex),
                    "mad_regime_ma": int(reg),
                    "mad_regime_ticker": str(regime_sym or ""),
                    "fee_rate": float(args.fee_rate),
                    "profit_factor": float(combined_metrics["profit_factor"]),
                    "sharpe_ratio": float(combined_metrics["sharpe_ratio"]),
                    "sortino_ratio": float(combined_metrics["sortino_ratio"]),
                    "trades": int(combined_metrics["trades"]),
                    "rebalance_days": int(combined_metrics.get("rebalance_days", combined_metrics["trades"])),
                    "days_with_position": int(combined_metrics.get("days_with_position", 0)),
                    "bars": int(combined_metrics["bars"]),
                    "net_total_return": float(combined_metrics["net_total_return"]),
                    "robustness_insights": insights,
                }
            ]
        )
        _save_outputs(split_metrics, sweep_df, split_sweep_df, summary_df, ref)

        results_all: dict[int, dict[str, object]] = {
            AVG_KEY: {
                "eval_df": combined_eval,
                "metrics": combined_metrics,
                "stats_df": comparison_stats_df(combined_metrics, combined_eval, bpy),
                "label": "All non-warmup splits",
            }
        }

        print(
            "\nFull-sample (split > 0) metrics:\n"
            f"  MRAT SMA            : {sh}/{lo} | exit MA : {ex if ex else 'off'} | "
            f"regime : {regime_sym or 'off'} SMA={reg if reg else 'off'}\n"
            f"  Profit Factor       : {combined_metrics['profit_factor']:.4f}\n"
            f"  Sharpe              : {combined_metrics['sharpe_ratio']:.4f}\n"
            f"  Bars scored         : {combined_metrics['bars']}\n"
            f"  Rebalance days      : {combined_metrics.get('rebalance_days', combined_metrics['trades'])} "
            f"(days |w| changed)\n"
            f"  Days with position  : {combined_metrics.get('days_with_position', 0)} "
            f"(non-flat book on scored days)\n"
        )
        _print_mad_cross_section_diag(combined_metrics)

        if not args.no_dashboard:
            print(f"\nDashboard → http://127.0.0.1:{args.port}\n")
            build_app(
                results_all,
                sweep_df,
                insights,
                ref,
                len(universe),
                combined_only=True,
                weight_banner=_format_weight_config_banner(wcfg),
            ).run(debug=False, port=args.port)
        return

    results: dict[int, dict[str, object]] = {}
    split_rows: list[dict[str, float | int]] = []
    split_sweep_rows: list[pd.DataFrame] = []

    total_is = len(is_splits)
    for split_idx, split_id in enumerate(is_splits, start=1):
        dates_this = split_by_d[split_by_d == split_id].index
        d_sub = daily_long[daily_long["date"].isin(dates_this)].copy()
        end_d = max(dates_this) if len(dates_this) else daily_long["date"].max()
        context = daily_long[daily_long["date"] <= end_d].copy()

        eval_dset = {mad_calendar_key(d) for d in dates_this}

        print(f"\n[Split {split_idx}/{total_is}] id={split_id} | grid search")

        best_pf = float("-inf")
        best_quad = (grid[0][0], grid[0][1], exit_grid[0], regime_grid[0])
        split_sweep_local: list[dict[str, float | int]] = []

        for sh, lo in grid:
            for ex in exit_grid:
                for reg_g in regime_grid:
                    metrics, _ = evaluate_mad(
                        context,
                        short_w=sh,
                        long_w=lo,
                        min_price=min_price,
                        min_history=min_hist,
                        fee_rate=float(args.fee_rate),
                        direction_mode=direction,
                        eval_dates=eval_dset,
                        bars_per_year_local=bpy,
                        exit_ma_period=int(ex),
                        regime_ma_period=int(reg_g),
                        regime_ticker=regime_sym,
                        granularity=gran,
                        aggregate_to_daily=daily_agg,
                    )
                    pf = float(metrics["profit_factor"])
                    split_sweep_local.append(
                        {
                            "mad_sma_short": sh,
                            "mad_sma_long": lo,
                            "mad_exit_ma": int(ex),
                            "mad_regime_ma": int(reg_g),
                            "profit_factor": pf,
                            "trades": int(metrics["trades"]),
                            "net_total_return": float(metrics["net_total_return"]),
                            "split": int(split_id),
                        }
                    )
                    if np.isfinite(pf) and pf > best_pf:
                        best_pf = pf
                        best_quad = (sh, lo, ex, reg_g)

        sh, lo, ex, reg = best_quad
        metrics, eval_ctx = evaluate_mad(
            context,
            short_w=sh,
            long_w=lo,
            min_price=min_price,
            min_history=min_hist,
            fee_rate=float(args.fee_rate),
            direction_mode=direction,
            eval_dates=eval_dset,
            bars_per_year_local=bpy,
            exit_ma_period=int(ex),
            regime_ma_period=int(reg),
            regime_ticker=regime_sym,
            granularity=gran,
            aggregate_to_daily=daily_agg,
        )
        d_keys = {mad_calendar_key(d) for d in d_sub["date"].unique()}
        eval_split = eval_ctx.loc[[i for i in eval_ctx.index if mad_calendar_key(i) in d_keys]].copy()
        split_rows.append({"split": int(split_id), "fee_rate": float(args.fee_rate), **metrics})
        split_sweep_rows.append(pd.DataFrame(split_sweep_local))
        results[int(split_id)] = {
            "eval_df": eval_split,
            "metrics": metrics,
            "stats_df": comparison_stats_df(metrics, eval_split, bpy),
            "label": f"Split {split_id}",
        }
        print(
            f"  Split {split_id}: best MRAT {sh}/{lo} exit_MA={ex} regime_MA={reg} | "
            f"PF={metrics['profit_factor']:.4f} | Sharpe={metrics['sharpe_ratio']:.4f}"
        )

    split_metrics = pd.DataFrame(split_rows)
    split_sweep_df = pd.concat(split_sweep_rows, ignore_index=True) if split_sweep_rows else pd.DataFrame()

    combined_dates = daily_long.loc[daily_long["split"].isin(is_splits), "date"].unique()
    eval_dset_combined = {mad_calendar_key(d) for d in combined_dates}
    context_all = daily_long.copy()

    if not split_sweep_df.empty:
        agg = (
            split_sweep_df.groupby(
                ["mad_sma_short", "mad_sma_long", "mad_exit_ma", "mad_regime_ma"], as_index=False
            )
            .agg(
                profit_factor=("profit_factor", "mean"),
                trades=("trades", "mean"),
                net_total_return=("net_total_return", "mean"),
            )
            .sort_values(["mad_sma_short", "mad_sma_long", "mad_exit_ma", "mad_regime_ma"])
        )
        pf_col = agg["profit_factor"].replace([-np.inf, np.inf], np.nan)
        if pf_col.notna().any():
            best_row = agg.loc[int(pf_col.idxmax())]
        else:
            best_row = agg.iloc[0]
        best_sh = int(best_row["mad_sma_short"])
        best_lo = int(best_row["mad_sma_long"])
        best_ex = int(best_row.get("mad_exit_ma", 0))
        best_reg = int(best_row.get("mad_regime_ma", 0))
        sweep_df = agg.copy()
    else:
        best_sh, best_lo = grid[0]
        best_ex = int(exit_grid[0])
        best_reg = int(regime_grid[0])
        sweep_df = pd.DataFrame(
            [
                {
                    "mad_sma_short": best_sh,
                    "mad_sma_long": best_lo,
                    "mad_exit_ma": best_ex,
                    "mad_regime_ma": best_reg,
                    "profit_factor": np.nan,
                    "trades": np.nan,
                    "net_total_return": np.nan,
                }
            ]
        )

    combined_metrics, combined_eval_ctx = evaluate_mad(
        context_all,
        short_w=best_sh,
        long_w=best_lo,
        min_price=min_price,
        min_history=min_hist,
        fee_rate=float(args.fee_rate),
        direction_mode=direction,
        eval_dates=eval_dset_combined,
        bars_per_year_local=bpy,
        exit_ma_period=int(best_ex),
        regime_ma_period=int(best_reg),
        regime_ticker=regime_sym,
        granularity=gran,
        aggregate_to_daily=daily_agg,
    )
    want_c = {mad_calendar_key(d) for d in combined_dates}
    combined_eval = combined_eval_ctx.loc[
        [i for i in combined_eval_ctx.index if mad_calendar_key(i) in want_c]
    ].copy()

    if len(sweep_df) == 1 and not np.isfinite(float(sweep_df.iloc[0]["profit_factor"])):
        sweep_df = pd.DataFrame(
            [
                {
                    "mad_sma_short": best_sh,
                    "mad_sma_long": best_lo,
                    "mad_exit_ma": int(best_ex),
                    "mad_regime_ma": int(best_reg),
                    "profit_factor": float(combined_metrics["profit_factor"]),
                    "trades": float(combined_metrics["trades"]),
                    "net_total_return": float(combined_metrics["net_total_return"]),
                }
            ]
        )

    insights = _mad_robustness_insights(sweep_df)
    summary_df = pd.DataFrame(
        [
            {
                "reference_ticker": ref,
                "granularity": gran,
                "universe": ",".join(universe),
                "is_splits": ",".join(str(s) for s in is_splits),
                "oos_splits": ",".join(str(s) for s in oos_splits),
                "mad_eval_all_splits": False,
                "non_warmup_split_ids": "",
                "mad_sma_short": int(best_sh),
                "mad_sma_long": int(best_lo),
                "mad_exit_ma": int(best_ex),
                "mad_regime_ma": int(best_reg),
                "mad_regime_ticker": str(regime_sym or ""),
                "fee_rate": float(args.fee_rate),
                "profit_factor": float(combined_metrics["profit_factor"]),
                "sharpe_ratio": float(combined_metrics["sharpe_ratio"]),
                "sortino_ratio": float(combined_metrics["sortino_ratio"]),
                "trades": int(combined_metrics["trades"]),
                "rebalance_days": int(combined_metrics.get("rebalance_days", combined_metrics["trades"])),
                "days_with_position": int(combined_metrics.get("days_with_position", 0)),
                "bars": int(combined_metrics["bars"]),
                "net_total_return": float(combined_metrics["net_total_return"]),
                "robustness_insights": insights,
            }
        ]
    )
    _save_outputs(split_metrics, sweep_df, split_sweep_df, summary_df, ref)

    results[AVG_KEY] = {
        "eval_df": combined_eval,
        "metrics": combined_metrics,
        "stats_df": comparison_stats_df(combined_metrics, combined_eval, bpy),
        "label": "Average (combined IS)",
    }

    print(
        "Combined in-sample metrics:\n"
        f"  Selected MRAT SMA : {best_sh}/{best_lo} | exit MA : {best_ex if best_ex else 'off'} | "
        f"regime : {regime_sym or 'off'} SMA={best_reg if best_reg else 'off'}\n"
        f"  Profit Factor     : {combined_metrics['profit_factor']:.4f}\n"
        f"  Sharpe            : {combined_metrics['sharpe_ratio']:.4f}\n"
    )
    _print_mad_cross_section_diag(combined_metrics)

    if not args.no_dashboard:
        print(f"\nDashboard → http://127.0.0.1:{args.port}\n")
        build_app(
            results,
            sweep_df,
            insights,
            ref,
            len(universe),
            combined_only=False,
            weight_banner=_format_weight_config_banner(wcfg),
        ).run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
