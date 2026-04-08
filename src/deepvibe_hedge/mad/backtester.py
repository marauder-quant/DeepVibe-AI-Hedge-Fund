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
from dataclasses import dataclass
from pathlib import Path

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
from deepvibe_hedge.walkforward_oos_common import read_explicit_split_plan_from_config, resolve_split_plan

# Panel bar index and ``split`` ids: all universe names reindexed onto this symbol (``build_panel_long``).
# Fixed so MRAT stays on an equity calendar even when ``TARGET_TICKER`` is a sleeve (e.g. BIL).
MAD_PANEL_REFERENCE_TICKER = "QQQ"

# Cross-sectional MRAT eligibility (not user config).
MAD_DEFAULT_MIN_PRICE = 5.0
MAD_DEFAULT_MIN_NAMES_PER_DATE = 30
MAD_DEFAULT_MIN_NAMES_UNIVERSE_SLACK = 5
MAD_DEFAULT_MIN_NAMES_ABS_FLOOR = 10

RESULTS_DB = MAD_DATA_DIR / f"{MAD_PANEL_REFERENCE_TICKER}_{config.TARGET_CANDLE_GRANULARITY}_mad_optim.db"
DATASETS_DIR = MAD_DATA_DIR
PORT = int(getattr(config, "MAD_DASHBOARD_PORT", 8063))


def mad_reference_ticker() -> str:
    return str(MAD_PANEL_REFERENCE_TICKER).strip().upper()


def mad_universe_tickers() -> tuple[str, ...]:
    raw = getattr(config, "MAD_UNIVERSE_TICKERS", (config.TARGET_TICKER,))
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


def _regime_entry_allow_series(close: pd.Series, ma_period: int) -> pd.Series:
    """Same bar timing as MRAT: risk-on today uses prior close vs SMA (shift(1))."""
    if int(ma_period) <= 0:
        return pd.Series(True, index=close.index, dtype=bool)
    sma = _sma(close, int(ma_period))
    above = close > sma
    return above.shift(1).fillna(False).astype(bool)


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


def effective_min_names_per_date(daily_long: pd.DataFrame, configured: int) -> int:
    """
    Cross-sectional filter cannot require more names than exist in the universe.
    If configured min is 30 but only 12 tickers load, use 12 or every day fails the filter.
    When the cap hits n exactly, optional slack avoids blanking almost every day when one or two
    symbols lack a bar (see ``MAD_DEFAULT_MIN_NAMES_UNIVERSE_SLACK``).
    """
    n = int(daily_long["ticker"].nunique())
    raw = min(int(configured), n)
    slack = int(MAD_DEFAULT_MIN_NAMES_UNIVERSE_SLACK)
    floor = int(MAD_DEFAULT_MIN_NAMES_ABS_FLOOR)
    if slack > 0 and raw == n and n > 1:
        raw = max(floor, n - slack)
    return raw


def compute_mrat_panel(
    daily_long: pd.DataFrame,
    *,
    short_w: int,
    long_w: int,
    min_price: float,
    min_history: int,
    min_names: int,
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
    n_per = work.groupby("date")["ticker"].transform("count")
    work.loc[n_per < int(min_names), ["sigma", "decile"]] = np.nan

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
    """Last completed calendar row MRAT targets for the next session (matches backtest signal → next-day hold)."""

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
    odir = ohlcv_dir or OHLCV_DIR
    ref = mad_reference_ticker()
    universe = mad_universe_tickers()
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

    min_names_cfg = int(MAD_DEFAULT_MIN_NAMES_PER_DATE)
    min_names = effective_min_names_per_date(daily_long, min_names_cfg)
    panel = compute_mrat_panel(
        daily_long,
        short_w=int(short_w),
        long_w=int(long_w),
        min_price=float(MAD_DEFAULT_MIN_PRICE),
        min_history=int(getattr(config, "MAD_MIN_HISTORY_BARS", 252)),
        min_names=min_names,
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

    rt = regime_ticker
    if rt is None:
        rt = mad_regime_ticker_symbol()
    regime_ok = _regime_risk_on_for_next_session(
        int(regime_ma_period or 0),
        rt,
        gran,
        daily_agg,
        odir,
        prefer_precomputed_sma=use_pc,
    )
    if regime_ok:
        w = _weights_from_entries(sig_series)
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
    )


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
) -> pd.DataFrame:
    """
    Build date-level gross/net log returns, BH equal-weight universe, turnover fees.
    Expects columns: date, ticker, entry_signal, daily_ret.

    regime_allow: optional Series bool indexed by UTC calendar date (mad_calendar_key).
    False => zero MRAT weights that day (full cash); True => use MRAT weights.
    """
    dates = sorted(df["date"].dropna().unique())
    if len(dates) < 2:
        return pd.DataFrame()

    entry_piv = df.pivot_table(index="date", columns="ticker", values="entry_signal", aggfunc="last")
    ret_piv = df.pivot_table(index="date", columns="ticker", values="daily_ret", aggfunc="last")
    entry_piv = entry_piv.reindex(dates)
    ret_piv = ret_piv.reindex(dates)

    ck_index = pd.DatetimeIndex([mad_calendar_key(d) for d in dates], tz="UTC")
    if regime_allow is not None:
        allow_arr = regime_allow.reindex(ck_index, fill_value=False).to_numpy(dtype=bool)
    else:
        allow_arr = np.ones(len(dates), dtype=bool)

    gross_list: list[float] = []
    net_log_list: list[float] = []
    bh_log_list: list[float] = []
    flip_list: list[int] = []
    abs_w_list: list[float] = []
    w_prev = pd.Series(0.0, index=ret_piv.columns, dtype=float)

    for j, d in enumerate(dates):
        er = entry_piv.loc[d]
        w = _weights_from_entries(er)
        if not allow_arr[j]:
            w = pd.Series(0.0, index=w.index, dtype=float)
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
    min_names: int,
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
        min_names=min_names,
        direction_mode=direction_mode,
        exit_ma_period=exit_ma_period,
    )
    gran = str(granularity or config.TARGET_CANDLE_GRANULARITY)
    if aggregate_to_daily is None:
        daily_agg = bool(getattr(config, "MAD_AGGREGATE_TO_DAILY", True)) and gran.lower() != "1d"
    else:
        daily_agg = bool(aggregate_to_daily)
    regime_allow = _build_regime_allow(
        int(regime_ma_period or 0),
        regime_ticker,
        gran,
        daily_agg,
        OHLCV_DIR,
    )
    path = portfolio_path_from_panel(panel, fee_rate=fee_rate, regime_allow=regime_allow)
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
                "Cross-sectional MRAT deciles + σ thresholds | Equal-weight portfolio",
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
        subtitle = (
            f"{label} | MRAT SMA {sh}/{lo} |{ex_part}{reg_part} | "
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
    args = parser.parse_args()

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
    min_names_cfg = int(MAD_DEFAULT_MIN_NAMES_PER_DATE)
    min_names = effective_min_names_per_date(daily_long, min_names_cfg)
    grid = _param_grid()
    exit_grid = _exit_ma_grid()
    exit_ma_enabled = bool(getattr(config, "MAD_EXIT_MA_ENABLED", True))
    regime_grid = _regime_ma_grid()
    regime_sym = mad_regime_ticker_symbol()
    eval_all = bool(getattr(config, "MAD_EVAL_ALL_SPLITS", False))
    all_research_dates = {mad_calendar_key(d) for d in split_by_d[split_by_d > 0].index}

    if any(int(x) > 0 for x in regime_grid):
        if not regime_sym:
            raise RuntimeError(
                "MAD_REGIME_MA grid includes a positive MA but MAD_REGIME_TICKER is empty or "
                "MAD_REGIME_MA_ENABLED is False. Set MAD_REGIME_TICKER (e.g. QQQ) and enable the filter."
            )
        _load_regime_daily_close(regime_sym, gran, OHLCV_DIR, aggregate_to_daily=daily_agg)

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
        f"  Regime (cash)    : "
        f"{'off' if regime_sym is None else f'{regime_sym} grid {regime_grid} (0 = off; ETF below SMA ⇒ full cash)'}\n"
        f"  Direction mode   : {direction}\n"
        f"  Long / short decile: ≥{getattr(config, 'MAD_LONG_DECILE_MIN', 10)} / ≤{getattr(config, 'MAD_SHORT_DECILE_MAX', 1)}\n"
        f"  Symmetric short σ: {getattr(config, 'MAD_SYMMETRIC_SHORT_SIGMA', False)} "
        f"(short k = long k when True; else MAD_SHORT_SIGMA_MULT)\n"
        f"  Fee rate         : {args.fee_rate:.4%}\n"
        f"  Min names / date : {min_names} (default={min_names_cfg}, universe n={n_univ})\n"
        f"  Min history bars : {min_hist} (daily bars after aggregation; IPOs join when warm)\n"
    )
    cap0 = min(min_names_cfg, n_univ)
    if min_names < cap0:
        print(
            f"  Note: σ/deciles need ≥{min_names} names per day (default {min_names_cfg}, "
            f"universe n={n_univ}, cap {cap0}; default universe slack relaxes a full-{n_univ} "
            f"requirement so sparse sessions still rank).\n"
        )
    elif min_names_cfg > n_univ:
        print(
            f"  Note: default min names/date {min_names_cfg} > n={n_univ}; effective min names = {min_names}.\n"
        )
    if eval_all:
        print(
            f"  All-splits eval    : {len(all_research_dates):,} calendar days (split > 0, excl. warmup)\n"
        )

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
                        min_names=min_names,
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
            min_names=min_names,
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
                        min_names=min_names,
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
            min_names=min_names,
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
        min_names=min_names,
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
        ).run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
