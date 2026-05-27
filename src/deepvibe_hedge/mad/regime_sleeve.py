"""
Multi-ticker regime filter + configurable risk-off sleeve.

Generalizes the single-ticker ``MAD_REGIME_TICKER`` gate in ``mad.backtester`` to an
ensemble of index ETFs (e.g. ``QQQ`` + ``SPY``) with a combination rule, and replaces
the single-ticker ``MAD_LIVE_REGIME_OFF_PROXY_TICKER`` pivot with a multi-asset
sleeve that can itself be trend-gated (each sleeve member only held if its own
trend filter passes, otherwise its weight reverts to a safe-harbor ticker).

Config
------
``MAD_REGIME_TICKERS``
    Tuple of ETF symbols, e.g. ``("QQQ", "SPY")``. If unset, falls back to
    ``(MAD_REGIME_TICKER,)`` for back-compat.

``MAD_REGIME_MA``
    Single MA period applied to *every* ticker in ``MAD_REGIME_TICKERS``. If unset,
    falls back to ``MAD_LIVE_REGIME_MA``.

``MAD_REGIME_MODE``
    ``"all_below"`` (default) — risk-off when **every** member is below its MA.
    User's SP500+QQQ rule fits this mode.
    ``"any_below"`` — risk-off when **any** member is below (more defensive; the
    first crack flips the book out).

``MAD_REGIME_OFF_SLEEVE``
    Tuple of ``(ticker, weight, trend_ma_period)`` triples describing the risk-off
    allocation. Weights are non-negative and should sum to ~1.0 (normalized on
    use). ``trend_ma_period == 0`` means "always hold at this weight". A positive
    value gates the member: if its close is below that SMA, its weight is
    reassigned to ``MAD_REGIME_OFF_SAFE_HARBOR``.

    Example (your choice, trend-following within the sleeve)::

        MAD_REGIME_OFF_SLEEVE = (
            ("GLD", 0.5, 200),   # hold gold only if GLD > 200D SMA, else reroute to safe harbor
            ("TLT", 0.5, 200),   # hold long bonds only if TLT > 200D SMA
        )

``MAD_REGIME_OFF_SAFE_HARBOR``
    Ticker that receives any sleeve weight rejected by the per-member trend gate
    (typically ``"BIL"`` — ultra-short T-bills, cash-equivalent). Also used as the
    whole sleeve when ``MAD_REGIME_OFF_SLEEVE`` is empty.

Back-compat
-----------
With no new config set, the module evaluates exactly like the legacy path:
single-ticker regime, single-ticker sleeve at 100% (``MAD_LIVE_REGIME_OFF_PROXY_TICKER``).

Pipeline + splitter integration
-------------------------------
``sleeve_and_regime_symbols(config)`` returns the union of regime tickers, sleeve
tickers, and safe harbor. Callers (``config.ohlcv_pipeline_tickers``,
``splitter_ma_periods``, ``ohlcv_live_append.live_ohlcv_append_symbols``) should
union this with the panel universe so the fetcher and splitter produce the DBs
these tickers need.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deepvibe_hedge.mad.weighting import (
    DEFAULT_EQUAL_BLEND,
    DEFAULT_REALIZED_VOL_LOOKBACK,
    DEFAULT_SOFTMAX_TAU,
    WeightConfig,
    WeightInputs,
    compute_weights,
)

SleeveEntry = tuple[str, float, int]  # (ticker, weight, trend_ma_period)

DEFAULT_REGIME_MA = 200
DEFAULT_REGIME_MODE = "all_below"
DEFAULT_SLEEVE_MRAT_SHORT = 21
DEFAULT_SLEEVE_MRAT_LONG = 200


@dataclass(frozen=True)
class RegimeState:
    """Single-point (live) regime evaluation + resulting sleeve allocation."""

    risk_on: bool
    sleeve_weights: dict[str, float]
    per_regime_ticker: dict[str, dict[str, float]] = field(default_factory=dict)
    per_sleeve_ticker: dict[str, dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def _as_upper_tuple(x: object) -> tuple[str, ...]:
    if x is None:
        return ()
    if isinstance(x, str):
        return (x.strip().upper(),) if x.strip() else ()
    try:
        return tuple(str(t).strip().upper() for t in x if str(t).strip())  # type: ignore[arg-type]
    except TypeError:
        return ()


def resolve_regime_tickers(config_module: object) -> tuple[str, ...]:
    """``MAD_REGIME_TICKERS`` if set, else ``(MAD_REGIME_TICKER,)`` (legacy)."""
    tickers = _as_upper_tuple(getattr(config_module, "MAD_REGIME_TICKERS", None))
    if tickers:
        return tickers
    legacy = getattr(config_module, "MAD_REGIME_TICKER", None)
    return _as_upper_tuple(legacy)


def resolve_regime_ma(config_module: object) -> int:
    """Single MA period applied to every regime ticker."""
    ma = getattr(config_module, "MAD_REGIME_MA", None)
    if ma is None:
        ma = getattr(config_module, "MAD_LIVE_REGIME_MA", None)
    if ma is None:
        return int(DEFAULT_REGIME_MA)
    try:
        return max(0, int(ma))
    except (TypeError, ValueError):
        return int(DEFAULT_REGIME_MA)


def resolve_regime_mode(config_module: object) -> str:
    mode = str(getattr(config_module, "MAD_REGIME_MODE", DEFAULT_REGIME_MODE)).strip().lower()
    if mode not in ("all_below", "any_below"):
        raise ValueError(
            f"Invalid MAD_REGIME_MODE={mode!r}. Use 'all_below' or 'any_below'."
        )
    return mode


def resolve_sleeve(config_module: object) -> tuple[SleeveEntry, ...]:
    """Parse ``MAD_REGIME_OFF_SLEEVE`` (list of ``(ticker, weight, trend_ma)``).

    Falls back to ``((MAD_LIVE_REGIME_OFF_PROXY_TICKER, 1.0, 0),)`` when unset, so
    existing configs with just ``MAD_LIVE_REGIME_OFF_PROXY_TICKER = "BIL"`` behave
    identically. Empty / all-zero-weight → empty tuple (sleeve = safe harbor 100%).
    """
    raw = getattr(config_module, "MAD_REGIME_OFF_SLEEVE", None)
    if raw is None:
        legacy = getattr(config_module, "MAD_LIVE_REGIME_OFF_PROXY_TICKER", None)
        sym = str(legacy).strip().upper() if legacy else ""
        return ((sym, 1.0, 0),) if sym else ()

    out: list[SleeveEntry] = []
    for item in raw:
        if isinstance(item, str):
            sym, w, trend = item, 1.0, 0
        elif len(item) == 2:
            sym, w = item  # type: ignore[misc]
            trend = 0
        elif len(item) >= 3:
            sym, w, trend = item[0], item[1], item[2]  # type: ignore[misc]
        else:
            continue
        sym_u = str(sym).strip().upper()
        if not sym_u:
            continue
        try:
            w_f = float(w)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(w_f) or w_f <= 0:
            continue
        try:
            trend_i = max(0, int(trend))
        except (TypeError, ValueError):
            trend_i = 0
        out.append((sym_u, w_f, trend_i))
    return tuple(out)


def resolve_safe_harbor(config_module: object) -> str:
    """Ticker that absorbs trend-rejected sleeve weight (typically ``BIL``)."""
    h = getattr(config_module, "MAD_REGIME_OFF_SAFE_HARBOR", None)
    if h:
        return str(h).strip().upper()
    legacy = getattr(config_module, "MAD_LIVE_REGIME_OFF_PROXY_TICKER", None)
    return str(legacy).strip().upper() if legacy else "BIL"


def resolve_sleeve_weighting_scheme(config_module: object) -> str | None:
    """``MAD_SLEEVE_WEIGHTING_SCHEME`` if set, else ``None`` (back-compat fixed weights)."""
    raw = getattr(config_module, "MAD_SLEEVE_WEIGHTING_SCHEME", None)
    if raw is None:
        return None
    s = str(raw).strip().lower()
    return s or None


def resolve_sleeve_mrat_pair(config_module: object) -> tuple[int, int]:
    """(short, long) MA pair used to score sleeve legs for dynamic weighting.

    Falls back to ``(21, 200)``. Only ``long`` > ``short`` is meaningful; short must be >= 1.
    """
    sh = int(getattr(config_module, "MAD_SLEEVE_MRAT_SHORT", DEFAULT_SLEEVE_MRAT_SHORT) or DEFAULT_SLEEVE_MRAT_SHORT)
    lg = int(getattr(config_module, "MAD_SLEEVE_MRAT_LONG", DEFAULT_SLEEVE_MRAT_LONG) or DEFAULT_SLEEVE_MRAT_LONG)
    return max(1, sh), max(sh + 1, lg)


def resolve_sleeve_weight_config(config_module: object) -> WeightConfig | None:
    """Return a ``WeightConfig`` for the sleeve when ``MAD_SLEEVE_WEIGHTING_SCHEME`` is set.

    Shares caps / blend / softmax τ / realized-vol lookback with the main
    ``MAD_WEIGHT_*`` knobs so users tune once and both books behave consistently.
    Returns ``None`` when no sleeve scheme is configured → callers keep the legacy
    fixed-weight + safe-harbor fallback path.
    """
    scheme = resolve_sleeve_weighting_scheme(config_module)
    if scheme is None:
        return None
    return WeightConfig(
        scheme=scheme,
        max_per_name=_opt_float_cfg(getattr(config_module, "MAD_WEIGHT_MAX_PER_NAME", None)),
        min_per_name=_opt_float_cfg(getattr(config_module, "MAD_WEIGHT_MIN_PER_NAME", None)),
        equal_blend=float(getattr(config_module, "MAD_WEIGHT_EQUAL_BLEND", DEFAULT_EQUAL_BLEND) or 0.0),
        softmax_tau=float(
            getattr(config_module, "MAD_WEIGHT_SOFTMAX_TAU", DEFAULT_SOFTMAX_TAU) or DEFAULT_SOFTMAX_TAU
        ),
        realized_vol_lookback=int(
            getattr(config_module, "MAD_WEIGHT_REALIZED_VOL_LOOKBACK", DEFAULT_REALIZED_VOL_LOOKBACK)
            or DEFAULT_REALIZED_VOL_LOOKBACK
        ),
    )


def _opt_float_cfg(x: object) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v > 0 else None


def sleeve_and_regime_symbols(config_module: object) -> tuple[str, ...]:
    """Union of regime + sleeve + safe-harbor tickers. Empty strings dropped."""
    syms: set[str] = set(resolve_regime_tickers(config_module))
    for sym, _w, _ma in resolve_sleeve(config_module):
        if sym:
            syms.add(sym)
    h = resolve_safe_harbor(config_module)
    if h:
        syms.add(h)
    return tuple(sorted(syms))


def sleeve_and_regime_ma_periods(config_module: object) -> tuple[int, ...]:
    """Union of regime MA + every sleeve member's trend MA (positive only).

    When a dynamic sleeve scheme is configured, also includes the sleeve MRAT
    short/long pair so the splitter precomputes those SMAs on sleeve tickers.
    """
    periods: set[int] = set()
    rm = resolve_regime_ma(config_module)
    if rm > 0:
        periods.add(rm)
    for _sym, _w, tma in resolve_sleeve(config_module):
        if tma > 0:
            periods.add(int(tma))
    if resolve_sleeve_weighting_scheme(config_module) is not None:
        sh, lg = resolve_sleeve_mrat_pair(config_module)
        periods.add(int(sh))
        periods.add(int(lg))
    return tuple(sorted(periods))


# ---------------------------------------------------------------------------
# Close / SMA loading (daily)
# ---------------------------------------------------------------------------


def _load_daily_close(
    ticker: str,
    granularity: str,
    ohlcv_dir: Path,
    *,
    aggregate_to_daily: bool,
) -> pd.Series:
    """Last close per UTC calendar day; tz-aware UTC normalized index.

    Mirrors ``mad.backtester._load_regime_daily_close`` but standalone so this module
    has no import cycle back to the backtester.
    """
    sym = str(ticker).strip().upper()
    path = ohlcv_dir / f"{sym}_{granularity}.db"
    if not path.exists():
        raise FileNotFoundError(
            f"OHLCV DB missing for regime/sleeve ticker {sym}: {path}. "
            "Ensure it is in ``config.ohlcv_pipeline_tickers()`` and fetch/split has run."
        )
    with sqlite3.connect(path) as con:
        df = pd.read_sql(
            "SELECT timestamp, close FROM ohlcv",
            con,
            parse_dates=["timestamp"],
        )
    if df.empty:
        raise RuntimeError(f"OHLCV DB for {sym} is empty: {path}")
    df = df.set_index("timestamp").sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    if aggregate_to_daily and str(granularity).lower() != "1d":
        tmp = df.reset_index()
        tmp["day"] = pd.to_datetime(tmp["timestamp"], utc=True).dt.normalize()
        s = tmp.groupby("day", sort=True)["close"].last()
        s.index = pd.DatetimeIndex(s.index, tz="UTC")
    else:
        s = pd.Series(
            df["close"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(df.index, tz="UTC").normalize(),
            dtype=float,
        ).groupby(level=0).last()
    if s.index.duplicated().any():
        s = s.groupby(level=0).last()
    return s.astype(float)


def _load_daily_close_and_sma(
    ticker: str,
    granularity: str,
    ohlcv_dir: Path,
    ma_period: int,
    *,
    aggregate_to_daily: bool,
    prefer_precomputed_sma: bool,
) -> tuple[pd.Series, pd.Series]:
    """Daily close + SMA. Uses splitter's precomputed ``sma_<n>`` when available on 1d bars."""
    close = _load_daily_close(
        ticker, granularity, ohlcv_dir, aggregate_to_daily=aggregate_to_daily
    )
    if int(ma_period) <= 0:
        return close, pd.Series(np.nan, index=close.index, dtype=float)

    gran_lc = str(granularity).strip().lower()
    if prefer_precomputed_sma and gran_lc == "1d":
        col = f"sma_{int(ma_period)}"
        path = ohlcv_dir / f"{str(ticker).strip().upper()}_{granularity}.db"
        with sqlite3.connect(path) as con:
            names = [r[1] for r in con.execute("PRAGMA table_info(ohlcv)").fetchall()]
            if col in names:
                df = pd.read_sql(
                    f"SELECT timestamp, {col} AS sx FROM ohlcv",
                    con,
                    parse_dates=["timestamp"],
                )
                if not df.empty and df["sx"].notna().any():
                    df = df.set_index("timestamp").sort_index()
                    df.index = pd.DatetimeIndex(df.index, tz="UTC").normalize()
                    sma = df.groupby(level=0)["sx"].last().astype(float)
                    sma = sma.reindex(close.index)
                    if sma.notna().any():
                        return close, sma

    sma = close.rolling(window=int(ma_period), min_periods=int(ma_period)).mean()
    return close, sma


# ---------------------------------------------------------------------------
# Evaluation (live — last bar)
# ---------------------------------------------------------------------------


def _above_sma_last_bar(close: pd.Series, sma: pd.Series) -> tuple[bool, float, float]:
    """Uses the last available close vs its SMA (no shift). Returns (above, close, sma)."""
    c_tail = close.dropna()
    s_tail = sma.dropna()
    if c_tail.empty or s_tail.empty:
        return False, float("nan"), float("nan")
    c = float(c_tail.iloc[-1])
    s = float(s_tail.iloc[-1])
    if not (np.isfinite(c) and np.isfinite(s)):
        return False, c, s
    return bool(c > s), c, s


def _compute_sleeve_mrat_and_rvol(
    sleeve_tickers: list[str],
    *,
    safe_harbor: str,
    granularity: str,
    ohlcv_dir: Path,
    aggregate_to_daily: bool,
    mrat_short: int,
    mrat_long: int,
    rvol_lookback: int,
    prefer_precomputed_sma: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Precompute sleeve-leg MRAT (date × ticker) and realized-vol pivots.

    MRAT uses the ``(short, long)`` pair; both SMAs are shifted by one bar so the
    value aligned on date ``d`` reflects bar ``d-1`` — matching the ``_shift_entry_allow``
    timing used for trend filters and the MRAT book. Realized vol is the rolling
    ``rvol_lookback``-day std of daily returns, also ``.shift(1)``.
    """
    mrat_cols: dict[str, pd.Series] = {}
    rvol_cols: dict[str, pd.Series] = {}
    idx: pd.DatetimeIndex | None = None
    # Safe harbor participates in dynamic weighting too (treated as an always-eligible
    # sleeve leg), so it needs MRAT + rvol series alongside the configured sleeve members.
    ordered = list(sleeve_tickers)
    if safe_harbor and safe_harbor not in ordered:
        ordered.append(safe_harbor)
    for sym in ordered:
        try:
            close_s, sma_s = _load_daily_close_and_sma(
                sym,
                granularity,
                ohlcv_dir,
                int(mrat_short),
                aggregate_to_daily=aggregate_to_daily,
                prefer_precomputed_sma=prefer_precomputed_sma,
            )
            _, sma_l = _load_daily_close_and_sma(
                sym,
                granularity,
                ohlcv_dir,
                int(mrat_long),
                aggregate_to_daily=aggregate_to_daily,
                prefer_precomputed_sma=prefer_precomputed_sma,
            )
        except FileNotFoundError:
            continue
        mrat = (sma_s / sma_l).shift(1).astype(float)
        # Rolling realized vol on daily log / simple returns, shifted to avoid look-ahead.
        ret = close_s.pct_change()
        rvol = ret.rolling(window=max(2, int(rvol_lookback)), min_periods=max(2, int(rvol_lookback))).std(ddof=1).shift(1)
        mrat_cols[sym] = mrat
        rvol_cols[sym] = rvol
        idx = mrat.index if idx is None else idx.union(mrat.index)

    if idx is None:
        return pd.DataFrame(), pd.DataFrame()
    idx = pd.DatetimeIndex(sorted(idx), tz="UTC")
    mrat_piv = pd.concat(
        [s.reindex(idx) for s in mrat_cols.values()], axis=1, keys=list(mrat_cols.keys()), sort=False,
    ) if mrat_cols else pd.DataFrame(index=idx)
    rvol_piv = pd.concat(
        [s.reindex(idx) for s in rvol_cols.values()], axis=1, keys=list(rvol_cols.keys()), sort=False,
    ) if rvol_cols else pd.DataFrame(index=idx)
    return mrat_piv, rvol_piv


def _dynamic_sleeve_weights_row(
    *,
    trend_ok: dict[str, bool],
    sleeve_tickers: list[str],
    safe_harbor: str,
    mrat_row: pd.Series,
    rvol_row: pd.Series | None,
    cfg: WeightConfig,
) -> dict[str, float]:
    """One date's sleeve allocation under a scheme.

    Eligible = trend-passing sleeve legs **plus** the safe-harbor ticker (always
    eligible — cash-equivalent is risk-free by construction). Scheme weights them
    via ``compute_weights``. If nothing gets a non-zero score, safe harbor gets 100%.
    """
    eligible = [s for s in sleeve_tickers if bool(trend_ok.get(s, False)) and s != safe_harbor]
    if safe_harbor and safe_harbor not in eligible:
        eligible.append(safe_harbor)
    if not eligible:
        return {safe_harbor: 1.0} if safe_harbor else {}

    ent = pd.Series(0, index=eligible, dtype=int)
    ent.loc[:] = 1
    # Decile: rank eligible legs by MRAT (1 = weakest, 10 = strongest). Scheme ``rank``
    # uses this; other schemes ignore it. With fewer than 10 legs we just use the rank.
    m_el = mrat_row.reindex(eligible).astype(float)
    order = m_el.rank(method="min", ascending=True).fillna(1.0)
    # Map to 1..10 scale so that ``_rank_mag`` (which computes ``d`` or ``11 - d``) works.
    n = max(1, len(eligible))
    decile = ((order - 1.0) * (9.0 / max(1, (n - 1)))) + 1.0 if n > 1 else pd.Series(10.0, index=eligible)

    # Cross-sectional sigma of MRAT on this date (may be NaN if only 1 eligible leg).
    sig = float(m_el.std(ddof=1)) if len(eligible) >= 2 else float("nan")

    rvol_el = rvol_row.reindex(eligible).astype(float) if rvol_row is not None else None

    inp = WeightInputs(
        entry_signal=ent,
        mrat=m_el,
        sigma=sig,
        decile=decile,
        realized_vol=rvol_el,
    )
    w = compute_weights(inp, cfg)
    # ``compute_weights`` already normalizes longs to sum to +1.0.
    out: dict[str, float] = {}
    for sym, val in w.items():
        v = float(val)
        if abs(v) > 1e-12:
            out[sym] = v
    if not out and safe_harbor:
        return {safe_harbor: 1.0}
    return out


def _compose_sleeve_weights(
    sleeve: tuple[SleeveEntry, ...],
    safe_harbor: str,
    trend_ok: dict[str, bool],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Apply trend gates; weights for failed members reroute to safe harbor; normalize."""
    if not sleeve:
        return ({safe_harbor: 1.0} if safe_harbor else {}, {})
    total = sum(float(w) for _t, w, _ in sleeve)
    if total <= 1e-15:
        return ({safe_harbor: 1.0} if safe_harbor else {}, {})

    weights: dict[str, float] = {}
    per_member: dict[str, dict[str, float]] = {}
    for sym, w, trend_ma in sleeve:
        norm = float(w) / total
        ok = True if trend_ma <= 0 else bool(trend_ok.get(sym, False))
        per_member[sym] = {
            "nominal_weight": norm,
            "trend_ma": float(trend_ma),
            "trend_ok": 1.0 if ok else 0.0,
        }
        if ok:
            weights[sym] = weights.get(sym, 0.0) + norm
        elif safe_harbor:
            weights[safe_harbor] = weights.get(safe_harbor, 0.0) + norm

    s = sum(weights.values())
    if s > 1e-15 and abs(s - 1.0) > 1e-9:
        weights = {k: v / s for k, v in weights.items()}
    return weights, per_member


def evaluate_regime_live(
    *,
    ohlcv_dir: Path,
    granularity: str,
    aggregate_to_daily: bool,
    regime_tickers: tuple[str, ...],
    regime_ma: int,
    mode: str,
    sleeve: tuple[SleeveEntry, ...],
    safe_harbor: str,
    prefer_precomputed_sma: bool = False,
    sleeve_weight_cfg: WeightConfig | None = None,
    sleeve_mrat_pair: tuple[int, int] | None = None,
) -> RegimeState:
    """Last-bar regime decision + resulting sleeve allocation.

    Timing matches the MRAT backtest signal: we check whether the regime tickers'
    *last completed close* is above their SMA on the same bar, which determines
    whether the **next session** runs MRAT (risk-on) or the sleeve (risk-off).
    """
    m = str(mode).strip().lower()
    if m not in ("all_below", "any_below"):
        raise ValueError(f"Invalid regime mode {mode!r}.")

    per_regime: dict[str, dict[str, float]] = {}
    flags: list[bool] = []
    for sym in regime_tickers:
        if int(regime_ma) <= 0:
            flags.append(True)
            per_regime[sym] = {"above": 1.0, "close": float("nan"), "sma": float("nan")}
            continue
        close, sma = _load_daily_close_and_sma(
            sym,
            granularity,
            ohlcv_dir,
            int(regime_ma),
            aggregate_to_daily=aggregate_to_daily,
            prefer_precomputed_sma=prefer_precomputed_sma,
        )
        above, c, s = _above_sma_last_bar(close, sma)
        flags.append(above)
        per_regime[sym] = {"above": 1.0 if above else 0.0, "close": c, "sma": s}

    if not flags:
        risk_on = True  # no regime tickers = always risk-on (legacy disabled state)
    elif m == "all_below":
        risk_on = any(flags)
    else:
        risk_on = all(flags)

    sleeve_weights: dict[str, float] = {}
    per_sleeve: dict[str, dict[str, float]] = {}
    if not risk_on:
        trend_ok: dict[str, bool] = {}
        for sym, _w, trend_ma in sleeve:
            if int(trend_ma) <= 0:
                trend_ok[sym] = True
                continue
            close, sma = _load_daily_close_and_sma(
                sym,
                granularity,
                ohlcv_dir,
                int(trend_ma),
                aggregate_to_daily=aggregate_to_daily,
                prefer_precomputed_sma=prefer_precomputed_sma,
            )
            above, _c, _s = _above_sma_last_bar(close, sma)
            trend_ok[sym] = above
        if sleeve_weight_cfg is not None and sleeve:
            # Dynamic scheme path: compute MRAT + rvol for last bar and size by scheme.
            sh, lg = sleeve_mrat_pair or (DEFAULT_SLEEVE_MRAT_SHORT, DEFAULT_SLEEVE_MRAT_LONG)
            mrat_piv, rvol_piv = _compute_sleeve_mrat_and_rvol(
                [s for s, _w, _ in sleeve],
                safe_harbor=safe_harbor,
                granularity=granularity,
                ohlcv_dir=ohlcv_dir,
                aggregate_to_daily=aggregate_to_daily,
                mrat_short=int(sh),
                mrat_long=int(lg),
                rvol_lookback=int(sleeve_weight_cfg.realized_vol_lookback),
                prefer_precomputed_sma=prefer_precomputed_sma,
            )
            if mrat_piv.empty:
                sleeve_weights, per_sleeve = _compose_sleeve_weights(sleeve, safe_harbor, trend_ok)
            else:
                m_row = mrat_piv.iloc[-1]
                r_row = rvol_piv.iloc[-1] if not rvol_piv.empty else None
                sleeve_weights = _dynamic_sleeve_weights_row(
                    trend_ok=trend_ok,
                    sleeve_tickers=[s for s, _w, _ in sleeve],
                    safe_harbor=safe_harbor,
                    mrat_row=m_row,
                    rvol_row=r_row,
                    cfg=sleeve_weight_cfg,
                )
                per_sleeve = {
                    sym: {
                        "trend_ma": float(tma),
                        "trend_ok": 1.0 if bool(trend_ok.get(sym, False)) else 0.0,
                        "mrat": float(m_row.get(sym, float("nan"))),
                        "scheme_weight": float(sleeve_weights.get(sym, 0.0)),
                    }
                    for sym, _w, tma in sleeve
                }
        else:
            sleeve_weights, per_sleeve = _compose_sleeve_weights(sleeve, safe_harbor, trend_ok)

    return RegimeState(
        risk_on=risk_on,
        sleeve_weights=sleeve_weights,
        per_regime_ticker=per_regime,
        per_sleeve_ticker=per_sleeve,
    )


def evaluate_sleeve_allocation_live(
    *,
    ohlcv_dir: Path,
    granularity: str,
    aggregate_to_daily: bool,
    sleeve: tuple[SleeveEntry, ...],
    safe_harbor: str,
    prefer_precomputed_sma: bool = False,
    sleeve_weight_cfg: WeightConfig | None = None,
    sleeve_mrat_pair: tuple[int, int] | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Last-bar sleeve composition, *unconditional* on top-level regime state.

    This is the multi-index live variant of the risk-off branch of
    ``evaluate_regime_live``: it always returns the per-leg sleeve weights and
    per-leg diagnostics even when no top-level regime gate has fired, because
    the multi-index allocator may be routing only a *fraction* of the book to
    the sleeve on a given day (one slot below trend, the other still passing).

    Per-leg trend filter still applies — failing legs route their share to
    ``safe_harbor`` (typically BIL). When a dynamic ``sleeve_weight_cfg`` is
    supplied, trend-passing legs are sized by the scheme (using
    ``sleeve_mrat_pair`` MRATs + realized vol); otherwise the static
    per-leg weights from ``sleeve`` are normalized after dropping failing legs.
    """
    if not sleeve:
        return {}, {}

    trend_ok: dict[str, bool] = {}
    for sym, _w, trend_ma in sleeve:
        if int(trend_ma) <= 0:
            trend_ok[sym] = True
            continue
        try:
            close, sma = _load_daily_close_and_sma(
                sym,
                granularity,
                ohlcv_dir,
                int(trend_ma),
                aggregate_to_daily=aggregate_to_daily,
                prefer_precomputed_sma=prefer_precomputed_sma,
            )
        except FileNotFoundError:
            trend_ok[sym] = False
            continue
        above, _c, _s = _above_sma_last_bar(close, sma)
        trend_ok[sym] = above

    if sleeve_weight_cfg is not None:
        sh, lg = sleeve_mrat_pair or (DEFAULT_SLEEVE_MRAT_SHORT, DEFAULT_SLEEVE_MRAT_LONG)
        mrat_piv, rvol_piv = _compute_sleeve_mrat_and_rvol(
            [s for s, _w, _ in sleeve],
            safe_harbor=safe_harbor,
            granularity=granularity,
            ohlcv_dir=ohlcv_dir,
            aggregate_to_daily=aggregate_to_daily,
            mrat_short=int(sh),
            mrat_long=int(lg),
            rvol_lookback=int(sleeve_weight_cfg.realized_vol_lookback),
            prefer_precomputed_sma=prefer_precomputed_sma,
        )
        if mrat_piv.empty:
            sleeve_weights, per_sleeve = _compose_sleeve_weights(sleeve, safe_harbor, trend_ok)
        else:
            m_row = mrat_piv.iloc[-1]
            r_row = rvol_piv.iloc[-1] if not rvol_piv.empty else None
            sleeve_weights = _dynamic_sleeve_weights_row(
                trend_ok=trend_ok,
                sleeve_tickers=[s for s, _w, _ in sleeve],
                safe_harbor=safe_harbor,
                mrat_row=m_row,
                rvol_row=r_row,
                cfg=sleeve_weight_cfg,
            )
            per_sleeve = {
                sym: {
                    "trend_ma": float(tma),
                    "trend_ok": 1.0 if bool(trend_ok.get(sym, False)) else 0.0,
                    "mrat": float(m_row.get(sym, float("nan"))),
                    "scheme_weight": float(sleeve_weights.get(sym, 0.0)),
                }
                for sym, _w, tma in sleeve
            }
    else:
        sleeve_weights, per_sleeve = _compose_sleeve_weights(sleeve, safe_harbor, trend_ok)

    return sleeve_weights, per_sleeve


# ---------------------------------------------------------------------------
# Evaluation (backtest — full time series)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeBacktestSeries:
    """Time-aligned regime + sleeve series for ``portfolio_path_from_panel``.

    ``risk_on`` uses the same ``shift(1)`` timing as MRAT entry (today's risk-on
    uses yesterday's close vs SMA). ``sleeve_weights_piv`` columns are sleeve /
    safe-harbor tickers and rows are UTC-normalized calendar dates; used when
    ``risk_on`` is False on the corresponding date.
    """

    risk_on: pd.Series  # index = UTC-normalized dates, values = bool
    sleeve_weights_piv: pd.DataFrame  # index = UTC-normalized dates, columns = ticker
    regime_tickers: tuple[str, ...]
    sleeve_tickers: tuple[str, ...]


def _shift_entry_allow(close: pd.Series, sma: pd.Series) -> pd.Series:
    """Same timing as MRAT: risk-on today uses close[t-1] > SMA[t-1]."""
    above = close > sma
    return above.shift(1).fillna(False).astype(bool)


def build_regime_backtest_series(
    *,
    ohlcv_dir: Path,
    granularity: str,
    aggregate_to_daily: bool,
    regime_tickers: tuple[str, ...],
    regime_ma: int,
    mode: str,
    sleeve: tuple[SleeveEntry, ...],
    safe_harbor: str,
    prefer_precomputed_sma: bool = False,
    sleeve_weight_cfg: WeightConfig | None = None,
    sleeve_mrat_pair: tuple[int, int] | None = None,
) -> RegimeBacktestSeries | None:
    """Build risk-on flag + sleeve weight pivot aligned on UTC calendar days.

    Returns ``None`` when no regime filter is active (``regime_ma <= 0`` or empty
    ``regime_tickers``) so callers can treat the result as "always risk-on" and
    skip the pivot merge entirely (preserves legacy fast path).
    """
    if int(regime_ma or 0) <= 0 or not regime_tickers:
        return None

    m = str(mode).strip().lower()
    if m not in ("all_below", "any_below"):
        raise ValueError(f"Invalid regime mode {mode!r}.")

    # 1. Per-ticker "above" series (shifted to MRAT entry timing).
    above_frames: list[pd.Series] = []
    master_idx: pd.DatetimeIndex | None = None
    for sym in regime_tickers:
        close, sma = _load_daily_close_and_sma(
            sym,
            granularity,
            ohlcv_dir,
            int(regime_ma),
            aggregate_to_daily=aggregate_to_daily,
            prefer_precomputed_sma=prefer_precomputed_sma,
        )
        flag = _shift_entry_allow(close, sma).rename(sym)
        above_frames.append(flag)
        master_idx = flag.index if master_idx is None else master_idx.union(flag.index)

    assert master_idx is not None
    master_idx = pd.DatetimeIndex(sorted(master_idx), tz="UTC")
    above_df = pd.concat(
        [f.reindex(master_idx, fill_value=False) for f in above_frames], axis=1
    )
    if m == "all_below":
        risk_on = above_df.any(axis=1)  # any ticker above SMA → risk-on
    else:  # any_below
        risk_on = above_df.all(axis=1)  # every ticker must be above → risk-on

    # 2. Per-sleeve-member trend flag, same timing.
    sleeve_tickers: list[str] = []
    trend_frames: dict[str, pd.Series] = {}
    for sym, _w, trend_ma in sleeve:
        if sym not in sleeve_tickers:
            sleeve_tickers.append(sym)
        if int(trend_ma) <= 0:
            trend_frames[sym] = pd.Series(True, index=master_idx, dtype=bool)
        else:
            close, sma = _load_daily_close_and_sma(
                sym,
                granularity,
                ohlcv_dir,
                int(trend_ma),
                aggregate_to_daily=aggregate_to_daily,
                prefer_precomputed_sma=prefer_precomputed_sma,
            )
            trend_frames[sym] = (
                _shift_entry_allow(close, sma).reindex(master_idx, fill_value=False)
            )
    if safe_harbor and safe_harbor not in sleeve_tickers:
        sleeve_tickers.append(safe_harbor)

    # 3. Per-date sleeve weights (only relevant on risk-off days, but built for all).
    total = sum(float(w) for _t, w, _ in sleeve) if sleeve else 0.0
    cols = sorted(set(sleeve_tickers))
    data = {c: np.zeros(len(master_idx), dtype=float) for c in cols}

    if sleeve_weight_cfg is not None and sleeve and total > 1e-15:
        # Dynamic path: recompute per-date sleeve weights using the chosen scheme
        # across trend-passing legs. Falls back to safe-harbor 100% when every leg fails.
        sh, lg = sleeve_mrat_pair or (DEFAULT_SLEEVE_MRAT_SHORT, DEFAULT_SLEEVE_MRAT_LONG)
        sleeve_syms_order = [s for s, _w, _ in sleeve]
        mrat_piv, rvol_piv = _compute_sleeve_mrat_and_rvol(
            sleeve_syms_order,
            safe_harbor=safe_harbor,
            granularity=granularity,
            ohlcv_dir=ohlcv_dir,
            aggregate_to_daily=aggregate_to_daily,
            mrat_short=int(sh),
            mrat_long=int(lg),
            rvol_lookback=int(sleeve_weight_cfg.realized_vol_lookback),
            prefer_precomputed_sma=prefer_precomputed_sma,
        )
        mrat_aligned = mrat_piv.reindex(master_idx) if not mrat_piv.empty else pd.DataFrame(index=master_idx)
        rvol_aligned = rvol_piv.reindex(master_idx) if not rvol_piv.empty else pd.DataFrame(index=master_idx)
        trend_aligned = {
            sym: trend_frames.get(sym, pd.Series(True, index=master_idx, dtype=bool)).reindex(master_idx, fill_value=False)
            for sym in sleeve_syms_order
        }
        for i, d in enumerate(master_idx):
            trend_ok_d = {sym: bool(trend_aligned[sym].iat[i]) for sym in sleeve_syms_order}
            m_row = mrat_aligned.loc[d] if d in mrat_aligned.index else pd.Series(dtype=float)
            r_row = rvol_aligned.loc[d] if not rvol_aligned.empty and d in rvol_aligned.index else None
            row = _dynamic_sleeve_weights_row(
                trend_ok=trend_ok_d,
                sleeve_tickers=sleeve_syms_order,
                safe_harbor=safe_harbor,
                mrat_row=m_row,
                rvol_row=r_row,
                cfg=sleeve_weight_cfg,
            )
            for sym, val in row.items():
                if sym in data:
                    data[sym][i] = float(val)
    elif sleeve and total > 1e-15:
        for sym, w, trend_ma in sleeve:
            norm = float(w) / total
            ok = trend_frames.get(sym, pd.Series(True, index=master_idx, dtype=bool))
            ok_arr = ok.reindex(master_idx, fill_value=False).to_numpy(dtype=bool)
            data[sym] = data[sym] + np.where(ok_arr, norm, 0.0)
            if safe_harbor:
                data[safe_harbor] = data[safe_harbor] + np.where(ok_arr, 0.0, norm)
    elif safe_harbor:
        data[safe_harbor] = np.ones(len(master_idx), dtype=float)

    sleeve_piv = pd.DataFrame(data, index=master_idx)
    # Renormalize (defensive — mass should already be 1.0 per row).
    row_sum = sleeve_piv.sum(axis=1).replace(0.0, np.nan)
    sleeve_piv = sleeve_piv.div(row_sum, axis=0).fillna(0.0)

    return RegimeBacktestSeries(
        risk_on=risk_on.astype(bool),
        sleeve_weights_piv=sleeve_piv,
        regime_tickers=tuple(regime_tickers),
        sleeve_tickers=tuple(cols),
    )


# ---------------------------------------------------------------------------
# Diagnostics helper (human-readable status line for logs)
# ---------------------------------------------------------------------------


def format_regime_state_line(
    state: RegimeState,
    *,
    regime_ma: int,
    mode: str,
) -> str:
    """One-line summary for the live bot / dashboard logs."""
    rt_parts: list[str] = []
    for sym, d in state.per_regime_ticker.items():
        above = bool(d.get("above", 0.0) >= 0.5)
        c = d.get("close", float("nan"))
        s = d.get("sma", float("nan"))
        c_s = f"{c:.2f}" if np.isfinite(c) else "nan"
        s_s = f"{s:.2f}" if np.isfinite(s) else "nan"
        rt_parts.append(f"{sym}={'ABOVE' if above else 'below'}({c_s}/{s_s})")

    if state.risk_on:
        sleeve_part = "sleeve=off (MRAT book)"
    else:
        items = ", ".join(f"{k}:{v:.2%}" for k, v in sorted(state.sleeve_weights.items()))
        sleeve_part = f"sleeve={{{items}}}" if items else "sleeve=cash"

    return (
        f"regime[{mode} {regime_ma}D] {'RISK-ON' if state.risk_on else 'RISK-OFF'} | "
        + ", ".join(rt_parts)
        + " | "
        + sleeve_part
    )


def describe_sleeve_config(config_module: object) -> dict[str, Any]:
    """Snapshot of the resolved regime/sleeve config (for dashboards + dry-run prints)."""
    return {
        "regime_tickers": resolve_regime_tickers(config_module),
        "regime_ma": resolve_regime_ma(config_module),
        "regime_mode": resolve_regime_mode(config_module),
        "sleeve": resolve_sleeve(config_module),
        "safe_harbor": resolve_safe_harbor(config_module),
        "sleeve_weighting_scheme": resolve_sleeve_weighting_scheme(config_module),
        "sleeve_mrat_pair": resolve_sleeve_mrat_pair(config_module),
    }
