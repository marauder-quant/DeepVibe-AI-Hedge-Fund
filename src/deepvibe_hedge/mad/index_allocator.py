"""
Multi-index allocator — top-level "Level 1" weighting across index sleeves.

When enabled, this module sits **one level above** the MRAT stock-picker. Each
configured index (``SPY`` / ``QQQ`` / ``IWM``) is scored by its own ETF MRAT
and its trend filter (``close > SMA(MAD_INDEX_REGIME_MA)``). The top-level
allocation rule (**progressive risk-off blend**) is:

* ``0 of N`` indexes fail → 100% equity-book, split across N indexes by the
  configured weighting scheme applied to their MRAT scores.
* ``k of N`` fail → ``(N-k)/N`` of the pie allocated across passing indexes by
  scheme, ``k/N`` rerouted to the risk-off sleeve.
* ``N of N`` fail → 100% risk-off sleeve.

The risk-off sleeve's **internal** composition is handled separately by
``mad.regime_sleeve`` — this module only decides the *share* that routes there.

Exclusive ticker assignment
---------------------------
Indexes share many constituents (e.g. ~85% of NASDAQ-100 is in S&P 500). To
avoid double-counting a stock across two slots, ``resolve_index_slots`` walks
``MAD_INDEX_SLOTS`` in order and claims each ticker for the **first** slot
that lists it. The default order (``SPY → QQQ → IWM``) means:

* SPY keeps its full list (504 names)
* QQQ's exclusive universe drops to ~15 non-S&P Nasdaq names
* IWM's exclusive universe is ~1930 small-caps with near-zero S&P overlap

Change ``MAD_INDEX_SLOTS`` order in config to flip priority.

Config
------
See ``config.py`` section headed "Multi-index allocator" for all keys:
``MAD_INDEX_ALLOCATOR_ENABLED``, ``MAD_INDEX_SLOTS``, ``MAD_INDEX_REGIME_MA``,
``MAD_INDEX_MRAT_SHORT``/``LONG``, ``MAD_INDEX_WEIGHTING_SCHEME``,
``MAD_MIN_DATA_COMPLETENESS``. Reuses ``MAD_WEIGHT_*`` knobs from the
stock-level weighting config.

Back-compat
-----------
With ``MAD_INDEX_ALLOCATOR_ENABLED = False`` (default), callers should skip
this module entirely and use the legacy single-universe path. Every public
resolver returns a well-defined "no-op" when the flag is off so callers can
branch cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deepvibe_hedge.mad.regime_sleeve import (
    _above_sma_last_bar,
    _load_daily_close,
    _load_daily_close_and_sma,
    _shift_entry_allow,
)
from deepvibe_hedge.mad.weighting import (
    DEFAULT_EQUAL_BLEND,
    DEFAULT_REALIZED_VOL_LOOKBACK,
    DEFAULT_SOFTMAX_TAU,
    WeightConfig,
    WeightInputs,
    compute_weights,
)

RISK_OFF_KEY = "__risk_off__"
"""Reserved column name emitted by ``build_index_allocation_series`` for the
share routed to the risk-off sleeve."""

DEFAULT_INDEX_REGIME_MA = 200
DEFAULT_INDEX_MRAT_SHORT = 21
DEFAULT_INDEX_MRAT_LONG = 200
DEFAULT_MIN_DATA_COMPLETENESS = 0.8


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexSlot:
    """One configured index slot after exclusive-assignment dedup."""

    etf: str                      # e.g. "SPY"
    universe: tuple[str, ...]     # constituent tickers exclusively assigned to this slot
    display_name: str             # e.g. "S&P 500"
    raw_universe_size: int = 0    # size before exclusive-assignment dedup (for banner)


@dataclass(frozen=True)
class IndexAllocationState:
    """Live (single-point) allocation output."""

    index_weights: dict[str, float]      # {"SPY": 0.35, "QQQ": 0.40, "IWM": 0.0}
    risk_off_weight: float               # 0.0 - 1.0
    index_trend_ok: dict[str, bool]      # which indexes passed their trend filter
    index_mrat: dict[str, float]         # raw MRATs (diagnostic)
    per_index_detail: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class IndexAllocationSeries:
    """Backtest (per-date) allocation output.

    ``index_weights_piv`` columns: each index ETF symbol + ``RISK_OFF_KEY``.
    Row sums equal 1.0 on every date (after warmup).
    """

    index_weights_piv: pd.DataFrame
    index_trend_ok: pd.DataFrame
    index_mrat: pd.DataFrame


# ---------------------------------------------------------------------------
# Config resolvers (all tolerant of missing attrs for back-compat)
# ---------------------------------------------------------------------------


def index_allocator_enabled(config_module: object) -> bool:
    return bool(getattr(config_module, "MAD_INDEX_ALLOCATOR_ENABLED", False))


def resolve_index_slots(config_module: object) -> tuple[IndexSlot, ...]:
    """Return slots with exclusive-assignment dedup applied in ``MAD_INDEX_SLOTS`` order.

    Honors ``MAD_INDEX_ENABLED_ETFS`` as a whitelist when set: slots whose ETF
    is not in the whitelist are skipped entirely (their tickers are NOT claimed
    for dedup purposes, so a later slot can pick them up).
    """
    raw_slots = getattr(config_module, "MAD_INDEX_SLOTS", ())
    if not raw_slots:
        return ()
    enabled_raw = getattr(config_module, "MAD_INDEX_ENABLED_ETFS", None)
    enabled: set[str] | None
    if enabled_raw is None:
        enabled = None
    else:
        enabled = {str(t).strip().upper() for t in enabled_raw if str(t).strip()}
        if not enabled:
            enabled = None  # treat empty tuple same as None (all enabled)
    claimed: set[str] = set()
    out: list[IndexSlot] = []
    for entry in raw_slots:
        if not entry:
            continue
        etf, universe_raw, display = (
            str(entry[0]).strip().upper(),
            entry[1],
            str(entry[2]) if len(entry) > 2 else str(entry[0]),
        )
        if enabled is not None and etf not in enabled:
            continue
        if isinstance(universe_raw, str):
            raw_universe = (universe_raw.strip().upper(),)
        else:
            raw_universe = tuple(
                str(x).strip().upper() for x in universe_raw if str(x).strip()
            )
        # Exclusive dedup: only keep tickers not already claimed by a prior slot.
        exclusive: list[str] = []
        for t in raw_universe:
            if t and t not in claimed:
                claimed.add(t)
                exclusive.append(t)
        out.append(
            IndexSlot(
                etf=etf,
                universe=tuple(exclusive),
                display_name=display,
                raw_universe_size=len(raw_universe),
            )
        )
    return tuple(out)


def resolve_index_regime_ma(config_module: object) -> int:
    v = getattr(config_module, "MAD_INDEX_REGIME_MA", DEFAULT_INDEX_REGIME_MA)
    return int(v) if v is not None else DEFAULT_INDEX_REGIME_MA


def resolve_index_mrat_pair(config_module: object) -> tuple[int, int]:
    s = int(getattr(config_module, "MAD_INDEX_MRAT_SHORT", DEFAULT_INDEX_MRAT_SHORT))
    l = int(getattr(config_module, "MAD_INDEX_MRAT_LONG", DEFAULT_INDEX_MRAT_LONG))
    return (s, l)


def resolve_index_weight_config(config_module: object) -> WeightConfig | None:
    scheme = getattr(config_module, "MAD_INDEX_WEIGHTING_SCHEME", None)
    if scheme is None or not str(scheme).strip():
        return None
    s = str(scheme).strip().lower()
    return WeightConfig(
        scheme=s,
        max_per_name=_opt_float(getattr(config_module, "MAD_WEIGHT_MAX_PER_NAME", None)),
        min_per_name=_opt_float(getattr(config_module, "MAD_WEIGHT_MIN_PER_NAME", None)),
        equal_blend=float(
            getattr(config_module, "MAD_WEIGHT_EQUAL_BLEND", DEFAULT_EQUAL_BLEND)
        ),
        softmax_tau=float(
            getattr(config_module, "MAD_WEIGHT_SOFTMAX_TAU", DEFAULT_SOFTMAX_TAU)
        ),
        realized_vol_lookback=int(
            getattr(
                config_module,
                "MAD_WEIGHT_REALIZED_VOL_LOOKBACK",
                DEFAULT_REALIZED_VOL_LOOKBACK,
            )
        ),
    )


def resolve_min_data_completeness(config_module: object) -> float:
    v = getattr(config_module, "MAD_MIN_DATA_COMPLETENESS", DEFAULT_MIN_DATA_COMPLETENESS)
    try:
        f = float(v)
    except Exception:
        return DEFAULT_MIN_DATA_COMPLETENESS
    return max(0.0, min(1.0, f))


def _opt_float(x: object) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def index_allocator_symbols(config_module: object) -> tuple[str, ...]:
    """Union of every index ETF + every constituent across all slots (deduped).

    Used by ``config.ohlcv_pipeline_tickers`` to ensure the fetcher + splitter
    produce DBs for every symbol the allocator can reference.
    """
    if not index_allocator_enabled(config_module):
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for slot in resolve_index_slots(config_module):
        for sym in (slot.etf, *slot.universe):
            s = str(sym).strip().upper()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return tuple(out)


# ---------------------------------------------------------------------------
# Progressive-blend primitive (shared by live + backtest)
# ---------------------------------------------------------------------------


def _progressive_blend_row(
    *,
    mrat_scores: dict[str, float],
    trend_ok: dict[str, bool],
    weight_cfg: WeightConfig | None,
    rvol_by_key: dict[str, float] | None = None,
) -> tuple[dict[str, float], float]:
    """Compute one date's per-index weights + risk-off share.

    Returns ``(index_weights, risk_off_share)``. ``index_weights`` always has the
    same keys as ``mrat_scores`` (failing indexes get ``0.0``). Row sums to 1.0
    with ``risk_off_share``.

    ``rvol_by_key`` — optional per-index realized vol (n-day rolling std of log
    returns, with ``n = weight_cfg.realized_vol_lookback``). Required for the
    ``inv_vol`` / ``mrat_distance_inv_vol`` schemes; when missing or ``NaN``
    those schemes silently fall back to equal (via ``_inv_vol_mag``).
    """
    keys = list(mrat_scores.keys())
    n_total = len(keys)
    if n_total == 0:
        return {}, 1.0
    passing = [k for k in keys if trend_ok.get(k, False) and not _nan(mrat_scores[k])]
    n_pass = len(passing)
    if n_pass == 0:
        return {k: 0.0 for k in keys}, 1.0

    equity_share = n_pass / n_total
    risk_off_share = 1.0 - equity_share

    if weight_cfg is None or weight_cfg.scheme == "equal":
        inner = {k: 1.0 / n_pass for k in passing}
    else:
        # Reuse stock-level weighting engine on the passing-index subset.
        # Long-only interpretation: entry_signal = +1 for every passer.
        sig = pd.Series({k: 1 for k in passing}, dtype=int)
        mrat_s = pd.Series({k: float(mrat_scores[k]) for k in passing}, dtype=float)
        # Cross-sectional sigma for the scoring set. Use sample std over passing
        # MRATs; fall back to a small positive value to avoid div-by-zero in
        # z-score-style schemes when all three indexes are nearly identical.
        if n_pass >= 2:
            sigma_cs = float(np.nanstd(mrat_s.to_numpy(dtype=float), ddof=1))
            if not np.isfinite(sigma_cs) or sigma_cs <= 0.0:
                sigma_cs = 1e-6
        else:
            sigma_cs = 1e-6
        # Decile-ranked score across passers (1..n_pass mapped into 1..10 bucket)
        ranks = mrat_s.rank(method="average", ascending=True)
        decile = (ranks / n_pass * 10.0).clip(lower=1.0, upper=10.0)

        # Per-index realized vol (required by inv_vol / mrat_distance_inv_vol).
        # Drop NaN / non-positive values — the weighting engine floors those
        # to the cross-sectional median internally.
        if rvol_by_key:
            rvol = pd.Series(
                {k: float(rvol_by_key.get(k, float("nan"))) for k in passing},
                dtype=float,
            )
            if not rvol.notna().any():
                rvol = None
        else:
            rvol = None

        inp = WeightInputs(
            entry_signal=sig,
            mrat=mrat_s,
            sigma=sigma_cs,
            decile=decile,
            realized_vol=rvol,
        )
        w = compute_weights(inp, weight_cfg)
        w = w.clip(lower=0.0)
        tot = float(w.sum())
        if tot <= 0.0 or not np.isfinite(tot):
            inner = {k: 1.0 / n_pass for k in passing}
        else:
            inner = {k: float(w.get(k, 0.0)) / tot for k in passing}

    out = {k: 0.0 for k in keys}
    for k, v in inner.items():
        out[k] = v * equity_share
    return out, risk_off_share


def _nan(x: float) -> bool:
    return x is None or not np.isfinite(x)


# ---------------------------------------------------------------------------
# Backtest (per-date pivot)
# ---------------------------------------------------------------------------


def build_index_allocation_series(
    *,
    slots: tuple[IndexSlot, ...],
    regime_ma: int,
    mrat_pair: tuple[int, int],
    weight_cfg: WeightConfig | None,
    granularity: str,
    ohlcv_dir: Path,
    aggregate_to_daily: bool,
    prefer_precomputed_sma: bool = False,
) -> IndexAllocationSeries:
    """Compute per-date index + risk-off weights for the backtest calendar.

    All inputs are shifted to MRAT entry timing (use previous bar's close / SMA)
    so the allocator does not peek at today's close.
    """
    if not slots:
        empty = pd.DataFrame()
        return IndexAllocationSeries(empty, empty, empty)

    short_w, long_w = int(mrat_pair[0]), int(mrat_pair[1])
    # Realized-vol lookback for inv_vol / mrat_distance_inv_vol schemes. Fall
    # back to the package default when no scheme config provided.
    rvol_lb = int(
        weight_cfg.realized_vol_lookback if weight_cfg is not None else DEFAULT_REALIZED_VOL_LOOKBACK
    )

    trend_frames: list[pd.Series] = []
    mrat_frames: list[pd.Series] = []
    rvol_frames: list[pd.Series] = []

    for slot in slots:
        close, sma_regime = _load_daily_close_and_sma(
            slot.etf,
            granularity,
            ohlcv_dir,
            int(regime_ma),
            aggregate_to_daily=aggregate_to_daily,
            prefer_precomputed_sma=prefer_precomputed_sma,
        )
        # Per-ETF trend filter shifted to match entry timing.
        trend_ok = _shift_entry_allow(close, sma_regime).rename(slot.etf)
        trend_frames.append(trend_ok.astype("boolean"))

        # ETF MRAT (short/long SMAs) shifted to entry timing.
        _, sma_s = _load_daily_close_and_sma(
            slot.etf,
            granularity,
            ohlcv_dir,
            short_w,
            aggregate_to_daily=aggregate_to_daily,
            prefer_precomputed_sma=prefer_precomputed_sma,
        )
        _, sma_l = _load_daily_close_and_sma(
            slot.etf,
            granularity,
            ohlcv_dir,
            long_w,
            aggregate_to_daily=aggregate_to_daily,
            prefer_precomputed_sma=prefer_precomputed_sma,
        )
        mrat_raw = (sma_s / sma_l).astype(float)
        mrat_shift = mrat_raw.shift(1).rename(slot.etf)
        mrat_frames.append(mrat_shift)

        # Realized vol: rolling std of log returns, shifted by 1 so we use
        # info known at ``t-1`` close (same entry timing as MRAT / trend).
        log_ret = np.log(close / close.shift(1))
        rvol_series = (
            log_ret.rolling(window=max(2, rvol_lb), min_periods=max(2, rvol_lb // 2))
            .std(ddof=1)
            .shift(1)
            .rename(slot.etf)
        )
        rvol_frames.append(rvol_series.astype(float))

    trend_df = pd.concat(trend_frames, axis=1, sort=True).sort_index()
    mrat_df = pd.concat(mrat_frames, axis=1, sort=True).sort_index()
    rvol_df = pd.concat(rvol_frames, axis=1, sort=True).sort_index()
    # Align indexes (use intersection to avoid NaN rows from mismatched calendars).
    idx = trend_df.index.intersection(mrat_df.index)
    trend_df = trend_df.reindex(idx).fillna(False)
    mrat_df = mrat_df.reindex(idx)
    rvol_df = rvol_df.reindex(idx)

    etf_cols = [slot.etf for slot in slots]
    out_rows: list[dict[str, float]] = []
    for dt in idx:
        scores_row = {c: float(mrat_df.at[dt, c]) if pd.notna(mrat_df.at[dt, c]) else float("nan") for c in etf_cols}
        trend_row = {c: bool(trend_df.at[dt, c]) for c in etf_cols}
        rvol_row = {
            c: float(rvol_df.at[dt, c]) if pd.notna(rvol_df.at[dt, c]) else float("nan")
            for c in etf_cols
        }
        per_idx, risk_off = _progressive_blend_row(
            mrat_scores=scores_row,
            trend_ok=trend_row,
            weight_cfg=weight_cfg,
            rvol_by_key=rvol_row,
        )
        row = {c: per_idx.get(c, 0.0) for c in etf_cols}
        row[RISK_OFF_KEY] = risk_off
        out_rows.append(row)

    weights_piv = pd.DataFrame(out_rows, index=idx, columns=[*etf_cols, RISK_OFF_KEY])
    return IndexAllocationSeries(
        index_weights_piv=weights_piv,
        index_trend_ok=trend_df,
        index_mrat=mrat_df,
    )


# ---------------------------------------------------------------------------
# Live (last bar, no shift)
# ---------------------------------------------------------------------------


def evaluate_index_allocation_live(
    *,
    slots: tuple[IndexSlot, ...],
    regime_ma: int,
    mrat_pair: tuple[int, int],
    weight_cfg: WeightConfig | None,
    granularity: str,
    ohlcv_dir: Path,
    aggregate_to_daily: bool,
    prefer_precomputed_sma: bool = False,
) -> IndexAllocationState:
    """Live snapshot allocation using each ETF's latest available bar (no shift)."""
    if not slots:
        return IndexAllocationState({}, 1.0, {}, {})

    short_w, long_w = int(mrat_pair[0]), int(mrat_pair[1])
    rvol_lb = int(
        weight_cfg.realized_vol_lookback if weight_cfg is not None else DEFAULT_REALIZED_VOL_LOOKBACK
    )

    trend_ok_live: dict[str, bool] = {}
    mrat_live: dict[str, float] = {}
    rvol_live: dict[str, float] = {}
    per_idx_detail: dict[str, dict[str, float]] = {}

    for slot in slots:
        close, sma_r = _load_daily_close_and_sma(
            slot.etf,
            granularity,
            ohlcv_dir,
            int(regime_ma),
            aggregate_to_daily=aggregate_to_daily,
            prefer_precomputed_sma=prefer_precomputed_sma,
        )
        above, close_v, sma_r_v = _above_sma_last_bar(close, sma_r)
        trend_ok_live[slot.etf] = bool(above)

        log_ret = np.log(close / close.shift(1)).dropna()
        if len(log_ret) >= max(2, rvol_lb // 2):
            rvol_tail = float(
                log_ret.tail(max(2, rvol_lb)).std(ddof=1)
            )
            rvol_live[slot.etf] = rvol_tail if np.isfinite(rvol_tail) else float("nan")
        else:
            rvol_live[slot.etf] = float("nan")

        _, sma_s = _load_daily_close_and_sma(
            slot.etf,
            granularity,
            ohlcv_dir,
            short_w,
            aggregate_to_daily=aggregate_to_daily,
            prefer_precomputed_sma=prefer_precomputed_sma,
        )
        _, sma_l = _load_daily_close_and_sma(
            slot.etf,
            granularity,
            ohlcv_dir,
            long_w,
            aggregate_to_daily=aggregate_to_daily,
            prefer_precomputed_sma=prefer_precomputed_sma,
        )
        s_tail = sma_s.dropna()
        l_tail = sma_l.dropna()
        if s_tail.empty or l_tail.empty:
            mrat_live[slot.etf] = float("nan")
        else:
            s_last = float(s_tail.iloc[-1])
            l_last = float(l_tail.iloc[-1])
            mrat_live[slot.etf] = s_last / l_last if l_last > 0 else float("nan")

        per_idx_detail[slot.etf] = {
            "close": float(close_v),
            f"sma_{int(regime_ma)}": float(sma_r_v),
            "mrat": float(mrat_live[slot.etf])
            if np.isfinite(mrat_live[slot.etf])
            else float("nan"),
            "trend_ok": 1.0 if above else 0.0,
        }

    idx_weights, risk_off = _progressive_blend_row(
        mrat_scores=mrat_live,
        trend_ok=trend_ok_live,
        weight_cfg=weight_cfg,
        rvol_by_key=rvol_live,
    )
    return IndexAllocationState(
        index_weights=idx_weights,
        risk_off_weight=risk_off,
        index_trend_ok=trend_ok_live,
        index_mrat=mrat_live,
        per_index_detail=per_idx_detail,
    )


# ---------------------------------------------------------------------------
# Per-stock data-completeness filter (for noisy Russell long tail)
# ---------------------------------------------------------------------------


def filter_universe_by_data_completeness(
    daily_long: pd.DataFrame,
    *,
    window_days: int,
    min_completeness: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop tickers whose bar coverage over the last ``window_days`` calendar
    days is below ``min_completeness`` (as a fraction).

    Returns ``(filtered_df, dropped_tickers)``. Pass-through when
    ``min_completeness <= 0.0`` or ``window_days <= 0``.
    """
    if min_completeness <= 0.0 or int(window_days) <= 0:
        return daily_long, []
    if daily_long.empty:
        return daily_long, []
    df = daily_long
    end = pd.Timestamp(df["date"].max())
    start = end - pd.Timedelta(days=int(window_days))
    recent = df[df["date"] >= start]
    if recent.empty:
        return daily_long, []
    bars_per_tk = recent.groupby("ticker", sort=False)["date"].nunique()
    expected = float(recent["date"].nunique())
    if expected <= 0.0:
        return daily_long, []
    completeness = bars_per_tk / expected
    keep = completeness[completeness >= float(min_completeness)].index
    dropped = [t for t in bars_per_tk.index if t not in keep]
    if not dropped:
        return daily_long, []
    filtered = df[df["ticker"].isin(keep)].copy()
    return filtered, dropped


# ---------------------------------------------------------------------------
# Banner / diagnostics
# ---------------------------------------------------------------------------


def describe_index_config(config_module: object) -> dict[str, Any]:
    """Human-readable summary of the index-allocator configuration."""
    enabled = index_allocator_enabled(config_module)
    slots = resolve_index_slots(config_module)
    wc = resolve_index_weight_config(config_module)
    return {
        "enabled": enabled,
        "regime_ma": resolve_index_regime_ma(config_module),
        "mrat_pair": resolve_index_mrat_pair(config_module),
        "weighting_scheme": wc.scheme if wc else "equal",
        "min_data_completeness": resolve_min_data_completeness(config_module),
        "slots": [
            {
                "etf": s.etf,
                "display_name": s.display_name,
                "raw_universe_size": s.raw_universe_size,
                "exclusive_universe_size": len(s.universe),
            }
            for s in slots
        ],
    }


def format_index_allocation_banner(config_module: object) -> list[str]:
    """Multi-line banner lines for the backtester / live bot startup block."""
    if not index_allocator_enabled(config_module):
        return ["  Index allocator  : disabled (MAD_INDEX_ALLOCATOR_ENABLED=False)"]
    desc = describe_index_config(config_module)
    slot_strs = [
        f"{s['etf']}({s['exclusive_universe_size']} excl. of {s['raw_universe_size']})"
        for s in desc["slots"]
    ]
    total_excl = sum(s["exclusive_universe_size"] for s in desc["slots"])
    lines = [
        f"  Index allocator  : scheme={desc['weighting_scheme']} | MRAT {desc['mrat_pair'][0]}/{desc['mrat_pair'][1]} | regime MA {desc['regime_ma']}",
        f"  Index slots      : {' + '.join(slot_strs)} \u2192 {total_excl} unique stocks",
        f"  Risk-off blend   : progressive (each failing index routes its 1/N slot share to the risk-off sleeve)",
    ]
    mdc = desc["min_data_completeness"]
    if mdc > 0.0:
        lines.append(
            f"  Stock filter     : min data completeness = {mdc:.0%} over MRAT long window"
        )
    return lines
