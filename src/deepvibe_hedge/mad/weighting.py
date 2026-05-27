"""
Pluggable position-weighting schemes for the MRAT cross-sectional book.

All schemes replace the default equal-weight sizing in
``mad.backtester._weights_from_entries``. Each scheme consumes per-date panel state
(entry signal + MRAT / sigma / decile / realized vol) and returns a ``pd.Series`` of
target weights with the same invariants as the equal-weight baseline:

  * longs sum to ``+1.0`` (if any names were elected long)
  * |shorts| sum to ``1.0`` (if any names were elected short)
  * everything else is ``0``

Shape invariants let the backtester / live bot drop new schemes in without changing
the downstream gross-notional or reconcile code paths.

Registered schemes (``MAD_WEIGHTING_SCHEME``)
---------------------------------------------
- ``equal`` (default): ``+1/nL`` / ``-1/nS``. Current production behavior.
- ``mrat_distance``: weight ∝ ``|MRAT - 1|``. Longs further above their long MA get
  more capital; shorts further below get more. "Winners build, losers shrink."
- ``mrat_zscore``: weight ∝ ``|MRAT - 1| / σ_cs`` (σ_cs = cross-sectional MRAT σ on
  that date). Same momentum tilt as ``mrat_distance`` but discounts broad ramps
  where σ_cs is wide (everyone is up, signal is less differentiated).
- ``softmax``: ``softmax((MRAT - 1) / τ)`` with temperature
  ``MAD_WEIGHT_SOFTMAX_TAU`` (default ``0.02``). Smooth, bounded alternative that
  resists any single name dominating.
- ``rank``: weight ∝ decile rank inside the elected book (10 gets more than 9).
  Robust to MRAT tails.
- ``inv_vol``: weight ∝ ``1 / σ_realized`` per name (n = ``MAD_WEIGHT_REALIZED_VOL_LOOKBACK``).
  Classic risk-parity flavor — high-vol names get *less*, not more.
- ``mrat_distance_inv_vol``: hybrid ``|MRAT - 1| / σ_realized``. Momentum tilt
  without letting a single volatile name eat the book.

Common controls (apply after the scheme produces raw weights)
-------------------------------------------------------------
- ``MAD_WEIGHT_MAX_PER_NAME`` (``float | None``): cap per-name |weight| before
  renormalize. ``None`` = no cap. Useful when a book shrinks to 2-3 names.
- ``MAD_WEIGHT_MIN_PER_NAME`` (``float | None``): drop positions below this
  |weight| (after caps) and renormalize the survivors. Reduces tiny fills / fees.
- ``MAD_WEIGHT_EQUAL_BLEND`` (``float`` in ``[0, 1]``, default ``0.0``):
  ``w = (1 - blend) * scheme + blend * equal``. Softens concentration when a
  scheme is confident but you don't fully trust the signal strength.

All per-name caps are applied independently per side (long book vs short book) to
preserve the ``sum(longs) = +1`` / ``sum(|shorts|) = 1`` invariants.

No look-ahead: callers are responsible for passing MRAT / σ / decile / realized vol
values that match the signal's timing. In the backtest, ``entry_signal[t] = signal[t-1]``
so the scheme inputs should also be ``.shift(1)`` relative to the date. In live,
the snapshot uses the last completed bar for both sides already.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

DEFAULT_SCHEME = "equal"
DEFAULT_SOFTMAX_TAU = 0.02
DEFAULT_REALIZED_VOL_LOOKBACK = 20
DEFAULT_EQUAL_BLEND = 0.0


@dataclass(frozen=True)
class WeightInputs:
    """Per-date panel slice used to size weights.

    entry_signal: int Series per ticker, values in {+1, 0, -1}.
    mrat: float Series per ticker (aligned with entry_signal timing).
    sigma: cross-sectional σ(MRAT) for the date; may be NaN when cross-section is thin.
    decile: float Series per ticker, values in ``[1, 10]``.
    realized_vol: optional float Series per ticker (n-day rolling std of daily_ret,
        already shifted to avoid look-ahead). Required for vol-based schemes.
    """

    entry_signal: pd.Series
    mrat: pd.Series
    sigma: float
    decile: pd.Series
    realized_vol: pd.Series | None = None


@dataclass(frozen=True)
class WeightConfig:
    """Static knobs applied to the scheme output."""

    scheme: str = DEFAULT_SCHEME
    max_per_name: float | None = None
    min_per_name: float | None = None
    equal_blend: float = DEFAULT_EQUAL_BLEND
    softmax_tau: float = DEFAULT_SOFTMAX_TAU
    realized_vol_lookback: int = DEFAULT_REALIZED_VOL_LOOKBACK


# ---------------------------------------------------------------------------
# Scheme primitives
#
# Each scheme returns ONE side (longs OR shorts) as non-negative magnitudes that
# sum to 1.0. The dispatcher then signs the short side and concatenates.
# ---------------------------------------------------------------------------


def _equal_magnitudes(tickers: pd.Index) -> pd.Series:
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / n, index=tickers, dtype=float)


def _proportional(raw: pd.Series) -> pd.Series:
    """Non-negative raw scores → weights summing to 1.0. All-zero / all-NaN → equal."""
    v = pd.to_numeric(raw, errors="coerce").astype(float)
    v = v.where(np.isfinite(v), 0.0)
    v = v.clip(lower=0.0)
    s = float(v.sum())
    if s <= 1e-15:
        return _equal_magnitudes(raw.index)
    return v / s


def _mrat_distance_mag(inp: WeightInputs, side: int) -> pd.Series:
    m = inp.mrat.reindex(inp.entry_signal.index).astype(float)
    if side > 0:
        raw = (m - 1.0).clip(lower=0.0)
    else:
        raw = (1.0 - m).clip(lower=0.0)
    return _proportional(raw)


def _mrat_zscore_mag(inp: WeightInputs, side: int) -> pd.Series:
    s = float(inp.sigma) if inp.sigma is not None else float("nan")
    if not np.isfinite(s) or s <= 0:
        return _equal_magnitudes(inp.entry_signal.index)
    m = inp.mrat.reindex(inp.entry_signal.index).astype(float)
    if side > 0:
        raw = ((m - 1.0) / s).clip(lower=0.0)
    else:
        raw = ((1.0 - m) / s).clip(lower=0.0)
    return _proportional(raw)


def _softmax_mag(inp: WeightInputs, side: int, tau: float) -> pd.Series:
    tau = float(tau) if float(tau) > 1e-9 else DEFAULT_SOFTMAX_TAU
    m = inp.mrat.reindex(inp.entry_signal.index).astype(float)
    score = (m - 1.0) / tau if side > 0 else (1.0 - m) / tau
    score = score.where(np.isfinite(score), -np.inf)
    finite = score[np.isfinite(score)]
    if finite.empty:
        return _equal_magnitudes(inp.entry_signal.index)
    shifted = score - float(finite.max())
    ex = np.exp(shifted.to_numpy(dtype=float))
    ex = np.where(np.isfinite(ex), ex, 0.0)
    total = float(ex.sum())
    if total <= 1e-15:
        return _equal_magnitudes(inp.entry_signal.index)
    return pd.Series(ex / total, index=score.index, dtype=float)


def _rank_mag(inp: WeightInputs, side: int) -> pd.Series:
    d = inp.decile.reindex(inp.entry_signal.index).astype(float)
    # Higher decile = stronger for longs; lower decile = stronger for shorts.
    if side > 0:
        raw = d.clip(lower=0.0)
    else:
        raw = (11.0 - d).clip(lower=0.0)
    return _proportional(raw)


def _inv_vol_mag(inp: WeightInputs, side: int) -> pd.Series:
    _ = side  # symmetric; sign is applied by caller
    rv = inp.realized_vol
    if rv is None:
        return _equal_magnitudes(inp.entry_signal.index)
    v = rv.reindex(inp.entry_signal.index).astype(float)
    # Floor tiny / NaN vols to the cross-sectional median so dead names don't hog the book.
    med = float(v.dropna().median()) if v.notna().any() else float("nan")
    floor = med if np.isfinite(med) and med > 0 else 1e-4
    v = v.where(v.notna() & (v > 0), floor)
    raw = 1.0 / v
    return _proportional(raw)


def _mrat_distance_inv_vol_mag(inp: WeightInputs, side: int) -> pd.Series:
    mag = _mrat_distance_mag(inp, side).to_numpy(dtype=float)
    iv = _inv_vol_mag(inp, side).to_numpy(dtype=float)
    idx = _mrat_distance_mag(inp, side).index
    raw = pd.Series(mag * iv, index=idx, dtype=float)
    return _proportional(raw)


SchemeFn = Callable[[WeightInputs, int, WeightConfig], pd.Series]


def _dispatch_scheme(name: str) -> SchemeFn:
    key = str(name or DEFAULT_SCHEME).strip().lower()
    if key == "equal":
        return lambda inp, side, _cfg: _equal_magnitudes(inp.entry_signal.index)
    if key == "mrat_distance":
        return lambda inp, side, _cfg: _mrat_distance_mag(inp, side)
    if key == "mrat_zscore":
        return lambda inp, side, _cfg: _mrat_zscore_mag(inp, side)
    if key == "softmax":
        return lambda inp, side, cfg: _softmax_mag(inp, side, cfg.softmax_tau)
    if key == "rank":
        return lambda inp, side, _cfg: _rank_mag(inp, side)
    if key == "inv_vol":
        return lambda inp, side, _cfg: _inv_vol_mag(inp, side)
    if key == "mrat_distance_inv_vol":
        return lambda inp, side, _cfg: _mrat_distance_inv_vol_mag(inp, side)
    raise ValueError(
        f"Unknown MAD_WEIGHTING_SCHEME={name!r}. "
        "Valid: equal, mrat_distance, mrat_zscore, softmax, rank, inv_vol, mrat_distance_inv_vol."
    )


# ---------------------------------------------------------------------------
# Caps / blend / renormalize
# ---------------------------------------------------------------------------


def _apply_caps_one_side(mag: pd.Series, *, max_per_name: float | None, min_per_name: float | None) -> pd.Series:
    """Cap then drop dust, renormalize after each step. Input sums to 1.0 (or empty)."""
    if mag.empty or float(mag.sum()) <= 1e-15:
        return mag

    if max_per_name is not None and float(max_per_name) > 0:
        cap = float(max_per_name)
        # Iterative cap-and-spread: names above cap get pinned; excess redistributed
        # across names below cap proportional to their current share. Stops when no
        # name exceeds cap or every name is pinned (infeasible cap → return equal-at-cap).
        w = mag.copy().astype(float)
        for _ in range(50):
            over = w > cap
            if not bool(over.any()):
                break
            excess = float((w[over] - cap).sum())
            w[over] = cap
            free = ~over & (w > 0)
            if not bool(free.any()):
                break
            free_sum = float(w[free].sum())
            if free_sum <= 1e-15:
                break
            w[free] = w[free] + excess * (w[free] / free_sum)
        mag = w

    if min_per_name is not None and float(min_per_name) > 0:
        floor = float(min_per_name)
        keep = mag >= floor
        if bool(keep.any()):
            mag = mag.where(keep, 0.0)
            s = float(mag.sum())
            if s > 1e-15:
                mag = mag / s

    s = float(mag.sum())
    if s > 1e-15 and abs(s - 1.0) > 1e-9:
        mag = mag / s
    return mag


def _blend_with_equal(mag: pd.Series, blend: float) -> pd.Series:
    b = float(blend)
    if b <= 0.0 or mag.empty:
        return mag
    b = min(1.0, b)
    eq = _equal_magnitudes(mag.index)
    return (1.0 - b) * mag + b * eq


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_weights(inputs: WeightInputs, cfg: WeightConfig) -> pd.Series:
    """Sign-aware weights. Same index as ``inputs.entry_signal``.

    Returns a Series with longs summing to ``+1.0`` and |shorts| summing to ``1.0``;
    names with ``entry_signal == 0`` are ``0.0``.
    """
    sig = inputs.entry_signal.fillna(0).astype(int)
    longs = sig[sig == 1].index
    shorts = sig[sig == -1].index

    fn = _dispatch_scheme(cfg.scheme)
    out = pd.Series(0.0, index=sig.index, dtype=float)

    if len(longs) > 0:
        inp_l = WeightInputs(
            entry_signal=sig.loc[longs],
            mrat=inputs.mrat,
            sigma=inputs.sigma,
            decile=inputs.decile,
            realized_vol=inputs.realized_vol,
        )
        mag_l = fn(inp_l, 1, cfg)
        mag_l = _blend_with_equal(mag_l, cfg.equal_blend)
        mag_l = _apply_caps_one_side(
            mag_l, max_per_name=cfg.max_per_name, min_per_name=cfg.min_per_name
        )
        out.loc[longs] = mag_l.reindex(longs).fillna(0.0).to_numpy()

    if len(shorts) > 0:
        inp_s = WeightInputs(
            entry_signal=sig.loc[shorts],
            mrat=inputs.mrat,
            sigma=inputs.sigma,
            decile=inputs.decile,
            realized_vol=inputs.realized_vol,
        )
        mag_s = fn(inp_s, -1, cfg)
        mag_s = _blend_with_equal(mag_s, cfg.equal_blend)
        mag_s = _apply_caps_one_side(
            mag_s, max_per_name=cfg.max_per_name, min_per_name=cfg.min_per_name
        )
        out.loc[shorts] = -mag_s.reindex(shorts).fillna(0.0).to_numpy()

    return out


# ---------------------------------------------------------------------------
# Panel helpers for backtest (pre-compute once, slice per date)
# ---------------------------------------------------------------------------


def realized_vol_pivot(
    panel: pd.DataFrame,
    *,
    lookback: int,
    return_col: str = "daily_ret",
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Rolling n-day std of ``daily_ret`` per ticker, shifted by one bar to avoid look-ahead.

    Returned as a pivot (date × ticker) so ``portfolio_path_from_panel`` can slice
    ``loc[d]`` cheaply. Values for ``d`` reflect volatility over the *prior* window.
    """
    n = max(2, int(lookback))
    df = panel[[date_col, ticker_col, return_col]].copy()
    df = df.sort_values([ticker_col, date_col])
    df["rvol"] = (
        df.groupby(ticker_col, sort=False)[return_col]
        .transform(lambda s: s.rolling(window=n, min_periods=n).std(ddof=1).shift(1))
    )
    piv = df.pivot_table(index=date_col, columns=ticker_col, values="rvol", aggfunc="last")
    return piv


def resolve_weight_config(config_module: object | None = None) -> WeightConfig:
    """Build a ``WeightConfig`` from the project config module (or defaults)."""
    cm = config_module
    if cm is None:
        return WeightConfig()
    return WeightConfig(
        scheme=str(getattr(cm, "MAD_WEIGHTING_SCHEME", DEFAULT_SCHEME)).strip().lower(),
        max_per_name=_opt_float(getattr(cm, "MAD_WEIGHT_MAX_PER_NAME", None)),
        min_per_name=_opt_float(getattr(cm, "MAD_WEIGHT_MIN_PER_NAME", None)),
        equal_blend=float(getattr(cm, "MAD_WEIGHT_EQUAL_BLEND", DEFAULT_EQUAL_BLEND) or 0.0),
        softmax_tau=float(
            getattr(cm, "MAD_WEIGHT_SOFTMAX_TAU", DEFAULT_SOFTMAX_TAU) or DEFAULT_SOFTMAX_TAU
        ),
        realized_vol_lookback=int(
            getattr(cm, "MAD_WEIGHT_REALIZED_VOL_LOOKBACK", DEFAULT_REALIZED_VOL_LOOKBACK)
            or DEFAULT_REALIZED_VOL_LOOKBACK
        ),
    )


def _opt_float(x: object) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v > 0 else None
