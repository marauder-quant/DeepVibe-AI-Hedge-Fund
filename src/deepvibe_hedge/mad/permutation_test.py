"""
Phase 1 in-sample permutation test for MAD / MRAT (block shuffle of portfolio bar log returns).

Uses the realised in-sample MAD portfolio net log returns; the null shuffles that series with
block bootstrap. This is a portfolio-level randomisation check (not a full panel re-draw).

Run:
    PYTHONPATH=src python -m deepvibe_hedge.mad.permutation_test
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html

from deepvibe_hedge import config
from deepvibe_hedge.mad.backtester import (
    MAD_DEFAULT_MIN_PRICE,
    aggregate_panel_to_daily,
    build_panel_long,
    evaluate_mad,
    mad_calendar_key,
    mad_reference_ticker,
    mad_universe_tickers,
    resolve_mad_split_plan,
    _bars_per_year_for_mad,
)
from deepvibe_hedge.paths import MAD_DATA_DIR, OHLCV_DIR
from deepvibe_hedge.permutation_helpers import (
    _available_best_rows,
    _normalize_optim_split,
    _verdict,
    block_shuffle,
)
def _results_db_path() -> Path:
    ref = mad_reference_ticker()
    return MAD_DATA_DIR / f"{ref}_{config.TARGET_CANDLE_GRANULARITY}_mad_optim.db"


def load_best_strategy(optim_split: str) -> dict:
    split_key = _normalize_optim_split(optim_split)
    db_path = _results_db_path()
    if not db_path.exists():
        raise FileNotFoundError(
            f"MAD optimizer DB not found: {db_path}. "
            "Run: PYTHONPATH=src python -m deepvibe_hedge.mad.backtester"
        )
    if split_key == "avg":
        with sqlite3.connect(db_path) as con:
            row = pd.read_sql("SELECT * FROM summary LIMIT 1", con)
        if not row.empty:
            r = row.iloc[0]
            return {
                "mode": "mad",
                "mad_sma_short": int(r["mad_sma_short"]),
                "mad_sma_long": int(r["mad_sma_long"]),
                "mad_exit_ma": int(r.get("mad_exit_ma", 0)),
                "mad_regime_ma": int(r.get("mad_regime_ma", 0)),
                "mad_regime_ticker": str(r.get("mad_regime_ticker", "") or ""),
                "fee_rate": float(r.get("fee_rate", getattr(config, "BACKTEST_FEE_RATE", 0.001))),
                "reference_profit_factor": float(r["profit_factor"]) if "profit_factor" in row.columns else np.nan,
                "source_db": str(db_path.name),
                "optim_split": split_key,
                "optim_source": "mad",
            }
        raise ValueError(f"No summary row in {db_path.name}.")
    with sqlite3.connect(db_path) as con:
        row = pd.read_sql(
            "SELECT * FROM is_split_metrics WHERE split = ? LIMIT 1",
            con,
            params=[int(split_key)],
        )
    if not row.empty:
        r = row.iloc[0]
        return {
            "mode": "mad",
            "mad_sma_short": int(r["mad_sma_short"]),
            "mad_sma_long": int(r["mad_sma_long"]),
            "mad_exit_ma": int(r.get("mad_exit_ma", 0)),
            "mad_regime_ma": int(r.get("mad_regime_ma", 0)),
            "mad_regime_ticker": str(r.get("mad_regime_ticker", "") or ""),
            "fee_rate": float(r.get("fee_rate", getattr(config, "BACKTEST_FEE_RATE", 0.001))),
            "reference_profit_factor": float(r["profit_factor"]) if "profit_factor" in row.columns else np.nan,
            "source_db": str(db_path.name),
            "optim_split": split_key,
            "optim_source": "mad",
        }
    avail = _available_best_rows(db_path, columns="split", table="is_split_metrics")
    available = sorted(str(x) for x in avail["split"].astype(str).unique().tolist()) if not avail.empty else []
    raise ValueError(
        f"Requested split '{split_key}' not found in is_split_metrics. Available: {available or 'none'}"
    )


def _pf_log_returns(vals: np.ndarray) -> float:
    wins = float(vals[vals > 0].sum())
    losses = float(abs(vals[vals < 0].sum()))
    if losses <= 0.0:
        return np.inf if wins > 0.0 else np.nan
    return wins / losses


def _load_is_dates_and_returns(strategy: dict) -> tuple[pd.Series, list[int]]:
    daily_long, _ = _prepare_daily_panel()
    if bool(getattr(config, "MAD_EVAL_ALL_SPLITS", False)):
        mask = daily_long["split"] > 0
        is_dates = daily_long.loc[mask, "date"].unique()
        is_splits = sorted({int(x) for x in daily_long.loc[mask, "split"].unique()})
    else:
        is_splits, _ = resolve_mad_split_plan(daily_long)
        split_key = str(strategy.get("optim_split", "avg"))
        if split_key != "avg":
            is_splits = [int(split_key)]
        is_dates = daily_long.loc[daily_long["split"].isin(is_splits), "date"].unique()
    eval_dset = {mad_calendar_key(d) for d in is_dates}
    end_d = pd.Timestamp(max(is_dates))
    context = daily_long[daily_long["date"] <= end_d].copy()
    gran = str(config.TARGET_CANDLE_GRANULARITY)
    daily_agg = bool(getattr(config, "MAD_AGGREGATE_TO_DAILY", True)) and gran.lower() != "1d"
    bpy = _bars_per_year_for_mad(gran, daily_agg or gran.lower() == "1d")
    direction = getattr(config, "MAD_DIRECTION_MODE", "both")
    min_price = float(MAD_DEFAULT_MIN_PRICE)
    min_hist = int(getattr(config, "MAD_MIN_HISTORY_BARS", 252))
    rt = str(strategy.get("mad_regime_ticker", "") or "").strip().upper() or None

    _, eval_ctx = evaluate_mad(
        context,
        short_w=int(strategy["mad_sma_short"]),
        long_w=int(strategy["mad_sma_long"]),
        min_price=min_price,
        min_history=min_hist,
        fee_rate=float(strategy["fee_rate"]),
        direction_mode=direction,
        eval_dates=eval_dset,
        bars_per_year_local=bpy,
        exit_ma_period=int(strategy.get("mad_exit_ma", 0)),
        regime_ma_period=int(strategy.get("mad_regime_ma", 0)),
        regime_ticker=rt,
        granularity=gran,
        aggregate_to_daily=daily_agg,
    )
    sub = eval_ctx.loc[[i for i in eval_ctx.index if mad_calendar_key(i) in eval_dset]]
    ser = sub["net_log_return"].dropna()
    return ser, is_splits


def _prepare_daily_panel() -> tuple[pd.DataFrame, pd.Series]:
    ref = mad_reference_ticker()
    universe = mad_universe_tickers()
    gran = str(config.TARGET_CANDLE_GRANULARITY)
    panel = build_panel_long(universe, gran, ref, OHLCV_DIR)
    if bool(getattr(config, "MAD_AGGREGATE_TO_DAILY", True)):
        daily_long = aggregate_panel_to_daily(panel)
    else:
        dl = panel.copy()
        dl["date"] = pd.to_datetime(dl["timestamp"], utc=True).dt.normalize()
        daily_long = dl.drop(columns=["timestamp"], errors="ignore")
    split_by_d = daily_long.groupby("date", sort=True)["split"].last()
    return daily_long, split_by_d


def run_phase1_mad(
    is_returns: np.ndarray,
    observed: float,
    n_perms: int,
    block_size: int,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    np.random.seed(seed)
    print(f"Running {n_perms} block-shuffle permutations (block = {block_size}) …")
    null_dist = np.empty(n_perms)
    t0 = time.time()
    for i in range(n_perms):
        shuffled = block_shuffle(is_returns.astype(float, copy=False), block_size)
        null_dist[i] = _pf_log_returns(shuffled)
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            per = elapsed / (i + 1)
            print(
                f"  [{i+1:>4}/{n_perms}] PF={null_dist[i]:.3f} | "
                f"median={np.nanmedian(null_dist[: i + 1]):.3f} | ETA {per * (n_perms - i - 1) / 60:.1f} min"
            )
    p_value = float(np.nanmean(null_dist >= observed))
    print(f"\nDone — p-value = {p_value:.4f}\n")
    return observed, p_value, null_dist


def _build_dashboard(
    observed: float,
    null_dist: np.ndarray,
    p_value: float,
    strategy: dict,
    alpha: float,
    n_perms: int,
    block_size: int,
    n_is_bars: int,
    is_splits: list[int],
) -> Dash:
    finite = null_dist[np.isfinite(null_dist)]
    null_95th = float(np.nanpercentile(finite, (1 - alpha) * 100))
    verdict_str, verdict_color = _verdict(p_value, alpha)
    combo = f"MRAT {strategy['mad_sma_short']}/{strategy['mad_sma_long']}, fee={float(strategy['fee_rate']):.4f}"

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=finite,
            nbinsx=max(50, n_perms // 100),
            name="Null — block-shuffled IS portfolio log returns",
            marker_color="#4a9eff",
            opacity=0.75,
        )
    )
    fig.add_vline(x=null_95th, line_color="#f9a825", line_dash="dash", annotation_text=f"{int((1-alpha)*100)}th pct")
    fig.add_vline(x=observed, line_color="#ff4d4d", line_width=2, annotation_text=f"Observed PF = {observed:.3f}")
    fig.update_layout(
        title=f"Phase 1 — MAD permutation | {mad_reference_ticker()} {config.TARGET_CANDLE_GRANULARITY}",
        xaxis_title="Profit factor (bar log returns)",
        template="plotly_dark",
        height=440,
    )

    app = Dash(__name__)
    app.layout = html.Div(
        style={"backgroundColor": "#111", "color": "#eee", "fontFamily": "monospace", "padding": "24px"},
        children=[
            html.H2("Phase 1 — MAD / MRAT permutation", style={"textAlign": "center"}),
            html.P(
                f"{mad_reference_ticker()} {config.TARGET_CANDLE_GRANULARITY} | IS splits {is_splits} | "
                f"{n_is_bars:,} daily bars | {combo} | block={block_size} | perms={n_perms} | α={alpha}",
                style={"textAlign": "center", "color": "#888", "fontSize": "12px"},
            ),
            dcc.Graph(figure=fig),
            html.Pre(verdict_str, style={"color": verdict_color, "fontSize": "15px", "fontWeight": "bold"}),
        ],
    )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 permutation — MAD portfolio IS returns.")
    parser.add_argument("--n-perms", type=int, default=int(getattr(config, "MAD_PERM_N", 10_000)))
    parser.add_argument("--block-size", type=int, default=int(getattr(config, "MAD_PERM_BLOCK_SIZE", 5)))
    parser.add_argument("--alpha", type=float, default=float(getattr(config, "MAD_PERM_ALPHA", 0.05)))
    parser.add_argument("--port", type=int, default=int(getattr(config, "MAD_PERM_PORT", 8065)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optim-split", type=str, default=str(getattr(config, "MAD_PERM_OPTIM_SPLIT", "avg")))
    args = parser.parse_args()

    strategy = load_best_strategy(args.optim_split)
    is_ser, is_splits = _load_is_dates_and_returns(strategy)
    vals = is_ser.to_numpy(dtype=float, copy=False)
    observed = _pf_log_returns(vals)
    ref_pf = float(strategy.get("reference_profit_factor", np.nan))
    if np.isfinite(ref_pf):
        print(f"Observed PF (recomputed): {observed:.6f} | optimizer reference: {ref_pf:.6f}")
        observed = ref_pf
    else:
        print(f"Observed PF: {observed:.6f}")

    print(
        f"\nPhase 1 MAD permutation\n"
        f"  IS splits   : {is_splits}\n"
        f"  IS bars     : {len(vals):,}\n"
        f"  Statistic   : profit factor on net log returns\n"
        f"  Null        : block shuffle of the same IS return series\n"
    )

    _, p_value, null_dist = run_phase1_mad(vals, observed, args.n_perms, args.block_size, seed=args.seed)
    print(_verdict(p_value, args.alpha)[0])

    app = _build_dashboard(
        observed,
        null_dist,
        p_value,
        strategy,
        args.alpha,
        args.n_perms,
        args.block_size,
        len(vals),
        is_splits,
    )
    print(f"\nDashboard → http://localhost:{args.port}\n")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
