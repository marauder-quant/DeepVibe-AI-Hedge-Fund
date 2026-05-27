"""
Walk-forward out-of-sample evaluation for MAD / MRAT (fixed IS winner from optimiser DB).

Run:
    PYTHONPATH=src python -m deepvibe_hedge.mad.walkforward_oos
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

from deepvibe_hedge import config
from deepvibe_hedge.breakout_plotting import comparison_stats_df, fig_equity, fig_trades, format_stats
from deepvibe_hedge.mad.backtester import (
    MAD_DEFAULT_MIN_PRICE,
    aggregate_panel_to_daily,
    build_panel_long,
    evaluate_mad,
    mad_calendar_key,
    mad_reference_ticker,
    mad_regime_ticker_symbol,
    mad_universe_tickers,
    resolve_mad_split_plan,
    _bars_per_year_for_mad,
)
from deepvibe_hedge.paths import MAD_DATA_DIR, OHLCV_DIR
from deepvibe_hedge.walkforward_oos_common import normalize_selector, select_oos_splits


def _results_db_path() -> Path:
    ref = mad_reference_ticker()
    return MAD_DATA_DIR / f"{ref}_{config.TARGET_CANDLE_GRANULARITY}_mad_optim.db"


DATASETS_DIR = MAD_DATA_DIR
PORT = int(getattr(config, "MAD_WF_DASHBOARD_PORT", 8064))


def _load_is_winner(optim_split: str) -> dict[str, int | float | str]:
    db_path = _results_db_path()
    if not db_path.exists():
        raise FileNotFoundError(
            f"MAD optimiser DB not found: {db_path}. "
            "Run: PYTHONPATH=src python -m deepvibe_hedge.mad.backtester"
        )
    split_key = normalize_selector(optim_split, allow_avg=True, allow_all=False)
    with sqlite3.connect(db_path) as con:
        if split_key == "avg":
            row = pd.read_sql("SELECT * FROM summary LIMIT 1", con)
            if row.empty:
                raise ValueError(
                    f"No avg summary row in {db_path.name}. Run mad.backtester first."
                )
            r = row.iloc[0]
        else:
            row = pd.read_sql(
                "SELECT * FROM is_split_metrics WHERE split = ? LIMIT 1",
                con,
                params=[int(split_key)],
            )
            if row.empty:
                raise ValueError(f"No is_split_metrics row for split={split_key}")
            r = row.iloc[0]
    return {
        "optim_split": str(split_key),
        "mad_sma_short": int(r["mad_sma_short"]),
        "mad_sma_long": int(r["mad_sma_long"]),
        "mad_exit_ma": int(r.get("mad_exit_ma", 0)),
        "mad_regime_ma": int(r.get("mad_regime_ma", 0)),
        "mad_regime_ticker": str(r.get("mad_regime_ticker", "") or ""),
        "fee_rate": float(r.get("fee_rate", getattr(config, "BACKTEST_FEE_RATE", 0.001))),
    }


def _robustness_sweep_df() -> pd.DataFrame:
    p = _results_db_path()
    if not p.exists():
        return pd.DataFrame()
    with sqlite3.connect(p) as con:
        try:
            return pd.read_sql(
                "SELECT mad_sma_short, mad_sma_long, mad_exit_ma, mad_regime_ma, profit_factor "
                "FROM robustness_sweep",
                con,
            )
        except Exception:
            try:
                return pd.read_sql(
                    "SELECT mad_sma_short, mad_sma_long, mad_exit_ma, profit_factor FROM robustness_sweep",
                    con,
                )
            except Exception:
                try:
                    return pd.read_sql(
                        "SELECT mad_sma_short, mad_sma_long, profit_factor FROM robustness_sweep",
                        con,
                    )
                except Exception:
                    return pd.DataFrame()


def _save_oos_result(
    winner: dict[str, int | float | str],
    selected_is_splits: list[int],
    selected_oos_splits: list[int],
    metrics: dict[str, float],
) -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    ref = mad_reference_ticker()
    stem = f"{ref}_{config.TARGET_CANDLE_GRANULARITY}_mad"
    out_csv = DATASETS_DIR / f"{stem}_walkforward_oos_result.csv"
    row = {
        "reference_ticker": ref,
        "granularity": config.TARGET_CANDLE_GRANULARITY,
        "optim_split": str(winner["optim_split"]),
        "is_splits": ",".join(str(s) for s in selected_is_splits),
        "oos_splits": ",".join(str(s) for s in selected_oos_splits),
        "mad_sma_short": int(winner["mad_sma_short"]),
        "mad_sma_long": int(winner["mad_sma_long"]),
        "mad_exit_ma": int(winner.get("mad_exit_ma", 0)),
        "mad_regime_ma": int(winner.get("mad_regime_ma", 0)),
        "mad_regime_ticker": str(winner.get("mad_regime_ticker", "") or ""),
        "fee_rate": float(winner["fee_rate"]),
        "bars": int(metrics["bars"]),
        "trades": int(metrics["trades"]),
        "rebalance_days": int(metrics.get("rebalance_days", metrics["trades"])),
        "days_with_position": int(metrics.get("days_with_position", 0)),
        "profit_factor": float(metrics["profit_factor"]),
        "sharpe_ratio": float(metrics["sharpe_ratio"]),
        "sortino_ratio": float(metrics["sortino_ratio"]),
        "net_total_return": float(metrics["net_total_return"]),
        "gross_total_log_return": float(metrics["gross_total_log_return"]),
        "net_total_log_return": float(metrics["net_total_log_return"]),
    }
    out_df = pd.DataFrame([row])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    db_path = _results_db_path()
    with sqlite3.connect(db_path) as con:
        out_df.to_sql("walkforward_oos_result", con, if_exists="replace", index=False)
    print(
        "Saved walk-forward OOS result:\n"
        f"  DB table : {db_path.name}::walkforward_oos_result\n"
        f"  CSV      : {out_csv}\n"
    )


def _oos_split_label(oos_splits: list[int]) -> str:
    if len(oos_splits) == 1:
        return f"OOS split {oos_splits[0]}"
    return f"OOS combined splits {','.join(str(s) for s in oos_splits)}"


def _prepare_daily_long() -> tuple[pd.DataFrame, pd.Series]:
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


def _oos_bundle_for_splits(
    daily_long: pd.DataFrame,
    split_by_d: pd.Series,
    winner: dict[str, int | float | str],
    selected_oos_splits: list[int],
    *,
    bpy: float,
    direction: str,
    min_price: float,
    min_hist: int,
) -> dict[str, object]:
    oos_dates = split_by_d[split_by_d.isin(selected_oos_splits)].index
    if len(oos_dates) == 0:
        raise ValueError(f"No dates for OOS splits {selected_oos_splits}")
    end_d = oos_dates.max()
    context = daily_long[daily_long["date"] <= end_d].copy()
    eval_dset = {mad_calendar_key(d) for d in oos_dates}
    gran = str(config.TARGET_CANDLE_GRANULARITY)
    daily_agg = bool(getattr(config, "MAD_AGGREGATE_TO_DAILY", True)) and gran.lower() != "1d"
    rt = str(winner.get("mad_regime_ticker", "") or "").strip().upper() or mad_regime_ticker_symbol()
    metrics, eval_ctx = evaluate_mad(
        context,
        short_w=int(winner["mad_sma_short"]),
        long_w=int(winner["mad_sma_long"]),
        min_price=min_price,
        min_history=min_hist,
        fee_rate=float(winner["fee_rate"]),
        direction_mode=direction,
        eval_dates=eval_dset,
        bars_per_year_local=bpy,
        exit_ma_period=int(winner.get("mad_exit_ma", 0)),
        regime_ma_period=int(winner.get("mad_regime_ma", 0)),
        regime_ticker=rt,
        granularity=gran,
        aggregate_to_daily=daily_agg,
    )
    eval_df = eval_ctx.loc[[i for i in eval_ctx.index if mad_calendar_key(i) in eval_dset]].copy()
    ref = mad_reference_ticker()
    ref_oos = daily_long[
        (daily_long["ticker"] == ref) & daily_long["date"].map(mad_calendar_key).isin(eval_dset)
    ].copy()
    ref_oos = ref_oos.set_index("date").sort_index()
    oos_ohlcv = pd.DataFrame(
        {
            "open": ref_oos["open"],
            "high": ref_oos["close"],
            "low": ref_oos["close"],
            "close": ref_oos["close"],
            "volume": 0.0,
            "split": ref_oos["split"],
        }
    )
    stats_df = comparison_stats_df(metrics, eval_df, bpy)
    return {
        "oos_ohlcv": oos_ohlcv,
        "eval_df": eval_df,
        "metrics": metrics,
        "stats_df": stats_df,
        "oos_splits": list(selected_oos_splits),
    }


def _empty_candle_fig(label: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=f"{label} | MAD is a cross-sectional panel — reference close shown in equity curve only",
        template="plotly_dark",
        height=260,
    )
    return fig


def _oos_bundle_index_for_cli(selected: list[int], reserved: list[int]) -> int:
    if len(selected) == len(reserved) and sorted(selected) == sorted(reserved):
        return 0
    if len(selected) == 1 and selected[0] in reserved:
        return 1 + reserved.index(selected[0])
    return 0


def build_app(
    *,
    winner: dict[str, int | float | str],
    is_splits: list[int],
    reserved_oos_splits: list[int],
    bundles: list[dict[str, object]],
    sweep_df: pd.DataFrame,
    initial_idx: int,
) -> Dash:
    if not bundles:
        raise ValueError("build_app requires at least one OOS bundle.")
    initial_idx = int(max(0, min(initial_idx, len(bundles) - 1)))
    marks: dict[int, dict[str, object]] = {0: {"label": "All OOS", "style": {"color": "#eee"}}}
    for i, sid in enumerate(reserved_oos_splits, start=1):
        marks[i] = {"label": str(sid), "style": {"color": "#eee"}}

    sh, lo = int(winner["mad_sma_short"]), int(winner["mad_sma_long"])
    ex = int(winner.get("mad_exit_ma", 0))
    reg = int(winner.get("mad_regime_ma", 0))
    rtk = str(winner.get("mad_regime_ticker", "") or "")

    app = Dash(__name__)
    app.layout = html.Div(
        style={"backgroundColor": "#111", "color": "#eee", "fontFamily": "monospace", "padding": "20px"},
        children=[
            html.H2(
                f"MAD Walk-Forward OOS — {mad_reference_ticker()} {config.TARGET_CANDLE_GRANULARITY}",
                style={"textAlign": "center", "marginBottom": "4px"},
            ),
            html.P(
                f"IS splits: {','.join(str(s) for s in is_splits)} | "
                f"Reserved OOS: {','.join(str(s) for s in reserved_oos_splits)}",
                style={"textAlign": "center", "color": "#aaa", "marginTop": 0},
            ),
            html.Div(
                [
                    html.Span("OOS view: ", style={"color": "#aaa", "marginRight": "8px"}),
                    dcc.Slider(
                        id="wf-mad-oos-slider",
                        min=0,
                        max=len(bundles) - 1,
                        step=1,
                        value=initial_idx,
                        marks=marks,
                    ),
                ],
                style={"maxWidth": "920px", "margin": "0 auto 16px"},
            ),
            html.P(id="wf-mad-summary-p", style={"textAlign": "center", "color": "#aaa"}),
            html.Div(id="wf-mad-subtitle", style={"textAlign": "center", "color": "#aaa", "marginBottom": "14px"}),
            dcc.Graph(id="wf-mad-candle"),
            dcc.Graph(id="wf-mad-equity"),
            dcc.Graph(id="wf-mad-trades"),
            dcc.Graph(id="wf-mad-robust"),
            html.H3(id="wf-mad-stats-title", style={"marginTop": "24px"}),
            dash_table.DataTable(
                id="wf-mad-stats",
                columns=[{"name": "Metric", "id": "Metric"}, {"name": "Value", "id": "Value"}],
                data=[],
                style_table={"maxWidth": "720px"},
                style_cell={
                    "backgroundColor": "#1a1a1a",
                    "color": "#eee",
                    "border": "1px solid #333",
                    "padding": "6px",
                },
                style_header={"backgroundColor": "#2a2a2a", "fontWeight": "bold"},
            ),
        ],
    )

    rb = sweep_df if not sweep_df.empty else pd.DataFrame(
        [
            {
                "mad_sma_short": sh,
                "mad_sma_long": lo,
                "mad_exit_ma": ex,
                "mad_regime_ma": reg,
                "profit_factor": float("nan"),
            }
        ],
    )

    @app.callback(
        [
            Output("wf-mad-summary-p", "children"),
            Output("wf-mad-subtitle", "children"),
            Output("wf-mad-candle", "figure"),
            Output("wf-mad-equity", "figure"),
            Output("wf-mad-trades", "figure"),
            Output("wf-mad-robust", "figure"),
            Output("wf-mad-stats-title", "children"),
            Output("wf-mad-stats", "data"),
        ],
        Input("wf-mad-oos-slider", "value"),
    )
    def _sync(idx: int | None) -> tuple:
        i = int(idx) if idx is not None else initial_idx
        i = max(0, min(i, len(bundles) - 1))
        b = bundles[i]
        oos_ohlcv = b["oos_ohlcv"]  # type: ignore[assignment]
        eval_df = b["eval_df"]  # type: ignore[assignment]
        metrics = b["metrics"]  # type: ignore[assignment]
        stats_df = b["stats_df"]  # type: ignore[assignment]
        oos_splits = b["oos_splits"]  # type: ignore[assignment]
        split_label = _oos_split_label(oos_splits)
        oos_start = str(oos_ohlcv.index.min().date())
        oos_end = str(oos_ohlcv.index.max().date())
        summary = f"OOS: {oos_start} -> {oos_end} ({len(oos_ohlcv):,} days) | {split_label}"
        ex_part = f" exit_MA={ex}" if ex else ""
        reg_part = f" regime_{rtk}_MA={reg}" if reg and rtk else (f" regime_MA={reg}" if reg else "")
        subtitle = (
            f"IS winner split={winner['optim_split']} | MRAT {sh}/{lo}{ex_part}{reg_part} | "
            f"PF={float(metrics['profit_factor']):.4f} | Sharpe={float(metrics['sharpe_ratio']):.4f}"
        )
        rbx = rb["mad_sma_short"].astype(str) + "/" + rb["mad_sma_long"].astype(str)
        if "mad_exit_ma" in rb.columns:
            rbx = rbx + "/ex" + rb["mad_exit_ma"].astype(str)
        if "mad_regime_ma" in rb.columns:
            rbx = rbx + "/r" + rb["mad_regime_ma"].astype(str)
        rb_fig = go.Figure(go.Bar(x=rbx, y=rb["profit_factor"]))
        rb_fig.update_layout(title="IS robustness sweep (PF)", template="plotly_dark", height=300)
        return (
            summary,
            subtitle,
            _empty_candle_fig(split_label),
            fig_equity(eval_df, split_label, float(metrics["profit_factor"]), strategy_curve_name="MAD"),
            fig_trades(eval_df, split_label),
            rb_fig,
            f"Portfolio stats — {split_label}",
            format_stats(stats_df).to_dict("records"),
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optim-split",
        type=str,
        default=str(getattr(config, "MAD_WF_OPTIM_SPLIT", "avg")),
    )
    parser.add_argument(
        "--oos-split",
        type=str,
        default=str(getattr(config, "MAD_WF_OOS_SPLIT", "all")),
    )
    parser.add_argument("--fee-rate", type=float, default=None)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-dashboard", action="store_true")
    args = parser.parse_args()

    daily_long, split_by_d = _prepare_daily_long()
    is_splits, reserved_oos = resolve_mad_split_plan(daily_long)
    selected_oos = select_oos_splits(reserved_oos, args.oos_split)
    winner = _load_is_winner(args.optim_split)
    if args.fee_rate is not None:
        winner["fee_rate"] = float(args.fee_rate)

    gran = str(config.TARGET_CANDLE_GRANULARITY)
    bpy = _bars_per_year_for_mad(gran, bool(getattr(config, "MAD_AGGREGATE_TO_DAILY", True)))
    direction = getattr(config, "MAD_DIRECTION_MODE", "both")
    min_price = float(MAD_DEFAULT_MIN_PRICE)
    min_hist = int(getattr(config, "MAD_MIN_HISTORY_BARS", 252))

    bundles: list[dict[str, object]] = [
        _oos_bundle_for_splits(
            daily_long,
            split_by_d,
            winner,
            reserved_oos,
            bpy=bpy,
            direction=direction,
            min_price=min_price,
            min_hist=min_hist,
        )
    ]
    for sid in reserved_oos:
        bundles.append(
            _oos_bundle_for_splits(
                daily_long,
                split_by_d,
                winner,
                [sid],
                bpy=bpy,
                direction=direction,
                min_price=min_price,
                min_hist=min_hist,
            )
        )

    save_idx = _oos_bundle_index_for_cli(selected_oos, reserved_oos)
    b_save = bundles[save_idx]
    metrics = b_save["metrics"]  # type: ignore[assignment]
    oos_df_save = b_save["oos_ohlcv"]  # type: ignore[assignment]

    print(
        f"\nMAD walk-forward OOS | {mad_reference_ticker()} {gran}\n"
        f"  IS splits       : {is_splits}\n"
        f"  Reserved OOS    : {reserved_oos}\n"
        f"  Tested OOS      : {selected_oos}\n"
        f"  IS winner       : split={winner['optim_split']} | MRAT {winner['mad_sma_short']}/"
        f"{winner['mad_sma_long']} exit_MA={winner.get('mad_exit_ma', 0)} "
        f"regime_MA={winner.get('mad_regime_ma', 0)} {winner.get('mad_regime_ticker', '')}\n"
        f"  OOS days        : {len(oos_df_save):,}\n"
        f"  OOS Profit Factor: {float(metrics['profit_factor']):.4f}\n"
        f"  OOS Sharpe      : {float(metrics['sharpe_ratio']):.4f}\n"
    )

    _save_oos_result(winner, is_splits, selected_oos, metrics)

    if not args.no_dashboard:
        sweep_df = _robustness_sweep_df()
        app = build_app(
            winner=winner,
            is_splits=is_splits,
            reserved_oos_splits=reserved_oos,
            bundles=bundles,
            sweep_df=sweep_df,
            initial_idx=save_idx,
        )
        print(f"Dashboard -> http://localhost:{args.port}\n")
        app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
