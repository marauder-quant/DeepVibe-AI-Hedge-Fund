"""
Plotly/Dash helpers and stat tables shared by modified Donchaian breakout tooling.

Extracted from the former classic Donchian backtester so that package could be removed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Dash slider: synthetic key for the cross-split average view
AVG_KEY = -1
AVG_SLIDER_VAL = 0


def bars_per_year(granularity: str) -> float:
    """Approximate bars per calendar year for annualising Sharpe/Sortino."""
    return {
        "1m": 252.0 * 390.0,
        "5m": 252.0 * 78.0,
        "15m": 252.0 * 26.0,
        "1h": 252.0 * 6.5,
        "4h": 252.0 * 2.0,
        "1d": 252.0,
        "1w": 52.0,
        "1mo": 12.0,
    }.get(granularity, 252.0)


def build_robustness_insights(sweep_df: pd.DataFrame, step: int) -> str:
    pf = sweep_df["profit_factor"].replace([np.inf, -np.inf], np.nan).dropna()
    if pf.empty:
        return "No finite Profit Factor values were produced in the sweep."

    best_idx = int(sweep_df["profit_factor"].replace(-np.inf, np.nan).idxmax())
    best_row = sweep_df.loc[best_idx]
    best_pf = float(best_row["profit_factor"])
    best_upper = int(best_row["upper_lookback"])
    best_lower = int(best_row["lower_lookback"])

    median_pf = float(pf.median())
    q25 = float(pf.quantile(0.25))
    q75 = float(pf.quantile(0.75))
    share_gt_1 = float((pf > 1.0).mean() * 100.0)

    return (
        f"Best Profit Factor = {best_pf:.4f} at upper={best_upper}, lower={best_lower}. "
        f"Median PF across sweep = {median_pf:.4f} (IQR {q25:.4f}–{q75:.4f}). "
        f"{share_gt_1:.1f}% of tested parameter pairs have PF > 1.0."
    )


def build_robustness_insights_sma(sweep_df: pd.DataFrame, step: int) -> str:
    """One-dimensional SMA period sweep (column ``sma_period``)."""
    if sweep_df.empty or "sma_period" not in sweep_df.columns:
        return "No SMA sweep results to summarise."
    pf = sweep_df["profit_factor"].replace([np.inf, -np.inf], np.nan).dropna()
    if pf.empty:
        return "No finite Profit Factor values were produced in the sweep."

    best_idx = int(sweep_df["profit_factor"].replace(-np.inf, np.nan).idxmax())
    best_row = sweep_df.loc[best_idx]
    best_pf = float(best_row["profit_factor"])
    best_n = int(best_row["sma_period"])

    median_pf = float(pf.median())
    q25 = float(pf.quantile(0.25))
    q75 = float(pf.quantile(0.75))
    share_gt_1 = float((pf > 1.0).mean() * 100.0)

    return (
        f"Best Profit Factor = {best_pf:.4f} at SMA period={best_n}. "
        f"Median PF across sweep = {median_pf:.4f} (IQR {q25:.4f}–{q75:.4f}). "
        f"{share_gt_1:.1f}% of tested periods (step {step}) have PF > 1.0."
    )


def _extract_trade_returns(eval_df: pd.DataFrame) -> pd.Series:
    valid = eval_df["next_log_return"].notna()
    seg = eval_df.loc[valid, ["flip", "net_log_return"]].copy()
    if seg.empty:
        return pd.Series(dtype=float)
    seg["trade_id"] = seg["flip"].astype(int).cumsum()
    trade_log = seg.groupby("trade_id", sort=True)["net_log_return"].sum()
    trade_ret = np.expm1(trade_log)
    trade_ret.name = "trade_return"
    return trade_ret


def _annualized_sharpe(log_returns: pd.Series, bars_per_y: float) -> float:
    vals = log_returns.dropna().to_numpy(dtype=float)
    if len(vals) < 2:
        return np.nan
    std = float(np.std(vals, ddof=1))
    if std <= 0.0:
        return np.nan
    return float(np.mean(vals) / std * np.sqrt(bars_per_y))


def _annualized_sortino(log_returns: pd.Series, bars_per_y: float) -> float:
    vals = log_returns.dropna().to_numpy(dtype=float)
    if len(vals) < 2:
        return np.nan
    downside = vals[vals < 0]
    if len(downside) < 2:
        return np.nan
    downside_std = float(np.std(downside, ddof=1))
    if downside_std <= 0.0:
        return np.nan
    return float(np.mean(vals) / downside_std * np.sqrt(bars_per_y))


def _max_drawdown_from_log(log_returns: pd.Series) -> float:
    eq = np.exp(log_returns.fillna(0.0).cumsum())
    dd = eq / eq.cummax() - 1.0
    return float(dd.min()) if len(dd) else np.nan


def comparison_stats_df(
    metrics: dict[str, float],
    eval_df: pd.DataFrame,
    bars_per_y: float,
) -> pd.DataFrame:
    valid = eval_df["next_log_return"].notna()
    strat_log = eval_df.loc[valid, "net_log_return"]
    bh_log = eval_df.loc[valid, "next_log_return"]
    bars = int(valid.sum())
    years = bars / bars_per_y if bars_per_y > 0 else np.nan

    strat_total = float(np.expm1(strat_log.sum())) if bars else np.nan
    bh_total = float(np.expm1(bh_log.sum())) if bars else np.nan
    strat_cagr = float(np.exp(strat_log.sum() / years) - 1.0) if bars and years > 0 else np.nan
    bh_cagr = float(np.exp(bh_log.sum() / years) - 1.0) if bars and years > 0 else np.nan
    strat_vol = float(np.std(strat_log.to_numpy(dtype=float), ddof=1) * np.sqrt(bars_per_y)) if bars > 1 else np.nan
    bh_vol = float(np.std(bh_log.to_numpy(dtype=float), ddof=1) * np.sqrt(bars_per_y)) if bars > 1 else np.nan
    strat_mdd = _max_drawdown_from_log(strat_log)
    bh_mdd = _max_drawdown_from_log(bh_log)
    trade_returns = _extract_trade_returns(eval_df)
    win_rate = float((trade_returns > 0).mean()) if len(trade_returns) else np.nan

    if "mad_sma_short" in metrics:
        param_rows = [
            {"Metric": "MAD SMA short", "Value": metrics["mad_sma_short"]},
            {"Metric": "MAD SMA long", "Value": metrics["mad_sma_long"]},
            {
                "Metric": "MAD exit SMA (0=off)",
                "Value": metrics.get("mad_exit_ma", 0),
            },
            {
                "Metric": "MAD regime ticker (blank=off)",
                "Value": metrics.get("mad_regime_ticker", "") or "—",
            },
            {
                "Metric": "MAD regime SMA (0=off; ETF below ⇒ cash)",
                "Value": metrics.get("mad_regime_ma", 0),
            },
        ]
    elif "sma_period" in metrics:
        param_rows = [
            {"Metric": "SMA period", "Value": metrics["sma_period"]},
        ]
    else:
        param_rows = [
            {"Metric": "Upper Lookback", "Value": metrics["upper_lookback"]},
            {"Metric": "Lower Lookback", "Value": metrics["lower_lookback"]},
        ]
        if "trend_sma_bars" in metrics:
            param_rows.append({"Metric": "Regime SMA (bars)", "Value": metrics["trend_sma_bars"]})
        elif "trend_sma_days" in metrics:
            param_rows.append({"Metric": "Regime SMA (bars)", "Value": metrics["trend_sma_days"]})

    mad_activity: list[dict[str, object]] = []
    if "mad_sma_short" in metrics:
        mad_activity.append(
            {
                "Metric": "Rebalance days (|weights| changed)",
                "Value": metrics.get("rebalance_days", metrics.get("trades", np.nan)),
            }
        )
        if "days_with_position" in metrics:
            mad_activity.append(
                {"Metric": "Days with position (|w|>0)", "Value": metrics["days_with_position"]},
            )
        if "mad_diag_eval_days" in metrics:
            mad_activity.extend(
                [
                    {
                        "Metric": "Diag: % days with valid cross-section (σ+decile)",
                        "Value": metrics.get("mad_diag_pct_days_valid_cross_section", np.nan),
                    },
                    {
                        "Metric": "Diag: % eval days with any top-decile name",
                        "Value": metrics.get("mad_diag_pct_days_any_top_decile", np.nan),
                    },
                    {
                        "Metric": "Diag: % eval days passing long gate (decile+MRAT)",
                        "Value": metrics.get("mad_diag_pct_days_pass_long_gate", np.nan),
                    },
                    {
                        "Metric": "Diag: % of top-decile days with no long (σ gate)",
                        "Value": metrics.get("mad_diag_pct_top_decile_days_no_long", np.nan),
                    },
                    {
                        "Metric": "Diag: mean long names on days with a long",
                        "Value": metrics.get("mad_diag_mean_long_names_when_long", np.nan),
                    },
                ]
            )

    trade_rows = (
        []
        if "mad_sma_short" in metrics
        else [{"Metric": "Trades", "Value": metrics["trades"]}]
    )

    rows = [
        *param_rows,
        *mad_activity,
        {"Metric": "Bars", "Value": metrics["bars"]},
        *trade_rows,
        {"Metric": "Trade Win Rate", "Value": win_rate},
        {"Metric": "Profit Factor", "Value": metrics["profit_factor"]},
        {"Metric": "Sharpe Ratio", "Value": metrics["sharpe_ratio"]},
        {"Metric": "Sortino Ratio", "Value": metrics["sortino_ratio"]},
        {"Metric": "Strategy Total Return", "Value": strat_total},
        {"Metric": "Buy & Hold Total Return", "Value": bh_total},
        {"Metric": "Strategy CAGR", "Value": strat_cagr},
        {"Metric": "Buy & Hold CAGR", "Value": bh_cagr},
        {"Metric": "Strategy Max Drawdown", "Value": strat_mdd},
        {"Metric": "Buy & Hold Max Drawdown", "Value": bh_mdd},
        {"Metric": "Strategy Volatility", "Value": strat_vol},
        {"Metric": "Buy & Hold Volatility", "Value": bh_vol},
        {"Metric": "Strategy Sharpe", "Value": _annualized_sharpe(strat_log, bars_per_y)},
        {"Metric": "Buy & Hold Sharpe", "Value": _annualized_sharpe(bh_log, bars_per_y)},
        {"Metric": "Strategy Sortino", "Value": _annualized_sortino(strat_log, bars_per_y)},
        {"Metric": "Buy & Hold Sortino", "Value": _annualized_sortino(bh_log, bars_per_y)},
        {"Metric": "Gross Total Log Return", "Value": metrics["gross_total_log_return"]},
        {"Metric": "Net Total Log Return", "Value": metrics["net_total_log_return"]},
    ]
    return pd.DataFrame(rows)


def format_stats(stats_df: pd.DataFrame) -> pd.DataFrame:
    out = stats_df.copy()

    def _fmt(metric: str, value) -> str:
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            val = float(value)
            if not np.isfinite(val):
                if str(metric).strip().lower() == "profit factor":
                    return "N/A (no losing returns)"
                return "N/A"
            if str(metric).strip().lower() == "net total return":
                return f"{val:.4%}"
            if "return" in str(metric).strip().lower() or "cagr" in str(metric).strip().lower():
                return f"{val:.4%}"
            if "drawdown" in str(metric).strip().lower():
                return f"{val:.4%}"
            if "volatility" in str(metric).strip().lower():
                return f"{val:.4%}"
            if "win rate" in str(metric).strip().lower():
                return f"{val:.2%}"
            return f"{val:.4f}"
        return str(value)

    out["Value"] = [_fmt(m, v) for m, v in zip(out["Metric"], out["Value"])]
    return out


def fig_candlestick(ohlcv: pd.DataFrame, eval_df: pd.DataFrame, label: str) -> go.Figure:
    o_col, h_col, l_col, c_col = ("open", "high", "low", "close")
    price = ohlcv[c_col]
    buy_times = eval_df.index[eval_df["exec_long_entry"].values.astype(bool)]
    short_times = eval_df.index[eval_df["exec_short_entry"].values.astype(bool)]
    exit_times = eval_df.index[eval_df["exec_exit_to_cash"].values.astype(bool)]

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=ohlcv.index,
            open=ohlcv[o_col],
            high=ohlcv[h_col],
            low=ohlcv[l_col],
            close=ohlcv[c_col],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=eval_df.index,
            y=eval_df["upper"],
            name="Upper band",
            line=dict(color="#f9a825", width=1.3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=eval_df.index,
            y=eval_df["lower"],
            name="Lower band",
            line=dict(color="#7b61ff", width=1.3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=buy_times,
            y=price.reindex(buy_times).values,
            mode="markers",
            name="Executed long entry",
            marker=dict(symbol="triangle-up", size=9, color="#00e676"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=short_times,
            y=price.reindex(short_times).values,
            mode="markers",
            name="Executed short entry",
            marker=dict(symbol="triangle-down", size=9, color="#ff1744"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=exit_times,
            y=price.reindex(exit_times).values,
            mode="markers",
            name="Executed exit to cash",
            marker=dict(symbol="x", size=8, color="#ffb74d"),
        )
    )
    fig.update_layout(
        title=f"{label} | Price + executed Modified Donchian signals",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=520,
        legend=dict(orientation="h"),
    )
    return fig


def fig_equity(
    eval_df: pd.DataFrame,
    label: str,
    pf: float,
    strategy_curve_name: str = "Modified Donchian",
) -> go.Figure:
    next_log = eval_df["next_log_return"].fillna(0.0)
    net_log = eval_df["net_log_return"].fillna(0.0)

    bh_curve = np.exp(next_log.cumsum())
    strat_curve = np.exp(net_log.cumsum())
    drawdown = strat_curve / strat_curve.cummax() - 1.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bh_curve.index,
            y=bh_curve.values,
            name="Buy & Hold",
            line=dict(color="#888", dash="dot"),
        )
    )
    strat_trace_kwargs: dict[str, object] = {}
    if "sleeve_desc" in eval_df.columns or "regime_state" in eval_df.columns:
        sleeve_desc = eval_df.get("sleeve_desc", pd.Series("", index=eval_df.index)).fillna("").astype(str)
        regime_state = eval_df.get("regime_state", pd.Series("", index=eval_df.index)).fillna("").astype(str)
        customdata = np.stack(
            [sleeve_desc.reindex(strat_curve.index, fill_value="").to_numpy(),
             regime_state.reindex(strat_curve.index, fill_value="").to_numpy()],
            axis=-1,
        )
        strat_trace_kwargs["customdata"] = customdata
        strat_trace_kwargs["hovertemplate"] = (
            "%{x|%Y-%m-%d}<br>"
            "Equity: %{y:.4f}<br>"
            "Regime: %{customdata[1]}<br>"
            "Allocation: %{customdata[0]}"
            "<extra>" + strategy_curve_name + "</extra>"
        )
    fig.add_trace(
        go.Scatter(
            x=strat_curve.index,
            y=strat_curve.values,
            name=strategy_curve_name,
            line=dict(color="#00d084"),
            **strat_trace_kwargs,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=(drawdown.values * 100.0),
            name="Drawdown %",
            yaxis="y2",
            line=dict(color="#ff4d4d"),
            fill="tozeroy",
            opacity=0.3,
        )
    )
    fig.update_layout(
        title=f"{label} | Equity curve | Profit Factor={pf:.4f}",
        yaxis=dict(title="Normalized value"),
        yaxis2=dict(
            title="Drawdown %",
            overlaying="y",
            side="right",
            showgrid=False,
            range=[-100, 0],
        ),
        template="plotly_dark",
        height=420,
        legend=dict(orientation="h"),
    )
    return fig


def fig_candlestick_sma_long(ohlcv: pd.DataFrame, eval_df: pd.DataFrame, label: str) -> go.Figure:
    """Long-only SMA: markers for entries to long and exits to cash; optional ``ma`` line on eval_df."""
    o_col, h_col, l_col, c_col = ("open", "high", "low", "close")
    price = ohlcv[c_col]
    buy_times = eval_df.index[eval_df["exec_long_entry"].values.astype(bool)]
    exit_times = eval_df.index[eval_df["exec_exit_to_cash"].values.astype(bool)]

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=ohlcv.index,
            open=ohlcv[o_col],
            high=ohlcv[h_col],
            low=ohlcv[l_col],
            close=ohlcv[c_col],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )
    if "ma" in eval_df.columns:
        ma = eval_df["ma"].astype(float)
        fig.add_trace(
            go.Scatter(
                x=ma.index,
                y=ma.values,
                mode="lines",
                name="SMA",
                line=dict(color="#00bcd4", width=1.5),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=buy_times,
            y=price.reindex(buy_times).values,
            mode="markers",
            name="Long (full)",
            marker=dict(symbol="triangle-up", size=9, color="#00e676"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=exit_times,
            y=price.reindex(exit_times).values,
            mode="markers",
            name="Exit to cash",
            marker=dict(symbol="x", size=8, color="#ffb74d"),
        )
    )
    fig.update_layout(
        title=f"{label} | Price + SMA long-only signals",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=520,
        legend=dict(orientation="h"),
    )
    return fig


def fig_trades(eval_df: pd.DataFrame, label: str) -> go.Figure:
    trade_returns = _extract_trade_returns(eval_df) * 100.0
    colors = ["#00d084" if r >= 0 else "#ff4d4d" for r in trade_returns.values]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=trade_returns.values, marker_color=colors, name="Trade return %"))
    fig.update_layout(
        title=f"{label} | Individual trade returns (%)",
        yaxis_title="Return %",
        template="plotly_dark",
        height=300,
    )
    return fig


def fig_robustness(sweep_df: pd.DataFrame, selected_upper: int, selected_lower: int) -> go.Figure:
    grid = (
        sweep_df.pivot_table(
            index="upper_lookback",
            columns="lower_lookback",
            values="profit_factor",
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=grid.columns.astype(int),
            y=grid.index.astype(int),
            z=grid.values,
            colorscale="Viridis",
            colorbar=dict(title="PF"),
            hovertemplate="Upper=%{y}<br>Lower=%{x}<br>PF=%{z:.4f}<extra></extra>",
            name="Profit Factor",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[int(selected_lower)],
            y=[int(selected_upper)],
            mode="markers",
            name=f"Selected (U={int(selected_upper)}, L={int(selected_lower)})",
            marker=dict(size=10, color="#f9a825", symbol="diamond"),
        )
    )
    fig.update_layout(
        title="Robustness sweep | PF heatmap (upper vs lower lookback)",
        xaxis_title="Lower lookback",
        yaxis_title="Upper lookback",
        template="plotly_dark",
        height=420,
        legend=dict(orientation="h"),
    )
    return fig


def fig_robustness_sma_period(sweep_df: pd.DataFrame, selected_period: int) -> go.Figure:
    """Line plot of profit factor vs SMA period (1D robustness)."""
    work = sweep_df.sort_values("sma_period")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=work["sma_period"].astype(int),
            y=work["profit_factor"].astype(float),
            mode="lines+markers",
            name="Profit Factor",
            line=dict(color="#4a9eff"),
            marker=dict(size=4),
        )
    )
    sel = float(
        work.loc[work["sma_period"] == int(selected_period), "profit_factor"].mean()
        if (work["sma_period"] == int(selected_period)).any()
        else np.nan
    )
    fig.add_trace(
        go.Scatter(
            x=[int(selected_period)],
            y=[sel],
            mode="markers",
            name=f"Selected (N={int(selected_period)})",
            marker=dict(size=12, color="#f9a825", symbol="diamond"),
        )
    )
    fig.update_layout(
        title="Robustness sweep | Profit factor vs SMA period",
        xaxis_title="SMA period",
        yaxis_title="Profit factor",
        template="plotly_dark",
        height=420,
        legend=dict(orientation="h"),
    )
    return fig


# Backward-compatible aliases (older code used leading underscores)
_bars_per_year = bars_per_year
_comparison_stats_df = comparison_stats_df
_format_stats = format_stats
