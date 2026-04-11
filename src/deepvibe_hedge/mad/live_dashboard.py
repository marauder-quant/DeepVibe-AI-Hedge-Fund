"""
Live Dash UI: equity (Alpaca), MRAT candle + watchlist (local SQLite panel), portfolio / orders.
User-visible times use **US/Eastern** (``America/New_York``, EST/EDT). APIs stay UTC internally.
Dark theme: ``mad/dash_assets/theme.css`` (loaded via ``Dash(assets_folder=...)``).

Defaults (no extra config keys): port 8066, MRAT + equity chart refresh 20s,
equity curve ranges: 24h (rolling, fixed x-axis), 1W / 1M / 1Y / YTD (US/Eastern), 250 closed orders.

Run::

    PYTHONPATH=src python -m deepvibe_hedge.mad.live_dashboard
"""
from __future__ import annotations

import sqlite3
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from alpaca.common.enums import Sort
from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest
from dash import Dash, Input, Output, dash_table, dcc, html, no_update
from plotly.subplots import make_subplots

from deepvibe_hedge import config
from deepvibe_hedge.alpaca_asset import _alpaca_trading_keys
from deepvibe_hedge.paths import OHLCV_DIR
from deepvibe_hedge.mad.backtester import (
    compute_mad_live_panel_and_snapshot,
    mad_live_watchlist_table,
    mad_reference_ticker,
    mad_regime_ticker_symbol,
    mad_universe_tickers,
)
from deepvibe_hedge.mad.live_bot import load_mad_live_strategy_params

# Defaults (intentionally not in config.py)
DASHBOARD_PORT = 8066
REFRESH_MS = 20_000
# ~6_000 sessions ≈ 24y of trading days — matches long OHLCV DBs (``TARGET_START_DATE`` / fetcher).
CANDLE_BARS = 6000
ORDER_HISTORY_LIMIT = 250
_PANEL_CACHE_TTL_SEC = 75.0
_panel_lock = threading.Lock()

_panel_cache: dict[str, Any] = {
    "until": 0.0,
    "panel": None,
    "snap": None,
    "sub": None,
    "params": None,
}

_APP_DIR = Path(__file__).resolve().parent

# Dark dashboard tokens (matches ``dash_assets/theme.css``).
UI: dict[str, str] = {
    "bg": "#0e0e14",
    "paper": "#12121a",
    "surface2": "#14141c",
    "grid": "#27272f",
    "text": "#e4e4e7",
    "muted": "#a1a1aa",
    "accent": "#2dd4bf",
    "accent2": "#a78bfa",
    "up": "#4ade80",
    "down": "#fb7185",
    "danger": "#fb7185",
    "border": "#2a2a36",
    "elevated": "#1a1a24",
    "font": "DM Sans, Inter, system-ui, sans-serif",
}

GRAPH_CONFIG: dict[str, Any] = {
    "displaylogo": False,
    "modeBarBackgroundColor": UI["paper"],
    "toImageButtonOptions": {"format": "png"},
}

# US/Eastern for all user-visible times (NYSE session clock; EST/EDT via zoneinfo).
_NY = ZoneInfo("America/New_York")


def _fmt_instant_ny(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_NY).strftime("%Y-%m-%d %H:%M %Z")


def _snap_as_of_ny_str(ts: Any) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.tz_convert(_NY).strftime("%Y-%m-%d %H:%M %Z")


def _format_order_submitted_et(raw: Any) -> str:
    if raw is None or raw == "":
        return "—"
    try:
        t = pd.Timestamp(raw)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        return t.tz_convert(_NY).strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        return str(raw)


def _paper_mode() -> bool:
    return config.bot_mode_is_paper()


def _data_feed() -> DataFeed:
    raw = str(getattr(config, "LIVE_BOT_DATA_FEED", "iex")).strip().lower()
    if raw == "sip":
        return DataFeed.SIP
    if raw in ("delayed_sip", "delayed"):
        return DataFeed.DELAYED_SIP
    return DataFeed.IEX


def _bar_adjustment() -> Adjustment | None:
    raw = str(getattr(config, "ALPACA_BAR_ADJUSTMENT", "split")).strip().lower()
    if raw == "raw":
        return Adjustment.RAW
    if raw == "dividend":
        return Adjustment.DIVIDEND
    if raw == "all":
        return Adjustment.ALL
    return Adjustment.SPLIT


def _portfolio_history_request_for_range(equity_range: str) -> GetPortfolioHistoryRequest:
    """Alpaca portfolio history by dashboard range (API uses UTC; chart displays US/Eastern)."""
    er = str(equity_range).strip().lower()
    if er == "24h":
        end = datetime.now(timezone.utc)
        return GetPortfolioHistoryRequest(
            start=end - timedelta(hours=24),
            end=end,
            timeframe="5Min",
            extended_hours=True,
        )
    if er == "1w":
        return GetPortfolioHistoryRequest(period="1W", timeframe="1H")
    if er == "1m":
        return GetPortfolioHistoryRequest(period="1M", timeframe="1D")
    if er == "1y":
        return GetPortfolioHistoryRequest(period="1A", timeframe="1D")
    if er == "ytd":
        end = datetime.now(timezone.utc)
        start_ny = datetime.now(_NY).replace(
            month=1, day=1, hour=0, minute=0, second=0, microsecond=0
        )
        start = start_ny.astimezone(timezone.utc)
        return GetPortfolioHistoryRequest(
            start=start,
            end=end,
            timeframe="1D",
        )
    end = datetime.now(timezone.utc)
    return GetPortfolioHistoryRequest(
        start=end - timedelta(hours=24),
        end=end,
        timeframe="5Min",
        extended_hours=True,
    )


def _parse_portfolio_history_raw(hist: Any) -> tuple[list[datetime], list[float]]:
    if isinstance(hist, dict):
        ts = list(hist.get("timestamp") or [])
        eq = list(hist.get("equity") or [])
    else:
        ts = list(getattr(hist, "timestamp", None) or [])
        eq = list(getattr(hist, "equity", None) or [])
    if not ts or not eq or len(ts) != len(eq):
        return [], []
    x_utc = [_ts_from_portfolio_hist(t) for t in ts]
    eq_f = [float(v) for v in eq]
    return x_utc, eq_f


def _trim_equity_from_first_trade(
    x_utc: list[datetime], eq_f: list[float], *, min_equity_usd: float = 1.0
) -> tuple[list[datetime], list[float]]:
    """Drop leading flat / zero-equity points (chart + all-time stats start when account had value)."""
    i0 = 0
    for i, v in enumerate(eq_f):
        if np.isfinite(v) and float(v) >= float(min_equity_usd):
            i0 = i
            break
    return x_utc[i0:], eq_f[i0:]


def _ny_calendar_date(dt: datetime) -> date:
    """Calendar date in US/Eastern (matches how users read Alpaca daily bars)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_NY).date()


def _bar_index_ny_day(ts: Any) -> date:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.tz_convert(_NY).date()


def _first_fill_ny_date(tc: TradingClient) -> date | None:
    """Earliest US/Eastern calendar day with a non-zero fill (chronological first trade day)."""
    try:
        orders = tc.get_orders(
            filter=GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                limit=500,
                direction=Sort.ASC,
                nested=True,
            )
        )
    except Exception:
        return None
    if not orders:
        return None
    best: date | None = None
    for o in orders:
        raw_fa = getattr(o, "filled_at", None)
        if raw_fa is None:
            continue
        try:
            fq = float(getattr(o, "filled_qty", 0) or 0)
        except (TypeError, ValueError):
            fq = 0.0
        if fq <= 0:
            continue
        t = pd.Timestamp(raw_fa)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        d = t.tz_convert(_NY).date()
        if best is None or d < best:
            best = d
    return best


def _trim_equity_from_first_fill_or_cash(
    tc: TradingClient,
    x_utc: list[datetime],
    eq_f: list[float],
    *,
    min_equity_usd: float = 1.0,
) -> tuple[list[datetime], list[float], date | None]:
    """
    Prefer first US/Eastern day with a filled order; else first day equity >= ``min_equity_usd``.
    Returns ``(x_trim, eq_trim, fill_ny_date_or_none)``.
    """
    fill_ny = _first_fill_ny_date(tc)
    if fill_ny is not None:
        for i, (t, v) in enumerate(zip(x_utc, eq_f)):
            if _ny_calendar_date(t) < fill_ny:
                continue
            if np.isfinite(v) and float(v) >= float(min_equity_usd):
                return x_utc[i:], eq_f[i:], fill_ny
        for i, (t, v) in enumerate(zip(x_utc, eq_f)):
            if _ny_calendar_date(t) >= fill_ny and np.isfinite(v) and float(v) > 0:
                return x_utc[i:], eq_f[i:], fill_ny
    tx, eq = _trim_equity_from_first_trade(x_utc, eq_f, min_equity_usd=min_equity_usd)
    return tx, eq, fill_ny


def _portfolio_return_pct_ny(
    x_utc: list[datetime],
    eq_f: list[float],
    start_ny_date: date,
    end_equity: float,
) -> float | None:
    """Return % from first daily sample on/after ``start_ny_date`` (ET calendar) to ``end_equity``."""
    start_eq: float | None = None
    for t, e in zip(x_utc, eq_f):
        if _ny_calendar_date(t) >= start_ny_date:
            if np.isfinite(e) and float(e) > 0:
                start_eq = float(e)
                break
    if start_eq is None or start_eq <= 0 or not np.isfinite(end_equity) or end_equity <= 0:
        return None
    return (float(end_equity) / start_eq - 1.0) * 100.0


def _benchmark_total_return_pct_ny(
    closes: pd.Series, start_ny_date: date, end_now_utc: datetime
) -> float | None:
    """Buy-and-hold % using first/last daily bar by **US/Eastern** session date."""
    if closes is None or closes.empty:
        return None
    s = closes.sort_index()
    end_ny_d = end_now_utc.astimezone(_NY).date()
    first_i: int | None = None
    last_i: int | None = None
    for i, ts in enumerate(s.index):
        bd = _bar_index_ny_day(ts)
        if bd >= start_ny_date and first_i is None:
            first_i = i
        if bd <= end_ny_d:
            last_i = i
    if (
        first_i is None
        or last_i is None
        or last_i < first_i
        or not np.isfinite(float(s.iloc[first_i]))
        or not np.isfinite(float(s.iloc[last_i]))
    ):
        return None
    first = float(s.iloc[first_i])
    last = float(s.iloc[last_i])
    if first <= 0 or last <= 0:
        return None
    return (last / first - 1.0) * 100.0


def _benchmark_daily_closes(
    symbol: str, start: datetime, end: datetime
) -> pd.Series:
    """Daily close series indexed by UTC midnight (``Timestamp``)."""
    sym = str(symbol).strip().upper()
    paper = _paper_mode()
    key, secret = _alpaca_trading_keys(paper=paper)
    dc = StockHistoricalDataClient(api_key=key, secret_key=secret)
    feeds: list[DataFeed] = []
    for fd in (_data_feed(), DataFeed.IEX, DataFeed.SIP):
        if fd not in feeds:
            feeds.append(fd)
    end_u = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
    start_u = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
    for fd in feeds:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame.Day,
                start=start_u,
                end=end_u,
                adjustment=_bar_adjustment(),
                feed=fd,
            )
            bars = dc.get_stock_bars(req)
            if bars is None or sym not in bars.data or not bars.data[sym]:
                continue
            out: dict[pd.Timestamp, float] = {}
            for b in bars.data[sym]:
                if isinstance(b, dict):
                    ts = pd.Timestamp(b["timestamp"])
                    cl = float(b["close"])
                else:
                    ts = pd.Timestamp(b.timestamp)
                    cl = float(b.close)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                out[ts.normalize()] = cl
            if not out:
                continue
            return pd.Series(out).sort_index()
        except (APIError, OSError, ValueError):
            continue
        except Exception:
            continue
    return pd.Series(dtype=float)


def _fmt_perf_pct(p: float | None) -> html.Td:
    if p is None or not np.isfinite(p):
        return html.Td("—", style={"color": UI["muted"]})
    color = UI["up"] if p > 1e-9 else (UI["down"] if p < -1e-9 else UI["muted"])
    return html.Td(
        f"{p:+.2f}%",
        style={"color": color, "fontFamily": UI["font"], "fontWeight": "600"},
    )


def _portfolio_performance_section(tc: TradingClient, acct: Any) -> html.Div:
    """
    Period returns vs DIA / SPY / QQQ (proxies for Dow, S&P 500, Nasdaq-100).
    Uses Alpaca portfolio history + same-calendar ETF bars.
    """
    end_now = datetime.now(timezone.utc)
    req = GetPortfolioHistoryRequest(
        timeframe="1D",
        start=datetime(2010, 1, 1, tzinfo=timezone.utc),
        end=end_now,
    )
    try:
        hist = tc.get_portfolio_history(req)
        x_utc, eq_f = _parse_portfolio_history_raw(hist)
    except Exception:
        x_utc, eq_f = [], []
    if not x_utc:
        return html.Div(
            html.P(
                "Performance vs benchmarks: no portfolio history yet.",
                className="dv-placeholder",
                style={"marginTop": "12px"},
            )
        )
    x_trim, eq_trim, fill_ny = _trim_equity_from_first_fill_or_cash(tc, x_utc, eq_f)
    if not x_trim:
        return html.Div(
            html.P(
                "Performance vs benchmarks: waiting for first equity above $1 in history.",
                className="dv-placeholder",
                style={"marginTop": "12px"},
            )
        )
    end_eq = float(acct.equity)
    end_ny = end_now.astimezone(_NY)
    ytd_ny_date = date(end_ny.year, 1, 1)
    m6_ny_date = (end_ny - timedelta(days=183)).date()
    m3_ny_date = (end_ny - timedelta(days=92)).date()
    all_ny_date = fill_ny if fill_ny is not None else _ny_calendar_date(x_trim[0])
    if fill_ny is not None:
        all_label = f"All-time (since first fill · {fill_ny.isoformat()} ET)"
    else:
        all_label = "All-time (since first activity)"

    min_bench = min(all_ny_date, ytd_ny_date, m6_ny_date, m3_ny_date)
    bench_start = datetime.combine(min_bench, datetime.min.time(), tzinfo=_NY) - timedelta(
        days=14
    )
    bench_start = bench_start.astimezone(timezone.utc)

    dia = _benchmark_daily_closes("DIA", bench_start, end_now)
    spy = _benchmark_daily_closes("SPY", bench_start, end_now)
    qqq = _benchmark_daily_closes("QQQ", bench_start, end_now)

    periods: list[tuple[str, date]] = [
        (all_label, all_ny_date),
        ("Year to date", ytd_ny_date),
        ("Last 6 months", m6_ny_date),
        ("Last 3 months", m3_ny_date),
    ]

    rows = [
        html.Tr(
            [
                html.Th("Period"),
                html.Th("Portfolio"),
                html.Th("Dow (DIA)"),
                html.Th("S&P 500 (SPY)"),
                html.Th("Nasdaq-100 (QQQ)"),
            ]
        )
    ]
    for label, ny_d in periods:
        pr = _portfolio_return_pct_ny(x_trim, eq_trim, ny_d, end_eq)
        br_d = _benchmark_total_return_pct_ny(dia, ny_d, end_now)
        br_s = _benchmark_total_return_pct_ny(spy, ny_d, end_now)
        br_q = _benchmark_total_return_pct_ny(qqq, ny_d, end_now)
        rows.append(
            html.Tr(
                [
                    html.Td(label, style={"color": UI["muted"]}),
                    _fmt_perf_pct(pr),
                    _fmt_perf_pct(br_d),
                    _fmt_perf_pct(br_s),
                    _fmt_perf_pct(br_q),
                ]
            )
        )

    return html.Div(
        [
            html.P(
                "Total return vs ETF proxies (same US/Eastern calendar window; not risk-adjusted). "
                "All-time starts on the first fill day when available; otherwise first equity slice. "
                "Portfolio uses Alpaca daily equity; benchmarks use adjusted daily closes.",
                style={
                    "fontSize": "0.78rem",
                    "color": UI["muted"],
                    "margin": "14px 0 8px 0",
                    "lineHeight": "1.45",
                },
            ),
            html.Table(rows, className="dv-portfolio-table dv-performance-table"),
        ]
    )


def _apply_equity_chart_axes(fig: go.Figure, *, title: str | None = None) -> None:
    """Dark theme + vertical spike/crosshair following the cursor (US/Eastern x-axis)."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=UI["paper"],
        plot_bgcolor=UI["bg"],
        font=dict(color=UI["text"], family=UI["font"], size=12),
        height=320,
        margin=dict(l=48, r=16, t=48, b=44),
        xaxis_title="Time (US/Eastern)",
        yaxis_title="USD",
        showlegend=False,
        hovermode="x unified",
        hoverdistance=-1,
        spikedistance=-1,
        hoverlabel=dict(
            bgcolor=UI["elevated"],
            font=dict(color=UI["text"], family=UI["font"]),
            bordercolor=UI["border"],
        ),
    )
    if title is not None:
        fig.update_layout(
            title=dict(text=title, font=dict(size=15, color=UI["text"], family=UI["font"]))
        )
    fig.update_xaxes(
        showspikes=True,
        spikecolor=UI["accent"],
        spikesnap="cursor",
        spikemode="across+marker",
        spikethickness=1,
        gridcolor=UI["grid"],
        zerolinecolor=UI["grid"],
        linecolor=UI["border"],
        tickfont=dict(color=UI["muted"]),
        title=dict(font=dict(color=UI["muted"], size=11)),
    )
    fig.update_yaxes(
        showspikes=False,
        gridcolor=UI["grid"],
        zerolinecolor=UI["grid"],
        linecolor=UI["border"],
        tickfont=dict(color=UI["muted"]),
        title=dict(font=dict(color=UI["muted"], size=11)),
    )


def _empty_equity_figure(title: str = "Account equity") -> go.Figure:
    fig = go.Figure()
    _apply_equity_chart_axes(fig, title=title)
    return fig


def _ts_from_portfolio_hist(sec: float | int) -> datetime:
    s = float(sec)
    if s > 1e15:
        s /= 1e9
    elif s > 1e12:
        s /= 1000.0
    return datetime.fromtimestamp(s, tz=timezone.utc)


def _pct_vs_baseline_label(current: Any, baseline: Any) -> tuple[str | None, str]:
    """Return ``("+1.23%", color)`` vs chart range start, or ``(None, _)`` if unknown."""
    try:
        c = float(current)
        b = float(baseline)
    except (TypeError, ValueError):
        return None, UI["muted"]
    if not np.isfinite(c) or not np.isfinite(b) or b <= 0:
        return None, UI["muted"]
    p = (c / b - 1.0) * 100.0
    if p > 1e-9:
        color = UI["up"]
    elif p < -1e-9:
        color = UI["down"]
    else:
        color = UI["muted"]
    return f"{p:+.2f}%", color


def _parse_equity_hover_ts(xv: Any) -> str:
    """Plotly hover ``x`` → US/Eastern label for the headline."""
    if xv is None:
        return "—"
    if isinstance(xv, datetime):
        if xv.tzinfo is None:
            xv = xv.replace(tzinfo=timezone.utc)
        return _fmt_instant_ny(xv)
    if isinstance(xv, (int, float)):
        s = float(xv)
        if s > 1e15:
            s /= 1e9
        elif s > 1e12:
            s /= 1000.0
        try:
            return _fmt_instant_ny(datetime.fromtimestamp(s, tz=timezone.utc))
        except (OSError, ValueError, OverflowError):
            return str(xv)
    try:
        dt = pd.Timestamp(str(xv))
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        else:
            dt = dt.tz_convert("UTC")
        return _fmt_instant_ny(dt.to_pydatetime())
    except Exception:
        return str(xv)


def _stitch_live_equity_end(
    x_utc: list[datetime],
    eq_f: list[float],
    now_ny: datetime,
    live_equity: float | None,
    *,
    merge_within_sec: float,
) -> None:
    """Append or refresh the last sample to **now** with live account equity (mutates lists)."""
    if live_equity is None or not np.isfinite(live_equity):
        return
    now_utc = now_ny.astimezone(timezone.utc)
    lv = float(live_equity)
    if not x_utc:
        x_utc.append(now_utc)
        eq_f.append(lv)
        return
    last_ny = x_utc[-1].astimezone(_NY)
    if (now_ny - last_ny).total_seconds() <= merge_within_sec:
        x_utc[-1] = now_utc
        eq_f[-1] = lv
    else:
        x_utc.append(now_utc)
        eq_f.append(lv)


def _equity_line_and_probe_traces(
    x_plot: list[datetime],
    eq_f: list[float],
    *,
    b0: float,
    pct_caption: str,
    n_probe: int = 400,
) -> list[Any]:
    """
    Visible line without hover plus a transparent dense scatter so the cursor always
    picks up a point (stable tooltip + crosshair along the full x domain).
    """
    if not x_plot or not eq_f or len(x_plot) != len(eq_f):
        return []
    if len(x_plot) == 1:
        v0 = float(eq_f[0])
        pct = (
            f"{(v0 / b0 - 1.0) * 100.0:+.2f}%"
            if b0 > 0 and np.isfinite(v0)
            else "—"
        )
        return [
            go.Scatter(
                x=x_plot,
                y=eq_f,
                mode="markers",
                marker=dict(size=12, color=UI["accent"]),
                name="Equity",
                customdata=[[v0, pct]],
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d %H:%M %Z}</b><br>"
                    "Equity: <b>$%{customdata[0]:,.2f}</b><br>"
                    f"{pct_caption}: <b>%{{customdata[1]}}</b><extra></extra>"
                ),
            )
        ]
    t_nums = np.array([t.timestamp() for t in x_plot], dtype=float)
    y_pts = np.array(eq_f, dtype=float)
    t0, t1 = float(t_nums[0]), float(t_nums[-1])
    if t1 <= t0:
        t1 = t0 + 1.0
    grid = np.linspace(t0, t1, int(max(32, min(n_probe, 800))))
    yi = np.interp(grid, t_nums, y_pts)
    x_dense = [
        datetime.fromtimestamp(float(g), tz=timezone.utc).astimezone(_NY) for g in grid
    ]
    pcts: list[str] = []
    for v in yi:
        if b0 > 0 and np.isfinite(v):
            pcts.append(f"{(float(v) / b0 - 1.0) * 100.0:+.2f}%")
        else:
            pcts.append("—")
    line = go.Scatter(
        x=x_plot,
        y=eq_f,
        mode="lines",
        line=dict(width=2.5, color=UI["accent"]),
        name="Equity",
        hoverinfo="skip",
    )
    cd_probe = [[float(v), p] for v, p in zip(yi, pcts)]
    probe = go.Scatter(
        x=x_dense,
        y=list(yi),
        mode="markers",
        marker=dict(size=22, opacity=0, color="rgba(0,0,0,0)", line=dict(width=0)),
        name="Equity",
        showlegend=False,
        customdata=cd_probe,
        hovertemplate=(
            "<b>%{x|%Y-%m-%d %H:%M %Z}</b><br>"
            "Equity: <b>$%{customdata[0]:,.2f}</b><br>"
            f"{pct_caption}: <b>%{{customdata[1]}}</b><extra></extra>"
        ),
    )
    return [line, probe]


def _equity_figure_and_snapshot(
    tc: TradingClient,
    equity_range: str,
) -> tuple[go.Figure, dict[str, Any] | None]:
    """
    Equity vs time in **US/Eastern**. Ranges: rolling 24h (fixed x window), 1W, 1M, 1Y, YTD.
    Last point uses live account equity at **now** where applicable.
    """
    er_l = str(equity_range or "24h").strip().lower()
    now_ny = datetime.now(_NY)
    window_start_24h = now_ny - timedelta(hours=24)

    live_equity: float | None = None
    try:
        live_equity = float(tc.get_account().equity)
    except Exception:
        live_equity = None

    req = _portfolio_history_request_for_range(er_l)
    hist = tc.get_portfolio_history(req)
    if isinstance(hist, dict):
        ts = list(hist.get("timestamp") or [])
        eq = list(hist.get("equity") or [])
    else:
        ts = list(getattr(hist, "timestamp", None) or [])
        eq = list(getattr(hist, "equity", None) or [])

    x_utc: list[datetime] = []
    eq_f: list[float] = []
    if ts and eq and len(ts) == len(eq):
        x_utc = [_ts_from_portfolio_hist(t) for t in ts]
        eq_f = [float(v) for v in eq]

    if er_l == "24h":
        xf: list[datetime] = []
        ef: list[float] = []
        for t, e in zip(x_utc, eq_f):
            tn = t.astimezone(_NY)
            if tn < window_start_24h or tn > now_ny:
                continue
            xf.append(t)
            ef.append(e)
        x_utc, eq_f = xf, ef
        merge_sec = 360.0
    elif er_l == "1w":
        merge_sec = 4000.0
    elif er_l in ("1m", "1y", "ytd"):
        merge_sec = 86400.0
    else:
        merge_sec = 360.0

    _stitch_live_equity_end(
        x_utc, eq_f, now_ny, live_equity, merge_within_sec=merge_sec
    )

    titles_map = {
        "24h": "Account equity (rolling 24h · US/Eastern)",
        "1w": "Account equity (1 week · US/Eastern)",
        "1m": "Account equity (1 month · US/Eastern)",
        "1y": "Account equity (1 year · US/Eastern)",
        "ytd": "Account equity (YTD · US/Eastern)",
    }
    title = titles_map.get(er_l, "Account equity · US/Eastern")

    if not x_utc:
        fig = go.Figure()
        _apply_equity_chart_axes(fig, title=f"{title} (no data)")
        if er_l == "24h":
            fig.update_xaxes(
                range=[window_start_24h, now_ny], autorange=False, fixedrange=True
            )
        return fig, None

    x_plot = [t.astimezone(_NY) for t in x_utc]
    b0 = float(eq_f[0])
    pct_caption = "vs 24h window start" if er_l == "24h" else "vs range start"
    traces = _equity_line_and_probe_traces(
        x_plot, eq_f, b0=b0, pct_caption=pct_caption
    )
    fig = go.Figure(data=traces)
    _apply_equity_chart_axes(fig, title=title)
    if er_l == "24h":
        fig.update_xaxes(
            range=[window_start_24h, now_ny], autorange=False, fixedrange=True
        )
    else:
        fig.update_xaxes(autorange=True, fixedrange=False)

    fig.update_layout(uirevision=f"equity-{er_l}")
    snap: dict[str, Any] = {
        "baseline_equity": b0,
        "baseline_ts_display": _fmt_instant_ny(x_utc[0]),
        "chart_last_equity": float(eq_f[-1]),
        "chart_ts_display": _fmt_instant_ny(x_utc[-1]),
        "equity_range": er_l,
    }
    return fig, snap


def _equity_tracker_block(
    equity_str: str,
    ts_str: str,
    *,
    caption: str,
    pct_text: str | None = None,
    pct_color: str | None = None,
) -> html.Div:
    pct_c = pct_color or UI["muted"]
    equity_row: list[Any] = [
        html.Span(
            equity_str,
            style={
                "fontSize": "2.35rem",
                "fontWeight": "700",
                "lineHeight": "1.15",
                "letterSpacing": "-0.03em",
                "color": UI["text"],
            },
        ),
    ]
    if pct_text:
        equity_row.append(
            html.Span(
                pct_text,
                style={
                    "marginLeft": "0.45rem",
                    "fontSize": "1.85rem",
                    "fontWeight": "600",
                    "color": pct_c,
                },
            )
        )
    return html.Div(
        [
            html.Div(
                equity_row,
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "alignItems": "baseline",
                },
            ),
            html.Div(
                ts_str,
                style={
                    "fontSize": "0.95rem",
                    "color": UI["muted"],
                    "marginTop": "6px",
                    "fontFamily": UI["font"],
                },
            ),
            html.Div(
                caption,
                style={
                    "fontSize": "0.65rem",
                    "color": UI["muted"],
                    "opacity": 0.85,
                    "textTransform": "uppercase",
                    "letterSpacing": "0.12em",
                    "marginTop": "8px",
                },
            ),
        ],
        style={"marginBottom": "16px"},
    )


def _alpaca_daily_ohlc(
    symbol: str,
    n_bars: int,
    sma_short: int,
    sma_long: int,
    *,
    feed: DataFeed | None = None,
) -> pd.DataFrame:
    sym = str(symbol).strip().upper()
    paper = _paper_mode()
    key, secret = _alpaca_trading_keys(paper=paper)
    dc = StockHistoricalDataClient(api_key=key, secret_key=secret)
    end = datetime.now(timezone.utc)
    fd = feed if feed is not None else _data_feed()
    req = StockBarsRequest(
        symbol_or_symbols=sym,
        timeframe=TimeFrame.Day,
        limit=int(n_bars),
        end=end,
        adjustment=_bar_adjustment(),
        feed=fd,
    )
    bars = dc.get_stock_bars(req)
    if bars is None or sym not in bars.data or not bars.data[sym]:
        return pd.DataFrame()
    rows = []
    for b in bars.data[sym]:
        if isinstance(b, dict):
            rows.append(
                {
                    "date": pd.Timestamp(b["timestamp"]),
                    "open": float(b["open"]),
                    "high": float(b["high"]),
                    "low": float(b["low"]),
                    "close": float(b["close"]),
                }
            )
        else:
            rows.append(
                {
                    "date": pd.Timestamp(b.timestamp),
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("date").reset_index(drop=True)
    sh, lo = int(sma_short), int(sma_long)
    df["sma_short"] = df["close"].rolling(sh, min_periods=sh).mean()
    df["sma_long"] = df["close"].rolling(lo, min_periods=lo).mean()
    return df


def _ohlcv_sqlite_daily_ohlc(
    symbol: str, n_bars: int, sma_short: int, sma_long: int
) -> pd.DataFrame:
    """Last ``n_bars`` rows from ``data/ohlcv/{SYM}_{gran}.db`` (same files as MRAT)."""
    sym = str(symbol).strip().upper()
    gran = str(config.TARGET_CANDLE_GRANULARITY)
    path: Path = OHLCV_DIR / f"{sym}_{gran}.db"
    if not path.exists():
        return pd.DataFrame()
    with sqlite3.connect(path) as con:
        cols = {r[1] for r in con.execute("PRAGMA table_info(ohlcv)").fetchall()}
        if "timestamp" not in cols or "close" not in cols:
            return pd.DataFrame()
        qcols = ["timestamp"]
        for c in ("open", "high", "low", "close"):
            if c in cols:
                qcols.append(c)
        placeholders = ", ".join(qcols)
        cur = con.execute(
            f"SELECT {placeholders} FROM ohlcv ORDER BY timestamp DESC LIMIT ?",
            (int(n_bars),),
        )
        raw = cur.fetchall()
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=qcols)
    df = df.iloc[::-1].reset_index(drop=True)
    if "open" not in df.columns:
        df["open"] = df["close"]
    if "high" not in df.columns:
        df["high"] = df["close"]
    if "low" not in df.columns:
        df["low"] = df["close"]
    df["date"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop(columns=["timestamp"])
    sh, lo = int(sma_short), int(sma_long)
    df["sma_short"] = df["close"].astype(float).rolling(sh, min_periods=sh).mean()
    df["sma_long"] = df["close"].astype(float).rolling(lo, min_periods=lo).mean()
    return df


def _daily_ohlc_for_chart(
    symbol: str, n_bars: int, sma_short: int, sma_long: int
) -> tuple[pd.DataFrame, str]:
    """
    OHLC for the price pane: pick the source with the **most bars**.

    Alpaca IEX often returns **one** daily bar for live accounts; we must not prefer that over
    ``data/ohlcv/*.db`` (hundreds of rows, same calendar as MRAT).
    """
    sym = str(symbol).strip().upper()
    best = pd.DataFrame()
    best_src = ""

    df_sql = _ohlcv_sqlite_daily_ohlc(sym, n_bars, sma_short, sma_long)
    if not df_sql.empty:
        best, best_src = df_sql, "SQLite OHLCV"

    primary = _data_feed()
    feeds: list[DataFeed] = []
    for fd in (primary, DataFeed.IEX, DataFeed.SIP):
        if fd not in feeds:
            feeds.append(fd)
    for fd in feeds:
        try:
            df = _alpaca_daily_ohlc(sym, n_bars, sma_short, sma_long, feed=fd)
        except (APIError, OSError, ValueError):
            continue
        except Exception:
            continue
        if df.empty:
            continue
        if len(df) > len(best):
            best, best_src = df, f"Alpaca {fd.name}"

    return best, best_src


def _sanitize_ohlc_for_candlestick(
    df: pd.DataFrame, sma_short: int, sma_long: int
) -> pd.DataFrame:
    """
    Plotly ``Candlestick`` rejects NaN/invalid OHLC (legend shows a broken trace, blank pane).

    Enforces high >= max(open,close) and low <= min(open,close), drops bad rows, then
    recomputes SMAs on the cleaned close.
    """
    if df.empty:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    out = out.dropna(subset=["date"])
    need = ("open", "high", "low", "close")
    for c in need:
        if c not in out.columns:
            return pd.DataFrame()
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=list(need))
    if out.empty:
        return out
    pos = (
        (out["open"] > 0)
        & (out["high"] > 0)
        & (out["low"] > 0)
        & (out["close"] > 0)
    )
    out = out.loc[pos]
    if out.empty:
        return out
    o = out["open"].to_numpy(dtype=float)
    h = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    c_ = out["close"].to_numpy(dtype=float)
    out["high"] = np.maximum(np.maximum(o, h), c_)
    out["low"] = np.minimum(np.minimum(o, low), c_)
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    sh, ln = int(sma_short), int(sma_long)
    out["sma_short"] = out["close"].rolling(sh, min_periods=1).mean()
    out["sma_long"] = out["close"].rolling(ln, min_periods=1).mean()
    du = pd.to_datetime(out["date"], utc=True)
    out["x_dt"] = du.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return out


def _load_panel_bundle():
    """Thread-safe; building ~100 SQLite MRAT panels can take minutes on a cold cache."""
    global _panel_cache
    params = load_mad_live_strategy_params()
    sh, lo, ex, reg_ma, reg_tick = params
    now = time.monotonic()
    with _panel_lock:
        if (
            _panel_cache["panel"] is not None
            and now < float(_panel_cache["until"])
            and _panel_cache["params"] == params
        ):
            return (
                _panel_cache["panel"],
                _panel_cache["snap"],
                _panel_cache["sub"],
                params,
            )
        panel, snap, sub = compute_mad_live_panel_and_snapshot(
            short_w=sh,
            long_w=lo,
            exit_ma_period=ex,
            regime_ma_period=reg_ma,
            regime_ticker=reg_tick,
        )
        _panel_cache = {
            "until": time.monotonic() + _PANEL_CACHE_TTL_SEC,
            "panel": panel,
            "snap": snap,
            "sub": sub,
            "params": params,
        }
        return panel, snap, sub, params


def _start_panel_cache_warmer() -> None:
    """Precompute MRAT panel in the background so the first browser request is not minutes long."""

    def _run() -> None:
        t0 = time.perf_counter()
        try:
            _load_panel_bundle()
            print(
                f"[live_dashboard] MRAT panel cache ready in {time.perf_counter() - t0:.1f}s "
                f"({len(mad_universe_tickers())} names)"
            )
        except Exception as exc:
            print(f"[live_dashboard] MRAT panel warm-up failed: {exc}")

    threading.Thread(target=_run, daemon=True, name="mad-panel-warm").start()


def _finalize_mrat_figure(fig: go.Figure) -> None:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=UI["paper"],
        plot_bgcolor=UI["bg"],
        font=dict(color=UI["text"], family=UI["font"], size=11),
        height=640,
        margin=dict(l=52, r=20, t=56, b=44),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            bgcolor="rgba(18,18,26,0.92)",
            bordercolor=UI["border"],
            borderwidth=1,
            font=dict(size=10, color=UI["muted"]),
        ),
        dragmode="pan",
        hoverlabel=dict(
            bgcolor=UI["elevated"],
            font=dict(color=UI["text"], family=UI["font"]),
            bordercolor=UI["border"],
        ),
    )
    fig.update_xaxes(
        gridcolor=UI["grid"],
        zerolinecolor=UI["grid"],
        linecolor=UI["border"],
        tickfont=dict(color=UI["muted"]),
        rangeslider_visible=False,
    )
    fig.update_yaxes(
        gridcolor=UI["grid"],
        zerolinecolor=UI["grid"],
        linecolor=UI["border"],
        tickfont=dict(color=UI["muted"]),
    )
    fig.update_annotations(font=dict(color=UI["muted"], size=11, family=UI["font"]))


def _candle_mrat_figure(
    symbol: str,
    panel: pd.DataFrame,
    *,
    sma_short: int,
    sma_long: int,
    regime_ok: bool,
) -> go.Figure:
    sym = str(symbol).strip().upper()
    ohlc, ohlc_src = _daily_ohlc_for_chart(sym, CANDLE_BARS, sma_short, sma_long)
    gran = str(config.TARGET_CANDLE_GRANULARITY)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.58, 0.42],
        subplot_titles=(
            f"{sym} daily ({ohlc_src or 'no OHLC'}) — SMA{sma_short} / SMA{sma_long}",
            f"MRAT vs 1+k·σ — regime {'risk-on' if regime_ok else 'risk-off'}",
        ),
    )
    if ohlc.empty:
        fig.update_layout(
            title=dict(
                text=(
                    f"{sym}: no OHLC for chart — try Alpaca data subscription, or ensure "
                    f"``data/ohlcv/{sym}_{gran}.db`` exists (fetcher + splitter)."
                ),
                font=dict(size=13, color=UI["text"], family=UI["font"]),
            ),
        )
        _finalize_mrat_figure(fig)
        return fig

    ohlc = _sanitize_ohlc_for_candlestick(ohlc, sma_short, sma_long)
    if ohlc.empty:
        fig.update_layout(
            title=dict(
                text=f"{sym}: OHLC rows invalid after cleaning (NaN/zero/bad high-low)",
                font=dict(size=13, color=UI["text"], family=UI["font"]),
            ),
        )
        _finalize_mrat_figure(fig)
        return fig

    x_row1 = ohlc["x_dt"]
    o = ohlc["open"].astype(float).tolist()
    hi = ohlc["high"].astype(float).tolist()
    lo = ohlc["low"].astype(float).tolist()
    cl = ohlc["close"].astype(float).tolist()

    fig.add_trace(
        go.Candlestick(
            x=x_row1,
            open=o,
            high=hi,
            low=lo,
            close=cl,
            name="OHLC",
            increasing_line_color="#34d399",
            decreasing_line_color="#f87171",
            increasing_fillcolor="rgba(52,211,153,0.35)",
            decreasing_fillcolor="rgba(248,113,113,0.35)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_row1,
            y=ohlc["sma_short"],
            mode="lines",
            line=dict(width=1.5, color=UI["accent"]),
            name=f"SMA{sma_short}",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_row1,
            y=ohlc["sma_long"],
            mode="lines",
            line=dict(width=1.5, color=UI["accent2"]),
            name=f"SMA{sma_long}",
        ),
        row=1,
        col=1,
    )

    ps = panel.loc[panel["ticker"] == sym].copy()
    if not ps.empty:
        ps["date"] = pd.to_datetime(ps["date"], utc=True)
        ps = ps.sort_values("date")
        ps["x_dt"] = ps["date"].dt.tz_convert("America/New_York").dt.tz_localize(None)
        lsm = float(getattr(config, "MAD_LONG_SIGMA_MULT", 1.0))
        ps["gate_long"] = 1.0 + lsm * ps["sigma"].astype(float)
        x2 = ps["x_dt"]
        fig.add_trace(
            go.Scatter(
                x=x2,
                y=ps["mrat"],
                mode="lines",
                line=dict(width=1.5, color=UI["accent"]),
                name="MRAT",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x2,
                y=ps["gate_long"],
                mode="lines",
                line=dict(width=1, color=UI["down"], dash="dash"),
                name="1 + k·σ",
            ),
            row=2,
            col=1,
        )
        long_days = ps.loc[ps["signal"].astype(int) == 1]
        if not long_days.empty:
            fig.add_trace(
                go.Scatter(
                    x=long_days["x_dt"],
                    y=long_days["mrat"],
                    mode="markers",
                    marker=dict(size=8, color=UI["up"], symbol="circle", line=dict(width=0)),
                    name="Long signal",
                ),
                row=2,
                col=1,
            )
    fig.add_hline(
        y=1.0,
        row=2,
        col=1,
        line=dict(color="rgba(228,228,231,0.2)", dash="dot", width=1),
    )

    _finalize_mrat_figure(fig)
    fig.update_yaxes(title_text="Price", title_font=dict(color=UI["muted"]), row=1, col=1)
    fig.update_yaxes(title_text="MRAT", title_font=dict(color=UI["muted"]), row=2, col=1)
    fig.update_xaxes(title_text="Date (US/Eastern)", row=2, col=1)
    return fig


def _fmt_money(x: Any) -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return "—"
        return f"${v:,.2f}"
    except (TypeError, ValueError):
        return "—"


def _positions_table(tc: TradingClient) -> list[dict[str, Any]]:
    out = []
    for p in tc.get_all_positions():
        out.append(
            {
                "symbol": str(p.symbol),
                "qty": str(p.qty),
                "market_value": _fmt_money(getattr(p, "market_value", None)),
                "avg_entry": _fmt_money(getattr(p, "avg_entry_price", None)),
                "current_price": _fmt_money(getattr(p, "current_price", None)),
                "unrealized_pl": _fmt_money(getattr(p, "unrealized_pl", None)),
            }
        )
    return sorted(out, key=lambda r: r["symbol"])


def _orders_rows(orders: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for o in orders:
        rows.append(
            {
                "symbol": str(getattr(o, "symbol", "")),
                "side": str(getattr(o, "side", "")),
                "qty": str(getattr(o, "qty", "")),
                "filled": str(getattr(o, "filled_qty", "")),
                "type": str(getattr(o, "type", "")),
                "status": str(getattr(o, "status", "")),
                "submitted": _format_order_submitted_et(
                    getattr(o, "submitted_at", "") or ""
                ),
            }
        )
    return rows


def _datatable_dark() -> dict[str, Any]:
    b = UI["border"]
    return {
        "style_table": {
            "overflowX": "auto",
            "borderRadius": "12px",
            "border": f"1px solid {b}",
            "backgroundColor": "#111118",
        },
        "style_cell": {
            "backgroundColor": UI["surface2"],
            "color": UI["text"],
            "border": f"1px solid {b}",
            "fontFamily": UI["font"],
            "fontSize": "12px",
            "padding": "10px 12px",
            "minWidth": "44px",
        },
        "style_header": {
            "backgroundColor": "#0f0f14",
            "color": UI["muted"],
            "fontWeight": "600",
            "fontSize": "10px",
            "letterSpacing": "0.08em",
            "textTransform": "uppercase",
            "border": f"1px solid {b}",
            "fontFamily": UI["font"],
            "padding": "12px 10px",
        },
        "style_data_conditional": [
            {"if": {"row_index": "odd"}, "backgroundColor": "#101016"},
        ],
        "css": [
            {
                "selector": ".dash-table-container .page-number",
                "rule": f"color: {UI['text']} !important;",
            },
            {
                "selector": ".dash-table-container .previous-next-container .arrow",
                "rule": f"color: {UI['muted']} !important;",
            },
        ],
    }


def build_app() -> Dash:
    paper = _paper_mode()
    try:
        key, secret = _alpaca_trading_keys(paper=paper)
        _tc = TradingClient(api_key=key, secret_key=secret, paper=paper)
        _tc.get_account()
        conn_err: str | None = None
    except Exception as exc:
        _tc = None
        conn_err = str(exc)

    app = Dash(
        __name__,
        assets_folder=str(_APP_DIR / "dash_assets"),
        suppress_callback_exceptions=True,
    )
    ref_sym = mad_reference_ticker()

    wl_table_kw = _datatable_dark()
    wl_table_kw["style_data_conditional"] = list(wl_table_kw["style_data_conditional"]) + [
        {"if": {"column_id": "reason"}, "textAlign": "left"}
    ]

    app.layout = html.Div(
        [
            html.Link(rel="preconnect", href="https://fonts.googleapis.com"),
            html.Link(
                rel="preconnect",
                href="https://fonts.gstatic.com",
                crossOrigin="anonymous",
            ),
            html.Link(
                rel="stylesheet",
                href=(
                    "https://fonts.googleapis.com/css2?"
                    "family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;"
                    "1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap"
                ),
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H2(
                                        "DeepVibe MAD",
                                        className="dv-title",
                                        style={"display": "inline"},
                                    ),
                                    html.Span("LIVE", className="dv-badge"),
                                ],
                                style={"marginBottom": "6px"},
                            ),
                            html.P(
                                (
                                    f"Alpaca · {'paper' if paper else 'live'} trading"
                                    + (f" · {conn_err}" if conn_err else "")
                                ),
                                className="dv-subtitle" + (" dv-error" if conn_err else ""),
                            ),
                        ],
                        className="dv-header",
                    ),
                    html.Div(
                        [
                            html.H4("Equity curve", className="dv-section-title"),
                            html.P(
                                "Times are US/Eastern (EST/EDT). Hover anywhere on the chart for "
                                "equity (interpolated along the line) and a vertical crosshair.",
                                className="dv-subtitle",
                                style={"marginBottom": "10px", "fontSize": "0.82rem"},
                            ),
                            dcc.RadioItems(
                                id="equity-range",
                                options=[
                                    {"label": " 24h ", "value": "24h"},
                                    {"label": " 1W ", "value": "1w"},
                                    {"label": " 1M ", "value": "1m"},
                                    {"label": " 1Y ", "value": "1y"},
                                    {"label": " YTD ", "value": "ytd"},
                                ],
                                value="24h",
                                inline=True,
                                className="dv-radio",
                                style={"marginBottom": "14px"},
                            ),
                            dcc.Store(id="equity-latest-store", data=None),
                            dcc.Store(id="live-account-store", data=None),
                            html.Div(id="equity-tracker"),
                            html.Div(
                                dcc.Graph(
                                    id="equity-graph",
                                    clear_on_unhover=False,
                                    config=GRAPH_CONFIG,
                                ),
                                className="dv-graph-wrap",
                            ),
                        ],
                        className="dv-card dv-card--chart",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    dcc.Loading(
                                        id="candle-mrat-loading",
                                        type="circle",
                                        color=UI["accent"],
                                        className="dv-candle-loading",
                                        style={"minHeight": "480px"},
                                        parent_className="dv-candle-loading-parent",
                                        children=[
                                            html.Label(
                                                "Chart symbol", className="dv-label"
                                            ),
                                            dcc.Dropdown(
                                                id="chart-symbol",
                                                clearable=False,
                                                value=ref_sym,
                                                options=[
                                                    {
                                                        "label": ref_sym,
                                                        "value": ref_sym,
                                                    }
                                                ],
                                                className="dv-dropdown",
                                                style={
                                                    "color": UI["text"],
                                                    "backgroundColor": UI["surface2"],
                                                },
                                            ),
                                            html.Div(
                                                dcc.Graph(
                                                    id="candle-mrat-graph",
                                                    config=GRAPH_CONFIG,
                                                ),
                                                className="dv-graph-wrap",
                                                style={"marginTop": "14px"},
                                            ),
                                        ],
                                    ),
                                ],
                                className="dv-col-main",
                            ),
                            html.Div(
                                [
                                    html.H4("Watchlist", className="dv-section-title"),
                                    html.P(
                                        "Target weight · MRAT gate · sizing",
                                        style={
                                            "fontSize": "0.8rem",
                                            "color": UI["muted"],
                                            "margin": "0 0 14px 0",
                                            "lineHeight": "1.4",
                                        },
                                    ),
                                    dash_table.DataTable(
                                        id="watchlist-table",
                                        columns=[
                                            {"name": "Ticker", "id": "ticker"},
                                            {
                                                "name": "Weight %",
                                                "id": "weight_pct",
                                                "type": "numeric",
                                            },
                                            {"name": "Sig", "id": "signal", "type": "numeric"},
                                            {"name": "Decile", "id": "decile", "type": "numeric"},
                                            {"name": "MRAT", "id": "mrat", "type": "numeric"},
                                            {"name": "sigma", "id": "sigma", "type": "numeric"},
                                            {
                                                "name": "1+k*sigma",
                                                "id": "one_plus_k_sigma",
                                                "type": "numeric",
                                            },
                                            {"name": "Reason", "id": "reason"},
                                        ],
                                        page_size=15,
                                        sort_action="native",
                                        data=[],
                                        **wl_table_kw,
                                    ),
                                ],
                                className="dv-col-side dv-card",
                                style={"padding": "18px 20px"},
                            ),
                        ],
                        className="dv-grid",
                    ),
                    html.Div(className="dv-hr"),
                    html.Div(
                        [
                            html.H4("Portfolio & MRAT", className="dv-section-title"),
                            html.Div(
                                id="portfolio-stats",
                                children=[
                                    html.P(
                                        "Loading Alpaca account…",
                                        className="dv-placeholder",
                                    )
                                ],
                            ),
                            html.Div(
                                id="mrat-summary",
                                children=[
                                    html.P(
                                        "MRAT / watchlist: building from local OHLCV SQLite (~100 names). "
                                        "First load often takes 1–5+ minutes; the table fills when ready. "
                                        "Watch the terminal for “MRAT panel cache ready”.",
                                        className="dv-mrat-summary",
                                    )
                                ],
                            ),
                        ],
                        className="dv-card",
                    ),
                    html.H4(
                        "Open positions",
                        className="dv-section-title",
                        style={"marginTop": "28px"},
                    ),
                    dash_table.DataTable(
                        id="positions-table",
                        columns=[
                            {"name": "Symbol", "id": "symbol"},
                            {"name": "Qty", "id": "qty"},
                            {"name": "Mkt value", "id": "market_value"},
                            {"name": "Avg entry", "id": "avg_entry"},
                            {"name": "Last", "id": "current_price"},
                            {"name": "Unreal P/L", "id": "unrealized_pl"},
                        ],
                        page_size=12,
                        data=[],
                        **_datatable_dark(),
                    ),
                    html.H4(
                        "Open orders",
                        className="dv-section-title",
                        style={"marginTop": "22px"},
                    ),
                    dash_table.DataTable(
                        id="open-orders-table",
                        columns=[
                            {"name": "Symbol", "id": "symbol"},
                            {"name": "Side", "id": "side"},
                            {"name": "Qty", "id": "qty"},
                            {"name": "Filled", "id": "filled"},
                            {"name": "Type", "id": "type"},
                            {"name": "Status", "id": "status"},
                            {"name": "Submitted (ET)", "id": "submitted"},
                        ],
                        page_size=12,
                        data=[],
                        **_datatable_dark(),
                    ),
                    html.H4(
                        "Order history (recent closed)",
                        className="dv-section-title",
                        style={"marginTop": "22px"},
                    ),
                    dash_table.DataTable(
                        id="closed-orders-table",
                        columns=[
                            {"name": "Symbol", "id": "symbol"},
                            {"name": "Side", "id": "side"},
                            {"name": "Qty", "id": "qty"},
                            {"name": "Filled", "id": "filled"},
                            {"name": "Type", "id": "type"},
                            {"name": "Status", "id": "status"},
                            {"name": "Submitted (ET)", "id": "submitted"},
                        ],
                        page_size=15,
                        data=[],
                        **_datatable_dark(),
                    ),
                    dcc.Interval(
                        id="tick",
                        interval=REFRESH_MS,
                        n_intervals=0,
                        max_intervals=-1,
                    ),
                ],
                className="dv-root",
            ),
        ]
    )

    @app.callback(
        Output("equity-graph", "figure"),
        Output("equity-latest-store", "data"),
        Input("tick", "n_intervals"),
        Input("equity-range", "value"),
        prevent_initial_call=False,
    )
    def _refresh_equity_chart(_tick_n, equity_range):
        """Alpaca portfolio history: selected range, US/Eastern; 24h uses a fixed rolling window."""
        disconnected = _empty_equity_figure("Account equity (disconnected)")
        if _tc is None:
            return disconnected, None
        er = equity_range or "24h"
        try:
            eq_fig, snap = _equity_figure_and_snapshot(_tc, er)
            return eq_fig, snap
        except Exception as exc:
            return _empty_equity_figure(f"Account / orders error: {exc}"), None

    @app.callback(
        Output("equity-tracker", "children"),
        Input("equity-graph", "hoverData"),
        Input("equity-latest-store", "data"),
        Input("live-account-store", "data"),
        prevent_initial_call=False,
    )
    def _equity_tracker(hover, latest, live_snap):
        baseline = (latest or {}).get("baseline_equity")
        if hover and hover.get("points"):
            pt = hover["points"][0]
            y = pt.get("y")
            xv = pt.get("x")
            pct_t, pct_c = _pct_vs_baseline_label(y, baseline)
            return _equity_tracker_block(
                _fmt_money(y),
                _parse_equity_hover_ts(xv),
                caption="Hover — US/Eastern",
                pct_text=pct_t,
                pct_color=pct_c,
            )
        live_eq = None
        ts_disp = "—"
        if live_snap and live_snap.get("live_equity") is not None:
            live_eq = live_snap["live_equity"]
            ts_disp = str(live_snap.get("as_of_display") or "—")
        elif latest and latest.get("chart_last_equity") is not None:
            live_eq = latest["chart_last_equity"]
            ts_disp = str(latest.get("chart_ts_display") or "—")
        if live_eq is None:
            return html.P(
                "Equity appears after the first successful chart load.",
                className="dv-placeholder",
            )
        pct_t, pct_c = _pct_vs_baseline_label(live_eq, baseline)
        er = str((latest or {}).get("equity_range") or "")
        cap_tail = (
            "vs 24h window start"
            if er == "24h"
            else "vs range start (first point on chart)"
        )
        return _equity_tracker_block(
            _fmt_money(live_eq),
            ts_disp,
            caption=f"Live account · US/Eastern ({cap_tail})",
            pct_text=pct_t,
            pct_color=pct_c,
        )

    @app.callback(
        Output("portfolio-stats", "children"),
        Output("positions-table", "data"),
        Output("open-orders-table", "data"),
        Output("closed-orders-table", "data"),
        Output("live-account-store", "data"),
        Input("tick", "n_intervals"),
        prevent_initial_call=False,
    )
    def _refresh_portfolio_and_orders(_n):
        """Account snapshot, positions, and orders (20s) — independent of equity chart cadence."""
        if _tc is None:
            return (
                [html.P("Connect Alpaca API keys in .env", className="dv-placeholder")],
                [],
                [],
                [],
                None,
            )
        try:
            acct = _tc.get_account()
            live_snap = {
                "live_equity": float(acct.equity),
                "as_of_display": datetime.now(_NY).strftime("%Y-%m-%d %H:%M %Z"),
            }
            stats = html.Div(
                [
                    html.Table(
                        [
                            html.Tr([html.Th("Field"), html.Th("Value")]),
                            html.Tr([html.Td("Equity"), html.Td(_fmt_money(acct.equity))]),
                            html.Tr([html.Td("Cash"), html.Td(_fmt_money(acct.cash))]),
                            html.Tr(
                                [
                                    html.Td("Buying power"),
                                    html.Td(_fmt_money(acct.buying_power)),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Portfolio value"),
                                    html.Td(
                                        _fmt_money(getattr(acct, "portfolio_value", None))
                                    ),
                                ]
                            ),
                        ],
                        className="dv-portfolio-table",
                    ),
                    _portfolio_performance_section(_tc, acct),
                ]
            )
            pos = _positions_table(_tc)
            try:
                open_o = _tc.get_orders(
                    filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=200)
                )
                open_rows = _orders_rows(list(open_o) if open_o else [])
            except Exception:
                open_rows = []
            try:
                closed_o = _tc.get_orders(
                    filter=GetOrdersRequest(
                        status=QueryOrderStatus.CLOSED,
                        limit=ORDER_HISTORY_LIMIT,
                    )
                )
                closed_rows = _orders_rows(list(closed_o) if closed_o else [])
            except Exception:
                closed_rows = []
        except Exception as exc:
            return (
                [html.P(str(exc), className="dv-mrat-summary dv-error")],
                [],
                [],
                [],
                None,
            )
        return ([stats], pos, open_rows, closed_rows, live_snap)

    @app.callback(
        Output("candle-mrat-graph", "figure"),
        Output("watchlist-table", "data"),
        Output("chart-symbol", "options"),
        Output("mrat-summary", "children"),
        Output("chart-symbol", "value"),
        Input("tick", "n_intervals"),
        Input("chart-symbol", "value"),
        prevent_initial_call=False,
    )
    def _refresh_panel(_n, chart_symbol):
        """Slow path: full MRAT panel from ``data/ohlcv/*.db`` (can take minutes when cold)."""
        busy = _empty_equity_figure(
            "MRAT chart: building panel from SQLite (see terminal / mrat-summary below)…"
        )
        if _tc is None:
            return (
                busy,
                [],
                [{"label": ref_sym, "value": ref_sym}],
                html.P(
                    "Alpaca not connected — MRAT panel skipped.",
                    className="dv-mrat-summary dv-error",
                ),
                ref_sym,
            )

        try:
            panel, snap, sub, params = _load_panel_bundle()
            sh, lo, ex, reg_ma, reg_tick = params
            regime_sym = (reg_tick or "").strip().upper() or (
                mad_regime_ticker_symbol() or "QQQ"
            )
            wl = mad_live_watchlist_table(
                sub,
                regime_ok=snap.regime_ok,
                weight_by_ticker=snap.weight_by_ticker,
                universe=snap.tickers,
                direction_mode=str(getattr(config, "MAD_DIRECTION_MODE", "long_only")),
                exit_ma_period=ex,
            )
            opts = [{"label": r["ticker"], "value": r["ticker"]} for r in wl]
            valid_syms = {o["value"] for o in opts}
            sym_in = (chart_symbol or ref_sym).strip().upper()
            if sym_in in valid_syms:
                sym = sym_in
            else:
                sym = next(iter(valid_syms)) if valid_syms else ref_sym
            prev_u = (chart_symbol or "").strip().upper()
            dd_val = (
                sym
                if (not prev_u or prev_u not in valid_syms or prev_u != sym)
                else no_update
            )
            candle_fig = _candle_mrat_figure(
                sym,
                panel,
                sma_short=sh,
                sma_long=lo,
                regime_ok=snap.regime_ok,
            )
            summary = html.P(
                f"MRAT as-of (US/Eastern): {_snap_as_of_ny_str(snap.as_of)} | SMA {sh}/{lo} | "
                f"exit MA={ex or 'off'} | regime MA={reg_ma or 'off'} ({regime_sym}) | "
                f"risk-on={snap.regime_ok} | panel long/short counts: {snap.n_long}/{snap.n_short}",
                className="dv-mrat-summary",
            )
            return candle_fig, wl, opts, summary, dd_val
        except Exception as exc:
            return (
                _empty_equity_figure(f"MRAT panel error: {exc}"),
                [],
                [{"label": ref_sym, "value": ref_sym}],
                html.P(f"MRAT panel: {exc}", className="dv-mrat-summary dv-error"),
                ref_sym,
            )

    if _tc is not None:
        _start_panel_cache_warmer()

    return app


def main() -> None:
    print(
        "[live_dashboard] Starting server. "
        "Equity/orders load first; MRAT watchlist follows SQLite panel build (warm-up in background)."
    )
    app = build_app()
    app.run(debug=False, host="0.0.0.0", port=DASHBOARD_PORT, threaded=True)


if __name__ == "__main__":
    main()
