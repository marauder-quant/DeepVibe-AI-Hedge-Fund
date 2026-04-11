"""
Live Alpaca bot for MAD / MRAT (equal-weight panel from local OHLCV SQLite).

Reads the same ``data/ohlcv/*.db`` files as ``mad.backtester`` (run ``alpaca_fetcher`` first so closes
are current). Optional ``MAD_LIVE_OHLCV_*`` checks staleness vs the reference ticker; optional
``MAD_LIVE_REFRESH_SPLITTER_DB`` re-runs ``data_splitter`` so ``sma_*`` stay in sync. When
``MAD_LIVE_USE_PRECOMPUTED_SMA`` is True (default), the live snapshot reads those columns for
MRAT (and regime on **1d** bars); otherwise it rolls SMAs from **close** only.

Parameters default to the ``summary`` table in ``{MAD_DATA_DIR}/{ref}_{gran}_mad_optim.db`` when
``MAD_LIVE_LOAD_PARAMS_FROM_DB`` is True.

``MAD_LIVE_REGIME_OFF_PROXY_TICKER`` (e.g. BIL): when regime is risk-off (e.g. QQQ below its SMA),
``MAD_LIVE_REGIME_OFF_CLOSE_ALL_NON_PROXY`` can flatten the whole account into that sleeve using
``MAD_LIVE_REGIME_OFF_EQUITY_FRACTION`` of equity. The sleeve is treated as cash-like: no OHLCV DB or
splitter work; order sizing uses an Alpaca **market quote** only.

**Rebalance schedule:** in the default long-running mode (no ``--once``), the bot does **not** trade on
every poll; it runs **one** reconcile per **NYSE session day**, **after** that day’s session close
from Alpaca’s calendar, and only if ``MAD_LIVE_REBALANCE_WINDOW_MINUTES`` allows (default places orders
in the first ~90 minutes after that close so a process started at 10 p.m. does not trade that session).
``MAD_LIVE_POLL_SECONDS`` is how often it wakes to check the clock. Use ``--once`` for an immediate
single cycle (e.g. cron right after the bell). Set ``MAD_LIVE_REBALANCE_WINDOW_MINUTES = 0`` to allow
submission any time after close (legacy).

Printed log timestamps and ``as_of`` lines use **US/Eastern** (``America/New_York``); internal
throttles (e.g. splitter once per UTC day) remain UTC-based where noted.

For after-hours tests: set ``MAD_LIVE_EXTENDED_HOURS_ORDERS = True``. Extended-hours **limit** prices
default to a live **ask/bid** anchor (see ``MAD_LIVE_EXT_HRS_LIMIT_FROM_DAILY_CLOSE``); printed ``px=``
is still MRAT's daily close for sizing context.

With ``MAD_LIVE_APPEND_DAILY_OHLCV`` (default True), each non-dry-run cycle **starts** by appending new
``1d`` bars from Alpaca into ``data/ohlcv/*.db`` and refreshing ``sma_21``/``sma_200`` (regime ETF:
``sma_200`` only), then runs MRAT on the updated DB.

Usage:
    PYTHONPATH=src python -m deepvibe_hedge.mad.live_bot --dry-run
    PYTHONPATH=src python -m deepvibe_hedge.mad.live_bot --once
    PYTHONPATH=src python -m deepvibe_hedge.mad.live_bot
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from alpaca.trading.client import TradingClient
from alpaca.trading.models import Calendar
from alpaca.trading.requests import GetCalendarRequest

from deepvibe_hedge import config
from deepvibe_hedge.alpaca_asset import _alpaca_trading_keys
from deepvibe_hedge.mad.backtester import (
    compute_mad_live_snapshot,
    mad_reference_ticker,
    mad_regime_ticker_symbol,
    mad_universe_tickers,
)
from deepvibe_hedge.alpaca_live import (
    _apply_live_short_constraints,
    _get_current_qty,
    _latest_stock_trade_price,
    _reconcile_symbol_net_qty,
    _round_alpaca_qty,
)
from deepvibe_hedge.data_splitter import run_pipeline_for_ticker
from deepvibe_hedge.mad.ohlcv_health import audit_mad_ohlcv_panel, print_health_report
from deepvibe_hedge.ohlcv_live_append import append_latest_daily_for_universe, summarize_append_status
from deepvibe_hedge.paths import MAD_DATA_DIR, OHLCV_DIR

_LAST_SPLITTER_REFRESH_UTC_DATE: date | None = None
_LAST_EOD_REBALANCE_SESSION_DATE: date | None = None

_ET = ZoneInfo("America/New_York")


def _fmt_now_et() -> str:
    """Log line timestamp in US/Eastern (EST/EDT)."""
    return datetime.now(_ET).strftime("%Y-%m-%d %H:%M:%S %Z")


def _snap_as_of_et_str(snap_as_of: object) -> str:
    """Panel ``as_of`` (stored UTC) → US/Eastern display."""
    t = pd.Timestamp(snap_as_of)
    if t.tzinfo is None:
        t = t.tz_localize(timezone.utc)
    else:
        t = t.tz_convert(timezone.utc)
    return t.tz_convert(_ET).strftime("%Y-%m-%d %H:%M %Z")


def _alpaca_calendar_open_close_to_et(dt: datetime) -> datetime:
    """
    Alpaca's ``Calendar`` model builds naive ``open``/``close`` from ``date`` + ``%H:%M`` strings —
    NYSE **local** wall time, not UTC. Tagging naive as UTC shifts the bell by several hours.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_ET)
    return dt.astimezone(_ET)


def _trading_session_for_date(
    tc: TradingClient, d: date
) -> tuple[Calendar | None, str | None]:
    """
    Returns ``(row, None)`` when the calendar call succeeds.

    ``(None, None)`` means no row (weekend/holiday). ``(None, err)`` means the API call failed
    (transient); callers must not treat that as a holiday.
    """
    try:
        rows = tc.get_calendar(GetCalendarRequest(start=d, end=d))
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not rows:
        return None, None
    return rows[0], None


def _eod_rebalance_should_run(tc: TradingClient) -> tuple[bool, str]:
    """
    True when (1) today is a trading session, (2) now is on or after session close in ET,
    (3) now is within ``MAD_LIVE_REBALANCE_WINDOW_MINUTES`` of that close (unless 0 = disabled),
    and (4) this session date is not already handled (rebalanced or skipped past window).
    """
    global _LAST_EOD_REBALANCE_SESSION_DATE
    now_et = datetime.now(_ET)
    d = now_et.date()
    sess, cal_err = _trading_session_for_date(tc, d)
    if cal_err is not None:
        return False, f"calendar API error — {cal_err}"
    if sess is None:
        return False, f"{d} — no exchange session (weekend/holiday)"
    close_et = _alpaca_calendar_open_close_to_et(sess.close)
    if now_et < close_et:
        return (
            False,
            f"before session close ({close_et.strftime('%Y-%m-%d %H:%M %Z')})",
        )
    sd = sess.date
    if _LAST_EOD_REBALANCE_SESSION_DATE == sd:
        return False, f"EOD already handled for session {sd}"

    win_min = int(getattr(config, "MAD_LIVE_REBALANCE_WINDOW_MINUTES", 0) or 0)
    win_end: datetime | None = None
    if win_min > 0:
        win_end = close_et + timedelta(minutes=win_min)
        if now_et > win_end:
            _LAST_EOD_REBALANCE_SESSION_DATE = sd
            return (
                False,
                f"past post-close window (close+{win_min}m → {win_end.strftime('%H:%M %Z')}); "
                f"skipped session {sd} — run with --once if you need a late manual rebalance",
            )

    win_note = (
        f", window +{win_min}m to {win_end.strftime('%H:%M %Z')}"
        if win_min > 0 and win_end is not None
        else ""
    )
    return True, f"EOD after close ({close_et.strftime('%H:%M %Z')}{win_note})"


def _ohlcv_health_reference_ticker() -> str:
    """
    Calendar for OHLCV gap/staleness checks. Default follows ``mad_reference_ticker()`` (QQQ panel clock).

    Override with ``MAD_LIVE_HEALTH_REFERENCE_TICKER`` when you need a different health baseline.
    """
    h = getattr(config, "MAD_LIVE_HEALTH_REFERENCE_TICKER", None)
    if h is not None and str(h).strip():
        return str(h).strip().upper()
    return mad_reference_ticker().strip().upper()


def _alpaca_ping_account(tc: TradingClient) -> None:
    """Validate API connectivity; retries on transient TLS / network failures."""
    n = max(1, int(getattr(config, "MAD_LIVE_ALPACA_CONNECT_RETRIES", 5)))
    base = float(getattr(config, "MAD_LIVE_ALPACA_CONNECT_RETRY_SEC", 2.0))
    last: BaseException | None = None
    for i in range(n):
        try:
            tc.get_account()
            return
        except (requests.exceptions.RequestException, OSError) as exc:
            last = exc
            if i + 1 < n:
                wait = base * float(i + 1)
                print(
                    f"  [Alpaca] connect failed ({type(exc).__name__}: {exc}); "
                    f"retry {i + 2}/{n} in {wait:.1f}s..."
                )
                time.sleep(wait)
    assert last is not None
    raise last


def _mad_optim_db_path() -> Path:
    ref = mad_reference_ticker()
    gran = str(config.TARGET_CANDLE_GRANULARITY)
    return MAD_DATA_DIR / f"{ref}_{gran}_mad_optim.db"


def load_mad_live_strategy_params() -> tuple[int, int, int, int, str | None]:
    """
    (sma_short, sma_long, exit_ma, regime_ma, regime_ticker_symbol).

    If ``MAD_LIVE_LOAD_PARAMS_FROM_DB`` and ``summary`` exists: start from that row, then apply any
    ``MAD_LIVE_*`` that is not None. Otherwise use strategy defaults, then the same overrides.
    """
    sh: int
    lo: int
    ex: int
    rg: int
    rt: str | None
    loaded_db = False

    use_db = bool(getattr(config, "MAD_LIVE_LOAD_PARAMS_FROM_DB", True))
    path = _mad_optim_db_path()
    if use_db and path.exists():
        with sqlite3.connect(path) as con:
            cur = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='summary'"
            )
            if cur.fetchone():
                df = pd.read_sql("SELECT * FROM summary LIMIT 1", con)
                if not df.empty:
                    row = df.iloc[0]
                    sh = int(row["mad_sma_short"])
                    lo = int(row["mad_sma_long"])
                    ex = int(row.get("mad_exit_ma", 0) or 0)
                    rg = int(row.get("mad_regime_ma", 0) or 0)
                    rt = str(row.get("mad_regime_ticker", "") or "").strip() or None
                    loaded_db = True

    if not loaded_db:
        sh = int(getattr(config, "MAD_SMA_SHORT", 21))
        lo = int(getattr(config, "MAD_SMA_LONG", 200))
        ex = int(getattr(config, "MAD_EXIT_MA_PERIOD", 0) or 0)
        rg = 0
        rt = mad_regime_ticker_symbol()

    def _apply_live_int(name: str, current: int) -> int:
        v = getattr(config, name, None)
        return int(v) if v is not None else current

    sh = _apply_live_int("MAD_LIVE_SMA_SHORT", sh)
    lo = _apply_live_int("MAD_LIVE_SMA_LONG", lo)
    ex = _apply_live_int("MAD_LIVE_EXIT_MA", ex)
    rg = _apply_live_int("MAD_LIVE_REGIME_MA", rg)

    live_sym = getattr(config, "MAD_LIVE_REGIME_TICKER", None)
    if live_sym is not None:
        s = str(live_sym).strip().upper()
        rt = s if s else None

    return sh, lo, ex, rg, rt


def _maybe_refresh_splitter_dbs(*, force: bool = False) -> None:
    """
    Recompute splitter indicators into each pipeline symbol DB (same as ``python -m deepvibe_hedge.data_splitter``).
    """
    global _LAST_SPLITTER_REFRESH_UTC_DATE
    if not bool(getattr(config, "MAD_LIVE_REFRESH_SPLITTER_DB", False)):
        return
    today = datetime.now(timezone.utc).date()
    if (
        not force
        and bool(getattr(config, "MAD_LIVE_REFRESH_SPLITTER_ONCE_PER_UTC_DAY", True))
        and _LAST_SPLITTER_REFRESH_UTC_DATE == today
    ):
        return
    syms = tuple(config.ohlcv_pipeline_tickers())
    print(
        f"\n[{_fmt_now_et()}] [MAD live] Splitter refresh: {len(syms)} symbol(s) → "
        "SMA columns + splits written to OHLCV DBs..."
    )
    failed: list[tuple[str, str]] = []
    for sym in syms:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                run_pipeline_for_ticker(sym)
        except Exception as exc:
            failed.append((sym, str(exc)))
    if failed:
        for s, err in failed[:15]:
            print(f"  [splitter] FAILED {s}: {err}")
        if len(failed) > 15:
            print(f"  [splitter] ... +{len(failed) - 15} more failures")
        print(
            "  [splitter] Throttle (UTC calendar day) not advanced — fix errors; "
            "will retry next poll."
        )
        return
    print(f"  [splitter] OK — updated {len(syms)} DB/CSV pair(s).")
    _LAST_SPLITTER_REFRESH_UTC_DATE = today


def _run_ohlcv_health_check(reg_ma: int, reg_tick: str | None) -> None:
    if not bool(getattr(config, "MAD_LIVE_OHLCV_HEALTH_CHECK", True)):
        return
    ref = _ohlcv_health_reference_ticker()
    panel = mad_universe_tickers()
    extra: list[str] = []
    rma = int(reg_ma or 0)
    if rma > 0:
        et = _display_regime_ticker(rma, reg_tick)
        if et != "off":
            extra.append(et)
    report = audit_mad_ohlcv_panel(
        ohlcv_dir=OHLCV_DIR,
        granularity=str(config.TARGET_CANDLE_GRANULARITY),
        ref_ticker=ref,
        panel_symbols=panel,
        extra_symbols=tuple(extra),
        recent_ref_bars=int(getattr(config, "MAD_LIVE_OHLCV_RECENT_REF_BARS", 60)),
        max_stale_calendar_days=int(getattr(config, "MAD_LIVE_OHLCV_MAX_STALE_CALENDAR_DAYS", 1)),
    )
    print(
        f"\n[MAD live] OHLCV health (panel + extras vs ref={ref}; "
        f"MAD panel calendar symbol={mad_reference_ticker()})\n"
    )
    print_health_report(report)
    if not report.ok and bool(getattr(config, "MAD_LIVE_ABORT_ON_OHLCV_ISSUES", False)):
        raise RuntimeError("MAD live: OHLCV health check failed — fix fetch/split or relax config.")


def _display_regime_ticker(reg_ma: int, reg_tick: str | None) -> str:
    if int(reg_ma or 0) <= 0:
        return "off"
    if reg_tick:
        return reg_tick
    sym = mad_regime_ticker_symbol()
    return sym or "QQQ"


def _gross_notional_usd(trading_client: TradingClient) -> float:
    frac = float(getattr(config, "MAD_LIVE_EQUITY_FRACTION", 0.98))
    cap = getattr(config, "MAD_LIVE_MAX_GROSS_USD", None)
    acct = trading_client.get_account()
    equity = float(acct.equity)
    raw = max(0.0, equity * frac)
    if cap is not None:
        raw = min(raw, float(cap))
    return raw


def _regime_off_sleeve_notional_usd(trading_client: TradingClient) -> float:
    """Notional to hold in BIL (or other sleeve) when regime is risk-off — full-equity pivot."""
    frac = float(getattr(config, "MAD_LIVE_REGIME_OFF_EQUITY_FRACTION", 0.995))
    cap = getattr(config, "MAD_LIVE_MAX_GROSS_USD", None)
    acct = trading_client.get_account()
    equity = float(acct.equity)
    raw = max(0.0, equity * frac)
    if cap is not None:
        raw = min(raw, float(cap))
    return raw


def _last_close_from_ohlcv_db(symbol: str) -> float:
    """Latest close in local OHLCV SQLite (same path convention as fetcher/splitter)."""
    sym = str(symbol).strip().upper()
    path = OHLCV_DIR / f"{sym}_{config.TARGET_CANDLE_GRANULARITY}.db"
    if not path.exists():
        return float("nan")
    with sqlite3.connect(path) as con:
        row = con.execute(
            "SELECT close FROM ohlcv ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
    if not row or row[0] is None:
        return float("nan")
    return float(row[0])


def _sleeve_market_price(symbol: str, *, paper: bool) -> float:
    """Regime sleeve (e.g. BIL): no OHLCV pipeline — quote from Alpaca for sizing/limits only."""
    try:
        return _latest_stock_trade_price(str(symbol).strip().upper(), paper=paper)
    except Exception:
        return float("nan")


def _px_for_reconcile(
    symbol: str,
    close_by_ticker: dict[str, float],
    *,
    paper: bool,
) -> float:
    """Reference price for extended-hours limits: snapshot → OHLCV → Alpaca last trade."""
    sym = str(symbol).strip().upper()
    p = float(close_by_ticker.get(sym, float("nan")))
    if np.isfinite(p) and p > 0:
        return p
    p2 = _last_close_from_ohlcv_db(sym)
    if np.isfinite(p2) and p2 > 0:
        return p2
    return _latest_stock_trade_price(sym, paper=paper)


def _fmt_net_qty(q: float | int, *, fractional: bool) -> str:
    if fractional:
        s = f"{float(q):+.6f}".rstrip("0").rstrip(".")
        return s if s not in ("+", "-") else f"{float(q):+g}"
    return f"{int(round(float(q))):+d}"


def _desired_qty_signed(
    weight: float, gross_usd: float, price: float, *, fractional: bool
) -> float:
    if not np.isfinite(price) or price <= 0.0:
        return 0.0
    usd = float(weight) * float(gross_usd)
    if abs(usd) < 1e-9:
        return 0.0
    if fractional:
        return float(math.copysign(round(abs(usd) / price, 6), usd))
    mag = abs(usd) / price
    return float(int(math.copysign(int(math.floor(mag)), usd)))


def _flatten_account_except_proxy(
    trading_client: TradingClient,
    *,
    proxy_sym: str,
    close_by_ticker: dict[str, float],
    ext_hrs: bool,
    paper: bool,
    fractional: bool,
) -> None:
    """Sell/cover every open position except the sleeve ETF (regime-off full pivot)."""
    px = proxy_sym.strip().upper()
    for pos in trading_client.get_all_positions():
        sym = str(pos.symbol).strip().upper()
        if sym == px:
            continue
        cur_f = float(pos.qty)
        cur = round(cur_f, 6) if fractional else int(round(cur_f))
        if abs(float(cur)) < 1e-8 if fractional else cur == 0:
            continue
        ref_px = _px_for_reconcile(sym, close_by_ticker, paper=paper)
        desired = 0
        dq_clamped, short_note = _apply_live_short_constraints(
            trading_client, sym, desired, fractional=fractional
        )
        cur_disp = _fmt_net_qty(cur, fractional=fractional)
        tgt_disp = _fmt_net_qty(dq_clamped, fractional=fractional)
        print(
            f"  {sym}: regime-off flatten px={ref_px:.4f} desired_net={tgt_disp} current={cur_disp}"
            f"{short_note}"
        )
        if dq_clamped != desired:
            desired = dq_clamped
        if (fractional and abs(float(desired) - float(cur)) >= 1e-8) or (
            not fractional and int(desired) != int(cur)
        ):
            _reconcile_symbol_net_qty(
                trading_client,
                sym,
                desired,
                extended_hours=ext_hrs,
                reference_price=ref_px if (ext_hrs or fractional) else None,
                paper=paper,
                fractional=fractional,
            )


def _run_cycle(
    trading_client: TradingClient | None,
    *,
    dry_run: bool,
    min_order_usd: float,
    paper: bool = True,
) -> None:
    if not dry_run and bool(getattr(config, "MAD_LIVE_APPEND_DAILY_OHLCV", True)):
        try:
            st = append_latest_daily_for_universe(quiet=True)
            summ = summarize_append_status(st)
            if summ:
                print(f"[ohlcv_append] {summ}")
        except Exception as exc:
            print(f"[ohlcv_append] ERROR: {exc}")

    et_now = _fmt_now_et()
    sh, lo, ex, reg_ma, reg_tick = load_mad_live_strategy_params()
    snap = compute_mad_live_snapshot(
        short_w=sh,
        long_w=lo,
        exit_ma_period=ex,
        regime_ma_period=reg_ma,
        regime_ticker=reg_tick,
    )
    rma = int(snap.mad_regime_ma or 0)
    reg_line = (
        "regime off"
        if rma <= 0
        else f"regime MA={rma} ETF={_display_regime_ticker(rma, reg_tick)!r}"
    )
    proxy_raw = getattr(config, "MAD_LIVE_REGIME_OFF_PROXY_TICKER", None)
    proxy_sym = str(proxy_raw).strip().upper() if proxy_raw else ""
    use_bil_sleeve = bool(proxy_sym) and not snap.regime_ok
    close_all_np = bool(
        use_bil_sleeve
        and getattr(config, "MAD_LIVE_REGIME_OFF_CLOSE_ALL_NON_PROXY", True)
    )
    ext_hrs = bool(getattr(config, "MAD_LIVE_EXTENDED_HOURS_ORDERS", False))
    frac = bool(getattr(config, "MAD_LIVE_FRACTIONAL_SHARES", True))

    if not dry_run:
        assert trading_client is not None
        gross = _gross_notional_usd(trading_client)
        sleeve_notional = (
            _regime_off_sleeve_notional_usd(trading_client) if use_bil_sleeve else float("nan")
        )
    else:
        gross = float("nan")
        sleeve_notional = float("nan")

    print(
        f"\n[{et_now}] MAD live cycle\n"
        f"  as_of (US/Eastern): {_snap_as_of_et_str(snap.as_of)}  (last panel bar)\n"
        f"  MRAT SMA         : {snap.mad_sma_short}/{snap.mad_sma_long} | exit MA={snap.mad_exit_ma or 'off'} | "
        f"{reg_line}\n"
        f"  regime risk-on   : {snap.regime_ok}\n"
        f"  raw long / short : {snap.n_long} / {snap.n_short}\n"
        f"  regime sleeve    : {proxy_sym or 'cash'} (active={'yes' if use_bil_sleeve else 'no — MRAT book'})\n"
        f"  extended_hours   : {ext_hrs}\n"
        f"  fractional_shares: {frac}\n"
    )

    if dry_run:
        print("  [dry-run] targets (weight, close, implied USD leg @ $100k gross example):")
        ex_gross = 100_000.0
        for t in snap.tickers:
            w = snap.weight_by_ticker.get(t, 0.0)
            px = snap.close_by_ticker.get(t, float("nan"))
            leg = ex_gross * abs(w)
            qtxt = ""
            if frac and np.isfinite(px) and px > 0 and abs(w) > 1e-12:
                dq = _desired_qty_signed(w, ex_gross, px, fractional=True)
                qtxt = f"  qty≈{_fmt_net_qty(dq, fractional=True)}"
            print(f"    {t:6s} w={w:+.4f} close={px:.4f} leg≈${leg:,.0f}{qtxt}")
        if proxy_sym:
            re_frac = float(getattr(config, "MAD_LIVE_REGIME_OFF_EQUITY_FRACTION", 0.995))
            ex_sleeve = ex_gross * re_frac if use_bil_sleeve else 0.0
            print(
                f"    {proxy_sym:6s} sleeve {'ON' if use_bil_sleeve else 'off'} "
                f"@ ${ex_sleeve:,.0f} regime-off notional (no OHLCV; size with Alpaca quote when live) | "
                f"close_all_non_proxy={close_all_np}"
            )
        return

    print(f"  gross notional USD : {gross:,.2f} (MRAT equity fraction + cap)")
    print(
        "  reconcile: one Alpaca order per symbol that still drifts vs target after "
        "cancel/re-read (normal to see many small fractional fills in one EOD pass).",
        flush=True,
    )
    if use_bil_sleeve:
        print(
            f"  regime-off sleeve  : ${sleeve_notional:,.2f} "
            f"({float(getattr(config, 'MAD_LIVE_REGIME_OFF_EQUITY_FRACTION', 0.995)):.4f} × equity) | "
            f"close_all_non_proxy={close_all_np}\n"
        )
    else:
        print()

    if use_bil_sleeve and close_all_np:
        _flatten_account_except_proxy(
            trading_client,
            proxy_sym=proxy_sym,
            close_by_ticker=snap.close_by_ticker,
            ext_hrs=ext_hrs,
            paper=paper,
            fractional=frac,
        )
    else:
        for t in snap.tickers:
            w = snap.weight_by_ticker.get(t, 0.0)
            if proxy_sym and t == proxy_sym and use_bil_sleeve:
                continue
            px = snap.close_by_ticker.get(t, float("nan"))
            leg_usd = abs(w) * gross
            if abs(w) > 1e-12 and leg_usd < float(min_order_usd):
                desired = 0.0
                note = f" | skipped leg ${leg_usd:.2f} < min_order"
            else:
                desired = _desired_qty_signed(w, gross, px, fractional=frac)
                note = ""

            cur_f = float(_get_current_qty(trading_client, t))
            cur = _round_alpaca_qty(cur_f) if frac else int(round(cur_f))
            zero_pos = (
                abs(float(desired)) < 1e-8 and abs(float(cur)) < 1e-8
                if frac
                else int(desired) == 0 and int(cur) == 0
            )
            if zero_pos:
                if abs(w) > 1e-12:
                    px_disp = f"{px:.4f}" if np.isfinite(px) and px > 0 else "nan"
                    zd = _fmt_net_qty(0, fractional=frac)
                    zc = _fmt_net_qty(0, fractional=frac)
                    if note:
                        print(f"  {t}: w={w:+.4f} px={px_disp} desired_net={zd} current={zc}{note}")
                    elif not np.isfinite(px) or px <= 0:
                        print(
                            f"  {t}: w={w:+.4f} px={px_disp} desired_net={zd} current={zc} "
                            f"| skipped (invalid price for sizing)"
                        )
                    elif not frac:
                        print(
                            f"  {t}: w={w:+.4f} px={px_disp} desired_net={zd} current={zc} "
                            f"| skipped whole-share floor (${leg_usd:.2f} alloc < 1 share @ ${px:.2f})"
                        )
                    else:
                        print(
                            f"  {t}: w={w:+.4f} px={px_disp} desired_net={zd} current={zc} "
                            f"| skipped (target rounds to 0 shares)"
                        )
                continue

            dq_clamped, short_note = _apply_live_short_constraints(
                trading_client, t, desired, fractional=frac
            )
            cd = _fmt_net_qty(dq_clamped, fractional=frac)
            cc = _fmt_net_qty(cur, fractional=frac)
            print(
                f"  {t}: w={w:+.4f} px={px:.4f} desired_net={cd} current={cc}{short_note}{note}"
            )
            if dq_clamped != desired and short_note:
                desired = float(dq_clamped) if frac else int(dq_clamped)

            ref_px = (
                px
                if np.isfinite(px) and px > 0 and (ext_hrs or frac)
                else None
            )
            need_rec = (
                abs(float(desired) - float(cur)) >= 1e-8
                if frac
                else int(round(desired)) != int(cur)
            )
            if need_rec:
                _reconcile_symbol_net_qty(
                    trading_client,
                    t,
                    desired,
                    extended_hours=ext_hrs,
                    reference_price=ref_px,
                    paper=paper,
                    fractional=frac,
                )

    if proxy_sym:
        bil_px = _sleeve_market_price(proxy_sym, paper=paper)
        if use_bil_sleeve:
            bil_usd = float(sleeve_notional)
            if bil_usd < float(min_order_usd):
                bil_desired = 0.0
                bil_note = f" | skipped sleeve ${bil_usd:.2f} < min_order"
            else:
                bil_desired = _desired_qty_signed(1.0, bil_usd, bil_px, fractional=frac)
                bil_note = ""
        else:
            bil_desired = 0.0
            bil_note = ""
        bil_cur_f = float(_get_current_qty(trading_client, proxy_sym))
        bil_cur = _round_alpaca_qty(bil_cur_f) if frac else int(round(bil_cur_f))
        bil_need_print = bool(bil_note) or (
            abs(float(bil_desired) - float(bil_cur)) >= 1e-8
            if frac
            else int(round(bil_desired)) != int(bil_cur)
        )
        if bil_need_print:
            px_disp = f"{bil_px:.4f}" if np.isfinite(bil_px) and bil_px > 0 else "market"
            bd = _fmt_net_qty(bil_desired, fractional=frac)
            bc = _fmt_net_qty(bil_cur, fractional=frac)
            print(
                f"  {proxy_sym}: sleeve={'risk-off full gross' if use_bil_sleeve else 'flat'} "
                f"quote={px_disp} desired_net={bd} current={bc}{bil_note}"
            )
        bil_ref = (
            bil_px
            if np.isfinite(bil_px) and bil_px > 0 and (ext_hrs or frac)
            else None
        )
        bil_need_rec = (
            abs(float(bil_desired) - float(bil_cur)) >= 1e-8
            if frac
            else int(round(bil_desired)) != int(bil_cur)
        )
        if bil_need_rec:
            _reconcile_symbol_net_qty(
                trading_client,
                proxy_sym,
                bil_desired,
                extended_hours=ext_hrs,
                reference_price=bil_ref,
                paper=paper,
                fractional=frac,
            )


def main() -> None:
    global _LAST_EOD_REBALANCE_SESSION_DATE

    parser = argparse.ArgumentParser(description="MAD / MRAT Alpaca live bot")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one reconcile cycle immediately, then exit (ignores EOD clock; use for cron).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print targets only; no orders (no account query for gross sizing).",
    )
    args = parser.parse_args()

    paper = config.bot_mode_is_paper()
    poll = max(int(getattr(config, "MAD_LIVE_POLL_SECONDS", 300)), 30)
    min_order = float(getattr(config, "MAD_LIVE_MIN_ORDER_USD", 1.0))

    sh, lo, ex, reg_ma, reg_tick = load_mad_live_strategy_params()
    regime_disp = _display_regime_ticker(reg_ma, reg_tick)
    _bm = str(getattr(config, "BOT_MODE", "paper")).strip().lower()
    mode = "PAPER" if _bm == "paper" else "CASH"
    panel_syms = mad_universe_tickers()
    pipe_mode = str(getattr(config, "OHLCV_PIPELINE_MODE", "mad_universe")).strip()
    fetch_syms = tuple(config.ohlcv_pipeline_tickers())
    fetch_set = set(fetch_syms)
    missing_panel = sorted(set(panel_syms) - fetch_set)
    print(
        f"\nMAD / MRAT live bot\n"
        f"  mode            : {mode}\n"
        f"  MAD ref (panel) : {mad_reference_ticker()}  (``mad.backtester.MAD_PANEL_REFERENCE_TICKER``)\n"
        f"  OHLCV health ref: {_ohlcv_health_reference_ticker()}  (equity calendar when sleeve == ref)\n"
        f"  bar granularity : {config.TARGET_CANDLE_GRANULARITY}\n"
        f"  MAD panel       : {len(panel_syms)} names (MAD_UNIVERSE_TICKERS)\n"
        f"  OHLCV pipeline  : {pipe_mode!r} → {len(fetch_syms)} fetch symbol(s) (``alpaca_fetcher``)\n"
        f"  params          : MRAT {sh}/{lo} exit_MA={ex} regime_MA={reg_ma} ticker={regime_disp!r}\n"
        f"  optim DB        : {_mad_optim_db_path()} (load={getattr(config, 'MAD_LIVE_LOAD_PARAMS_FROM_DB', True)})\n"
        f"  rebalance       : EOD once/session after Alpaca calendar close (US/Eastern); "
        f"poll={poll}s is wake interval only\n"
        f"  --once          : run one reconcile immediately (ignores EOD schedule)\n"
        f"  extended_hours  : {getattr(config, 'MAD_LIVE_EXTENDED_HOURS_ORDERS', False)}\n"
        f"  fractional      : {getattr(config, 'MAD_LIVE_FRACTIONAL_SHARES', True)}\n"
        f"  regime sleeve   : {getattr(config, 'MAD_LIVE_REGIME_OFF_PROXY_TICKER', None) or 'cash'}\n"
        f"  dry_run         : {args.dry_run}\n"
    )
    if missing_panel:
        sample = ", ".join(missing_panel[:12])
        more = f" (+{len(missing_panel) - 12} more)" if len(missing_panel) > 12 else ""
        print(
            "  WARNING: MAD universe includes symbols not in ``ohlcv_pipeline_tickers()`` — "
            "those DBs will be missing unless you fetched them another way.\n"
            f"    Missing vs pipeline: {sample}{more}\n"
        )

    _run_ohlcv_health_check(reg_ma, reg_tick)

    if args.dry_run:
        _run_cycle(None, dry_run=True, min_order_usd=min_order, paper=paper)
        return

    key, secret = _alpaca_trading_keys(paper=paper)
    tc = TradingClient(api_key=key, secret_key=secret, paper=paper)
    _alpaca_ping_account(tc)

    startup_refresh = bool(getattr(config, "MAD_LIVE_REFRESH_SPLITTER_ON_STARTUP", True))
    first_loop = True

    while True:
        if first_loop and startup_refresh:
            _maybe_refresh_splitter_dbs(force=False)
        first_loop = False

        now_txt = _fmt_now_et()
        if args.once:
            try:
                print(f"\n[{now_txt}] Single cycle (--once)")
                _run_cycle(tc, dry_run=False, min_order_usd=min_order, paper=paper)
                _maybe_refresh_splitter_dbs(force=False)
            except Exception as exc:
                print(f"  ERROR: {exc}")
            break

        run_eod, eod_reason = _eod_rebalance_should_run(tc)
        if run_eod:
            now_et = datetime.now(_ET)
            sess, cal_err = _trading_session_for_date(tc, now_et.date())
            if cal_err is not None:
                print(
                    f"[{now_txt}] EOD gate passed but calendar re-fetch failed — {cal_err}"
                )
            elif sess is None:
                print(f"[{now_txt}] EOD gate passed but no calendar row — skipping")
            else:
                sd = sess.date
                try:
                    print(f"\n[{now_txt}] EOD rebalance — {eod_reason}")
                    _run_cycle(tc, dry_run=False, min_order_usd=min_order, paper=paper)
                    _LAST_EOD_REBALANCE_SESSION_DATE = sd
                    _maybe_refresh_splitter_dbs(force=False)
                except Exception as exc:
                    print(f"  ERROR: {exc}")
        else:
            print(f"[{now_txt}] Idle — {eod_reason}. Next check in {poll}s.")

        time.sleep(poll)


if __name__ == "__main__":
    main()
