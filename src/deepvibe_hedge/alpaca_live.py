"""
Alpaca order helpers for MAD live trading (extracted from the parent project's Donchian live bot).

MAD only needs net-quantity reconcile, market clock, latest trade price, and shortability checks.
"""
from __future__ import annotations

import math

from alpaca.common.exceptions import APIError
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)

from deepvibe_hedge import config
from deepvibe_hedge.alpaca_asset import _alpaca_trading_keys

_SYMBOL_SELL_SHORT_BLOCKED: set[str] = set()

_ALPACA_QTY_DECIMALS = 6


def _round_alpaca_qty(q: float) -> float:
    return round(float(q), _ALPACA_QTY_DECIMALS)


def _alpaca_short_sale_forbidden(exc: BaseException) -> bool:
    raw = str(exc)
    low = raw.lower()
    return "42210000" in raw or "cannot be sold short" in low


def _get_current_qty(trading_client: TradingClient, symbol: str) -> float:
    try:
        pos = trading_client.get_open_position(symbol)
    except APIError as exc:
        if "position does not exist" in str(exc).lower():
            return 0.0
        raise
    return float(pos.qty)


def _apply_live_short_constraints(
    trading_client: TradingClient,
    symbol: str,
    desired_qty_net: float | int,
    *,
    fractional: bool = False,
) -> tuple[float | int, str]:
    if fractional:
        dq = _round_alpaca_qty(float(desired_qty_net))
    else:
        dq = int(round(float(desired_qty_net)))
    if dq >= 0:
        return dq, ""
    sym = symbol.strip().upper()
    if sym in _SYMBOL_SELL_SHORT_BLOCKED:
        ts = f"{dq:+.6f}".rstrip("0").rstrip(".") if fractional else f"{int(dq):+d}"
        print(
            f"  [{symbol}] short target {ts} suppressed "
            "(Alpaca rejected sell-to-open earlier this session; restart bot to retry)."
        )
        return (0.0 if fractional else 0), " | broker_no_short→0"
    if not bool(getattr(config, "LIVE_BOT_ALLOW_SHORT", True)):
        ts = f"{dq:+.6f}".rstrip("0").rstrip(".") if fractional else f"{int(dq):+d}"
        print(
            f"  [{symbol}] net target {ts} (short); LIVE_BOT_ALLOW_SHORT=False → flat."
        )
        return (0.0 if fractional else 0), " | short_disabled→0"
    try:
        asset = trading_client.get_asset(symbol)
    except APIError as exc:
        print(f"  [{symbol}] get_asset failed ({exc}); short order may still be attempted.")
        return dq, ""
    if not bool(asset.shortable):
        ts = f"{dq:+.6f}".rstrip("0").rstrip(".") if fractional else f"{int(dq):+d}"
        print(
            f"  [{symbol}] net target {ts} (short) skipped: Alpaca asset.shortable=false "
            f"for {symbol!r}. Paper/live still require a shortable ticker for sell-to-open."
        )
        return (0.0 if fractional else 0), " | not_shortable→0"
    if not bool(asset.easy_to_borrow):
        print(
            f"  [{symbol}] short warning: easy_to_borrow=false; broker may reject the sell anyway."
        )
    return dq, ""


def _latest_stock_trade_price(symbol: str, *, paper: bool) -> float:
    sym = symbol.strip().upper()
    key, secret = _alpaca_trading_keys(paper=paper)
    dc = StockHistoricalDataClient(api_key=key, secret_key=secret)
    out = dc.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=sym))
    if isinstance(out, dict):
        tr = out.get(sym)
        if tr is None and out:
            tr = next(iter(out.values()))
    else:
        tr = out
    if tr is None:
        raise RuntimeError(f"No latest trade returned for {sym!r}")
    return float(tr.price)


def _ext_hours_limit_anchor_price(symbol: str, *, paper: bool, buy: bool) -> float:
    """
    Price to anchor extended-hours limits: **ask** for buys, **bid** for sells when Alpaca returns
    finite quotes; else **last trade**. Daily OHLCV close is a poor AH anchor (often far from NBBO).
    """
    sym = symbol.strip().upper()
    key, secret = _alpaca_trading_keys(paper=paper)
    dc = StockHistoricalDataClient(api_key=key, secret_key=secret)
    try:
        out = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=sym))
        q = out.get(sym) if isinstance(out, dict) else None
        if q is None and isinstance(out, dict) and out:
            q = next(iter(out.values()))
        if q is not None:
            ap = float(getattr(q, "ask_price", None) or float("nan"))
            bp = float(getattr(q, "bid_price", None) or float("nan"))
            if buy and math.isfinite(ap) and ap > 0:
                return ap
            if not buy and math.isfinite(bp) and bp > 0:
                return bp
    except Exception:
        pass
    return _latest_stock_trade_price(sym, paper=paper)


def _cancel_open_orders_for_symbol(trading_client: TradingClient, symbol: str) -> int:
    """
    Cancel every open order for ``symbol`` (same session / account as ``trading_client``).

    Returns the number of cancel API calls that succeeded. Ignores errors for orders that are already
    closed or not cancelable (race with fills).
    """
    sym = symbol.strip().upper()
    req = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        symbols=[sym],
        limit=500,
    )
    orders = trading_client.get_orders(filter=req)
    if not orders:
        return 0
    n_ok = 0
    for o in orders:
        try:
            trading_client.cancel_order_by_id(o.id)
            n_ok += 1
        except APIError as exc:
            raw = str(exc).lower()
            if (
                "not found" in raw
                or "cannot be canceled" in raw
                or "already" in raw
                or "42210000" in str(exc)
            ):
                continue
            raise
    return n_ok


def _extended_hours_limit_price(reference_price: float, *, buy: bool) -> float:
    r = float(reference_price)
    if not math.isfinite(r) or r <= 0:
        raise ValueError(f"invalid reference_price for extended-hours limit: {reference_price!r}")
    if buy:
        return round(r * 1.01, 2)
    return round(max(r * 0.99, 0.01), 2)


def _submit_delta_order(
    trading_client: TradingClient,
    symbol: str,
    delta_qty: float | int,
    *,
    extended_hours: bool = False,
    reference_price: float | None = None,
    paper: bool | None = None,
    fractional: bool = False,
) -> None:
    if fractional:
        dq = _round_alpaca_qty(float(delta_qty))
        if abs(dq) < 1e-8:
            return
    else:
        dq = int(delta_qty)
        if dq == 0:
            return
    side = OrderSide.BUY if dq > 0 else OrderSide.SELL
    qty_abs = abs(float(dq)) if fractional else abs(int(dq))
    if extended_hours:
        pb = config.bot_mode_is_paper() if paper is None else paper
        if bool(getattr(config, "MAD_LIVE_EXT_HRS_LIMIT_FROM_DAILY_CLOSE", False)):
            ref = reference_price
            if ref is None or not math.isfinite(float(ref)) or float(ref) <= 0:
                ref = _latest_stock_trade_price(symbol, paper=pb)
        else:
            ref = _ext_hours_limit_anchor_price(symbol, paper=pb, buy=(dq > 0))
        limit_px = _extended_hours_limit_price(float(ref), buy=(dq > 0))
        order = LimitOrderRequest(
            symbol=symbol,
            qty=qty_abs,
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_px,
            extended_hours=True,
        )
    else:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty_abs,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
    try:
        trading_client.submit_order(order_data=order)
    except APIError as exc:
        if dq < 0:
            print(
                "  [order] SELL rejected (sell-to-open or larger sell). Typical causes: symbol not "
                "shortable, not easy to borrow, shorting disabled on the account, or insufficient "
                f"buying power for margin. Raw API error: {exc}"
            )
        else:
            print(f"  [order] BUY rejected: {exc}")
        raise


def _reconcile_symbol_net_qty(
    trading_client: TradingClient,
    symbol: str,
    desired_qty_net: float | int,
    *,
    _short_retry: bool = True,
    extended_hours: bool = False,
    reference_price: float | None = None,
    paper: bool | None = None,
    fractional: bool = False,
) -> tuple[float | int, float | int, float | int]:
    current_qty = _get_current_qty(trading_client, symbol)
    min_usd = float(getattr(config, "MAD_LIVE_MIN_ORDER_USD", 1.0))

    if fractional:
        cur = _round_alpaca_qty(float(current_qty))
        d_tgt = _round_alpaca_qty(float(desired_qty_net))
        delta_qty = _round_alpaca_qty(d_tgt - cur)
        px = float(reference_price) if reference_price is not None else float("nan")
        if math.isfinite(px) and px > 0 and abs(delta_qty) * px < min_usd:
            return cur, d_tgt, delta_qty
        if abs(delta_qty) < 1e-8:
            return cur, d_tgt, delta_qty
    else:
        current_qty_int = int(round(current_qty))
        d_int = int(round(float(desired_qty_net)))
        delta_qty = d_int - current_qty_int
        cur = current_qty_int
        d_tgt = d_int

    try:
        submit_delta = delta_qty if fractional else int(delta_qty)
        if (fractional and abs(float(submit_delta)) >= 1e-8) or (
            not fractional and int(submit_delta) != 0
        ):
            if bool(getattr(config, "MAD_LIVE_CANCEL_OPEN_BEFORE_RECONCILE", True)):
                n_cx = _cancel_open_orders_for_symbol(trading_client, symbol)
                if n_cx:
                    sym = symbol.strip().upper()
                    if fractional:
                        print(
                            f"  [{sym}] cancelled {n_cx} open order(s) before reconcile "
                            f"(filled_net={cur:+.6f} → target {d_tgt:+.6f}, delta {float(submit_delta):+.6f})"
                        )
                    else:
                        print(
                            f"  [{sym}] cancelled {n_cx} open order(s) before reconcile "
                            f"(filled_net={int(cur):+d} → target {int(d_tgt):+d}, delta {int(submit_delta):+d})"
                        )
        _submit_delta_order(
            trading_client,
            symbol,
            submit_delta,
            extended_hours=extended_hours,
            reference_price=reference_price,
            paper=paper,
            fractional=fractional,
        )
    except APIError as exc:
        sd = float(submit_delta) if fractional else int(submit_delta)
        if (
            _short_retry
            and sd < 0
            and _alpaca_short_sale_forbidden(exc)
        ):
            _SYMBOL_SELL_SHORT_BLOCKED.add(symbol.strip().upper())
            print(
                f"  [{symbol}] sell-to-open rejected by Alpaca ({exc}). "
                "Symbol is cached as no-short for this session; targeting flat/long only."
            )
            flat_tgt = max(0.0, float(desired_qty_net)) if fractional else max(0, int(round(float(desired_qty_net))))
            return _reconcile_symbol_net_qty(
                trading_client,
                symbol,
                flat_tgt,
                _short_retry=False,
                extended_hours=extended_hours,
                reference_price=reference_price,
                paper=paper,
                fractional=fractional,
            )
        raise
    out_d = delta_qty if fractional else int(delta_qty)
    return cur, d_tgt, out_d


def _market_is_open(trading_client: TradingClient) -> bool:
    try:
        return bool(trading_client.get_clock().is_open)
    except Exception:
        return False
