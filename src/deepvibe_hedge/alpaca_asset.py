"""
Alpaca asset metadata helpers (shortable, easy_to_borrow, …).

These use the Trading API ``get_asset`` endpoint — **not** market data — so they work when
the cash session is closed (weekends, overnight), subject only to API availability.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv

from deepvibe_hedge import config


def _alpaca_trading_keys(*, paper: bool) -> tuple[str, str]:
    load_dotenv()

    def first_nonempty(*names: str) -> str:
        for name in names:
            value = os.getenv(name, "").strip()
            if value:
                return value
        return ""

    if paper:
        key = first_nonempty(
            "ALPACA_API_KEY_PAPER",
            "ALPACA_PAPER_API_KEY",
            "APCA_PAPER_API_KEY_ID",
            "ALPACA_API_KEY",
            "APCA_API_KEY_ID",
        )
        secret = first_nonempty(
            "ALPACA_SECRET_KEY_PAPER",
            "ALPACA_PAPER_SECRET_KEY",
            "APCA_PAPER_API_SECRET_KEY",
            "ALPACA_SECRET_KEY",
            "APCA_API_SECRET_KEY",
        )
    else:
        key = first_nonempty(
            "ALPACA_API_KEY_LIVE",
            "ALPACA_LIVE_API_KEY",
            "APCA_LIVE_API_KEY_ID",
            "ALPACA_API_KEY",
            "APCA_API_KEY_ID",
        )
        secret = first_nonempty(
            "ALPACA_SECRET_KEY_LIVE",
            "ALPACA_LIVE_SECRET_KEY",
            "APCA_LIVE_API_SECRET_KEY",
            "ALPACA_SECRET_KEY",
            "APCA_API_SECRET_KEY",
        )
    if not key or not secret:
        mode = "paper" if paper else "live"
        raise RuntimeError(f"Missing Alpaca {mode} API credentials in environment / .env.")
    return key, secret


def trading_client_for_assets(*, paper: bool | None = None) -> TradingClient:
    if paper is None:
        paper = config.bot_mode_is_paper()
    key, secret = _alpaca_trading_keys(paper=paper)
    return TradingClient(api_key=key, secret_key=secret, paper=paper)


@dataclass(frozen=True)
class EquityShortability:
    """Subset of Alpaca ``Asset`` fields relevant to sell-to-open."""

    symbol: str
    tradable: bool
    shortable: bool
    easy_to_borrow: bool
    marginable: bool


def fetch_equity_shortability(
    symbol: str,
    *,
    trading_client: TradingClient | None = None,
    paper: bool | None = None,
) -> EquityShortability:
    """
    Load Alpaca asset flags for *symbol* (uppercased). Works when the market is closed.

    Note: ``shortable`` / ``easy_to_borrow`` can still disagree with order-time checks
    (e.g. borrow availability); use as a preflight only.
    """
    sym = symbol.strip().upper()
    client = trading_client or trading_client_for_assets(paper=paper)
    try:
        asset = client.get_asset(sym)
    except APIError as exc:
        raise RuntimeError(f"Alpaca get_asset({sym!r}) failed: {exc}") from exc
    return EquityShortability(
        symbol=str(asset.symbol),
        tradable=bool(asset.tradable),
        shortable=bool(asset.shortable),
        easy_to_borrow=bool(asset.easy_to_borrow),
        marginable=bool(asset.marginable),
    )


def is_equity_shortable(
    symbol: str,
    *,
    require_easy_to_borrow: bool = False,
    trading_client: TradingClient | None = None,
    paper: bool | None = None,
) -> bool:
    """
    True if Alpaca marks the symbol shortable (and optionally easy-to-borrow).

    Same session / API caveats as :func:`fetch_equity_shortability`.
    """
    info = fetch_equity_shortability(symbol, trading_client=trading_client, paper=paper)
    if not info.tradable or not info.shortable:
        return False
    if require_easy_to_borrow and not info.easy_to_borrow:
        return False
    return True


def _default_ticker_from_config() -> str:
    raw = getattr(config, "TARGET_TICKER", None)
    if raw is None or not str(raw).strip():
        raise RuntimeError("Set TARGET_TICKER in deepvibe_hedge.config (or pass a symbol on the CLI).")
    return str(raw).strip().upper()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Alpaca shortable / ETB flags (works when market closed).")
    parser.add_argument(
        "symbol",
        nargs="?",
        default=None,
        help="Ticker (omit to use TARGET_TICKER from deepvibe_hedge.config)",
    )
    parser.add_argument("--live", action="store_true", help="Use live keys (default: paper from config/env).")
    parser.add_argument("--require-etb", action="store_true", help="Also require easy_to_borrow.")
    args = parser.parse_args()
    symbol = (args.symbol or "").strip().upper() or _default_ticker_from_config()
    paper = not bool(args.live)
    client = trading_client_for_assets(paper=paper)
    info = fetch_equity_shortability(symbol, trading_client=client)
    ok = is_equity_shortable(
        symbol,
        require_easy_to_borrow=bool(args.require_etb),
        trading_client=client,
    )
    print(
        f"{info.symbol}: tradable={info.tradable} shortable={info.shortable} "
        f"easy_to_borrow={info.easy_to_borrow} marginable={info.marginable}\n"
        f"is_equity_shortable(require_etb={args.require_etb}) -> {ok}"
        + (f"\n(symbol from deepvibe_hedge.config TARGET_TICKER)" if not args.symbol else "")
    )


if __name__ == "__main__":
    main()
