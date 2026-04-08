from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timezone
import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed

from deepvibe_hedge.alpaca_bar_adjustment import historical_bar_adjustment
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

from deepvibe_hedge import config
from deepvibe_hedge.paths import OHLCV_DIR


def _filename(symbol: str) -> str:
    return f"{symbol.strip().upper()}_{config.TARGET_CANDLE_GRANULARITY}"


def save_to_db(df: pd.DataFrame, symbol: str) -> None:
    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    path = OHLCV_DIR / f"{_filename(symbol)}.db"
    with sqlite3.connect(path) as con:
        df.reset_index().to_sql("ohlcv", con, if_exists="replace", index=False)
    print(f"Saved to DB  → {path}")


def save_to_csv(df: pd.DataFrame, symbol: str) -> None:
    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    path = OHLCV_DIR / f"{_filename(symbol)}.csv"
    df.to_csv(path)
    print(f"Saved to CSV → {path}")

_UNITS = {"m": TimeFrameUnit.Minute, "h": TimeFrameUnit.Hour, "d": TimeFrameUnit.Day,
          "w": TimeFrameUnit.Week, "mo": TimeFrameUnit.Month}


def _parse_timeframe(value: str) -> TimeFrame:
    m = re.fullmatch(r"(\d+)(mo|m|h|d|w)", value.strip().lower())
    if not m:
        raise ValueError(f"Bad CANDLE_GRANULARITY {value!r} — use e.g. 5m, 1h, 1d, 1w, 1mo")
    return TimeFrame(amount=int(m.group(1)), unit=_UNITS[m.group(2)])


def _make_client() -> StockHistoricalDataClient:
    load_dotenv()

    def _first_nonempty(*names: str) -> str:
        for name in names:
            value = os.getenv(name, "").strip()
            if value:
                return value
        return ""

    key = _first_nonempty(
        "ALPACA_API_KEY",
        "APCA_API_KEY_ID",
        "ALPACA_API_KEY_PAPER",
        "ALPACA_PAPER_API_KEY",
        "APCA_PAPER_API_KEY_ID",
        "ALPACA_API_KEY_LIVE",
        "ALPACA_LIVE_API_KEY",
        "APCA_LIVE_API_KEY_ID",
    )
    secret = _first_nonempty(
        "ALPACA_SECRET_KEY",
        "APCA_API_SECRET_KEY",
        "ALPACA_SECRET_KEY_PAPER",
        "ALPACA_PAPER_SECRET_KEY",
        "APCA_PAPER_API_SECRET_KEY",
        "ALPACA_SECRET_KEY_LIVE",
        "ALPACA_LIVE_SECRET_KEY",
        "APCA_LIVE_API_SECRET_KEY",
    )
    if not key or not secret:
        raise RuntimeError(
            "Missing Alpaca keys in .env. Set one of: "
            "ALPACA_API_KEY/ALPACA_SECRET_KEY, "
            "ALPACA_API_KEY_PAPER/ALPACA_SECRET_KEY_PAPER, or "
            "ALPACA_API_KEY_LIVE/ALPACA_SECRET_KEY_LIVE."
        )
    return StockHistoricalDataClient(api_key=key, secret_key=secret)


def fetch_ohlcv(symbol: str, client: StockHistoricalDataClient | None = None) -> pd.DataFrame:
    """Fetch OHLCV bars for one symbol using settings from config.py."""
    sym = str(symbol).strip().upper()
    client = client or _make_client()
    timeframe = _parse_timeframe(config.TARGET_CANDLE_GRANULARITY)
    raw_feed = str(getattr(config, "LIVE_BOT_DATA_FEED", "iex")).strip().lower()
    feed = {
        "iex": DataFeed.IEX,
        "sip": DataFeed.SIP,
        "delayed_sip": DataFeed.DELAYED_SIP,
    }.get(raw_feed, DataFeed.IEX)
    adj = historical_bar_adjustment()
    start_utc = config.ohlcv_download_start_utc()
    end_utc = config.ohlcv_download_end_utc()
    req = StockBarsRequest(
        symbol_or_symbols=sym,
        timeframe=timeframe,
        start=start_utc,
        end=end_utc,
        feed=feed,
        adjustment=adj,
    )
    try:
        df = client.get_stock_bars(req).df
    except APIError as exc:
        if "subscription does not permit querying recent sip data" in str(exc).lower() and feed != DataFeed.IEX:
            fallback_req = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=timeframe,
                start=start_utc,
                end=end_utc,
                feed=DataFeed.IEX,
                adjustment=adj,
            )
            df = client.get_stock_bars(fallback_req).df
        else:
            raise
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(sym, level="symbol")
    df.index = pd.to_datetime(df.index, utc=True)
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[cols].sort_index()


def fetch_ohlcv_between(
    symbol: str,
    start_utc: datetime,
    end_utc: datetime,
    *,
    client: StockHistoricalDataClient | None = None,
) -> pd.DataFrame:
    """Historical bars for ``[start_utc, end_utc)`` (Alpaca semantics); same feed/adjustment as bulk fetch."""
    sym = str(symbol).strip().upper()
    client = client or _make_client()
    timeframe = _parse_timeframe(config.TARGET_CANDLE_GRANULARITY)
    raw_feed = str(getattr(config, "LIVE_BOT_DATA_FEED", "iex")).strip().lower()
    feed = {
        "iex": DataFeed.IEX,
        "sip": DataFeed.SIP,
        "delayed_sip": DataFeed.DELAYED_SIP,
    }.get(raw_feed, DataFeed.IEX)
    adj = historical_bar_adjustment()
    s = start_utc if start_utc.tzinfo else start_utc.replace(tzinfo=timezone.utc)
    e = end_utc if end_utc.tzinfo else end_utc.replace(tzinfo=timezone.utc)
    req = StockBarsRequest(
        symbol_or_symbols=sym,
        timeframe=timeframe,
        start=s,
        end=e,
        feed=feed,
        adjustment=adj,
    )
    try:
        df = client.get_stock_bars(req).df
    except APIError as exc:
        if "subscription does not permit querying recent sip data" in str(exc).lower() and feed != DataFeed.IEX:
            fallback_req = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=timeframe,
                start=s,
                end=e,
                feed=DataFeed.IEX,
                adjustment=adj,
            )
            df = client.get_stock_bars(fallback_req).df
        else:
            raise
    if df is None or (hasattr(df, "empty") and df.empty):
        return pd.DataFrame()
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(sym, level="symbol")
    df.index = pd.to_datetime(df.index, utc=True)
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[cols].sort_index()


if __name__ == "__main__":
    from deepvibe_hedge.paths import ensure_data_dirs

    ensure_data_dirs()
    symbols = config.ohlcv_pipeline_tickers()
    _adj = historical_bar_adjustment()
    _end = config.ohlcv_download_end_utc()
    print(
        f"Fetching {len(symbols)} symbol(s): {', '.join(symbols)} | "
        f"bar adjustment={_adj.value} | end_mode={config.OHLCV_DOWNLOAD_END_MODE!r} | end_utc={_end.isoformat()}"
    )
    client = _make_client()
    for sym in symbols:
        df = fetch_ohlcv(sym, client=client)
        save_to_db(df, sym)
        save_to_csv(df, sym)
        print(f"{sym} | {config.TARGET_CANDLE_GRANULARITY} | {len(df)} bars")
        print(df.head())
        print()
