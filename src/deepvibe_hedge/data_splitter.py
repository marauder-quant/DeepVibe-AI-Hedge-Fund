"""
Assigns walk-forward splits and pre-calculates SMA columns on the existing OHLCV dataset.

Split 0 — warmup: enough bars (and, via ``splitter_warmup_min_calendar_days``, enough distinct
          calendar days when applicable) so the longest SMA in ``config.splitter_ma_periods()`` is
          defined on the first bar of split 1. Split 0 is not used for IS/OOS scoring.

Split 1..NUM_SPLITS — remaining rows divided evenly for walk-forward optimisation and OOS testing.

The MAD backtester and live snapshot **recompute** MRAT/regime moving averages from ``close`` when
they run; ``sma_<n>`` in SQLite is for inspection, SQL, and keeping the DB aligned with the same
periods used in research.

Run after alpaca_fetcher. Processes every symbol in ``config.ohlcv_pipeline_tickers()``; each DB/CSV
is updated in place.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from deepvibe_hedge import config
from deepvibe_hedge.paths import OHLCV_DIR


def _progress(done: int, total: int, label: str) -> None:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    pct = 100.0 * done / total
    bar_len = 24
    filled = int(round(bar_len * done / total))
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"[{bar}] {done}/{total} ({pct:6.2f}%)  {label}")


def print_loaded_config() -> None:
    cfg_path = Path(getattr(config, "__file__", "unknown")).resolve()
    periods = config.splitter_ma_periods()
    print(
        "Loaded config:\n"
        f"  file               : {cfg_path}\n"
        f"  splitter_ma_periods: {periods}\n"
        f"  warmup_min_cal_days: {config.splitter_warmup_min_calendar_days()}\n"
        f"  SPLITTER_NUM_SPLITS: {config.SPLITTER_NUM_SPLITS}\n"
        f"  SPLITTER_ENABLE_SPLIT_ASSIGNMENT : {getattr(config, 'SPLITTER_ENABLE_SPLIT_ASSIGNMENT', True)}\n"
        f"  SPLITTER_ENABLE_MA_PRECOMPUTE    : {getattr(config, 'SPLITTER_ENABLE_MA_PRECOMPUTE', True)}\n"
        f"  SPLITTER_WARMUP_BARS (effective): {_required_warmup_bars()}\n"
        f"  OHLCV pipeline tickers: {', '.join(config.ohlcv_pipeline_tickers())}\n"
    )


def _filename(ticker: str | None = None) -> str:
    t = str(ticker or config.TARGET_TICKER).strip().upper()
    return f"{t}_{config.TARGET_CANDLE_GRANULARITY}"


def load_ohlcv(ticker: str | None = None) -> pd.DataFrame:
    path = OHLCV_DIR / f"{_filename(ticker)}.db"
    if not path.exists():
        raise FileNotFoundError(
            f"No DB found at {path} — run: PYTHONPATH=src python -m deepvibe_hedge.alpaca_fetcher"
        )
    with sqlite3.connect(path) as con:
        cols = [row[1] for row in con.execute("PRAGMA table_info(ohlcv)").fetchall()]
        wanted = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in cols]
        if "timestamp" not in wanted:
            raise RuntimeError("ohlcv table is missing required 'timestamp' column.")
        query = f"SELECT {', '.join(wanted)} FROM ohlcv"
        df = pd.read_sql(query, con, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    warmup = _required_warmup_bars(df)

    if n <= warmup:
        raise ValueError(
            f"Only {n} rows of data — need more than {warmup} bars for warmup."
        )

    remaining = n - warmup
    # evenly distribute remaining rows across SPLITTER_NUM_SPLITS
    indices = np.arange(remaining)
    split_labels = (indices * config.SPLITTER_NUM_SPLITS // remaining) + 1  # 1..NUM_SPLITS

    splits = np.zeros(n, dtype=int)   # warmup rows = split 0
    splits[warmup:] = split_labels

    df["split"] = splits
    return df


def _warmup_bars_for_min_days(df: pd.DataFrame, min_days: int) -> int:
    if min_days <= 0:
        return 0
    day_series = pd.Series(df.index.normalize(), index=df.index)
    unique_days = day_series.drop_duplicates()
    if len(unique_days) <= min_days:
        # Not enough unique days to leave any non-warmup rows.
        return len(df)
    # Warmup ends at the first bar of day (min_days + 1),
    # so warmup contains at least `min_days` unique daily bars.
    day_cutoff = unique_days.iloc[min_days]
    day_vals = day_series.to_numpy()
    idx = np.flatnonzero(day_vals == day_cutoff)
    first_test_pos = int(idx[0]) if len(idx) else len(df)
    return max(1, first_test_pos)


def _required_warmup_bars(df: pd.DataFrame | None = None) -> int:
    periods = list(config.splitter_ma_periods())
    pmax = max(periods) if periods else 1
    warmups: list[int] = []
    if bool(getattr(config, "SPLITTER_ENABLE_MA_PRECOMPUTE", True)):
        warmups.append(pmax)
    nd = int(config.splitter_warmup_min_calendar_days())
    if nd > 0:
        if df is not None and not df.empty:
            warmups.append(_warmup_bars_for_min_days(df, nd))
        else:
            bars_per_day = {
                "1m": 390.0,
                "5m": 78.0,
                "15m": 26.0,
                "1h": 6.5,
                "4h": 2.0,
                "1d": 1.0,
                "1w": 1.0 / 5.0,
                "1mo": 1.0 / 21.0,
            }.get(str(getattr(config, "TARGET_CANDLE_GRANULARITY", "1d")).lower(), 1.0)
            warmups.append(max(1, int(np.ceil(float(nd) * float(bars_per_day)))))
    if not warmups:
        warmups.append(1)
    return max(warmups)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    price_col = "close"
    periods = config.splitter_ma_periods()
    new_cols: dict[str, pd.Series] = {}
    for period in periods:
        p = int(period)
        new_cols[f"sma_{p}"] = df[price_col].rolling(p, min_periods=p).mean().round(4)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    print(f"Calculated {len(periods)} SMAs from {price_col}: {list(periods)}")
    return df


def save_back(df: pd.DataFrame, ticker: str | None = None, *, verbose: bool = True) -> None:
    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    db_path = OHLCV_DIR / f"{_filename(ticker)}.db"
    retries = int(getattr(config, "SPLITTER_DB_WRITE_RETRIES", 6))
    retry_sec = float(getattr(config, "SPLITTER_DB_WRITE_RETRY_SEC", 5))
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with sqlite3.connect(db_path, timeout=max(5.0, retry_sec)) as con:
                con.execute(f"PRAGMA busy_timeout = {int(max(5000, retry_sec * 1000))}")
                df.reset_index().to_sql("ohlcv", con, if_exists="replace", index=False)
            if verbose:
                print(f"Updated DB  → {db_path}")
            last_err = None
            break
        except Exception as exc:
            last_err = exc
            msg = str(exc).lower()
            if "database is locked" in msg and attempt < retries:
                wait = retry_sec * attempt
                print(
                    f"DB write locked (attempt {attempt}/{retries}); retrying in {wait:.1f}s..."
                )
                time.sleep(wait)
                continue
            raise
    if last_err is not None:
        raise RuntimeError(
            "Failed to save ohlcv due to persistent DB lock. "
            "Close other processes using this DB and retry."
        ) from last_err

    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OHLCV_DIR / f"{_filename(ticker)}.csv"
    df.to_csv(csv_path)
    if verbose:
        print(f"Updated CSV → {csv_path}")


def print_summary(df: pd.DataFrame, *, ticker_label: str = "") -> None:
    label = f" [{ticker_label}]" if ticker_label else ""
    counts = df.groupby("split").size()
    print(f"\n{'Split':<10} {'Bars':<8} {'From':<28} {'To'}{label}")
    print("-" * 70)
    for split_num, count in counts.items():
        chunk = df[df["split"] == split_num]
        split_lbl = f"0 (warmup)" if split_num == 0 else str(split_num)
        print(f"{split_lbl:<10} {count:<8} {str(chunk.index.min()):<28} {chunk.index.max()}")


def run_pipeline_for_ticker(ticker: str) -> None:
    """Load one OHLCV DB, assign splits, precompute SMA columns, save back."""
    sym = str(ticker).strip().upper()
    do_split = bool(getattr(config, "SPLITTER_ENABLE_SPLIT_ASSIGNMENT", True))
    do_ma = bool(getattr(config, "SPLITTER_ENABLE_MA_PRECOMPUTE", True))

    steps = [
        ("Load OHLCV", True),
        ("Assign splits", do_split),
        ("Precompute SMA columns", do_ma),
        ("Save DB/CSV", True),
        ("Print summary", True),
    ]
    total_steps = sum(1 for _name, enabled in steps if enabled)
    done_steps = 0

    _progress(done_steps, total_steps, f"[{sym}] Starting data splitter")
    df = load_ohlcv(sym)
    done_steps += 1
    _progress(done_steps, total_steps, f"[{sym}] Loaded {len(df):,} OHLCV rows")

    if do_split:
        df = assign_splits(df)
        done_steps += 1
        _progress(done_steps, total_steps, f"[{sym}] Assigned walk-forward splits")
    else:
        print(f"[{sym}] Skipping split assignment (SPLITTER_ENABLE_SPLIT_ASSIGNMENT=False)")

    if do_ma:
        df = add_indicators(df)
        done_steps += 1
        _progress(done_steps, total_steps, f"[{sym}] Precomputed SMA columns")
    else:
        print(f"[{sym}] Skipping MA precompute (SPLITTER_ENABLE_MA_PRECOMPUTE=False)")

    save_back(df, sym)
    done_steps += 1
    _progress(done_steps, total_steps, f"[{sym}] Saved updated dataset to DB/CSV")
    if "split" in df.columns:
        print_summary(df, ticker_label=sym)
    else:
        print(f"\n[{sym}] No split column present; split summary skipped.")
    done_steps += 1
    _progress(done_steps, total_steps, f"[{sym}] Completed")


if __name__ == "__main__":
    print_loaded_config()
    tickers = config.ohlcv_pipeline_tickers()
    print(f"\nData splitter — {len(tickers)} symbol(s): {', '.join(tickers)}\n")
    for sym in tickers:
        print(f"{'=' * 16} {sym} {'=' * 16}")
        run_pipeline_for_ticker(sym)
        print()
