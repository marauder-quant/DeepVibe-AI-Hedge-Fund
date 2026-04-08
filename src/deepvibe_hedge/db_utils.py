"""
OHLCV SQLite inspector (same role as ``reference_old_folder/.../swing_trade/db_utils.py``).

Outputs are written under ``data/ohlcv/`` (see ``deepvibe_hedge.paths.OHLCV_DIR``). That directory is
gitignored so local DB/CSV do not appear in ``git status``; they still exist on disk after fetcher /
splitter runs.

Usage
-----
# Overview of every OHLCV DB in data/ohlcv/
PYTHONPATH=src python -m deepvibe_hedge.db_utils

# Head / tail of a specific DB (stem without .db)
PYTHONPATH=src python -m deepvibe_hedge.db_utils head QQQ_1d
PYTHONPATH=src python -m deepvibe_hedge.db_utils tail QQQ_1d --rows 10

# Walk-forward splits
PYTHONPATH=src python -m deepvibe_hedge.db_utils splits QQQ_1d

# One split id
PYTHONPATH=src python -m deepvibe_hedge.db_utils split QQQ_1d 1

# SMA columns from splitter
PYTHONPATH=src python -m deepvibe_hedge.db_utils indicators QQQ_1d
PYTHONPATH=src python -m deepvibe_hedge.db_utils sma QQQ_1d 200
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from deepvibe_hedge.paths import DATA_ROOT, OHLCV_DIR


def _all_dbs() -> list[Path]:
    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(OHLCV_DIR.glob("*.db"))


def _load(name: str) -> pd.DataFrame:
    path = OHLCV_DIR / (name if name.endswith(".db") else f"{name}.db")
    if not path.exists():
        raise FileNotFoundError(
            f"No DB named {name!r} under {OHLCV_DIR}. "
            f"Run: PYTHONPATH=src python -m deepvibe_hedge.alpaca_fetcher"
        )
    with sqlite3.connect(path) as con:
        df = pd.read_sql("SELECT * FROM ohlcv", con, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def _sma_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("sma_")]


def _overview_row_sqlite(path: Path) -> tuple[str, str, str, str, str, str, str]:
    """Fast metadata without loading the full table into pandas."""
    with sqlite3.connect(path) as con:
        n = int(con.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0])
        t0, t1 = con.execute("SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv").fetchone()
        info = con.execute("PRAGMA table_info(ohlcv)").fetchall()
        col_names = [row[1] for row in info]
        has_split = "split" in col_names
        splits = (
            str(int(con.execute("SELECT MAX(split) FROM ohlcv").fetchone()[0] or 0))
            if has_split
            else "0"
        )
        sma_cols = [c for c in col_names if c.startswith("sma_")]
        if sma_cols:
            periods = sorted(int(c.split("_")[1]) for c in sma_cols)
            sma_range = f"{periods[0]}–{periods[-1]} ({len(periods)})"
        else:
            sma_range = "—"
    ts_min = str(t0)[:25] if t0 is not None else "—"
    ts_max = str(t1)[:25] if t1 is not None else "—"
    return (
        path.stem,
        f"{n:,}",
        splits,
        ts_min,
        ts_max,
        str(len(sma_cols)),
        sma_range,
    )


def cmd_overview(_args: argparse.Namespace) -> None:
    dbs = _all_dbs()
    print(f"DATA_ROOT = {DATA_ROOT.resolve()}")
    print(f"OHLCV_DIR = {OHLCV_DIR.resolve()}\n")
    if not dbs:
        print(f"No databases found in {OHLCV_DIR} — run alpaca_fetcher first.")
        return

    col_w = [22, 8, 8, 28, 28, 6, 14]
    headers = ["DB", "Rows", "Splits", "From", "To", "SMAs", "SMA periods"]
    sep = "  ".join("─" * w for w in col_w)
    row_fmt = "  ".join(f"{{:<{w}}}" for w in col_w)

    print(f"{'─' * (sum(col_w) + len(col_w) * 2)}")
    print(row_fmt.format(*headers))
    print(sep)

    for path in dbs:
        try:
            row = _overview_row_sqlite(path)
            print(row_fmt.format(*row))
        except Exception as e:
            print(f"  {path.stem:<22} ERROR: {e}")

    print(f"{'─' * (sum(col_w) + len(col_w) * 2)}\n")


def cmd_head(args: argparse.Namespace) -> None:
    df = _load(args.name)
    ohlcv = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    cols = ohlcv + (["split"] if "split" in df.columns else [])
    print(f"\n{args.name}  —  first {args.rows} rows\n{df[cols].head(args.rows)}\n")


def cmd_tail(args: argparse.Namespace) -> None:
    df = _load(args.name)
    ohlcv = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    cols = ohlcv + (["split"] if "split" in df.columns else [])
    print(f"\n{args.name}  —  last {args.rows} rows\n{df[cols].tail(args.rows)}\n")


def cmd_splits(args: argparse.Namespace) -> None:
    df = _load(args.name)
    if "split" not in df.columns:
        print(f"{args.name} has no split column — run data_splitter first.")
        return
    counts = df.groupby("split").size()
    print(f"\n{args.name}  —  split summary\n")
    print(f"  {'Split':<12} {'Bars':<8} {'From':<28} To")
    print(f"  {'─' * 12} {'─' * 8} {'─' * 28} {'─' * 28}")
    for num, count in counts.items():
        chunk = df[df["split"] == num]
        label = "0 (warmup)" if num == 0 else str(num)
        print(f"  {label:<12} {count:<8} {str(chunk.index.min()):<28} {chunk.index.max()}")
    print()


def cmd_split(args: argparse.Namespace) -> None:
    df = _load(args.name)
    if "split" not in df.columns:
        print(f"{args.name} has no split column — run data_splitter first.")
        return
    chunk = df[df["split"] == args.num]
    if chunk.empty:
        print(f"No data for split {args.num}")
        return
    ohlcv = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    print(f"\n{args.name}  —  split {args.num}  ({len(chunk)} bars)\n{chunk[ohlcv]}\n")


def cmd_indicators(args: argparse.Namespace) -> None:
    df = _load(args.name)
    smas = _sma_cols(df)
    if not smas:
        print(f"{args.name} has no SMA columns — run data_splitter first.")
        return
    periods = sorted(int(c.split("_")[1]) for c in smas)
    print(f"\n{args.name}  —  {len(smas)} SMA columns")
    print(f"  Periods: {', '.join(str(p) for p in periods)}\n")


def cmd_sma(args: argparse.Namespace) -> None:
    df = _load(args.name)
    col = f"sma_{args.period}"
    if col not in df.columns:
        available = sorted(int(c.split("_")[1]) for c in _sma_cols(df))
        print(f"No {col} in {args.name}. Available: {available}")
        return
    ohlcv = [c for c in ("open", "high", "low", "close") if c in df.columns]
    print(f"\n{args.name}  —  close + {col}\n{df[ohlcv + [col]].dropna(subset=[col])}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="deepvibe_hedge.db_utils",
        description="Inspect OHLCV SQLite files under data/ohlcv/.",
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("overview", help="Summary table of all DBs (default)")

    for cmd in ("head", "tail"):
        p = sub.add_parser(cmd)
        p.add_argument("name")
        p.add_argument("--rows", type=int, default=5)

    p = sub.add_parser("splits")
    p.add_argument("name")

    p = sub.add_parser("split")
    p.add_argument("name")
    p.add_argument("num", type=int)

    p = sub.add_parser("indicators")
    p.add_argument("name")

    p = sub.add_parser("sma")
    p.add_argument("name")
    p.add_argument("period", type=int)

    args = parser.parse_args()

    dispatch: dict[str | None, object] = {
        None: cmd_overview,
        "overview": cmd_overview,
        "head": cmd_head,
        "tail": cmd_tail,
        "splits": cmd_splits,
        "split": cmd_split,
        "indicators": cmd_indicators,
        "sma": cmd_sma,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
