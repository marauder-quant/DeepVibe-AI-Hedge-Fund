"""
Data directories under the project ``data/`` tree.

- ``OHLCV_DIR``: ``<repo>/data/ohlcv/{SYMBOL}_{granularity}.db`` and matching ``.csv`` from
  ``alpaca_fetcher`` / ``data_splitter``. The whole ``data/`` tree is gitignored (see ``.gitignore``);
  files are created on first fetch.
- ``MAD_DATA_DIR``: backtest / optimiser SQLite outputs.
"""
from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = _PROJECT_ROOT / "data"
OHLCV_DIR = DATA_ROOT / "ohlcv"
MAD_DATA_DIR = DATA_ROOT / "mad"


def ensure_data_dirs() -> None:
    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    MAD_DATA_DIR.mkdir(parents=True, exist_ok=True)
