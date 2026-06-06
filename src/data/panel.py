"""Build wide panels from cleaned per-ticker parquet files.

Outputs (under a dataset's panel dir — ``data/panel/`` for the default dataset,
``data/panel/{name}/`` otherwise):
- ``prices.parquet``       wide close panel, shape (date x ticker)
- ``volumes.parquet``      wide volume panel, same shape
- ``index.parquet``        single-column OHLCV for the market index ticker
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from config import BIST100_INDEX_TICKER, PANEL_DIR
from src.data.clean import clean_path


def _wide(field: str, tickers: Iterable[str]) -> pd.DataFrame:
    cols: dict[str, pd.Series] = {}
    for t in tickers:
        p = clean_path(t)
        if not p.exists():
            continue
        s = pd.read_parquet(p)[field]
        s.name = t
        cols[t] = s
    if not cols:
        raise RuntimeError("no cleaned tickers found; run ingest+clean first")
    df = pd.concat(cols.values(), axis=1).sort_index()
    df.columns.name = "ticker"
    return df


def build_prices_panel(tickers: Iterable[str], panel_dir: Path = PANEL_DIR) -> Path:
    df = _wide("close", tickers)
    out = panel_dir / "prices.parquet"
    df.to_parquet(out)
    return out


def build_volumes_panel(tickers: Iterable[str], panel_dir: Path = PANEL_DIR) -> Path:
    df = _wide("volume", tickers)
    out = panel_dir / "volumes.parquet"
    df.to_parquet(out)
    return out


def build_index_series(
    index_ticker: str = BIST100_INDEX_TICKER, panel_dir: Path = PANEL_DIR
) -> Path:
    """Persist the index OHLCV (not just close) for use as a regime filter."""
    p = clean_path(index_ticker)
    if not p.exists():
        raise FileNotFoundError(
            f"index ticker {index_ticker} not cleaned; run ingest+clean first"
        )
    df = pd.read_parquet(p)
    out = panel_dir / "index.parquet"
    df.to_parquet(out)
    return out


def build_all(
    constituents: Iterable[str],
    index_ticker: str = BIST100_INDEX_TICKER,
    panel_dir: Path = PANEL_DIR,
) -> dict[str, Path]:
    return {
        "prices": build_prices_panel(constituents, panel_dir),
        "volumes": build_volumes_panel(constituents, panel_dir),
        "index": build_index_series(index_ticker, panel_dir),
    }
