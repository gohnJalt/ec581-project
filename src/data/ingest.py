"""Ingest raw OHLCV bars and persist to data/raw/{ticker}.parquet.

Two sources are supported:

- ``yfinance``: pulls daily history for a Yahoo symbol (e.g. ``AKBNK.IS``).
- ``csv``: reads a vendor CSV from a user-supplied directory.

The output is immutable: once a ticker's parquet exists, ``ingest_ticker``
returns it directly unless ``force=True``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import END_DATE, OHLCV_COLS, RAW_DIR, START_DATE


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
def _validate_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{ticker}: index must be DatetimeIndex, got {type(df.index)}")
    missing = [c for c in OHLCV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing}; have {list(df.columns)}")
    out = df[OHLCV_COLS].copy()
    out.index = out.index.tz_localize(None) if out.index.tz is not None else out.index
    out.index.name = "date"
    return out.sort_index()


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def fetch_yfinance(ticker: str, start: str = START_DATE, end: str | None = END_DATE) -> pd.DataFrame:
    """Pull daily OHLCV from Yahoo Finance. Adjusted close (auto_adjust=True)."""
    import yfinance as yf

    df = yf.Ticker(ticker).history(
        start=start, end=end, interval="1d", auto_adjust=True, actions=False,
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned empty frame for {ticker}")
    df = df.rename(columns=str.lower)
    return _validate_ohlcv(df, ticker)


def fetch_csv(ticker: str, csv_dir: Path) -> pd.DataFrame:
    """Read a vendor CSV named ``{ticker}.csv`` from ``csv_dir``.

    Expected columns (case-insensitive): date, open, high, low, close, volume.
    """
    path = csv_dir / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No CSV for {ticker} at {path}")
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError(f"{ticker}: CSV missing 'date' column; have {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return _validate_ohlcv(df, ticker)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def raw_path(ticker: str) -> Path:
    return RAW_DIR / f"{ticker}.parquet"


def ingest_ticker(
    ticker: str,
    source: str = "yfinance",
    csv_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Fetch one ticker and persist to data/raw. Returns the parquet path."""
    out = raw_path(ticker)
    if out.exists() and not force:
        return out

    if source == "yfinance":
        df = fetch_yfinance(ticker)
    elif source == "csv":
        if csv_dir is None:
            raise ValueError("csv_dir required when source='csv'")
        df = fetch_csv(ticker, csv_dir)
    else:
        raise ValueError(f"unknown source: {source!r}")

    df.to_parquet(out)
    meta = {
        "ticker": ticker,
        "source": source,
        "rows": int(len(df)),
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return out


def ingest_universe(
    tickers: Iterable[str],
    source: str = "yfinance",
    csv_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Path | Exception]:
    """Ingest many tickers; return a {ticker: path-or-exception} map.

    We swallow per-ticker errors so one failure doesn't kill the batch.
    """
    results: dict[str, Path | Exception] = {}
    for t in tickers:
        try:
            results[t] = ingest_ticker(t, source=source, csv_dir=csv_dir, force=force)
        except Exception as e:
            results[t] = e
    return results
