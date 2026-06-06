"""Clean per-ticker OHLCV: dedupe, sort, drop short histories.

Reads ``data/raw/{ticker}.parquet`` and writes ``data/clean/{ticker}.parquet``.
The cleaning rules are intentionally light — yfinance auto_adjust already
handles splits/dividends, so this stage mostly exists to enforce a
trading-day-indexed frame and reject stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from config import CLEAN_DIR, MIN_HISTORY_DAYS, OHLCV_COLS, RAW_DIR


def clean_path(ticker: str) -> Path:
    return CLEAN_DIR / f"{ticker}.parquet"


SPLIT_RATIO_THRESHOLD = 3.0


def _adjust_splits(df: pd.DataFrame, threshold: float = SPLIT_RATIO_THRESHOLD) -> pd.DataFrame:
    # Detect single-day OHLC jumps with ratio > threshold (or < 1/threshold) and
    # back-adjust all prior OHLC by the ratio. Catches the 2005-01-03 Turkish
    # 1M-to-1 TL redenomination (1000x jump) that yfinance leaves unadjusted,
    # corporate-action gaps yfinance auto_adjust misses (e.g. MGROS 4x on
    # 2009-08-04), and any real stock splits not picked up upstream. Threshold
    # 3x is safely above the largest legitimate single-day move in our
    # BIST100 universe (HEKTS +44% on 2021-04-30).
    closes = df["close"].to_numpy()
    if len(closes) < 2:
        return df
    ratios = closes[1:] / closes[:-1]
    big = (ratios > threshold) | (ratios < 1.0 / threshold)
    if not big.any():
        return df
    df = df.copy()
    for i, is_big in enumerate(big):
        if not is_big:
            continue
        # ratio[i] applies between row i and row i+1; multiply prior rows (0..i) by ratio.
        r = float(ratios[i])
        for col in ("open", "high", "low", "close"):
            df.iloc[: i + 1, df.columns.get_loc(col)] = (
                df.iloc[: i + 1][col] * r
            )
    return df


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    df = _adjust_splits(df)
    return df[OHLCV_COLS]


def clean_ticker(ticker: str, force: bool = False) -> Path | None:
    """Clean one ticker. Returns the output path, or None if dropped."""
    src = RAW_DIR / f"{ticker}.parquet"
    if not src.exists():
        raise FileNotFoundError(f"raw file missing for {ticker}: {src}")
    out = clean_path(ticker)
    if out.exists() and not force:
        return out

    df = pd.read_parquet(src)
    df = _clean_frame(df)

    if len(df) < MIN_HISTORY_DAYS:
        # short history — drop. Don't write a stub.
        if out.exists():
            out.unlink()
        return None

    df.to_parquet(out)
    return out


def clean_universe(
    tickers: Iterable[str], force: bool = False
) -> dict[str, Path | None | Exception]:
    results: dict[str, Path | None | Exception] = {}
    for t in tickers:
        try:
            results[t] = clean_ticker(t, force=force)
        except Exception as e:
            results[t] = e
    return results
