"""Central project configuration. Imported by every pipeline module."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
PANEL_DIR = DATA_DIR / "panel"
FEATURES_DIR = DATA_DIR / "features"
RESULTS_DIR = ROOT / "results"

for _d in (RAW_DIR, CLEAN_DIR, PANEL_DIR, FEATURES_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
RANDOM_SEED = 42 #answer to the ultimate question of life, the universe, and everything

# ---------------------------------------------------------------------------
# Capital and trade sizing
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 1_000_000.0
CASH_PER_TRADE = 100_000.0
COMMISSION = 0.0

# ---------------------------------------------------------------------------
# Time window
# ---------------------------------------------------------------------------
# Design doc: "expect ~2003–present for most names". END_DATE=None means "today".
START_DATE = "2015-01-01"
END_DATE: str | None = None

# Drop tickers with fewer than this many trading-day observations.
MIN_HISTORY_DAYS = 750

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
# BIST100 index symbol on Yahoo Finance.
BIST100_INDEX_TICKER = "XU100.IS"

# Phase-1 (index) work uses BIST100_INDEX_TICKER.
# Phase-2 (single names) uses BIST100_CONSTITUENTS below.
#
# This is a curated list of well-known BIST100 names — sufficient for M1
# (pipeline smoke test on BIST100 + 5 stocks) and for early single-name work.
# TODO: replace with the full official current BIST100 membership before the
# Phase-2 walk-forward. The course brief says we accept survivorship bias on
# the *current* membership snapshot.
#
# TODO (later, do not implement yet): all XU100 names will be tagged by
# sector (banks, holdings, transport, energy, industrials, etc.) so that
# Phase-2 results can be sliced and aggregated per-sector. The grouping
# comments below are placeholders for that future taxonomy — they are NOT
# the canonical sector codes and should not be relied on programmatically
# until the proper sector mapping is added.
BIST100_CONSTITUENTS: list[str] = [
    # Banks
    "AKBNK.IS", "GARAN.IS", "ISCTR.IS", "YKBNK.IS", "HALKB.IS", "VAKBN.IS",
    # Holdings / conglomerates
    "KCHOL.IS", "SAHOL.IS", "ENKAI.IS", "TKFEN.IS",
    # Transport
    "THYAO.IS", "PGSUS.IS",
    # Energy / refining / chemicals
    "TUPRS.IS", "PETKM.IS", "HEKTS.IS",
    # Steel / industrials
    "EREGL.IS", "KRDMD.IS",
    # Glass / soda
    "SISE.IS", # "SODA.IS",
    # Defense / telecom
    "ASELS.IS", "TCELL.IS", "TTKOM.IS",
    # Retail / consumer
    "BIMAS.IS", "MGROS.IS", "ULKER.IS",
    # White goods / autos
    "ARCLK.IS", "VESTL.IS", "FROTO.IS", "TOASO.IS", "DOAS.IS",
    # Mining
    "TRALT.IS", "TRMET.IS",
]

# Five-name subset used for the M1 smoke test.
SMOKE_TICKERS: list[str] = [
    "AKBNK.IS", "GARAN.IS", "THYAO.IS", "EREGL.IS", "ASELS.IS",
]

# ---------------------------------------------------------------------------
# Schema for raw OHLCV parquet
# ---------------------------------------------------------------------------
OHLCV_COLS = ["open", "high", "low", "close", "volume"]
