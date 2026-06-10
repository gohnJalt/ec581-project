"""Central project configuration. Imported by every pipeline module."""

from __future__ import annotations

from dataclasses import dataclass
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
START_DATE = "2015-01-01"
END_DATE: str | None = None

# Drop tickers with fewer than this many trading-day observations.
MIN_HISTORY_DAYS = 750

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
# A Dataset bundles everything a driver needs to run the whole pipeline on one
# market: the index ticker (Phase-1 + regime filter), the Phase-2 constituent
# universe, a smoke subset, and the spec used to (re)fetch the constituent list
# from the web. Add a market by appending a Dataset to DATASETS; every CLI
# subcommand then accepts `--dataset <name>`.
#
# Constituents live in data/universe/{name}.txt (full Yahoo symbols, one per
# line), written by `python -m src.data.fetch_universe --dataset <name>`. If the
# file is missing we fall back to the curated subset baked in below so the
# pipeline still imports cleanly on a fresh checkout.
#
# Survivorship bias on *current* membership is accepted per the course brief.

DEFAULT_DATASET = "bist100"


@dataclass(frozen=True)
class Dataset:
    name: str
    index_ticker: str                       # Yahoo symbol of the market index
    fallback_constituents: tuple[str, ...]   # used when the universe file is absent
    smoke_tickers: tuple[str, ...]           # 5-name subset for --smoke
    # --- fetch_universe spec ---
    source_url: str                          # HTML page listing current members
    code_columns: tuple[str, ...]            # preferred header names for the code column
    ticker_regex: str                        # whole-cell match for a valid bare code
    yahoo_suffix: str = ""                   # appended to a bare code (".IS" for BIST)
    dot_to_dash: bool = False                # BRK.B -> BRK-B for Yahoo US class shares
    fixed_universe: bool = False             # no scrape: fetch writes fallback_constituents

    @property
    def universe_file(self) -> Path:
        return DATA_DIR / "universe" / f"{self.name}.txt"

    @property
    def panel_dir(self) -> Path:
        # Default dataset keeps the legacy root paths; others get a subdir so
        # their panels don't clobber each other.
        d = PANEL_DIR if self.name == DEFAULT_DATASET else PANEL_DIR / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def results_dir(self) -> Path:
        d = RESULTS_DIR if self.name == DEFAULT_DATASET else RESULTS_DIR / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def to_yahoo(self, code: str) -> str:
        """Map a bare exchange code to its Yahoo Finance symbol."""
        c = code.strip().upper()
        if self.dot_to_dash:
            c = c.replace(".", "-")
        return f"{c}{self.yahoo_suffix}"

    def load_constituents(self) -> list[str]:
        if not self.universe_file.exists():
            return list(self.fallback_constituents)
        tickers: list[str] = []
        for line in self.universe_file.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                tickers.append(s)
        return tickers or list(self.fallback_constituents)


# TODO (later, do not implement yet): tag BIST names by sector (banks, holdings,
# transport, energy, industrials, etc.) for per-sector Phase-2 aggregation. The
# grouping comments below are placeholders, not canonical sector codes.
_BIST100 = Dataset(
    name="bist100",
    index_ticker="XU100.IS",
    fallback_constituents=(
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
        "SISE.IS",
        # Defense / telecom
        "ASELS.IS", "TCELL.IS", "TTKOM.IS",
        # Retail / consumer
        "BIMAS.IS", "MGROS.IS", "ULKER.IS",
        # White goods / autos
        "ARCLK.IS", "VESTL.IS", "FROTO.IS", "TOASO.IS", "DOAS.IS",
        # Mining
        "TRALT.IS", "TRMET.IS",
    ),
    smoke_tickers=("AKBNK.IS", "GARAN.IS", "THYAO.IS", "EREGL.IS", "ASELS.IS"),

    source_url=(
        "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/"
        "Temel-Degerler-Ve-Oranlar.aspx?endeks=01"
    ),
    code_columns=("Kod", "Hisse Kodu", "Code", "Symbol", "Ticker"),
    # BIST equity codes: 4-5 uppercase letters, occasionally a trailing class marker.
    ticker_regex=r"^[A-Z]{4,5}[A-Z0-9]?$",
    yahoo_suffix=".IS",
)

_SP500 = Dataset(
    name="sp500",
    index_ticker="^GSPC",
    fallback_constituents=(
        # Tech / comms
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "ORCL",
        # Financials
        "JPM", "BAC", "WFC", "GS", "BRK-B", "V", "MA",
        # Health care
        "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK",
        # Consumer
        "PG", "KO", "PEP", "WMT", "HD", "MCD", "COST",
        # Industrials / energy
        "XOM", "CVX", "CAT", "BA", "GE",
    ),
    smoke_tickers=("AAPL", "MSFT", "JPM", "XOM", "JNJ"),

    source_url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    code_columns=("Symbol", "Ticker", "Code"),

    ticker_regex=r"^[A-Z]{1,5}(\.[A-Z])?$",
    yahoo_suffix="",
    dot_to_dash=True,
)

# G20 currencies quoted *vs USD* as `XXXUSD=X` — the value of one unit of the
# foreign currency in USD (so each series behaves like a USD-priced asset, the
# right convention for the trend engine). The EU/France/Germany/Italy seats all
# share EUR, so the unique-currency universe is 16 names. SAR is effectively
# USD-pegged and CNY is managed — both stay in for completeness but their trend
# signal is near-degenerate. Index = ICE US Dollar Index (`DX-Y.NYB`); Yahoo's
# `^DXY`/`DX=F` return no data. There is no clean web table of G20-currency Yahoo
# symbols, so this is a `fixed_universe` dataset: `python main.py universe
# --dataset currency` just writes the `fallback_constituents` list below to
# data/universe/currency.txt (no scrape). The fetch-spec fields are kept only to
# satisfy the dataclass.
_CURRENCY = Dataset(
    name="currency",
    index_ticker="DX-Y.NYB",
    fallback_constituents=(
        # Majors
        "EURUSD=X", "GBPUSD=X", "JPYUSD=X", "AUDUSD=X", "CADUSD=X",
        # Asia
        "CNYUSD=X", "INRUSD=X", "IDRUSD=X", "KRWUSD=X",
        # Americas
        "BRLUSD=X", "MXNUSD=X", "ARSUSD=X",
        # EMEA
        "TRYUSD=X", "RUBUSD=X", "ZARUSD=X", "SARUSD=X",
    ),
    smoke_tickers=("EURUSD=X", "GBPUSD=X", "JPYUSD=X", "AUDUSD=X", "CADUSD=X"),
    # Fixed universe — fetch is not wired for FX. Wikipedia G20 page kept only as
    # a human reference; the regex matches Yahoo's 6-letter FX symbols.
    source_url="https://en.wikipedia.org/wiki/G20",
    code_columns=("Symbol", "Code", "Ticker"),
    ticker_regex=r"^[A-Z]{6}=X$",
    yahoo_suffix="",
    fixed_universe=True,
)

DATASETS: dict[str, Dataset] = {d.name: d for d in (_BIST100, _SP500, _CURRENCY)}
DATASET_NAMES: list[str] = sorted(DATASETS)


def get_dataset(name: str = DEFAULT_DATASET) -> Dataset:
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; choices: {DATASET_NAMES}")


# Backward-compatible module-level aliases for the default (BIST100) dataset.
# Existing code and notebooks import these directly.
BIST100_INDEX_TICKER = _BIST100.index_ticker
BIST100_CONSTITUENTS: list[str] = _BIST100.load_constituents()
SMOKE_TICKERS: list[str] = list(_BIST100.smoke_tickers)

# ---------------------------------------------------------------------------
# Schema for raw OHLCV parquet
# ---------------------------------------------------------------------------
OHLCV_COLS = ["open", "high", "low", "close", "volume"]
