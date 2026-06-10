"""Fetch the current constituent list for a dataset (BIST100, S&P 500, ...).

One-shot snapshot script. The course brief accepts current-membership
survivorship bias, so we don't reconstruct historical composition — we just
pull whatever the source publishes today and persist it to
``data/universe/{dataset}.txt`` (one Yahoo symbol per line). ``config.py``
reads from that file at import time if it exists; otherwise it falls back to
the curated subset baked into each ``Dataset``.

USAGE
-----

    python -m src.data.fetch_universe                       # bist100 (default)
    python -m src.data.fetch_universe --dataset sp500       # write data/universe/sp500.txt
    python -m src.data.fetch_universe --dataset sp500 --dry-run
    python -m src.data.fetch_universe --url URL             # override the source page

The source URL, the code-column candidates, the per-cell ticker regex, and the
Yahoo-symbol mapping all come from the selected ``Dataset`` in ``config.py``:

  - BIST100 -> IsYatirim's BIST 100 fundamentals page (`Kod` column, ``.IS`` suffix).
  - S&P 500 -> Wikipedia "List of S&P 500 companies" (`Symbol` column; class
    shares like ``BRK.B`` are mapped to Yahoo's ``BRK-B`` form).
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import pandas as pd
import requests

from config import DEFAULT_DATASET, Dataset, get_dataset

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
_HTTP_TIMEOUT = 30


def _codes_from_named_column(
    tables: list[pd.DataFrame], code_columns: tuple[str, ...], ticker_rx: re.Pattern[str]
) -> list[str]:
    """Prefer extracting from a table that has an obvious code column —
    cleaner than regex-scanning every cell (which catches navbox headers
    and other false positives on Wikipedia-style pages)."""
    for tbl in tables:
        cols = [str(c).strip() for c in tbl.columns]
        match = next((c for c in cols if c in code_columns), None)
        if match is None:
            continue
        series = tbl[tbl.columns[cols.index(match)]].dropna().astype(str)
        codes = [s.strip().upper() for s in series if ticker_rx.match(s.strip().upper())]
        if len(codes) >= 50:  # plausibly a full-index table, not a partial widget
            return codes
    return []


def _codes_from_regex_sweep(
    tables: list[pd.DataFrame], ticker_rx: re.Pattern[str]
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for tbl in tables:
        for col in tbl.columns:
            for val in tbl[col].dropna().astype(str):
                s = val.strip().upper()
                if ticker_rx.match(s) and s not in seen:
                    seen.add(s)
                    out.append(s)
    return out


def fetch_constituents(ds: Dataset, url: str | None = None) -> list[str]:
    url = url or ds.source_url
    ticker_rx = re.compile(ds.ticker_regex)
    resp = requests.get(url, headers={"User-Agent": _UA}, timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    codes = (
        _codes_from_named_column(tables, ds.code_columns, ticker_rx)
        or _codes_from_regex_sweep(tables, ticker_rx)
    )
    # Dedupe preserving order, then map bare codes to Yahoo symbols.
    seen: set[str] = set()
    uniq = [c for c in codes if not (c in seen or seen.add(c))]
    return [ds.to_yahoo(c) for c in uniq]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="dataset to refresh (default: bist100)")
    p.add_argument("--url", default=None, help="override the source page")
    p.add_argument("--out", type=Path, default=None,
                   help="output file (default: data/universe/{dataset}.txt)")
    p.add_argument("--dry-run", action="store_true",
                   help="print parsed tickers to stdout and exit without writing")
    ns = p.parse_args(argv)

    ds = get_dataset(ns.dataset)
    url = ns.url or ds.source_url
    out = ns.out or ds.universe_file

    if ds.fixed_universe:
        # No scrapable source (e.g. FX): the curated fallback IS the universe.
        tickers = list(ds.fallback_constituents)
        print(f"[{ds.name}] fixed universe — writing {len(tickers)} curated tickers",
              file=sys.stderr)
    else:
        print(f"[{ds.name}] fetching {url}", file=sys.stderr)
        tickers = fetch_constituents(ds, url=url)
    if not tickers:
        print(f"no tickers parsed from {url} — check the URL or pass --url",
              file=sys.stderr)
        return 1
    print(f"parsed {len(tickers)} tickers", file=sys.stderr)

    if ns.dry_run:
        print("\n".join(tickers))
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    src = "curated fixed list" if ds.fixed_universe else f"snapshot from {url}"
    header = f"# {ds.name} constituents, {src}\n"
    out.write_text(header + "\n".join(tickers) + "\n")
    print(f"wrote {len(tickers)} tickers to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
