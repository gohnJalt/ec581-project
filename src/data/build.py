"""End-to-end data pipeline orchestrator.

Usage:

    python -m src.data.build                  # full BIST100 universe (default)
    python -m src.data.build --dataset sp500  # full S&P 500 universe
    python -m src.data.build --smoke          # M1 smoke: index + 5 stocks
    python -m src.data.build --source csv --csv-dir path/to/vendor

Stages: ingest -> clean -> panel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import DEFAULT_DATASET, PANEL_DIR, get_dataset
from src.data.clean import clean_universe
from src.data.ingest import ingest_universe
from src.data.panel import build_all


def _summarize(stage: str, results: dict) -> None:
    ok = [k for k, v in results.items() if not isinstance(v, Exception) and v is not None]
    dropped = [k for k, v in results.items() if v is None]
    failed = {k: str(v) for k, v in results.items() if isinstance(v, Exception)}
    print(f"[{stage}] ok={len(ok)} dropped={len(dropped)} failed={len(failed)}")
    if dropped:
        print(f"  dropped (short history): {dropped}")
    if failed:
        for t, err in failed.items():
            print(f"  FAILED {t}: {err}")


def run(
    constituents: list[str],
    index_ticker: str,
    source: str,
    csv_dir: Path | None,
    force: bool,
    panel_dir: Path = PANEL_DIR,
) -> int:
    universe = [index_ticker] + list(constituents)

    print(f"[ingest] {len(universe)} tickers from source={source!r}")
    ingest_results = ingest_universe(universe, source=source, csv_dir=csv_dir, force=force)
    _summarize("ingest", ingest_results)

    ingested = [t for t, v in ingest_results.items() if not isinstance(v, Exception)]
    print(f"[clean] {len(ingested)} tickers")
    clean_results = clean_universe(ingested, force=force)
    _summarize("clean", clean_results)

    surviving_constituents = [
        t for t in constituents
        if not isinstance(clean_results.get(t), Exception) and clean_results.get(t) is not None
    ]
    if isinstance(clean_results.get(index_ticker), Exception) or clean_results.get(index_ticker) is None:
        print(f"[panel] FATAL: index {index_ticker} did not survive cleaning")
        return 2
    if not surviving_constituents:
        print("[panel] FATAL: no constituents survived cleaning")
        return 2

    print(f"[panel] {len(surviving_constituents)} constituents + 1 index")
    paths = build_all(surviving_constituents, index_ticker=index_ticker, panel_dir=panel_dir)
    for k, p in paths.items():
        print(f"  {k}: {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="EC581 data pipeline")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="dataset to build (default: bist100)")
    p.add_argument("--smoke", action="store_true", help="run on M1 5-stock subset")
    p.add_argument("--source", choices=("yfinance", "csv"), default="yfinance")
    p.add_argument("--csv-dir", type=Path, default=None)
    p.add_argument("--force", action="store_true", help="re-fetch and re-clean")
    args = p.parse_args(argv)

    ds = get_dataset(args.dataset)
    constituents = list(ds.smoke_tickers) if args.smoke else ds.load_constituents()
    print(f"[dataset] {ds.name}  index={ds.index_ticker}  "
          f"constituents={len(constituents)}{' (smoke)' if args.smoke else ''}")
    return run(
        constituents=constituents,
        index_ticker=ds.index_ticker,
        source=args.source,
        csv_dir=args.csv_dir,
        force=args.force,
        panel_dir=ds.panel_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
