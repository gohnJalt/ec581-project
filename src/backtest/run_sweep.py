"""Phase-1 grid-search driver: run sweep.default_specs() on the BIST100 Index.

Reads the index close from data/clean/, runs the full S1..S4 grid through
``src.backtest.sweep.sweep``, and writes ``results/phase1_sweep.parquet``.

Usage:
    python -m src.backtest.run_sweep
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from config import CASH_PER_TRADE, DEFAULT_DATASET, INITIAL_CAPITAL, get_dataset
from src.backtest.sweep import default_specs, sweep
from src.data.clean import clean_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="dataset whose index to sweep (default: bist100)")
    ap.add_argument("--min-trades", type=int, default=30,
                    help="floor for the enough_trades flag (does not drop rows)")
    args = ap.parse_args()

    ds = get_dataset(args.dataset)
    ohlcv = pd.read_parquet(clean_path(ds.index_ticker))
    prices = ohlcv["close"]
    print(f"dataset={ds.name}  index={ds.index_ticker}  bars={len(prices)}  "
          f"range=[{prices.index.min().date()}, {prices.index.max().date()}]")
    print(f"initial capital={INITIAL_CAPITAL:,}  cash per trade={CASH_PER_TRADE:,}")

    t0 = time.time()
    df = sweep(prices, ohlcv, specs=default_specs(),
               initial_capital=INITIAL_CAPITAL, cash_per_trade=CASH_PER_TRADE,
               min_trades=args.min_trades, verbose=True)
    print(f"\nsweep finished in {time.time()-t0:.1f}s")

    out = ds.results_dir / "phase1_sweep.parquet"
    df.to_parquet(out)
    print(f"\nwrote {out}  shape={df.shape}")

    elig = df[df["enough_trades"]]
    winners = (elig.sort_values("sharpe", ascending=False)
                   .groupby("strategy", as_index=False).first())
    print("\nWinners (highest Sharpe per strategy, ≥ min_trades):")
    print(winners[["strategy", "sharpe", "n_trades", "cagr",
                   "max_drawdown"]].round({"sharpe": 3, "cagr": 4,
                                            "max_drawdown": 4}).to_string(index=False))


if __name__ == "__main__":
    main()
