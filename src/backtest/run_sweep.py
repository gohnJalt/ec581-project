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

from config import BIST100_INDEX_TICKER, CASH_PER_TRADE, INITIAL_CAPITAL, RESULTS_DIR
from src.backtest.sweep import default_specs, sweep
from src.data.clean import clean_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-trades", type=int, default=30,
                    help="floor for the enough_trades flag (does not drop rows)")
    args = ap.parse_args()

    ohlcv = pd.read_parquet(clean_path(BIST100_INDEX_TICKER))
    prices = ohlcv["close"]
    print(f"index={BIST100_INDEX_TICKER}  bars={len(prices)}  "
          f"range=[{prices.index.min().date()}, {prices.index.max().date()}]")
    print(f"initial capital={INITIAL_CAPITAL:,}  cash per trade={CASH_PER_TRADE:,}")

    t0 = time.time()
    df = sweep(prices, ohlcv, specs=default_specs(),
               initial_capital=INITIAL_CAPITAL, cash_per_trade=CASH_PER_TRADE,
               min_trades=args.min_trades, verbose=True)
    print(f"\nsweep finished in {time.time()-t0:.1f}s")

    out = RESULTS_DIR / "phase1_sweep.parquet"
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
