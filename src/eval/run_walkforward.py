"""Phase-2 walk-forward driver: S3 HP on every cleaned ticker, base + regime.

Reads cleaned OHLCV from ``data/clean/`` (whatever is there — populated by
``python -m src.data.build``), runs walk-forward for the primary strategy S3
HP-direction in two variants:

  - ``base``   plain TrendStrategy (no regime gate)
  - ``regime`` BIST100 vs HP(1600, 504) regime gate per DESIGN.md §3.4

Outputs two parquets under ``results/``:
  - ``phase2_walkforward_base.parquet``
  - ``phase2_walkforward_regime.parquet``
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd

from config import (
    BIST100_INDEX_TICKER,
    DEFAULT_DATASET,
    RESULTS_DIR,
    get_dataset,
)
from src.data.clean import clean_path
from src.eval.walkforward import walk_forward
from src.features.hp import hp_direction_signal
from src.features.regime import bist_regime_flag


# Walk-forward HP grid. Tighter than the Phase-1 sweep: drop the 1260-bar
# window (warmup eats 5y, leaving too little room for per-fold retraining)
# and drop lam=100 (uncompetitive in Phase 1 per CLAUDE.md results table).
WF_HP_GRID_FULL = [
    {"lam": lam, "window": w}
    for lam, w in product([1600, 14400, 129600], [252, 504])
]

# Smoke grid for fast iteration.
WF_HP_GRID_SMOKE = [
    {"lam": 14400, "window": 504},
    {"lam": 1600, "window": 252},
]


def _available_tickers(universe: list[str]) -> list[str]:
    return [t for t in universe if clean_path(t).exists()]


def _load_close(ticker: str) -> pd.Series:
    df = pd.read_parquet(clean_path(ticker))
    return df["close"].rename(ticker)


def _load_ohlcv(ticker: str) -> pd.DataFrame:
    return pd.read_parquet(clean_path(ticker))


def run(
    tickers: list[str],
    grid: list[dict],
    regime_lam: float = 1600,
    regime_window: int = 504,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
    min_trades_train: int = 30,
    index_ticker: str = BIST100_INDEX_TICKER,
    out_dir: Path = RESULTS_DIR,
    verbose: bool = True,
) -> dict[str, Path]:
    if not tickers:
        raise RuntimeError("no tickers available — populate data/clean/ first")

    bist_close = _load_close(index_ticker)
    regime = bist_regime_flag(bist_close, lam=regime_lam, window=regime_window)
    if verbose:
        print(f"regime flag: lam={regime_lam} window={regime_window} "
              f"long={int((regime > 0).sum())} short={int((regime < 0).sum())} "
              f"flat={int((regime == 0).sum())}")

    base_rows: list[pd.DataFrame] = []
    regime_rows: list[pd.DataFrame] = []

    for ticker in tickers:
        if verbose:
            print(f"\n=== {ticker} ===")
        ohlcv = _load_ohlcv(ticker)
        prices = ohlcv["close"]

        if verbose:
            print(" [base]")
        base = walk_forward(
            prices, ohlcv, hp_direction_signal, grid,
            train_years=train_years, test_years=test_years,
            step_years=step_years, min_trades_train=min_trades_train,
            label=ticker, verbose=verbose,
        )
        base["ticker"] = ticker
        base["variant"] = "base"
        base_rows.append(base)

        if verbose:
            print(" [regime]")
        reg_aligned = regime.reindex(prices.index).fillna(0.0)
        regd = walk_forward(
            prices, ohlcv, hp_direction_signal, grid,
            regime=reg_aligned,
            train_years=train_years, test_years=test_years,
            step_years=step_years, min_trades_train=min_trades_train,
            label=ticker, verbose=verbose,
        )
        regd["ticker"] = ticker
        regd["variant"] = "regime"
        regime_rows.append(regd)

    base_df = pd.concat(base_rows, ignore_index=True) if base_rows else pd.DataFrame()
    regime_df = pd.concat(regime_rows, ignore_index=True) if regime_rows else pd.DataFrame()

    base_path = out_dir / "phase2_walkforward_base.parquet"
    regime_path = out_dir / "phase2_walkforward_regime.parquet"
    base_df.to_parquet(base_path)
    regime_df.to_parquet(regime_path)

    if verbose:
        _print_summary("base", base_df)
        _print_summary("regime", regime_df)

    return {"base": base_path, "regime": regime_path}


def _print_summary(name: str, df: pd.DataFrame) -> None:
    if df.empty:
        print(f"\n[{name}] empty")
        return
    print(f"\n[{name}] {len(df)} fold rows across {df['ticker'].nunique()} tickers")
    agg = (
        df.groupby("ticker")
        .agg(folds=("fold_id", "count"),
             mean_sharpe=("test_sharpe", "mean"),
             mean_cagr=("test_cagr", "mean"),
             mean_mdd=("test_max_drawdown", "mean"),
             total_trades=("test_n_trades", "sum"))
        .round(3)
    )
    print(agg.to_string())
    print(f"\n[{name}] overall mean test Sharpe: {df['test_sharpe'].mean():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="dataset to run (default: bist100)")
    ap.add_argument("--smoke", action="store_true",
                    help="use the dataset's smoke subset and a 2-config HP grid")
    ap.add_argument("--min-trades", type=int, default=30,
                    help="minimum training-fold trade count for a candidate "
                         "to be selectable")
    args = ap.parse_args()

    ds = get_dataset(args.dataset)
    universe = list(ds.smoke_tickers) if args.smoke else ds.load_constituents()
    tickers = _available_tickers(universe)
    grid = WF_HP_GRID_SMOKE if args.smoke else WF_HP_GRID_FULL

    print(f"dataset={ds.name}  tickers ({len(tickers)}): {tickers}")
    print(f"grid ({len(grid)} configs): {grid}")
    print(f"min trades per train fold: {args.min_trades}")

    run(tickers, grid, min_trades_train=args.min_trades,
        index_ticker=ds.index_ticker, out_dir=ds.results_dir)


if __name__ == "__main__":
    main()
