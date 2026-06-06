"""Phase-2 per-stock Monte Carlo Sharpe p-values.

For each cleaned constituent, generate ``n_iter`` random matched-frequency
signal series and compute the one-sided p-value
``P(MC_Sharpe >= strat_Sharpe)`` using the fast simulator in
``src.eval.montecarlo``.

Only the ungated (``base``) variant is MC-tested. The regime variant position
series is path-dependent (close on blocked entry, then flat until the next
allowed flip), which the fast simulator does not model — comparing the regime
Sharpe against a random pool that ignores the same path-dependence would not
be apples-to-apples.

The strategy is selectable via ``--strategy {hp,lowess}`` and outputs go to:
  ``results/phase2_mc_{stub}_base.parquet`` (stub = hp or lowess).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import (
    DEFAULT_DATASET,
    RESULTS_DIR,
    get_dataset,
)
from src.data.clean import clean_path
from src.eval.montecarlo import monte_carlo_sharpe
from src.eval.strategies import StrategySpec, get as get_strategy, signal as compute_signal


DEFAULT_N_ITER = 1000


def _available_tickers(universe: list[str]) -> list[str]:
    return [t for t in universe if clean_path(t).exists()]


def _load_close(ticker: str) -> pd.Series:
    return pd.read_parquet(clean_path(ticker))["close"]


def _output_path(spec: StrategySpec, out_dir: Path) -> Path:
    # Preserve the legacy "phase2_mc_base.parquet" name for hp; namespace
    # lowess and any future strategies.
    if spec.key == "hp":
        return out_dir / "phase2_mc_base.parquet"
    return out_dir / f"phase2_mc_{spec.output_stub}_base.parquet"


def run(
    tickers: list[str],
    spec: StrategySpec,
    param: float | None = None,
    window: int | None = None,
    n_iter: int = DEFAULT_N_ITER,
    out_dir: Path = RESULTS_DIR,
    verbose: bool = True,
) -> Path:
    rows: list[dict] = []
    for ticker in tickers:
        prices = _load_close(ticker)
        sig = compute_signal(prices, spec, param=param, window=window)
        mc = monte_carlo_sharpe(prices, sig, n_iter=n_iter)
        row = {"ticker": ticker, **mc.to_dict()}
        rows.append(row)
        if verbose:
            print(f"{ticker:9s}  Sharpe {mc.strat_sharpe:6.2f}  "
                  f"MC mean {mc.mc_sharpes.mean():6.2f}  "
                  f"q95 {row['mc_sharpe_q95']:6.2f}  "
                  f"p={mc.p_value:.3f}  "
                  f"runs L/S {mc.n_long_runs}/{mc.n_short_runs}")

    df = pd.DataFrame(rows)
    out_path = _output_path(spec, out_dir)
    df.to_parquet(out_path)
    if verbose:
        _print_summary(df)
    return out_path


def _print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("\n[mc] empty")
        return
    print(f"\n[mc] {len(df)} tickers")
    sig01 = int((df["p_value"] < 0.01).sum())
    sig05 = int((df["p_value"] < 0.05).sum())
    print(f"  p<0.01: {sig01}/{len(df)}    p<0.05: {sig05}/{len(df)}")
    print(f"  mean strat Sharpe {df['strat_sharpe'].mean():.3f}   "
          f"mean MC mean {df['mc_sharpe_mean'].mean():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="dataset to run (default: bist100)")
    ap.add_argument("--smoke", action="store_true",
                    help="use the dataset's smoke subset")
    ap.add_argument("--strategy", choices=["hp", "lowess"], default="hp")
    ap.add_argument("--param", type=float, default=None,
                    help="lam for hp, frac for lowess; defaults to Phase-1 winner")
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    args = ap.parse_args()

    ds = get_dataset(args.dataset)
    spec = get_strategy(args.strategy)
    universe = list(ds.smoke_tickers) if args.smoke else ds.load_constituents()
    tickers = _available_tickers(universe)
    p = spec.default_param if args.param is None else args.param
    w = spec.default_window if args.window is None else args.window
    print(f"dataset={ds.name}  tickers ({len(tickers)}): {tickers}")
    print(f"strategy: {spec.label} {spec.param_name}={p} window={w}")
    print(f"n_iter: {args.n_iter}")
    run(tickers, spec, param=args.param, window=args.window, n_iter=args.n_iter,
        out_dir=ds.results_dir)


if __name__ == "__main__":
    main()
