"""Phase-2 base+regime driver: chosen strategy on every cleaned constituent.

For each ticker, run the chosen Phase-2 strategy (S3 HP-direction by default,
or S4 LOWESS-direction with ``--strategy lowess``) at its Phase-1 winning
parameters in two variants:

  - ``base``   plain TrendStrategy (no regime gate)
  - ``regime`` BIST100 vs HP(1600, 504) regime gate per DESIGN.md §3.4

Single full-history backtest per (ticker, variant). For the per-fold version
that re-picks parameters every year, see ``src/eval/run_walkforward.py``.

Outputs two parquets under ``results/`` (stub depends on strategy):
  - ``phase2_{stub}_base.parquet``
  - ``phase2_{stub}_regime.parquet``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import (
    BIST100_CONSTITUENTS,
    BIST100_INDEX_TICKER,
    CASH_PER_TRADE,
    INITIAL_CAPITAL,
    RESULTS_DIR,
    SMOKE_TICKERS,
)
from src.backtest.runner import run_backtest
from src.data.clean import clean_path
from src.eval.metrics import compute_metrics
from src.eval.strategies import StrategySpec, get as get_strategy, signal as compute_signal
from src.features.regime import bist_regime_flag


# Regime filter parameters per DESIGN.md §3.4 — λ=1600 is the brief's default.
REGIME_LAM = 1600
REGIME_WINDOW = 504


def _available_tickers(universe: list[str]) -> list[str]:
    return [t for t in universe if clean_path(t).exists()]


def _load_ohlcv(ticker: str) -> pd.DataFrame:
    return pd.read_parquet(clean_path(ticker))


def _row(ticker: str, ohlcv: pd.DataFrame, trend: pd.Series,
         regime: pd.Series | None) -> dict:
    res = run_backtest(ohlcv, trend, regime=regime,
                       initial_cash=INITIAL_CAPITAL,
                       cash_per_trade=CASH_PER_TRADE)
    m = compute_metrics(res.equity_curve, res.trades, INITIAL_CAPITAL, CASH_PER_TRADE)
    d = m.to_dict()
    d["ticker"] = ticker
    return d


def run(
    tickers: list[str],
    spec: StrategySpec,
    param: float | None = None,
    window: int | None = None,
    regime_lam: float = REGIME_LAM,
    regime_window: int = REGIME_WINDOW,
    out_dir: Path = RESULTS_DIR,
    verbose: bool = True,
) -> dict[str, Path]:
    if not tickers:
        raise RuntimeError("no tickers available — populate data/clean/ first")

    p = spec.default_param if param is None else param
    w = spec.default_window if window is None else window

    bist_close = _load_ohlcv(BIST100_INDEX_TICKER)["close"]
    regime = bist_regime_flag(bist_close, lam=regime_lam, window=regime_window)
    if verbose:
        print(f"regime flag: lam={regime_lam} window={regime_window} "
              f"long={int((regime > 0).sum())} short={int((regime < 0).sum())} "
              f"flat={int((regime == 0).sum())}")
        print(f"strategy: {spec.label} {spec.param_name}={p} window={w}\n")

    base_rows: list[dict] = []
    regime_rows: list[dict] = []

    for ticker in tickers:
        ohlcv = _load_ohlcv(ticker)
        prices = ohlcv["close"]
        trend = compute_signal(prices, spec, param=p, window=w)

        b = _row(ticker, ohlcv, trend, regime=None)
        base_rows.append(b)

        reg_aligned = regime.reindex(prices.index).fillna(0.0)
        r = _row(ticker, ohlcv, trend, regime=reg_aligned)
        regime_rows.append(r)

        if verbose:
            print(f"{ticker:9s} base   Sharpe {b['sharpe']:6.2f}  "
                  f"trades {b['n_trades']:4d}  CAGR {b['cagr']:7.2%}  "
                  f"MDD {b['max_drawdown']:7.2%}")
            print(f"{ticker:9s} regime Sharpe {r['sharpe']:6.2f}  "
                  f"trades {r['n_trades']:4d}  CAGR {r['cagr']:7.2%}  "
                  f"MDD {r['max_drawdown']:7.2%}")

    base_df = pd.DataFrame(base_rows)
    regime_df = pd.DataFrame(regime_rows)

    base_path = out_dir / f"phase2_{spec.output_stub}_base.parquet"
    regime_path = out_dir / f"phase2_{spec.output_stub}_regime.parquet"
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
    print(f"\n[{name}] {len(df)} tickers")
    print(f"  mean Sharpe  {df['sharpe'].mean():.3f}   "
          f"median {df['sharpe'].median():.3f}")
    print(f"  mean CAGR    {df['cagr'].mean():.2%}   "
          f"median {df['cagr'].median():.2%}")
    print(f"  mean MDD     {df['max_drawdown'].mean():.2%}   "
          f"median {df['max_drawdown'].median():.2%}")
    print(f"  total trades {int(df['n_trades'].sum())}   "
          f"mean per ticker {df['n_trades'].mean():.0f}")
    print(f"  Sharpe>0     {int((df['sharpe'] > 0).sum())}/{len(df)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="use SMOKE_TICKERS instead of full BIST100_CONSTITUENTS")
    ap.add_argument("--strategy", choices=["hp", "lowess"], default="hp",
                    help="Phase-2 strategy: S3 HP-direction (default) or S4 LOWESS-direction")
    ap.add_argument("--param", type=float, default=None,
                    help="smoother param (lam for hp, frac for lowess); defaults to Phase-1 winner")
    ap.add_argument("--window", type=int, default=None,
                    help="rolling window (defaults to Phase-1 winner for the chosen strategy)")
    args = ap.parse_args()

    spec = get_strategy(args.strategy)
    universe = SMOKE_TICKERS if args.smoke else BIST100_CONSTITUENTS
    tickers = _available_tickers(universe)

    print(f"tickers ({len(tickers)}): {tickers}")
    run(tickers, spec, param=args.param, window=args.window)


if __name__ == "__main__":
    main()
