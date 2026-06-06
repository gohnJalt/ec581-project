"""Resumable per-ticker Phase-2 driver: writes incrementally after each ticker.

Same logic as run_phase2.py but each ticker is processed in isolation and the
result parquet is rewritten after each, with explicit gc.collect() between
iterations. Solves two issues seen with the bulk run_phase2.py on LOWESS:

  1. If the process stalls on any one ticker, completed tickers are persisted.
  2. Per-ticker gc avoids the slow memory accumulation that caused 'U'-state
     throttling on the bulk run.

Usage:
    python -u -m src.eval.run_phase2_resumable --strategy lowess
    python -u -m src.eval.run_phase2_resumable --strategy hp
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import pandas as pd

from config import (
    CASH_PER_TRADE,
    DEFAULT_DATASET,
    INITIAL_CAPITAL,
    get_dataset,
)
from src.backtest.runner import run_backtest
from src.data.clean import clean_path
from src.eval.metrics import compute_metrics
from src.eval.strategies import get as get_strategy, signal as compute_signal
from src.features.regime import bist_regime_flag


REGIME_LAM = 1600
REGIME_WINDOW = 504


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="dataset to run (default: bist100)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--strategy", choices=["hp", "lowess"], default="hp")
    ap.add_argument("--param", type=float, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--start-from", type=str, default=None,
                    help="Skip tickers until this one (resume mode)")
    args = ap.parse_args()

    ds = get_dataset(args.dataset)
    spec = get_strategy(args.strategy)
    universe = list(ds.smoke_tickers) if args.smoke else ds.load_constituents()
    tickers = [t for t in universe if clean_path(t).exists()]
    if args.start_from:
        i = tickers.index(args.start_from)
        tickers = tickers[i:]
        print(f"[resume] starting from {args.start_from}, {len(tickers)} tickers left")

    p = spec.default_param if args.param is None else args.param
    w = spec.default_window if args.window is None else args.window

    base_path = ds.results_dir / f"phase2_{spec.output_stub}_base.parquet"
    regime_path = ds.results_dir / f"phase2_{spec.output_stub}_regime.parquet"

    base_existing = pd.read_parquet(base_path) if base_path.exists() and args.start_from else pd.DataFrame()
    regime_existing = pd.read_parquet(regime_path) if regime_path.exists() and args.start_from else pd.DataFrame()
    if not base_existing.empty:
        base_existing = base_existing[~base_existing["ticker"].isin(tickers)]
        regime_existing = regime_existing[~regime_existing["ticker"].isin(tickers)]

    bist_close = pd.read_parquet(clean_path(ds.index_ticker))["close"]
    regime = bist_regime_flag(bist_close, lam=REGIME_LAM, window=REGIME_WINDOW)
    print(f"regime: lam={REGIME_LAM} window={REGIME_WINDOW} "
          f"long={int((regime>0).sum())} short={int((regime<0).sum())} flat={int((regime==0).sum())}")
    print(f"strategy: {spec.label} {spec.param_name}={p} window={w}")
    print(f"writing to {base_path.name} and {regime_path.name}\n")

    base_rows: list[dict] = base_existing.to_dict("records") if not base_existing.empty else []
    regime_rows: list[dict] = regime_existing.to_dict("records") if not regime_existing.empty else []

    for idx, ticker in enumerate(tickers, 1):
        t0 = time.time()
        ohlcv = pd.read_parquet(clean_path(ticker))
        prices = ohlcv["close"]

        trend = compute_signal(prices, spec, param=p, window=w)
        t_sig = time.time() - t0

        res_b = run_backtest(ohlcv, trend, regime=None,
                             initial_cash=INITIAL_CAPITAL, cash_per_trade=CASH_PER_TRADE)
        m_b = compute_metrics(res_b.equity_curve, res_b.trades, INITIAL_CAPITAL, CASH_PER_TRADE)
        base_rows.append({**m_b.to_dict(), "ticker": ticker})

        reg_aligned = regime.reindex(prices.index).fillna(0.0)
        res_r = run_backtest(ohlcv, trend, regime=reg_aligned,
                             initial_cash=INITIAL_CAPITAL, cash_per_trade=CASH_PER_TRADE)
        m_r = compute_metrics(res_r.equity_curve, res_r.trades, INITIAL_CAPITAL, CASH_PER_TRADE)
        regime_rows.append({**m_r.to_dict(), "ticker": ticker})

        pd.DataFrame(base_rows).to_parquet(base_path)
        pd.DataFrame(regime_rows).to_parquet(regime_path)

        elapsed = time.time() - t0
        print(f"[{idx:2d}/{len(tickers)}] {ticker:9s} "
              f"base S={m_b.sharpe:5.2f} CAGR={m_b.cagr:6.2%} MDD={m_b.max_drawdown:7.2%}  "
              f"reg S={m_r.sharpe:5.2f} CAGR={m_r.cagr:6.2%} MDD={m_r.max_drawdown:7.2%}  "
              f"({elapsed:.1f}s sig={t_sig:.1f}s)", flush=True)

        del ohlcv, prices, trend, res_b, res_r, reg_aligned
        gc.collect()

    base_df = pd.DataFrame(base_rows)
    regime_df = pd.DataFrame(regime_rows)
    print(f"\n[base]   {len(base_df)} tickers  mean Sharpe {base_df['sharpe'].mean():.3f}  "
          f"mean CAGR {base_df['cagr'].mean():.2%}  mean MDD {base_df['max_drawdown'].mean():.2%}")
    print(f"[regime] {len(regime_df)} tickers  mean Sharpe {regime_df['sharpe'].mean():.3f}  "
          f"mean CAGR {regime_df['cagr'].mean():.2%}  mean MDD {regime_df['max_drawdown'].mean():.2%}")


if __name__ == "__main__":
    main()
