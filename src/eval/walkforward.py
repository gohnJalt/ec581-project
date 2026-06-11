"""Walk-forward fold runner.

For each ticker we slide a (train, test) window across the full price history:
re-pick the best parameter combination on the training slice by Sharpe, then
evaluate those parameters on the immediately-following test slice.

The candidate trend signals are causal by construction (the rolling wrappers in
``src/features/``), so the full-series signal can be computed once per
parameter combination and sliced per fold without leaking future information.

Defaults: 3y train / 1y test / 1y step. A minimum-trade-count floor on the
training slice avoids degenerate winners that traded too rarely to be measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from config import CASH_PER_TRADE, INITIAL_CAPITAL
from src.backtest.runner import run_backtest
from src.eval.metrics import compute_metrics

TRADING_DAYS = 252


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp     # exclusive
    test_start: pd.Timestamp
    test_end: pd.Timestamp      # exclusive


def make_folds(
    index: pd.DatetimeIndex,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
    warmup_bars: int = 0,
) -> list[Fold]:
    train_n = train_years * TRADING_DAYS
    test_n = test_years * TRADING_DAYS
    step_n = step_years * TRADING_DAYS
    n = len(index)
    folds: list[Fold] = []
    fold_id = 0
    start = warmup_bars
    while start + train_n + test_n <= n:
        train_end = start + train_n
        test_end = train_end + test_n
        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=index[start],
                train_end=index[train_end - 1],
                test_start=index[train_end],
                test_end=index[test_end - 1],
            )
        )
        fold_id += 1
        start += step_n
    return folds


def _slice(series_or_df, start: pd.Timestamp, end: pd.Timestamp):
    return series_or_df.loc[start:end]


def _params_key(params: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(params.items()))


def _precompute_signals(
    prices: pd.Series, signal_fn: Callable[..., pd.Series], grid: list[dict]
) -> dict[str, pd.Series]:
    cache: dict[str, pd.Series] = {}
    for params in grid:
        cache[_params_key(params)] = signal_fn(prices, **params)
    return cache


def _run_slice(
    ohlcv: pd.DataFrame,
    trend: pd.Series,
    regime: pd.Series | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_capital: float,
    cash_per_trade: float,
) -> dict:
    o = _slice(ohlcv, start, end)
    t = _slice(trend, start, end)
    r = _slice(regime, start, end) if regime is not None else None
    if len(o) < 2:
        return {"sharpe": 0.0, "n_trades": 0, "cagr": 0.0, "max_drawdown": 0.0,
                "net_pnl_pct": 0.0, "calmar": 0.0, "ann_vol": 0.0,
                "sortino": 0.0, "win_rate": 0.0, "avg_holding_period": 0.0,
                "turnover": 0.0, "n_bars": len(o)}
    res = run_backtest(o, t, regime=r, initial_cash=initial_capital,
                       cash_per_trade=cash_per_trade)
    m = compute_metrics(res.equity_curve, res.trades, initial_capital, cash_per_trade)
    return m.to_dict()


def walk_forward(
    prices: pd.Series,
    ohlcv: pd.DataFrame,
    signal_fn: Callable[..., pd.Series],
    grid: list[dict],
    *,
    regime: pd.Series | None = None,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
    min_trades_train: int = 30,
    selection_metric: str = "sharpe",
    initial_capital: float = INITIAL_CAPITAL,
    cash_per_trade: float = CASH_PER_TRADE,
    label: str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """Run walk-forward for one (price series, strategy) pair across a grid.

    Returns one row per fold with the selected params and test-slice metrics.
    Folds where no grid entry hits ``min_trades_train`` are skipped.
    """
    warmup = max((p.get("window", 0) for p in grid), default=0)
    folds = make_folds(prices.index, train_years, test_years, step_years,
                       warmup_bars=warmup)
    if not folds:
        return pd.DataFrame()

    signals = _precompute_signals(prices, signal_fn, grid)

    rows: list[dict] = []
    for fold in folds:
        best = None
        for params in grid:
            key = _params_key(params)
            trend = signals[key]
            train_metrics = _run_slice(
                ohlcv, trend, regime, fold.train_start, fold.train_end,
                initial_capital, cash_per_trade,
            )
            if train_metrics["n_trades"] < min_trades_train:
                continue
            score = train_metrics[selection_metric]
            if best is None or score > best["train_score"]:
                best = {"params": params, "train_score": score,
                        "train_n_trades": train_metrics["n_trades"],
                        "train_sharpe": train_metrics["sharpe"]}

        row: dict = {
            "label": label,
            "fold_id": fold.fold_id,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
        }

        if best is None:
            row["selected_params"] = None
            row["train_sharpe"] = np.nan
            row["train_n_trades"] = 0
            for k in ("sharpe", "cagr", "max_drawdown", "n_trades",
                      "net_pnl_pct", "calmar", "ann_vol", "sortino",
                      "win_rate", "avg_holding_period", "turnover"):
                row[f"test_{k}"] = np.nan
            rows.append(row)
            if verbose:
                print(f"  fold {fold.fold_id:2d} {fold.test_start.date()}..{fold.test_end.date()} "
                      f"no candidate met min_trades={min_trades_train}")
            continue

        trend = signals[_params_key(best["params"])]
        test_metrics = _run_slice(
            ohlcv, trend, regime, fold.test_start, fold.test_end,
            initial_capital, cash_per_trade,
        )

        row["selected_params"] = _params_key(best["params"])
        row["train_sharpe"] = best["train_sharpe"]
        row["train_n_trades"] = best["train_n_trades"]
        for k in ("sharpe", "cagr", "max_drawdown", "n_trades",
                  "net_pnl_pct", "calmar", "ann_vol", "sortino",
                  "win_rate", "avg_holding_period", "turnover"):
            row[f"test_{k}"] = test_metrics[k]
        rows.append(row)

        if verbose:
            print(f"  fold {fold.fold_id:2d} {fold.test_start.date()}..{fold.test_end.date()} "
                  f"params=[{row['selected_params']}] "
                  f"train Sharpe {best['train_sharpe']:.2f} ({best['train_n_trades']} trades) -> "
                  f"test Sharpe {test_metrics['sharpe']:.2f} ({test_metrics['n_trades']} trades)")

    return pd.DataFrame(rows)


def aggregate_walkforward(
    df: pd.DataFrame, by: Iterable[str] = ("label",)
) -> pd.DataFrame:
    """Summary stats per (ticker, variant) across folds — useful for the
    final M5 side-by-side table."""
    if df.empty:
        return df
    grp = df.groupby(list(by), dropna=False)
    agg = grp.agg(
        n_folds=("fold_id", "count"),
        mean_test_sharpe=("test_sharpe", "mean"),
        median_test_sharpe=("test_sharpe", "median"),
        mean_test_cagr=("test_cagr", "mean"),
        mean_test_mdd=("test_max_drawdown", "mean"),
        total_test_trades=("test_n_trades", "sum"),
    ).reset_index()
    return agg
