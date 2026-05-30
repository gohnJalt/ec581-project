"""Phase-1 grid search over S1..S4 on a single price series.

Each strategy has its own parameter grid (DESIGN.md §6.3). We run every
combination through the backtest engine and collect metrics. Configs whose
trade count falls below `min_trades` are kept in the table but flagged so they
can be filtered out before selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable

import pandas as pd

from config import CASH_PER_TRADE, INITIAL_CAPITAL
from src.backtest.runner import run_backtest
from src.eval.metrics import compute_metrics
from src.features.ema import ema_crossover_signal, ema_direction_signal
from src.features.hp import hp_direction_signal
from src.features.lowess import lowess_direction_signal


@dataclass
class StrategySpec:
    name: str
    signal_fn: Callable[..., pd.Series]
    grid: list[dict]


def _ema_crossover_grid() -> list[dict]:
    fasts = [5, 10, 20, 30]
    slows = [50, 100, 200]
    return [{"n_fast": f, "n_slow": s} for f, s in product(fasts, slows) if f < s]


def _ema_direction_grid() -> list[dict]:
    return [{"n": n} for n in [10, 20, 50, 100, 200]]


def _hp_grid() -> list[dict]:
    return [{"lam": lam, "window": w}
            for lam, w in product([100, 1600, 14400, 129600], [252, 504, 1260])]


def _lowess_grid() -> list[dict]:
    return [{"frac": f, "window": w}
            for f, w in product([0.05, 0.1, 0.2, 0.3], [252, 504, 1260])]


def default_specs() -> list[StrategySpec]:
    return [
        StrategySpec("S1_ema_crossover", ema_crossover_signal, _ema_crossover_grid()),
        StrategySpec("S2_ema_direction", ema_direction_signal, _ema_direction_grid()),
        StrategySpec("S3_hp_direction", hp_direction_signal, _hp_grid()),
        StrategySpec("S4_lowess_direction", lowess_direction_signal, _lowess_grid()),
    ]


def run_one(
    spec_name: str,
    signal_fn: Callable[..., pd.Series],
    params: dict,
    prices: pd.Series,
    ohlcv: pd.DataFrame,
    initial_capital: float,
    cash_per_trade: float,
) -> dict:
    trend = signal_fn(prices, **params)
    result = run_backtest(ohlcv, trend, initial_cash=initial_capital, cash_per_trade=cash_per_trade)
    m = compute_metrics(result.equity_curve, result.trades, initial_capital, cash_per_trade)
    return {"strategy": spec_name, **params, **m.to_dict()}


def sweep(
    prices: pd.Series,
    ohlcv: pd.DataFrame,
    specs: Iterable[StrategySpec] | None = None,
    initial_capital: float = INITIAL_CAPITAL,
    cash_per_trade: float = CASH_PER_TRADE,
    min_trades: int = 30,
    verbose: bool = True,
) -> pd.DataFrame:
    specs = list(specs) if specs is not None else default_specs()
    rows: list[dict] = []
    for spec in specs:
        for params in spec.grid:
            row = run_one(
                spec.name, spec.signal_fn, params, prices, ohlcv,
                initial_capital, cash_per_trade,
            )
            row["enough_trades"] = row["n_trades"] >= min_trades
            rows.append(row)
            if verbose:
                pstr = ", ".join(f"{k}={v}" for k, v in params.items())
                print(f"  {spec.name:24s} {pstr:30s} -> Sharpe {row['sharpe']:6.2f}  trades {row['n_trades']:4d}")
    return pd.DataFrame(rows)


def best_per_strategy(df: pd.DataFrame, min_trades: int = 30) -> pd.DataFrame:
    """Return the highest-Sharpe row per strategy, restricted to rows with
    enough trades. If no row in a strategy meets the floor, that strategy is
    dropped from the result."""
    eligible = df[df["n_trades"] >= min_trades]
    if eligible.empty:
        return eligible
    return (
        eligible.sort_values("sharpe", ascending=False)
        .groupby("strategy", as_index=False)
        .head(1)
        .sort_values("sharpe", ascending=False)
        .reset_index(drop=True)
    )
