"""Phase-2 strategy registry.

Maps a strategy key ("hp", "lowess") to its signal function and Phase-1
winning hyperparameters. The Phase-2 drivers (run_phase2, run_phase2_mc,
run_phase2_portfolio) all accept ``--strategy`` and look up this table so a
single CLI can swap S3 HP for the S4 LOWESS benchmark without duplicating
infrastructure.

Phase-1 winners (per CLAUDE.md sweep results):
  - S3 HP-direction:     lam=14400,  window=504
  - S4 LOWESS-direction: frac=0.2,   window=252
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.features.hp import hp_direction_signal, rolling_hp_trend
from src.features.lowess import lowess_direction_signal, rolling_lowess_trend


@dataclass(frozen=True)
class StrategySpec:
    key: str
    label: str
    signal_fn: Callable[..., pd.Series]
    trend_fn: Callable[..., pd.Series]   # causal rolling trend (for conviction sizing)
    param_name: str        # name of the smoother parameter ("lam" or "frac")
    default_param: float   # Phase-1 winner for that parameter
    default_window: int    # Phase-1 winner window
    output_stub: str       # file-naming stub: phase2_{stub}_base.parquet, ...


_REGISTRY: dict[str, StrategySpec] = {
    "hp": StrategySpec(
        key="hp",
        label="S3 HP-direction",
        signal_fn=hp_direction_signal,
        trend_fn=rolling_hp_trend,
        param_name="lam",
        default_param=14400,
        default_window=504,
        output_stub="hp",
    ),
    "lowess": StrategySpec(
        key="lowess",
        label="S4 LOWESS-direction",
        signal_fn=lowess_direction_signal,
        trend_fn=rolling_lowess_trend,
        param_name="frac",
        default_param=0.2,
        default_window=252,
        output_stub="lowess",
    ),
}


def get(key: str) -> StrategySpec:
    if key not in _REGISTRY:
        raise KeyError(f"unknown strategy {key!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def signal(prices: pd.Series, spec: StrategySpec,
           param: float | None = None, window: int | None = None) -> pd.Series:
    """Compute the chosen strategy's signal at given (param, window).

    Falls back to the Phase-1 winning values when either is None.
    """
    p = spec.default_param if param is None else param
    w = spec.default_window if window is None else window
    return spec.signal_fn(prices, **{spec.param_name: p}, window=w)


def trend(prices: pd.Series, spec: StrategySpec,
          param: float | None = None, window: int | None = None) -> pd.Series:
    """Causal rolling trend level (its sign-of-diff is the strategy signal).

    The conviction sub-model reads the *magnitude* of this trend's slope, so it
    needs the level, not just the sign produced by ``signal``.
    """
    p = spec.default_param if param is None else param
    w = spec.default_window if window is None else window
    return spec.trend_fn(prices, **{spec.param_name: p}, window=w)
