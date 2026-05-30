"""EMA-based trend extractors (S1 crossover and S2 direction).

Pure functions over price series. Causal by construction — `pandas.Series.ewm`
uses only past data — so no rolling-window wrapper is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(prices: pd.Series, n: int) -> pd.Series:
    return prices.ewm(span=n, adjust=False).mean()


def ema_crossover_signal(prices: pd.Series, n_fast: int, n_slow: int) -> pd.Series:
    """S1: sign of (EMA_fast - EMA_slow). +1 long, -1 short, 0 before warmup."""
    if n_fast >= n_slow:
        raise ValueError(f"need n_fast < n_slow, got {n_fast} >= {n_slow}")
    diff = ema(prices, n_fast) - ema(prices, n_slow)
    sig = np.sign(diff).astype(float)
    sig.iloc[:n_slow] = 0.0
    return sig.rename("trend")


def ema_direction_signal(prices: pd.Series, n: int) -> pd.Series:
    """S2: sign of the change in EMA_n."""
    e = ema(prices, n)
    sig = np.sign(e.diff()).astype(float)
    sig.iloc[: n + 1] = 0.0
    return sig.rename("trend")
