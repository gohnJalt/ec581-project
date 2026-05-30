"""LOWESS trend extractor. Bare function from the brief plus a causal wrapper.

Like HP, plain LOWESS uses the whole series and is non-causal at any historical
bar. ``rolling_lowess_trend`` re-fits on a trailing window and returns only the
right-edge trend value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess


def lowess_trend(prices: pd.Series, frac: float = 0.1) -> pd.Series:
    """LOWESS smoothed trend (verbatim from NB_Projects.pdf, page 14)."""
    x_num = np.arange(len(prices))
    smoothed = lowess(prices.values, x_num, frac=frac, return_sorted=False)
    return pd.Series(smoothed, index=prices.index, name="lowess_trend")


def rolling_lowess_trend(
    prices: pd.Series, frac: float = 0.1, window: int = 504
) -> pd.Series:
    """Causal LOWESS trend: refit on prices[t-window+1 .. t] each bar."""
    n = len(prices)
    out = np.full(n, np.nan)
    if window > n:
        return pd.Series(out, index=prices.index, name="lowess_trend")

    arr = prices.to_numpy(dtype=float)
    x_num = np.arange(window, dtype=float)

    for t in range(window - 1, n):
        y = arr[t - window + 1 : t + 1]
        smoothed = lowess(y, x_num, frac=frac, return_sorted=False)
        out[t] = smoothed[-1]

    return pd.Series(out, index=prices.index, name="lowess_trend")


def lowess_direction_signal(
    prices: pd.Series, frac: float = 0.1, window: int = 504
) -> pd.Series:
    """S4: sign of the change in the rolling LOWESS trend."""
    trend = rolling_lowess_trend(prices, frac=frac, window=window)
    sig = np.sign(trend.diff()).astype(float)
    sig.iloc[:window] = 0.0
    return sig.rename("trend")
