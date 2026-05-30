"""Hodrick-Prescott filter. Bare function from the brief plus a causal wrapper.

The bare ``hp_filter`` is non-causal — it solves a single linear system over the
entire input and so the value at index t is influenced by data at t' > t. For
plotting and EDA that is fine; for backtesting it leaks future information.

``rolling_hp_trend`` re-fits HP on a fixed-length trailing window ending at
each bar and returns the right-edge trend value, which is causal by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve


def hp_filter(x, lam: float = 1600) -> np.ndarray:
    """Hodrick-Prescott filter (verbatim from NB_Projects.pdf, page 13)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    eye = np.eye(n)
    D = np.diff(eye, n=2, axis=0)
    trend = np.linalg.solve(eye + lam * D.T @ D, x)
    return trend


def rolling_hp_trend(
    prices: pd.Series, lam: float = 1600, window: int = 504
) -> pd.Series:
    """Causal HP trend: at each bar t, refit HP on prices[t-window+1 .. t]."""
    n = len(prices)
    out = np.full(n, np.nan)
    if window > n:
        return pd.Series(out, index=prices.index, name="hp_trend")

    arr = prices.to_numpy(dtype=float)
    # Pre-factor (I + lam D'D) once — same window length and lam every step.
    eye = np.eye(window)
    D = np.diff(eye, n=2, axis=0)
    A = eye + lam * D.T @ D
    cho = cho_factor(A, lower=True)

    for t in range(window - 1, n):
        x = arr[t - window + 1 : t + 1]
        trend = cho_solve(cho, x)
        out[t] = trend[-1]

    return pd.Series(out, index=prices.index, name="hp_trend")


def hp_direction_signal(
    prices: pd.Series, lam: float = 1600, window: int = 504
) -> pd.Series:
    """S3: sign of the change in the rolling HP trend."""
    trend = rolling_hp_trend(prices, lam=lam, window=window)
    sig = np.sign(trend.diff()).astype(float)
    sig.iloc[:window] = 0.0
    return sig.rename("trend")
