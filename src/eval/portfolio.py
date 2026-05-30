"""Cross-sectional equal-weight portfolio aggregator (DESIGN.md M4).

The Backtrader runner produces one equity curve per (ticker, variant). For the
M4 deliverable we want a single "strategy as a whole" view: an equal-weight
portfolio across all cleaned constituents.

The brief's fixed-cash sizing (100k per trade out of 1M) is enforced per
ticker. Aggregating 29 such per-ticker runs into a single portfolio with only
1M total capital doesn't divide cleanly — so we report the natural
cross-sectional average instead: for each bar, the portfolio return is the
mean of the per-ticker strategy returns (over tickers with an active signal
that day). This is the standard cross-sectional summary and is documented as
an approximation alongside the per-ticker tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def effective_position(
    signal: pd.Series, regime: pd.Series | None = None
) -> pd.Series:
    """Replicate TrendStrategy / TrendRegimeStrategy position over time.

    Returns the held position ∈ {-1, 0, +1} *at the close of each bar*. Multiply
    by ``prices.pct_change()`` shifted appropriately to get realized P&L.
    """
    sig = signal.fillna(0.0).to_numpy(dtype=float)
    reg = regime.fillna(0.0).to_numpy(dtype=float) if regime is not None else None
    n = len(sig)
    out = np.zeros(n, dtype=float)
    prev = 0.0
    cur = 0.0
    for t in range(n):
        s = sig[t]
        if s == 0.0:
            out[t] = cur
            prev = s
            continue
        if s > 0 and prev <= 0:
            if cur < 0:
                cur = 0.0
            if reg is None or reg[t] > 0:
                cur = 1.0
        elif s < 0 and prev >= 0:
            if cur > 0:
                cur = 0.0
            if reg is None or reg[t] < 0:
                cur = -1.0
        out[t] = cur
        prev = s
    return pd.Series(out, index=signal.index, name="position")


def strategy_returns(
    prices: pd.Series, position: pd.Series
) -> pd.Series:
    """Per-bar P&L = yesterday's held position × today's pct_change."""
    pos = position.shift(1).fillna(0.0)
    rets = prices.pct_change().fillna(0.0)
    return (pos * rets).rename("ret")


def portfolio_returns(returns_panel: pd.DataFrame) -> pd.Series:
    """Equal-weight cross-sectional portfolio return per bar.

    Mean across tickers, ignoring NaN (pre-IPO bars where the ticker doesn't
    exist yet) but including 0.0 (the ticker exists but is flat) — so days
    where most names are flat are properly dampened toward zero.
    """
    return returns_panel.mean(axis=1).fillna(0.0).rename("portfolio_ret")


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    return (initial * (1.0 + returns).cumprod()).rename("equity")
