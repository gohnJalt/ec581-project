"""PandasData feed extended with a precomputed `trend` line.

Strategies read `self.data.trend[0]` and act on its sign-flips. Computing the
trend in pandas (outside Backtrader) keeps the strategy class trivial and lets
hyperparameter sweeps reuse cached feature parquets instead of recomputing
inside cerebro.
"""

from __future__ import annotations

import backtrader as bt
import pandas as pd


class TrendPandasData(bt.feeds.PandasData):
    """OHLCV + a `trend` line carrying the precomputed signal in {-1, 0, +1}."""

    lines = ("trend",)
    params = (("trend", -1),)  # -1 means "use column named 'trend'"


class TrendRegimePandasData(bt.feeds.PandasData):
    """OHLCV + `trend` and `regime` lines for the Section-4.3 regime variant."""

    lines = ("trend", "regime")
    params = (("trend", -1), ("regime", -1))


def make_feed(
    ohlcv: pd.DataFrame, trend: pd.Series, regime: pd.Series | None = None
) -> bt.feeds.PandasData:
    """Build a Backtrader feed from a cleaned OHLCV frame plus signal series.

    `ohlcv` must have columns [open, high, low, close, volume] and a
    DatetimeIndex. `trend` (and optional `regime`) must align to that index.
    """
    df = ohlcv.copy()
    df["trend"] = trend.reindex(df.index).fillna(0.0).astype(float)
    if regime is not None:
        df["regime"] = regime.reindex(df.index).fillna(0.0).astype(float)
        return TrendRegimePandasData(dataname=df)
    return TrendPandasData(dataname=df)
