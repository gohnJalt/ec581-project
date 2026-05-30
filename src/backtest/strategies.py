"""Trend-following strategies S1..S4 plus the regime-filtered variant.

All four index-level strategies share one driver: read `self.data.trend[0]`
(precomputed +/-1 sign), trade on sign-flips, always-in-market. The trend line
is generated outside cerebro so the only difference between S1..S4 is which
indicator was used to fill it.

The regime variant adds a second precomputed line `regime` (also +/-1) carrying
the BIST100 vs. HP-filtered-BIST100 flag; long entries require regime>0 and
short entries require regime<0. Exits ignore the regime, per the brief.

On a signal flip, the strategy deploys total broker equity by firing
``floor(equity / cash_per_trade)`` market orders in one bar — each order is
sized to ``cash_per_trade`` by ``FixedCashSizer`` (course-mandated per-trade
cap), and the sum reaches the full equity. Holding multiple concurrent lots on
the same instrument so capital is fully deployed (not 10% deployed / 90% in
cash).
"""

from __future__ import annotations

import backtrader as bt


def _n_lots(strategy: bt.Strategy) -> int:
    """How many cash_per_trade-sized orders fit in current equity."""
    cash_per_trade = strategy.sizer.params.cash_per_trade
    if cash_per_trade <= 0:
        return 0
    return max(int(strategy.broker.getvalue() // cash_per_trade), 0)


class TrendStrategy(bt.Strategy):
    """Always-in-market trend follower driven by a precomputed `trend` line."""

    params = (("name", "trend"),)

    def __init__(self):
        self.trend = self.data.trend
        self._prev = 0.0

    def _deploy(self, direction: int) -> None:
        n = _n_lots(self)
        for _ in range(n):
            if direction > 0:
                self.buy()
            else:
                self.sell()

    def next(self):
        sig = self.trend[0]
        if sig == 0.0:
            self._prev = sig
            return

        pos = self.position.size
        if sig > 0 and self._prev <= 0:
            if pos < 0:
                self.close()
            self._deploy(+1)
        elif sig < 0 and self._prev >= 0:
            if pos > 0:
                self.close()
            self._deploy(-1)

        self._prev = sig


class TrendRegimeStrategy(bt.Strategy):
    """Section-4.3 variant: gate entries by the BIST100 regime flag."""

    params = (("name", "trend_regime"),)

    def __init__(self):
        self.trend = self.data.trend
        self.regime = self.data.regime
        self._prev = 0.0

    def _deploy(self, direction: int) -> None:
        n = _n_lots(self)
        for _ in range(n):
            if direction > 0:
                self.buy()
            else:
                self.sell()

    def next(self):
        sig = self.trend[0]
        reg = self.regime[0]
        pos = self.position.size

        if sig == 0.0:
            self._prev = sig
            return

        if sig > 0 and self._prev <= 0:
            # Always close any opposite position; only open new long if regime allows.
            if pos < 0:
                self.close()
            if reg > 0:
                self._deploy(+1)
        elif sig < 0 and self._prev >= 0:
            if pos > 0:
                self.close()
            if reg < 0:
                self._deploy(-1)

        self._prev = sig
