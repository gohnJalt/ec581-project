"""FixedCashSizer — verbatim from NB_Projects.pdf, page 3.

Course-mandated: allocates a fixed cash amount per trade regardless of price.
"""

from __future__ import annotations

import backtrader as bt


class FixedCashSizer(bt.Sizer):
    """Allocate a fixed cash amount per trade."""

    params = (("cash_per_trade", 100_000),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        close_price = data.close[0]
        if close_price <= 0:
            return 0
        size = int(self.params.cash_per_trade / close_price)
        return size
