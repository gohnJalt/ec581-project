"""Cerebro orchestration: build a configured Cerebro for one (strategy, asset).

Returns the strategy instance after running so callers can pull analyzers /
the broker value curve out for downstream metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import backtrader as bt
import pandas as pd

from config import CASH_PER_TRADE, COMMISSION, INITIAL_CAPITAL
from src.backtest.feeds import make_feed
from src.backtest.sizers import FixedCashSizer
from src.backtest.strategies import TrendRegimeStrategy, TrendStrategy


@dataclass
class RunResult:
    final_value: float
    initial_value: float
    equity_curve: pd.Series   # broker value per bar
    trades: pd.DataFrame      # one row per closed trade
    strat: bt.Strategy        # for further analyzer extraction


class _EquityRecorder(bt.Analyzer):
    """Records broker.value at the close of every bar."""

    def start(self):
        self.values: list[tuple[pd.Timestamp, float]] = []

    def next(self):
        dt = bt.num2date(self.data.datetime[0])
        self.values.append((pd.Timestamp(dt).normalize(), float(self.strategy.broker.getvalue())))

    def get_analysis(self):
        if not self.values:
            return pd.Series(dtype=float, name="equity")
        idx, vals = zip(*self.values)
        return pd.Series(vals, index=pd.DatetimeIndex(idx), name="equity")


class _TradeRecorder(bt.Analyzer):
    """Records closed trades with entry/exit dates and pnl."""

    def start(self):
        self.rows: list[dict] = []

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        # trade.dtopen / dtclose are matplotlib floats
        entry_dt = bt.num2date(trade.dtopen)
        exit_dt = bt.num2date(trade.dtclose)
        self.rows.append({
            "entry": pd.Timestamp(entry_dt).normalize(),
            "exit": pd.Timestamp(exit_dt).normalize(),
            "pnl": float(trade.pnl),
            "pnl_net": float(trade.pnlcomm),
            "size": int(trade.size) if trade.size else 0,
            "bars_held": int(trade.barlen),
        })

    def get_analysis(self):
        return pd.DataFrame(self.rows)


def run_backtest(
    ohlcv: pd.DataFrame,
    trend: pd.Series,
    regime: pd.Series | None = None,
    initial_cash: float = INITIAL_CAPITAL,
    cash_per_trade: float = CASH_PER_TRADE,
    commission: float = COMMISSION,
) -> RunResult:
    """Run one backtest. If `regime` is provided, the regime-filtered strategy
    is used; otherwise the plain trend strategy."""
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addsizer(FixedCashSizer, cash_per_trade=cash_per_trade)

    cerebro.adddata(make_feed(ohlcv, trend, regime))

    if regime is None:
        cerebro.addstrategy(TrendStrategy)
    else:
        cerebro.addstrategy(TrendRegimeStrategy)

    cerebro.addanalyzer(_EquityRecorder, _name="equity")
    cerebro.addanalyzer(_TradeRecorder, _name="trades")

    initial_value = cerebro.broker.getvalue()
    strats = cerebro.run()
    strat = strats[0]
    final_value = cerebro.broker.getvalue()

    equity = strat.analyzers.equity.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    return RunResult(
        final_value=final_value,
        initial_value=initial_value,
        equity_curve=equity,
        trades=trades,
        strat=strat,
    )
