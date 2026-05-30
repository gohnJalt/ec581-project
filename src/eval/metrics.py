"""Performance metrics computed from a RunResult.

Every metric is a pure function of (equity_curve, trades). We don't pull from
Backtrader analyzers beyond what the runner already records, so the same
machinery applies to non-Backtrader equity curves (e.g., the Monte-Carlo
random-strategy benchmark).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass
class Metrics:
    n_bars: int
    n_trades: int
    net_pnl: float
    net_pnl_pct: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar: float
    win_rate: float
    avg_win: float
    avg_loss: float
    avg_holding_period: float
    turnover: float

    def to_dict(self) -> dict:
        return asdict(self)


def _drawdown(equity: pd.Series) -> tuple[float, int]:
    if equity.empty:
        return 0.0, 0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min())

    in_dd = equity < peak
    longest = cur = 0
    for flag in in_dd:
        cur = cur + 1 if flag else 0
        longest = max(longest, cur)
    return max_dd, longest


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    initial_capital: float,
    cash_per_trade: float,
) -> Metrics:
    if equity.empty:
        return Metrics(
            n_bars=0, n_trades=0, net_pnl=0.0, net_pnl_pct=0.0,
            cagr=0.0, ann_vol=0.0, sharpe=0.0, sortino=0.0,
            max_drawdown=0.0, max_drawdown_duration=0, calmar=0.0,
            win_rate=0.0, avg_win=0.0, avg_loss=0.0,
            avg_holding_period=0.0, turnover=0.0,
        )

    rets = equity.pct_change().dropna()
    n_bars = len(equity)
    years = n_bars / TRADING_DAYS

    final = float(equity.iloc[-1])
    net_pnl = final - initial_capital
    net_pnl_pct = net_pnl / initial_capital

    cagr = (final / initial_capital) ** (1 / years) - 1 if years > 0 and final > 0 else 0.0
    ann_vol = float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(rets) > 1 else 0.0
    sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(rets) > 1 and rets.std(ddof=1) > 0 else 0.0

    downside = rets[rets < 0]
    dn_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = float(rets.mean() / dn_std * np.sqrt(TRADING_DAYS)) if dn_std > 0 else 0.0

    max_dd, max_dd_dur = _drawdown(equity)
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    if not trades.empty:
        wins = trades[trades["pnl"] > 0]["pnl"]
        losses = trades[trades["pnl"] <= 0]["pnl"]
        win_rate = float(len(wins) / len(trades))
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        avg_hold = float(trades["bars_held"].mean())
        turnover = float(len(trades) * cash_per_trade / initial_capital / years) if years > 0 else 0.0
    else:
        win_rate = avg_win = avg_loss = avg_hold = turnover = 0.0

    return Metrics(
        n_bars=n_bars,
        n_trades=int(len(trades)),
        net_pnl=net_pnl,
        net_pnl_pct=net_pnl_pct,
        cagr=float(cagr),
        ann_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration=int(max_dd_dur),
        calmar=float(calmar),
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_holding_period=avg_hold,
        turnover=turnover,
    )


def format_metrics(m: Metrics) -> str:
    lines = [
        f"  bars                {m.n_bars:>10d}",
        f"  trades              {m.n_trades:>10d}",
        f"  net P/L             {m.net_pnl:>14,.0f} TL  ({m.net_pnl_pct:>6.1%})",
        f"  CAGR                {m.cagr:>10.2%}",
        f"  annualized vol      {m.ann_vol:>10.2%}",
        f"  Sharpe              {m.sharpe:>10.2f}",
        f"  Sortino             {m.sortino:>10.2f}",
        f"  max drawdown        {m.max_drawdown:>10.2%}  (dur {m.max_drawdown_duration} bars)",
        f"  Calmar              {m.calmar:>10.2f}",
        f"  win rate            {m.win_rate:>10.2%}",
        f"  avg win / loss      {m.avg_win:>10,.0f} / {m.avg_loss:,.0f}",
        f"  avg holding bars    {m.avg_holding_period:>10.1f}",
        f"  turnover (per yr)   {m.turnover:>10.2f}",
    ]
    return "\n".join(lines)
