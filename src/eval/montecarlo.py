"""Monte Carlo Sharpe p-value test (course brief Section 1.3).

For a candidate strategy with signal series ``s`` in {-1, 0, +1}, generate
``n_iter`` random alternating sign series that match the empirical:

- run-length distribution (i.e. average holding period and trade frequency)
- long/short proportion

Run each through a fast vectorized simulator that produces the same per-bar
P&L as the Backtrader engine up to fixed-shares-vs-fixed-cash scaling (which
doesn't affect Sharpe), and compute the one-sided p-value
``P(MC_Sharpe >= strat_Sharpe)``.

The fast simulator is used for both the strategy and the MC pool so the
comparison is apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import RANDOM_SEED


TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Fast simulator
# ---------------------------------------------------------------------------
def strategy_returns(prices: pd.Series, signal: pd.Series) -> pd.Series:
    """Per-bar P&L of a long/short trader who holds yesterday's signal sign."""
    pos = signal.shift(1).fillna(0.0)
    rets = prices.pct_change().fillna(0.0)
    return (pos * rets).rename("ret")


def sharpe(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Annualized Sharpe; returns 0 if the series is degenerate."""
    r = returns[returns.notna()]
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods))


# ---------------------------------------------------------------------------
# Run-length bootstrap
# ---------------------------------------------------------------------------
def _run_lengths(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (long_run_lengths, short_run_lengths, first_active_idx)."""
    nonzero_mask = signal != 0
    if not nonzero_mask.any():
        return np.array([], int), np.array([], int), len(signal)
    first = int(np.argmax(nonzero_mask))
    s = signal[first:]

    longs: list[int] = []
    shorts: list[int] = []
    cur_sign = s[0]
    cur_len = 1
    for v in s[1:]:
        if v == cur_sign:
            cur_len += 1
        else:
            (longs if cur_sign > 0 else shorts).append(cur_len)
            cur_sign = v
            cur_len = 1
    (longs if cur_sign > 0 else shorts).append(cur_len)
    return np.asarray(longs, int), np.asarray(shorts, int), first


def _random_signal(
    long_runs: np.ndarray,
    short_runs: np.ndarray,
    long_prob: float,
    length: int,
    first_active: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a random {-1,+1} sequence of `length-first_active` bars by drawing
    alternating runs from the empirical long/short run-length pools, then pad
    with zeros for the warmup region."""
    out = np.zeros(length, dtype=float)
    if long_runs.size == 0 and short_runs.size == 0:
        return out

    cur = 1.0 if rng.random() < long_prob else -1.0
    pos = first_active
    while pos < length:
        pool = long_runs if cur > 0 else short_runs
        if pool.size == 0:
            pool = short_runs if cur > 0 else long_runs
        run = int(rng.choice(pool))
        end = min(pos + run, length)
        out[pos:end] = cur
        pos = end
        cur = -cur
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class MCResult:
    strat_sharpe: float
    mc_sharpes: np.ndarray
    p_value: float
    n_iter: int
    long_prob: float
    n_long_runs: int
    n_short_runs: int

    def to_dict(self) -> dict:
        return {
            "strat_sharpe": self.strat_sharpe,
            "p_value": self.p_value,
            "n_iter": self.n_iter,
            "long_prob": self.long_prob,
            "n_long_runs": self.n_long_runs,
            "n_short_runs": self.n_short_runs,
            "mc_sharpe_mean": float(self.mc_sharpes.mean()),
            "mc_sharpe_std": float(self.mc_sharpes.std(ddof=1)),
            "mc_sharpe_q05": float(np.quantile(self.mc_sharpes, 0.05)),
            "mc_sharpe_q95": float(np.quantile(self.mc_sharpes, 0.95)),
        }


def monte_carlo_sharpe(
    prices: pd.Series,
    signal: pd.Series,
    n_iter: int = 1000,
    seed: int = RANDOM_SEED,
) -> MCResult:
    """One-sided p-value of the strategy's Sharpe against a matched-randomness pool."""
    sig_arr = signal.to_numpy(dtype=float)
    long_runs, short_runs, first = _run_lengths(sig_arr)
    active = sig_arr[first:]
    long_prob = float((active > 0).sum() / max(1, (active != 0).sum()))

    strat_sharpe = sharpe(strategy_returns(prices, signal))

    rng = np.random.default_rng(seed)
    mc_sharpes = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        rand_sig_arr = _random_signal(long_runs, short_runs, long_prob, len(sig_arr), first, rng)
        rand_sig = pd.Series(rand_sig_arr, index=signal.index)
        mc_sharpes[i] = sharpe(strategy_returns(prices, rand_sig))

    p_value = float((mc_sharpes >= strat_sharpe).mean())
    return MCResult(
        strat_sharpe=strat_sharpe,
        mc_sharpes=mc_sharpes,
        p_value=p_value,
        n_iter=n_iter,
        long_prob=long_prob,
        n_long_runs=int(long_runs.size),
        n_short_runs=int(short_runs.size),
    )
