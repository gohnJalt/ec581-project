"""Patch only MGROS into the existing MC and portfolio parquets.

These parquets use the fast simulator (effective_position + pos.shift(1) *
pct_change), which is NOT affected by the Backtrader strategy change. They
were stale only because MGROS's *price data* changed (SPLIT_RATIO_THRESHOLD
50 -> 3 caught a 4x corporate-action gap on 2009-08-04 that was previously
missed).

This script rebuilds only the MGROS contribution and splices it in, leaving
all other 28 tickers untouched.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import BIST100_INDEX_TICKER, RANDOM_SEED, RESULTS_DIR
from src.data.clean import clean_path
from src.eval.montecarlo import monte_carlo_sharpe
from src.eval.portfolio import effective_position, equity_curve, portfolio_returns, strategy_returns
from src.eval.strategies import get as get_strategy, signal as compute_signal
from src.features.regime import bist_regime_flag


TICKER = "MGROS.IS"
REGIME_LAM, REGIME_WINDOW = 1600, 504
TRADING_DAYS = 252


def _returns_metrics(returns: pd.Series) -> dict:
    r = returns.dropna()
    n = len(r)
    eq = equity_curve(r)
    final = float(eq.iloc[-1])
    years = n / TRADING_DAYS
    cagr = final ** (1 / years) - 1 if years > 0 and final > 0 else 0.0
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else 0.0
    dn = r[r < 0]
    dn_std = float(dn.std(ddof=1)) if len(dn) > 1 else 0.0
    sortino = float(r.mean() / dn_std * np.sqrt(TRADING_DAYS)) if dn_std > 0 else 0.0
    peak = eq.cummax()
    max_dd = float((eq / peak - 1.0).min())
    calmar = float(cagr) / abs(max_dd) if max_dd < 0 else 0.0
    return {"n_bars": int(n), "cagr": float(cagr), "ann_vol": ann_vol,
            "sharpe": sharpe, "sortino": sortino, "max_drawdown": max_dd,
            "calmar": calmar, "net_pnl_pct": final - 1.0}


def patch_mc(strategy_key: str) -> None:
    spec = get_strategy(strategy_key)
    prices = pd.read_parquet(clean_path(TICKER))["close"]
    sig = compute_signal(prices, spec)
    print(f"[{strategy_key} MC] recomputing MGROS with {spec.label}...", flush=True)
    t0 = time.time()
    mc = monte_carlo_sharpe(prices, sig, n_iter=1000, seed=RANDOM_SEED)
    print(f"  done in {time.time()-t0:.1f}s  Sharpe={mc.strat_sharpe:.3f}  p={mc.p_value:.3f}")

    fname = "phase2_mc_base.parquet" if strategy_key == "hp" else "phase2_mc_lowess_base.parquet"
    p = RESULTS_DIR / fname
    df = pd.read_parquet(p)
    new_row = {"ticker": TICKER, **mc.to_dict()}
    mask = df["ticker"] == TICKER
    if mask.any():
        for k, v in new_row.items():
            df.loc[mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_parquet(p)
    print(f"  patched {fname}: new MGROS row Sharpe={df.loc[df['ticker']==TICKER, 'strat_sharpe'].iloc[0]:.3f}")


def patch_portfolio(strategy_key: str) -> None:
    spec = get_strategy(strategy_key)
    bist_close = pd.read_parquet(clean_path(BIST100_INDEX_TICKER))["close"]
    regime = bist_regime_flag(bist_close, lam=REGIME_LAM, window=REGIME_WINDOW)

    prices = pd.read_parquet(clean_path(TICKER))["close"]
    sig = compute_signal(prices, spec)
    pos_base = effective_position(sig, regime=None)
    reg_aligned = regime.reindex(prices.index).fillna(0.0)
    pos_reg = effective_position(sig, regime=reg_aligned)
    rets_base = strategy_returns(prices, pos_base)
    rets_reg = strategy_returns(prices, pos_reg)

    stub = "" if strategy_key == "hp" else f"_{spec.output_stub}"
    paths = {
        "panel_base": RESULTS_DIR / f"phase2_portfolio{stub}_panel_base.parquet",
        "panel_regime": RESULTS_DIR / f"phase2_portfolio{stub}_panel_regime.parquet",
        "equity": RESULTS_DIR / f"phase2_portfolio{stub}_equity.parquet",
        "summary": RESULTS_DIR / f"phase2_portfolio{stub}_summary.parquet",
    }

    # Patch panels
    panel_base = pd.read_parquet(paths["panel_base"])
    panel_reg = pd.read_parquet(paths["panel_regime"])
    panel_base[TICKER] = rets_base.reindex(panel_base.index)
    panel_reg[TICKER] = rets_reg.reindex(panel_reg.index)
    panel_base.to_parquet(paths["panel_base"])
    panel_reg.to_parquet(paths["panel_regime"])
    print(f"[{strategy_key} portfolio] patched MGROS column in {paths['panel_base'].name} + regime variant")

    # Rebuild portfolio returns + equity + summary from the panel
    port_base = portfolio_returns(panel_base)
    port_reg = portfolio_returns(panel_reg)
    eq_base = equity_curve(port_base)
    eq_reg = equity_curve(port_reg)

    equity_df = pd.concat({"base": eq_base, "regime": eq_reg}, axis=1)
    equity_df.to_parquet(paths["equity"])

    m_base = _returns_metrics(port_base)
    m_base["variant"] = "base"
    m_base["n_tickers"] = panel_base.shape[1]
    m_base["strategy"] = strategy_key
    m_base["mean_active_tickers"] = float((panel_base != 0).sum(axis=1).mean())

    m_reg = _returns_metrics(port_reg)
    m_reg["variant"] = "regime"
    m_reg["n_tickers"] = panel_reg.shape[1]
    m_reg["strategy"] = strategy_key
    m_reg["mean_active_tickers"] = float((panel_reg != 0).sum(axis=1).mean())

    summary_df = pd.DataFrame([m_base, m_reg])
    summary_df.to_parquet(paths["summary"])
    print(f"  new portfolio Sharpe base={m_base['sharpe']:.3f} regime={m_reg['sharpe']:.3f}  "
          f"CAGR base={m_base['cagr']:.2%} regime={m_reg['cagr']:.2%}")


if __name__ == "__main__":
    for key in ("hp", "lowess"):
        patch_mc(key)
        patch_portfolio(key)
    print("\nAll MGROS patches done. MC + portfolio parquets are now consistent with the new MGROS data.")
