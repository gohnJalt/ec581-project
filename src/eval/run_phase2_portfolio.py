"""Phase-2 equal-weight portfolio aggregator driver (DESIGN.md M4 deliverable).

For each cleaned constituent, compute the chosen Phase-2 strategy's signal at
the Phase-1 winning parameters (``--strategy hp`` for S3 HP-direction,
``--strategy lowess`` for S4 LOWESS-direction), reconstruct the effective held
position (base or regime), and produce per-ticker daily strategy returns.

Aggregate cross-sectionally: portfolio return on bar t = mean of available
ticker strategy returns. Then compute portfolio-level Sharpe / CAGR / MDD /
Sortino in both base and regime variants.

Outputs under ``results/`` (stub depends on strategy):
  - ``phase2_portfolio_{stub}_equity.parquet``     wide: index date, columns base/regime
  - ``phase2_portfolio_{stub}_panel_base.parquet`` per-ticker strategy returns (base)
  - ``phase2_portfolio_{stub}_panel_regime.parquet`` per-ticker strategy returns (regime)
  - ``phase2_portfolio_{stub}_summary.parquet``    one row per variant with metrics

For the HP strategy, the legacy unsuffixed names (``phase2_portfolio_equity.parquet``
etc.) are kept for backward compatibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BIST100_CONSTITUENTS,
    BIST100_INDEX_TICKER,
    RESULTS_DIR,
    SMOKE_TICKERS,
)
from src.data.clean import clean_path
from src.eval.portfolio import (
    effective_position,
    equity_curve,
    portfolio_returns,
    strategy_returns,
)
from src.eval.strategies import StrategySpec, get as get_strategy, signal as compute_signal
from src.features.regime import bist_regime_flag


REGIME_LAM = 1600
REGIME_WINDOW = 504
TRADING_DAYS = 252


def _available_tickers(universe: list[str]) -> list[str]:
    return [t for t in universe if clean_path(t).exists()]


def _load_close(ticker: str) -> pd.Series:
    return pd.read_parquet(clean_path(ticker))["close"].rename(ticker)


def _returns_metrics(returns: pd.Series) -> dict:
    r = returns.dropna()
    n_bars = len(r)
    if n_bars < 2:
        return {"n_bars": n_bars, "cagr": 0.0, "ann_vol": 0.0, "sharpe": 0.0,
                "sortino": 0.0, "max_drawdown": 0.0, "calmar": 0.0,
                "net_pnl_pct": 0.0}
    eq = equity_curve(r)
    years = n_bars / TRADING_DAYS
    final = float(eq.iloc[-1])
    cagr = final ** (1 / years) - 1 if years > 0 and final > 0 else 0.0
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else 0.0
    downside = r[r < 0]
    dn_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = float(r.mean() / dn_std * np.sqrt(TRADING_DAYS)) if dn_std > 0 else 0.0
    peak = eq.cummax()
    dd = (eq / peak - 1.0)
    max_dd = float(dd.min())
    calmar = float(cagr) / abs(max_dd) if max_dd < 0 else 0.0
    return {"n_bars": int(n_bars), "cagr": float(cagr), "ann_vol": ann_vol,
            "sharpe": sharpe, "sortino": sortino, "max_drawdown": max_dd,
            "calmar": calmar, "net_pnl_pct": final - 1.0}


def _build_panel(
    tickers: list[str], spec: StrategySpec, param: float, window: int,
    regime: pd.Series | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (positions_panel, strategy_returns_panel) for one variant."""
    pos_cols: dict[str, pd.Series] = {}
    ret_cols: dict[str, pd.Series] = {}
    for t in tickers:
        prices = _load_close(t)
        sig = compute_signal(prices, spec, param=param, window=window)
        reg_aligned = regime.reindex(prices.index).fillna(0.0) if regime is not None else None
        pos = effective_position(sig, reg_aligned)
        rets = strategy_returns(prices, pos)
        pos_cols[t] = pos
        ret_cols[t] = rets
    pos_df = pd.concat(pos_cols, axis=1)
    ret_df = pd.concat(ret_cols, axis=1)
    return pos_df, ret_df


def _output_paths(spec: StrategySpec, out_dir: Path) -> dict[str, Path]:
    # Preserve legacy unsuffixed names for the HP strategy.
    stub = "" if spec.key == "hp" else f"_{spec.output_stub}"
    return {
        "equity":       out_dir / f"phase2_portfolio{stub}_equity.parquet",
        "panel_base":   out_dir / f"phase2_portfolio{stub}_panel_base.parquet",
        "panel_regime": out_dir / f"phase2_portfolio{stub}_panel_regime.parquet",
        "summary":      out_dir / f"phase2_portfolio{stub}_summary.parquet",
    }


def run(
    tickers: list[str],
    spec: StrategySpec,
    param: float | None = None,
    window: int | None = None,
    regime_lam: float = REGIME_LAM,
    regime_window: int = REGIME_WINDOW,
    out_dir: Path = RESULTS_DIR,
    verbose: bool = True,
) -> dict[str, Path]:
    if not tickers:
        raise RuntimeError("no tickers available — populate data/clean/ first")

    p = spec.default_param if param is None else param
    w = spec.default_window if window is None else window

    bist_close = pd.read_parquet(clean_path(BIST100_INDEX_TICKER))["close"]
    regime = bist_regime_flag(bist_close, lam=regime_lam, window=regime_window)

    if verbose:
        print(f"{spec.label} {spec.param_name}={p} window={w}")
        print(f"Regime: lam={regime_lam} window={regime_window}\n")

    if verbose:
        print(f"building base panel for {len(tickers)} tickers...")
    _, rets_base = _build_panel(tickers, spec, p, w, regime=None)
    port_base = portfolio_returns(rets_base)
    eq_base = equity_curve(port_base)
    m_base = _returns_metrics(port_base)
    m_base["variant"] = "base"
    m_base["n_tickers"] = len(tickers)
    m_base["strategy"] = spec.key

    if verbose:
        print(f"building regime panel for {len(tickers)} tickers...")
    _, rets_reg = _build_panel(tickers, spec, p, w, regime=regime)
    port_reg = portfolio_returns(rets_reg)
    eq_reg = equity_curve(port_reg)
    m_reg = _returns_metrics(port_reg)
    m_reg["variant"] = "regime"
    m_reg["n_tickers"] = len(tickers)
    m_reg["strategy"] = spec.key

    m_base["mean_active_tickers"] = float((rets_base != 0).sum(axis=1).mean())
    m_reg["mean_active_tickers"] = float((rets_reg != 0).sum(axis=1).mean())

    paths = _output_paths(spec, out_dir)
    equity_df = pd.concat({"base": eq_base, "regime": eq_reg}, axis=1)
    equity_df.to_parquet(paths["equity"])
    rets_base.to_parquet(paths["panel_base"])
    rets_reg.to_parquet(paths["panel_regime"])
    summary_df = pd.DataFrame([m_base, m_reg])
    summary_df.to_parquet(paths["summary"])

    if verbose:
        print()
        _print_summary(summary_df)

    return paths


def _print_summary(df: pd.DataFrame) -> None:
    cols = ["strategy", "variant", "n_tickers", "mean_active_tickers", "n_bars",
            "sharpe", "sortino", "cagr", "ann_vol", "max_drawdown",
            "calmar", "net_pnl_pct"]
    print(df[cols].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--strategy", choices=["hp", "lowess"], default="hp")
    ap.add_argument("--param", type=float, default=None,
                    help="lam for hp, frac for lowess; defaults to Phase-1 winner")
    ap.add_argument("--window", type=int, default=None)
    args = ap.parse_args()

    spec = get_strategy(args.strategy)
    universe = SMOKE_TICKERS if args.smoke else BIST100_CONSTITUENTS
    tickers = _available_tickers(universe)
    print(f"tickers ({len(tickers)}): {tickers}")
    run(tickers, spec, param=args.param, window=args.window)


if __name__ == "__main__":
    main()
