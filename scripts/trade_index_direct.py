"""Trial: trade the BIST100 index *directly* with HP / LOWESS, vs buy-and-hold.

Isolated, intentionally basic. Instead of the cross-sectional equal-weight
portfolio over the BIST100 constituents (src/eval/run_phase2_portfolio.py), this
applies the S3 HP and S4 LOWESS direction signals to the index series itself
and benchmarks against BIST100 buy-and-hold.

Position is the held {-1,0,+1} state from src.eval.portfolio.effective_position
(same state machine as TrendStrategy), so a signal at bar t is realized on the
t->t+1 return — no look-ahead. We report long/short and long-only-flat variants;
shorting a positively-drifting index is the obvious place this can bleed.

    python scripts/trade_index_direct.py
    python scripts/trade_index_direct.py --dataset sp500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data.clean import clean_path
from src.eval.portfolio import effective_position, strategy_returns, equity_curve
from src.eval.strategies import get as get_strategy, signal as compute_signal

TRADING_DAYS = 252


def _metrics(returns: pd.Series, position: pd.Series | None = None) -> dict:
    r = returns.dropna()
    n = len(r)
    if n < 2:
        return dict(n_bars=n, cagr=0.0, ann_vol=0.0, sharpe=0.0, sortino=0.0,
                    max_dd=0.0, calmar=0.0, total_x=1.0, n_trades=0, exposure=0.0)
    eq = equity_curve(r)
    years = n / TRADING_DAYS
    final = float(eq.iloc[-1])
    cagr = final ** (1 / years) - 1 if final > 0 else 0.0
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else 0.0
    dn = r[r < 0]
    dn_std = float(dn.std(ddof=1)) if len(dn) > 1 else 0.0
    sortino = float(r.mean() / dn_std * np.sqrt(TRADING_DAYS)) if dn_std > 0 else 0.0
    dd = float((eq / eq.cummax() - 1.0).min())
    calmar = cagr / abs(dd) if dd < 0 else 0.0
    # a "trade" = a change in held position
    n_trades = int((position.diff().fillna(0) != 0).sum()) if position is not None else 0
    exposure = float((position != 0).mean()) if position is not None else 1.0
    return dict(n_bars=n, cagr=cagr, ann_vol=vol, sharpe=sharpe, sortino=sortino,
                max_dd=dd, calmar=calmar, total_x=final, n_trades=n_trades,
                exposure=exposure)


def run(dataset: str = config.DEFAULT_DATASET) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Returns (metrics table, {variant -> equity curve starting at 1.0})."""
    ds = config.get_dataset(dataset)
    index_ticker = ds.index_ticker
    prices = pd.read_parquet(clean_path(index_ticker))["close"].rename(index_ticker)

    rows: list[dict] = []
    curves: dict[str, pd.Series] = {}

    # Benchmark: buy-and-hold the index.
    bh_ret = prices.pct_change().fillna(0.0)
    bh_label = f"{index_ticker} buy & hold"
    rows.append({"variant": bh_label, **_metrics(bh_ret)})
    curves[bh_label] = equity_curve(bh_ret)

    for key in ("hp", "lowess"):
        spec = get_strategy(key)
        sig = compute_signal(prices, spec)  # Phase-1 winning params from registry

        # Long/short: trade the raw {-1,0,+1} signal.
        pos_ls = effective_position(sig)
        ret_ls = strategy_returns(prices, pos_ls)
        ls_label = f"{spec.label} long/short"
        rows.append({"variant": ls_label, **_metrics(ret_ls, pos_ls)})
        curves[ls_label] = equity_curve(ret_ls)

        # Long-only-flat: go flat instead of short (clamp shorts to 0).
        pos_lo = pos_ls.clip(lower=0)
        ret_lo = strategy_returns(prices, pos_lo)
        lo_label = f"{spec.label} long-only"
        rows.append({"variant": lo_label, **_metrics(ret_lo, pos_lo)})
        curves[lo_label] = equity_curve(ret_lo)

    return pd.DataFrame(rows), curves


def plot(curves: dict[str, pd.Series], index_ticker: str,
         save_path: Path | None = None) -> None:
    """Equity overlay (log) on top, drawdown of each variant below."""
    import matplotlib.pyplot as plt

    from src.plots.figures import apply_style, equity_curve_overlay

    apply_style()
    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(11, 9), height_ratios=[2, 1], sharex=True)

    equity_curve_overlay(
        curves, log=True,
        title=f"{index_ticker} traded directly — HP / LOWESS vs buy & hold",
        ax=ax_eq)

    for label, eq in curves.items():
        dd = eq / eq.cummax() - 1.0
        ax_dd.plot(dd.index, dd.values, label=label, linewidth=1.0)
    import matplotlib.ticker as mticker
    ax_dd.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax_dd.set_ylabel("drawdown")
    ax_dd.set_xlabel("date")
    ax_dd.set_title("drawdown by variant")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"saved figure -> {save_path}")
    else:
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=config.DEFAULT_DATASET,
                    choices=list(config.DATASETS))
    ap.add_argument("--plot", action="store_true",
                    help="draw equity overlay + drawdown panel")
    ap.add_argument("--save", type=Path, default=None,
                    help="save the figure to this path instead of showing it "
                         "(implies --plot)")
    args = ap.parse_args()

    df, curves = run(args.dataset)

    show = df.copy()
    show["cagr"] = (show["cagr"] * 100).map("{:.2f}%".format)
    show["ann_vol"] = (show["ann_vol"] * 100).map("{:.2f}%".format)
    show["max_dd"] = (show["max_dd"] * 100).map("{:.2f}%".format)
    show["exposure"] = (show["exposure"] * 100).map("{:.1f}%".format)
    show["total_x"] = show["total_x"].map("{:.2f}x".format)
    for c in ("sharpe", "sortino", "calmar"):
        show[c] = show[c].map("{:.3f}".format)
    cols = ["variant", "sharpe", "cagr", "ann_vol", "max_dd", "calmar",
            "total_x", "sortino", "n_trades", "exposure", "n_bars"]
    print(show[cols].to_string(index=False))

    if args.plot or args.save is not None:
        plot(curves, config.get_dataset(args.dataset).index_ticker,
             save_path=args.save)


if __name__ == "__main__":
    main()
