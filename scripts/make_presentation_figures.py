"""Regenerate the curated (slide-friendly) presentation figures.

The notebook-extracted figures pack all 84 constituents onto one page, which is
unreadable on a projector. These replacements distil the same stories into one
clean panel each. Output overwrites presentation/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PANEL = ROOT / "data" / "panel"
FIG = ROOT / "presentation" / "figures"

ACCENT = "#1f5fa8"
WARM = "#d4762a"
RED = "#c0392b"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    })


def data_overview() -> None:
    """Single clean BIST100 index log-price line: the Phase-1 training series."""
    idx = pd.read_parquet(PANEL / "index.parquet")["close"].dropna()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(idx.index, idx.values, color=ACCENT, linewidth=1.6)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("date")
    ax.set_ylabel("BIST100 index (log scale)")
    ax.set_title("BIST100 index, 2015 to 2026")
    fig.savefig(FIG / "data_overview.png")
    plt.close(fig)


def regime_scatter() -> None:
    """Base vs regime per-stock Sharpe over the 84 names, with the y=x line."""
    base = pd.read_parquet(RESULTS / "phase2_hp_base.parquet").set_index("ticker")["sharpe"]
    reg = pd.read_parquet(RESULTS / "phase2_hp_regime.parquet").set_index("ticker")["sharpe"]
    df = pd.concat([base.rename("base"), reg.rename("regime")], axis=1).dropna()

    lo = float(min(df.min().min(), 0) - 0.1)
    hi = float(df.max().max() + 0.1)
    helped = df["regime"] >= df["base"]

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot([lo, hi], [lo, hi], color="0.5", linewidth=1.0, zorder=1)
    ax.scatter(df.loc[~helped, "base"], df.loc[~helped, "regime"],
               s=34, color=WARM, alpha=0.8, label="regime lowers Sharpe", zorder=2)
    ax.scatter(df.loc[helped, "base"], df.loc[helped, "regime"],
               s=34, color=ACCENT, alpha=0.8, label="regime raises Sharpe", zorder=2)
    ax.axhline(0, color="0.8", linewidth=0.8)
    ax.axvline(0, color="0.8", linewidth=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("base Sharpe")
    ax.set_ylabel("regime Sharpe")
    ax.set_title("Regime filter, per stock")
    ax.legend(loc="upper left", fontsize=10)
    n_helped = int(helped.sum())
    ax.text(0.97, 0.04, f"helps {n_helped} of {len(df)} names",
            transform=ax.transAxes, ha="right", fontsize=10, color="0.4")
    fig.savefig(FIG / "regime_scatter.png")
    plt.close(fig)


def mc_hist() -> None:
    """Per-stock strategy Sharpe vs the matched-random pool mean."""
    mc = pd.read_parquet(RESULTS / "phase2_mc_base.parquet")
    strat = mc["strat_sharpe"].dropna()
    pool_mean = float(mc["mc_sharpe_mean"].mean())
    n_sig01 = int((mc["p_value"] < 0.01).sum())
    n_sig05 = int((mc["p_value"] < 0.05).sum())
    n = len(mc)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.hist(strat, bins=22, color=ACCENT, alpha=0.85, edgecolor="white")
    ax.axvline(pool_mean, color=RED, linewidth=1.8,
               label=f"random-pool mean {pool_mean:.2f}")
    ax.axvline(float(strat.mean()), color="0.25", linewidth=1.8, linestyle="--",
               label=f"strategy mean {strat.mean():.2f}")
    ax.set_xlabel("per-stock Sharpe")
    ax.set_ylabel("number of stocks")
    ax.set_title("Per-stock HP signal vs matched randomness")
    ax.legend(loc="upper right", fontsize=10)
    ax.text(0.02, 0.95,
            f"significant: {n_sig01} of {n} at p<0.01\n"
            f"{n_sig05} of {n} at p<0.05",
            transform=ax.transAxes, va="top", fontsize=10, color="0.3")
    fig.savefig(FIG / "phase2_mc_hist.png")
    plt.close(fig)


def wf_sharpe_dist() -> None:
    """Out-of-sample test Sharpe distribution by fold, base vs regime."""
    base = pd.read_parquet(RESULTS / "phase2_walkforward_base.parquet")
    reg = pd.read_parquet(RESULTS / "phase2_walkforward_regime.parquet")
    folds = sorted(base["fold_id"].dropna().unique())

    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.34
    for off, df, color, lab in [(-width / 2, base, ACCENT, "base"),
                                 (width / 2, reg, WARM, "regime")]:
        data = [df.loc[df["fold_id"] == f, "test_sharpe"].dropna().values for f in folds]
        pos = [i + off for i in range(len(folds))]
        bp = ax.boxplot(data, positions=pos, widths=width, patch_artist=True,
                        showfliers=False, medianprops=dict(color="0.15"))
        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=0.7, edgecolor=color)
        for w in bp["whiskers"] + bp["caps"]:
            w.set(color=color)
        ax.plot([], [], color=color, linewidth=6, alpha=0.7, label=lab)
    ax.axhline(0, color="0.5", linewidth=1.0)
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels([f"year {i + 1}" for i in range(len(folds))])
    ax.set_xlabel("out-of-sample fold")
    ax.set_ylabel("test Sharpe")
    ax.set_title("Walk-forward out-of-sample Sharpe by fold")
    ax.legend(loc="upper right", fontsize=10)
    fig.savefig(FIG / "wf_sharpe_dist.png")
    plt.close(fig)


def main() -> None:
    _style()
    data_overview()
    regime_scatter()
    mc_hist()
    wf_sharpe_dist()
    print("wrote: data_overview, regime_scatter, phase2_mc_hist, wf_sharpe_dist")


if __name__ == "__main__":
    main()
