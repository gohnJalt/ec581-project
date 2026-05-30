"""Shared matplotlib helpers reused across notebooks/01..04 and the deck.

Each helper takes already-loaded DataFrames or Series from results/*.parquet
and either creates a Figure or draws into a caller-supplied Axes. Matplotlib
only — no seaborn — so the deck figures stay visually consistent.

All helpers return the Axes (or Figure for multi-panel layouts) so callers can
post-tweak titles, save figures, etc.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


FIGSIZE_WIDE = (10, 5)
FIGSIZE_SQUARE = (7, 6)
FIGSIZE_TALL = (8, 10)


def apply_style() -> None:
    """Call once at the top of a notebook to get consistent fonts/grid/spines."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.frameon": False,
    })


# ---------------------------------------------------------------------------
# Equity / drawdown
# ---------------------------------------------------------------------------


def equity_curve_overlay(
    curves: dict[str, pd.Series],
    log: bool = True,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlay several equity curves on one axis. ``curves`` maps label -> series."""
    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for label, eq in curves.items():
        ax.plot(eq.index, eq.values, label=label, linewidth=1.4)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("date")
    ax.set_ylabel("equity (log scale)" if log else "equity")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    return ax


def drawdown_curve(
    equity: pd.Series,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Underwater plot: percentage drop from the running peak."""
    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    peak = equity.cummax()
    dd = equity / peak - 1.0
    ax.fill_between(dd.index, dd.values, 0, color="C3", alpha=0.35)
    ax.plot(dd.index, dd.values, color="C3", linewidth=0.8)
    ax.set_xlabel("date")
    ax.set_ylabel("drawdown")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    if title:
        ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


def mc_distribution(
    strat_sharpe: float,
    mc_sharpes: np.ndarray | pd.Series,
    label: str = "",
    bins: int = 40,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Histogram of MC Sharpes with the strategy Sharpe drawn as a vertical line.

    The title carries the implied one-sided p-value computed from the empirical
    distribution so the reader sees significance without flipping to a table.
    """
    arr = np.asarray(mc_sharpes)
    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.hist(arr, bins=bins, color="0.7", edgecolor="0.4")
    ax.axvline(strat_sharpe, color="C3", linewidth=2,
               label=f"strategy = {strat_sharpe:.2f}")
    p = float(np.mean(arr >= strat_sharpe))
    ax.set_xlabel("Sharpe ratio")
    ax.set_ylabel("frequency")
    head = f"{label} — " if label else ""
    ax.set_title(f"{head}MC p-value = {p:.3f}  (N={len(arr)})")
    ax.legend(loc="best")
    return ax


# ---------------------------------------------------------------------------
# Per-ticker bars
# ---------------------------------------------------------------------------


def per_ticker_sharpe_bar(
    df: pd.DataFrame,
    sharpe_col: str = "sharpe",
    ticker_col: str = "ticker",
    sig_p_values: pd.Series | None = None,
    sig_threshold: float = 0.05,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Horizontal bar of per-ticker Sharpe, sorted ascending.

    If ``sig_p_values`` (indexed by ticker) is given, mark tickers below
    ``sig_threshold`` with a "*" so significance reads off the same chart.
    """
    d = df[[ticker_col, sharpe_col]].copy().sort_values(sharpe_col, ascending=True)
    tickers = d[ticker_col].to_list()
    sharpes = d[sharpe_col].to_numpy()
    colors = ["C0" if s >= 0 else "C3" for s in sharpes]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(tickers))))
    ax.barh(tickers, sharpes, color=colors)
    ax.axvline(0, color="0.3", linewidth=0.8)
    ax.set_xlabel("Sharpe ratio")
    if sig_p_values is not None:
        sig = pd.Series(sig_p_values).reindex(tickers)
        for i, p in enumerate(sig.to_numpy()):
            if pd.notna(p) and p < sig_threshold:
                ax.text(sharpes[i], i, "  *", va="center", fontsize=11)
    if title:
        ax.set_title(title)
    return ax


def per_ticker_sharpe_compare(
    base: pd.DataFrame,
    regime: pd.DataFrame,
    sharpe_col: str = "sharpe",
    ticker_col: str = "ticker",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Side-by-side base vs regime per-ticker Sharpe — for notebook 04."""
    b = base.set_index(ticker_col)[sharpe_col]
    r = regime.set_index(ticker_col)[sharpe_col]
    order = b.sort_values(ascending=True).index
    y = np.arange(len(order))
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(order))))
    ax.barh(y - 0.2, b.reindex(order).values, height=0.4, label="base", color="C0")
    ax.barh(y + 0.2, r.reindex(order).values, height=0.4, label="regime", color="C1")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.axvline(0, color="0.3", linewidth=0.8)
    ax.set_xlabel("Sharpe ratio")
    ax.legend(loc="best")
    return ax


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------


def walkforward_heatmap(
    df: pd.DataFrame,
    ticker_col: str = "ticker",
    fold_col: str = "fold_id",
    value_col: str = "test_sharpe",
    cmap: str = "RdBu_r",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Ticker × fold heatmap of fold-level Sharpe (or any value column)."""
    mat = df.pivot_table(index=ticker_col, columns=fold_col, values=value_col)
    bound = float(np.nanmax(np.abs(mat.values))) if mat.size else 1.0
    if ax is None:
        _, ax = plt.subplots(figsize=(max(8, 0.35 * mat.shape[1]),
                                       max(4, 0.25 * mat.shape[0])))
    im = ax.imshow(mat.values, aspect="auto", cmap=cmap,
                   vmin=-bound, vmax=bound)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns)
    ax.set_xlabel("fold")
    ax.set_ylabel("ticker")
    ax.figure.colorbar(im, ax=ax, label=value_col)
    return ax


def param_selection_stack(
    df: pd.DataFrame,
    params_col: str = "selected_params",
    group_col: str = "fold_id",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Stacked bar: for each fold (or ticker), count how often each param won."""
    counts = (
        df.groupby([group_col, params_col])
        .size()
        .unstack(fill_value=0)
    )
    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    counts.plot(kind="bar", stacked=True, ax=ax, width=0.85)
    ax.set_xlabel(group_col)
    ax.set_ylabel("ticker count")
    ax.legend(title=params_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    return ax
