"""Phase-2 conviction-sizing ablation driver (progress.md Part 5 deliverable).

Reconstructs the cross-sectional portfolio under a ladder of sizing / short-side
schemes and reports each as a Sharpe delta against the brief-compliant binary
equal-weight baseline. This is the Part 5 "scoreboard": the two causal levers
that work (conviction sizing, short-side policy) over the Full Universe.

Every scheme is a signed unit-gross weight panel combined with the *next* bar's
raw per-name return (``src/eval/sizing.py``), so all rows share one
reconstruction and only the relative deltas are interpreted (absolute Sharpes
are inflated vs the realistic Backtrader portfolio — see Part 5 caveats).

Outputs under ``results/`` (stub depends on strategy; HP keeps the unsuffixed name):
  - ``phase2_sizing_{stub}_scoreboard.parquet``  one row per scheme + Δ vs baseline
  - ``phase2_sizing_{stub}_equity.parquet``      wide equity curves, one column per scheme
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import (
    BIST100_INDEX_TICKER,
    DEFAULT_DATASET,
    RESULTS_DIR,
    get_dataset,
)
from src.data.clean import clean_path
from src.eval import sizing
from src.eval.strategies import StrategySpec, get as get_strategy
from src.features.regime import bist_regime_flag

REGIME_LAM = 1600
REGIME_WINDOW = 504

# (label, sizing-kind, conviction-method, cross-sectional?, short-policy)
#   kind ∈ {"binary", "conviction"}; the conviction methods mirror the four
#   Part-5 definitions plus the two short-side levers and their stack.
_SCHEMES = [
    ("binary / both (baseline)",   "binary",     None,     False, "both"),
    ("conviction raw / both",      "conviction", "raw",    False, "both"),
    ("conviction retvol / both",   "conviction", "retvol", False, "both"),
    ("conviction zstd / both",     "conviction", "zstd",   False, "both"),
    ("conviction rank / both",     "conviction", "retvol", True,  "both"),
    ("binary / long_only",         "binary",     None,     False, "long_only"),
    ("conviction retvol / long_only", "conviction", "retvol", False, "long_only"),
    ("binary / regime_short",      "binary",     None,     False, "regime_short"),
    ("conviction retvol / regime_short", "conviction", "retvol", False, "regime_short"),
]


def _available_tickers(universe: list[str]) -> list[str]:
    return [t for t in universe if clean_path(t).exists()]


def _load_prices(tickers: list[str]) -> dict[str, pd.Series]:
    return {t: pd.read_parquet(clean_path(t))["close"].rename(t) for t in tickers}


def _weights_for(panels: sizing.SizingPanels, kind: str, cross_sectional: bool) -> pd.DataFrame:
    if kind == "binary":
        return sizing.binary_weights(panels.positions)
    return sizing.conviction_weights(panels.positions, panels.conviction,
                                     cross_sectional=cross_sectional)


def run(
    tickers: list[str],
    spec: StrategySpec,
    param: float | None = None,
    window: int | None = None,
    regime_lam: float = REGIME_LAM,
    regime_window: int = REGIME_WINDOW,
    index_ticker: str = BIST100_INDEX_TICKER,
    out_dir: Path = RESULTS_DIR,
    verbose: bool = True,
) -> dict[str, Path]:
    if not tickers:
        raise RuntimeError("no tickers available — populate data/clean/ first")

    prices_by_ticker = _load_prices(tickers)
    bist_close = pd.read_parquet(clean_path(index_ticker))["close"]
    regime = bist_regime_flag(bist_close, lam=regime_lam, window=regime_window)

    if verbose:
        print(f"{spec.label}: conviction-sizing ablation over {len(tickers)} tickers\n")

    # Panels depend only on (conviction_method, short_policy); cache to avoid
    # recomputing the rolling trend per scheme.
    panel_cache: dict[tuple[str, str], sizing.SizingPanels] = {}

    def _panels(method: str | None, short_policy: str) -> sizing.SizingPanels:
        key = (method or "raw", short_policy)   # binary ignores conviction; reuse "raw"
        if key not in panel_cache:
            panel_cache[key] = sizing.build_panels(
                prices_by_ticker, spec, param=param, window=window,
                regime=regime, regime_gate=False, short_policy=short_policy,
                conviction_method=method or "raw",
            )
        return panel_cache[key]

    rows: list[dict] = []
    curves: dict[str, pd.Series] = {}
    baseline_sharpe: float | None = None
    for label, kind, method, xsec, short_policy in _SCHEMES:
        panels = _panels(method, short_policy)
        weights = _weights_for(panels, kind, xsec)
        port = sizing.weighted_portfolio_returns(weights, panels.returns)
        m = sizing.returns_metrics(port)
        m["scheme"] = label
        m["kind"] = kind
        m["conviction"] = method or ""
        m["cross_sectional"] = xsec
        m["short_policy"] = short_policy
        m["turnover"] = sizing.turnover(weights)
        m["strategy"] = spec.key
        if baseline_sharpe is None:
            baseline_sharpe = m["sharpe"]
        m["sharpe_delta"] = m["sharpe"] - baseline_sharpe
        rows.append(m)
        curves[label] = sizing.equity_curve(port)

    summary = pd.DataFrame(rows)
    equity = pd.concat(curves, axis=1)

    stub = "" if spec.key == "hp" else f"_{spec.output_stub}"
    paths = {
        "scoreboard": out_dir / f"phase2_sizing{stub}_scoreboard.parquet",
        "equity":     out_dir / f"phase2_sizing{stub}_equity.parquet",
    }
    summary.to_parquet(paths["scoreboard"])
    equity.to_parquet(paths["equity"])

    if verbose:
        cols = ["scheme", "sharpe", "sharpe_delta", "cagr", "ann_vol",
                "max_drawdown", "calmar", "turnover"]
        print(summary[cols].to_string(index=False))

    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="dataset to run (default: bist100)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--strategy", choices=["hp", "lowess"], default="hp")
    ap.add_argument("--param", type=float, default=None,
                    help="lam for hp, frac for lowess; defaults to Phase-1 winner")
    ap.add_argument("--window", type=int, default=None)
    args = ap.parse_args()

    ds = get_dataset(args.dataset)
    spec = get_strategy(args.strategy)
    universe = list(ds.smoke_tickers) if args.smoke else ds.load_constituents()
    tickers = _available_tickers(universe)
    print(f"dataset={ds.name}  tickers ({len(tickers)}): {tickers}")
    run(tickers, spec, param=args.param, window=args.window,
        index_ticker=ds.index_ticker, out_dir=ds.results_dir)


if __name__ == "__main__":
    main()
