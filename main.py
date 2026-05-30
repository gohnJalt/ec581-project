"""Centralized CLI for the EC581 Trend Following project.

One entry point for every runnable thing in the repo. Each subcommand forwards
to its underlying ``python -m src.<module>`` script (preserving exit codes and
stdout/stderr), so behavior is identical to running the modules directly.

USAGE
-----

    python main.py <command> [options]
    python main.py <command> --help        # per-command help

Examples:

    python main.py data --smoke
    python main.py sweep
    python main.py phase2 --strategy lowess
    python main.py phase2-mc --n-iter 1000
    python main.py walkforward --smoke
    python main.py notebooks

COMMANDS AND OPTIONS
--------------------

data
    Build the data pipeline: ingest -> clean -> panel.
    Wraps: python -m src.data.build
      --smoke              run on the M1 5-stock SMOKE_TICKERS subset
      --source {yfinance,csv}
                           data source for ingestion (default: yfinance)
      --csv-dir PATH       directory of CSVs when --source csv
      --force              re-fetch and re-clean (ignore parquet cache)

sweep
    Phase-1 grid search: run S1..S4 (default_specs) on the BIST100 Index and
    write results/phase1_sweep.parquet.
    Wraps: python -m src.backtest.run_sweep
      --min-trades N       floor for the enough_trades flag (default: 30)

phase2
    Phase-2 base + regime: chosen strategy on every cleaned constituent at
    Phase-1 winning params. Writes results/phase2_{stub}_{base,regime}.parquet.
    Wraps: python -m src.eval.run_phase2
      --smoke              use SMOKE_TICKERS subset
      --strategy {hp,lowess}
                           HP-direction (default) or LOWESS-direction
      --param FLOAT        smoother param (lam for hp, frac for lowess);
                           defaults to Phase-1 winner
      --window INT         rolling window (defaults to Phase-1 winner)

phase2-mc
    Per-stock Monte Carlo Sharpe p-values for the chosen Phase-2 strategy
    (base variant only). Writes results/phase2_mc_{stub}_base.parquet.
    Wraps: python -m src.eval.run_phase2_mc
      --smoke              use SMOKE_TICKERS subset
      --strategy {hp,lowess}
                           HP-direction (default) or LOWESS-direction
      --param FLOAT        smoother param; defaults to Phase-1 winner
      --window INT         rolling window; defaults to Phase-1 winner
      --n-iter N           MC iterations per ticker (default: 1000)

phase2-portfolio
    Phase-2 equal-weight cross-sectional portfolio aggregator (DESIGN.md M4).
    Writes results/phase2_portfolio_{stub}_{equity,panel_*,summary}.parquet.
    Wraps: python -m src.eval.run_phase2_portfolio
      --smoke              use SMOKE_TICKERS subset
      --strategy {hp,lowess}
                           HP-direction (default) or LOWESS-direction
      --param FLOAT        smoother param; defaults to Phase-1 winner
      --window INT         rolling window; defaults to Phase-1 winner

phase2-resumable
    Same as `phase2` but writes after each ticker and supports resume — for
    long LOWESS runs that may stall midway.
    Wraps: python -u -m src.eval.run_phase2_resumable
      --smoke              use SMOKE_TICKERS subset
      --strategy {hp,lowess}
                           HP-direction (default) or LOWESS-direction
      --param FLOAT        smoother param; defaults to Phase-1 winner
      --window INT         rolling window; defaults to Phase-1 winner
      --start-from TICKER  skip tickers until this one (resume mode)

walkforward
    Phase-2 walk-forward driver: 3y train / 1y test / 1y step over the HP grid.
    Writes results/phase2_walkforward_{base,regime}.parquet.
    Wraps: python -m src.eval.run_walkforward
      --smoke              use SMOKE_TICKERS and a 2-config HP grid
      --min-trades N       minimum training-fold trade count (default: 30)

notebooks
    Re-execute the four analysis notebooks in place (writes outputs into the
    same .ipynb). Pass --only to run a subset.
    Wraps: .venv/bin/python -m jupyter nbconvert --to notebook --execute ...
      --only NAMES         comma-separated subset (e.g. "02,04"); default: all
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"
NOTEBOOKS = {
    "01": "notebooks/01_eda.ipynb",
    "02": "notebooks/02_index_strategy_comparison.ipynb",
    "03": "notebooks/03_walk_forward.ipynb",
    "04": "notebooks/04_regime_filter.ipynb",
}


def _python() -> str:
    # Prefer the project venv interpreter (stale-shebang note in CLAUDE.md),
    # fall back to whatever is running this script.
    return str(VENV_PY) if VENV_PY.exists() else sys.executable


def _forward(module: str, args: list[str], unbuffered: bool = False) -> int:
    cmd = [_python()]
    if unbuffered:
        cmd.append("-u")
    cmd.extend(["-m", module, *args])
    print(f"+ {shlex.join(cmd)}", file=sys.stderr)
    return subprocess.call(cmd, cwd=REPO_ROOT)


def _strategy_args(ns: argparse.Namespace) -> list[str]:
    out: list[str] = []
    if ns.smoke:
        out.append("--smoke")
    out += ["--strategy", ns.strategy]
    if ns.param is not None:
        out += ["--param", str(ns.param)]
    if ns.window is not None:
        out += ["--window", str(ns.window)]
    return out


def _add_strategy_opts(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--smoke", action="store_true")
    sp.add_argument("--strategy", choices=["hp", "lowess"], default="hp")
    sp.add_argument("--param", type=float, default=None,
                    help="lam for hp, frac for lowess; defaults to Phase-1 winner")
    sp.add_argument("--window", type=int, default=None)


def cmd_data(ns: argparse.Namespace) -> int:
    args: list[str] = []
    if ns.smoke:
        args.append("--smoke")
    args += ["--source", ns.source]
    if ns.csv_dir is not None:
        args += ["--csv-dir", str(ns.csv_dir)]
    if ns.force:
        args.append("--force")
    return _forward("src.data.build", args)


def cmd_sweep(ns: argparse.Namespace) -> int:
    return _forward("src.backtest.run_sweep", ["--min-trades", str(ns.min_trades)])


def cmd_phase2(ns: argparse.Namespace) -> int:
    return _forward("src.eval.run_phase2", _strategy_args(ns))


def cmd_phase2_mc(ns: argparse.Namespace) -> int:
    args = _strategy_args(ns) + ["--n-iter", str(ns.n_iter)]
    return _forward("src.eval.run_phase2_mc", args)


def cmd_phase2_portfolio(ns: argparse.Namespace) -> int:
    return _forward("src.eval.run_phase2_portfolio", _strategy_args(ns))


def cmd_phase2_resumable(ns: argparse.Namespace) -> int:
    args = _strategy_args(ns)
    if ns.start_from:
        args += ["--start-from", ns.start_from]
    return _forward("src.eval.run_phase2_resumable", args, unbuffered=True)


def cmd_walkforward(ns: argparse.Namespace) -> int:
    args: list[str] = []
    if ns.smoke:
        args.append("--smoke")
    args += ["--min-trades", str(ns.min_trades)]
    return _forward("src.eval.run_walkforward", args)


def cmd_notebooks(ns: argparse.Namespace) -> int:
    keys = list(NOTEBOOKS) if not ns.only else [k.strip() for k in ns.only.split(",")]
    rc = 0
    for k in keys:
        nb = NOTEBOOKS.get(k)
        if nb is None:
            print(f"unknown notebook key {k!r}; valid: {list(NOTEBOOKS)}", file=sys.stderr)
            rc = 2
            continue
        nb_path = REPO_ROOT / nb
        cmd = [_python(), "-m", "jupyter", "nbconvert", "--to", "notebook",
               "--execute", str(nb_path), "--output", os.path.basename(nb)]
        print(f"+ {shlex.join(cmd)}", file=sys.stderr)
        rc = subprocess.call(cmd, cwd=REPO_ROOT) or rc
    return rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ec581",
        description="Centralized CLI for the EC581 Trend Following project.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("data", help="ingest -> clean -> panel")
    sp.add_argument("--smoke", action="store_true")
    sp.add_argument("--source", choices=("yfinance", "csv"), default="yfinance")
    sp.add_argument("--csv-dir", type=Path, default=None)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_data)

    sp = sub.add_parser("sweep", help="Phase-1 S1..S4 grid search on BIST100 Index")
    sp.add_argument("--min-trades", type=int, default=30)
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("phase2", help="Phase-2 base + regime on every constituent")
    _add_strategy_opts(sp)
    sp.set_defaults(func=cmd_phase2)

    sp = sub.add_parser("phase2-mc", help="Per-stock Monte Carlo Sharpe p-values")
    _add_strategy_opts(sp)
    sp.add_argument("--n-iter", type=int, default=1000)
    sp.set_defaults(func=cmd_phase2_mc)

    sp = sub.add_parser("phase2-portfolio", help="Equal-weight cross-sectional portfolio")
    _add_strategy_opts(sp)
    sp.set_defaults(func=cmd_phase2_portfolio)

    sp = sub.add_parser("phase2-resumable", help="Per-ticker resumable Phase-2 driver")
    _add_strategy_opts(sp)
    sp.add_argument("--start-from", type=str, default=None)
    sp.set_defaults(func=cmd_phase2_resumable)

    sp = sub.add_parser("walkforward", help="Phase-2 walk-forward (HP grid)")
    sp.add_argument("--smoke", action="store_true")
    sp.add_argument("--min-trades", type=int, default=30)
    sp.set_defaults(func=cmd_walkforward)

    sp = sub.add_parser("notebooks", help="Re-execute analysis notebooks in place")
    sp.add_argument("--only", type=str, default=None,
                    help="comma-separated subset of notebook keys (e.g. '02,04')")
    sp.set_defaults(func=cmd_notebooks)

    return p


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    return ns.func(ns)


if __name__ == "__main__":
    sys.exit(main())
