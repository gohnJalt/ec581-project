"""Regenerate the Phase-1 Monte-Carlo Sharpe parquets on the current window.

Reads the fresh Phase-1 sweep winners (results/phase1_sweep.parquet), rebuilds
each winning signal on the BIST100 index, runs the matched-randomness MC, and
writes results/phase1_mc.parquet + results/phase1_mc_distributions.parquet.
There is no main.py subcommand for this (Phase-1 MC was a one-off), hence this
small driver mirroring what generated the original parquets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import RANDOM_SEED, RESULTS_DIR, get_dataset
from src.data.clean import clean_path
from src.eval.montecarlo import monte_carlo_sharpe
from src.features.ema import ema_crossover_signal, ema_direction_signal
from src.features.hp import hp_direction_signal
from src.features.lowess import lowess_direction_signal

N_ITER = 1000


def build_signal(strategy: str, prices: pd.Series, row: pd.Series) -> pd.Series:
    if strategy == "S1_ema_crossover":
        return ema_crossover_signal(prices, int(row["n_fast"]), int(row["n_slow"]))
    if strategy == "S2_ema_direction":
        return ema_direction_signal(prices, int(row["n"]))
    if strategy == "S3_hp_direction":
        return hp_direction_signal(prices, lam=float(row["lam"]), window=int(row["window"]))
    if strategy == "S4_lowess_direction":
        return lowess_direction_signal(prices, frac=float(row["frac"]), window=int(row["window"]))
    raise ValueError(strategy)


def main() -> None:
    ds = get_dataset("bist100")
    prices = pd.read_parquet(clean_path(ds.index_ticker))["close"]
    print(f"index={ds.index_ticker}  bars={len(prices)}  "
          f"range=[{prices.index.min().date()}, {prices.index.max().date()}]")

    sweep = pd.read_parquet(RESULTS_DIR / "phase1_sweep.parquet")
    elig = sweep[sweep["enough_trades"]]
    winners = (elig.sort_values("sharpe", ascending=False)
                   .groupby("strategy", as_index=False).first())

    order = ["S1_ema_crossover", "S2_ema_direction",
             "S3_hp_direction", "S4_lowess_direction"]
    rows, dists = [], {}
    for strat in order:
        row = winners[winners["strategy"] == strat].iloc[0]
        sig = build_signal(strat, prices, row)
        res = monte_carlo_sharpe(prices, sig, n_iter=N_ITER, seed=RANDOM_SEED)
        d = res.to_dict()
        d["strategy"] = strat
        rows.append(d)
        dists[strat] = res.mc_sharpes
        print(f"{strat:22s} sharpe={res.strat_sharpe:.3f}  p={res.p_value:.3f}  "
              f"mc_mean={res.mc_sharpes.mean():.3f}  q95={np.quantile(res.mc_sharpes,0.95):.3f}")

    mc = pd.DataFrame(rows)
    mc.to_parquet(RESULTS_DIR / "phase1_mc.parquet")
    pd.DataFrame(dists).to_parquet(RESULTS_DIR / "phase1_mc_distributions.parquet")
    print(f"\nwrote {RESULTS_DIR/'phase1_mc.parquet'} and phase1_mc_distributions.parquet")


if __name__ == "__main__":
    main()
