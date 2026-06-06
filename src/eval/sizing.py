"""Conviction sizing / short-side selection sub-model (progress.md Part 5).

Part 5's pressure-test concluded that the *risk-based* sizing branch of Part 4
(inverse-vol, risk-parity, Kelly, cross-sectional name selection) is dead on
this universe — the whole panel is one TL-inflation factor, the signed signal
already harvests the diversification, and trailing covariance is too noisy to
invert out-of-sample. What *does* move portfolio Sharpe are two causal levers:

  1. **Conviction sizing** — replace the binary ±1 signal with a vol-normalized
     trend-slope magnitude so a just-flipped weak trend gets less capital than
     an established steep one. ~+0.25 Sharpe at neutral drawdown, robust across
     four conviction definitions. Works because it is a *within-name,
     time-series* tilt (weak slopes are universally noisier), not a
     cross-sectional bet on which names are good (which does not persist).
  2. **Short-side policy** — the short leg is a standalone money-loser
     (Sharpe ≈ −0.80, fighting a 136× positive-drift index) but a crash hedge.
     Dropping it (``long_only``) lifts Sharpe ~+0.16 at deeper MDD; gating
     shorts through the Section 4.3 regime filter (``regime_short``) is the
     middle ground that keeps most of the hedge.

This is the brief-deviating **overlay**: the binary equal-weight portfolio in
``src/eval/portfolio.py`` stays the brief-compliant baseline, and everything
here is reported as a delta against it (only relative deltas are meaningful —
see the caveats in progress.md Part 5). All scores are causal: every quantity
at bar t uses prices ≤ t only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.eval.portfolio import effective_position
from src.eval.strategies import StrategySpec, trend as compute_trend

TRADING_DAYS = 252
DEFAULT_VOL_WINDOW = 63   # ~one quarter, for the trailing-vol normalizers

CONVICTION_METHODS = ("raw", "retvol", "zstd")
SHORT_POLICIES = ("both", "long_only", "regime_short")


# ---------------------------------------------------------------------------
# Per-ticker primitives
# ---------------------------------------------------------------------------

def trend_slope(prices: pd.Series, spec: StrategySpec,
                param: float | None = None, window: int | None = None) -> pd.Series:
    """Signed one-bar change in the causal rolling trend (sign() of it = signal)."""
    return compute_trend(prices, spec, param, window).diff().rename("slope")


def conviction_score(prices: pd.Series, slope: pd.Series, method: str = "retvol",
                     vol_window: int = DEFAULT_VOL_WINDOW) -> pd.Series:
    """Non-negative per-bar conviction magnitude (steeper / cleaner trend → larger).

    Methods (progress.md Part 5, all causal):
      raw     |slope| / price                                    (the +0.42 def)
      retvol  |slope| / (price · trailing return vol)            (the +0.28 def)
      zstd    (|slope|/price) / rolling-std(|slope|/price)       (the +0.25 def)

    The fourth Part-5 definition (cross-sectional rank) is a panel transform —
    see ``cross_sectional_rank`` — because it ranks across names per bar.
    """
    base = (slope.abs() / prices)
    if method == "raw":
        out = base
    elif method == "retvol":
        retvol = prices.pct_change().rolling(
            vol_window, min_periods=vol_window // 2).std()
        out = base / retvol.replace(0.0, np.nan)
    elif method == "zstd":
        sd = base.rolling(vol_window, min_periods=vol_window // 2).std()
        out = base / sd.replace(0.0, np.nan)
    else:
        raise ValueError(f"unknown conviction method {method!r}; "
                         f"available: {CONVICTION_METHODS}")
    return out.replace([np.inf, -np.inf], np.nan).rename("conviction")


def apply_short_policy(position: pd.Series, regime: pd.Series | None,
                       policy: str = "both") -> pd.Series:
    """Mask the held position's short leg per the chosen short-side policy.

    both          keep shorts as-is
    long_only     drop every short (Sharpe↑ but MDD↑ — loses the crash hedge)
    regime_short  keep a short only on bars where BIST100 regime confirms (<0)
    """
    if policy == "both":
        return position
    if policy == "long_only":
        return position.clip(lower=0.0)
    if policy == "regime_short":
        if regime is None:
            raise ValueError("regime_short policy needs a regime series")
        reg = regime.reindex(position.index).fillna(0.0)
        keep_short = (position < 0) & (reg < 0)
        return position.where((position >= 0) | keep_short, 0.0)
    raise ValueError(f"unknown short policy {policy!r}; available: {SHORT_POLICIES}")


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------

@dataclass
class SizingPanels:
    """Aligned (date × ticker) frames the weighting schemes consume."""
    returns: pd.DataFrame      # raw per-name pct_change (NaN pre-IPO)
    positions: pd.DataFrame    # signed held position ∈ {-1,0,+1} after short policy
    conviction: pd.DataFrame   # non-negative conviction magnitude (NaN in warmup)


def build_panels(
    prices_by_ticker: dict[str, pd.Series],
    spec: StrategySpec,
    param: float | None = None,
    window: int | None = None,
    regime: pd.Series | None = None,
    regime_gate: bool = False,
    short_policy: str = "both",
    conviction_method: str = "retvol",
    vol_window: int = DEFAULT_VOL_WINDOW,
) -> SizingPanels:
    """Build the return / position / conviction panels for one experiment.

    ``regime_gate`` toggles the Section 4.3 entry gate inside ``effective_position``
    (the existing regime variant); ``short_policy`` masks shorts *after* the
    position is formed (the Part 5 short-side lever). They are orthogonal — the
    sizing ablation leaves the entry gate off and varies only the short policy.
    """
    ret_cols: dict[str, pd.Series] = {}
    pos_cols: dict[str, pd.Series] = {}
    conv_cols: dict[str, pd.Series] = {}
    for ticker, prices in prices_by_ticker.items():
        sig = np.sign(compute_trend(prices, spec, param, window).diff()).astype(float)
        reg_gate = regime.reindex(prices.index).fillna(0.0) if (regime is not None and regime_gate) else None
        pos = effective_position(sig, reg_gate)
        pos = apply_short_policy(pos, regime, short_policy)

        slope = trend_slope(prices, spec, param, window)
        conv = conviction_score(prices, slope, method=conviction_method,
                                vol_window=vol_window)

        ret_cols[ticker] = prices.pct_change().rename(ticker)
        pos_cols[ticker] = pos.rename(ticker)
        conv_cols[ticker] = conv.rename(ticker)

    return SizingPanels(
        returns=pd.concat(ret_cols, axis=1),
        positions=pd.concat(pos_cols, axis=1),
        conviction=pd.concat(conv_cols, axis=1),
    )


# ---------------------------------------------------------------------------
# Weighting schemes (signed, close-of-bar; combined with NEXT bar's return)
# ---------------------------------------------------------------------------

def _l1_normalize(weights: pd.DataFrame) -> pd.DataFrame:
    """Scale each bar so gross (Σ|w|) = 1; leave all-flat bars at 0."""
    gross = weights.abs().sum(axis=1)
    return weights.div(gross.where(gross > 0, np.nan), axis=0).fillna(0.0)


def cross_sectional_rank(score: pd.DataFrame) -> pd.DataFrame:
    """Per-bar rank of each name's conviction across the active universe (Part-5 'rank')."""
    return score.where(score > 0).rank(axis=1, method="average")


def binary_weights(positions: pd.DataFrame) -> pd.DataFrame:
    """Brief-compliant baseline: equal ±1/N_active per bar."""
    return _l1_normalize(positions.fillna(0.0))


def conviction_weights(positions: pd.DataFrame, conviction: pd.DataFrame,
                       cross_sectional: bool = False) -> pd.DataFrame:
    """Signed weights = held direction × conviction magnitude, unit-gross per bar.

    With ``cross_sectional=True`` the magnitude is the per-bar cross-sectional
    rank of the conviction instead of its level (Part 5's 'rank' definition).
    """
    score = conviction.reindex_like(positions).fillna(0.0)
    if cross_sectional:
        score = cross_sectional_rank(score).fillna(0.0)
    return _l1_normalize(positions.fillna(0.0) * score)


def weighted_portfolio_returns(weights: pd.DataFrame,
                               returns: pd.DataFrame) -> pd.Series:
    """Portfolio return_t = Σ_i weight_{i,t-1} · raw_return_{i,t} (weights lag one bar)."""
    aligned = weights.reindex_like(returns)
    return (aligned.shift(1) * returns).sum(axis=1).rename("portfolio_ret")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    return (initial * (1.0 + returns.fillna(0.0)).cumprod()).rename("equity")


def returns_metrics(returns: pd.Series) -> dict:
    """Sharpe / Sortino / CAGR / vol / MDD / Calmar from a daily return series."""
    r = returns.dropna()
    n = len(r)
    if n < 2 or r.std(ddof=1) == 0:
        return {"n_bars": int(n), "sharpe": 0.0, "sortino": 0.0, "cagr": 0.0,
                "ann_vol": 0.0, "max_drawdown": 0.0, "calmar": 0.0,
                "total_return": 0.0, "turnover": 0.0}
    eq = equity_curve(r)
    years = n / TRADING_DAYS
    final = float(eq.iloc[-1])
    cagr = final ** (1 / years) - 1 if years > 0 and final > 0 else 0.0
    sd = float(r.std(ddof=1))
    sharpe = float(r.mean() / sd * np.sqrt(TRADING_DAYS))
    downside = r[r < 0]
    dn = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = float(r.mean() / dn * np.sqrt(TRADING_DAYS)) if dn > 0 else 0.0
    dd = eq / eq.cummax() - 1.0
    mdd = float(dd.min())
    return {"n_bars": int(n), "sharpe": sharpe, "sortino": sortino,
            "cagr": float(cagr), "ann_vol": sd * np.sqrt(TRADING_DAYS),
            "max_drawdown": mdd, "calmar": float(cagr) / abs(mdd) if mdd < 0 else 0.0,
            "total_return": final - 1.0}


def turnover(weights: pd.DataFrame) -> float:
    """Mean per-bar gross weight change — churn proxy (free at commission=0, flag for live)."""
    return float(weights.fillna(0.0).diff().abs().sum(axis=1).mean())
