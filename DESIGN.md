# Trend Following on BIST100 — Design Document

**Course:** EC581 — Algorithmic Trading and Quantitative Strategies
**Project:** Section 4 — Trend Following
**Reference article:** Harris, R. D. F. and Yilmaz, F. (2009), *A Momentum Trading Strategy Based on the Low Frequency Component of the Exchange Rate*, Journal of Banking and Finance, 33(9), 1575–1585.

---

## 1. Executive Summary

We design and evaluate a long/short trend-following strategy for Turkish equities (BIST100 constituents), inspired by Harris & Yilmaz (2008/09). The central idea is that an asset price contains a slow-moving, persistent **trend component** that can be separated from short-term noise with a low-pass filter, and that the **direction of that trend** is a tradable signal. We implement four trend extractors — EMA crossover, EMA direction, the Hodrick–Prescott (HP) filter, and LOWESS — tune them on the BIST100 Index, select a primary strategy, and deploy it on individual stocks with a market-regime filter gated by the index's own HP trend.

**Headline results (2015–2026, 84 cleaned constituents).** All four strategies were tuned on the BIST100 Index (Phase 1). The low-frequency trend strategies (HP, LOWESS) and EMA direction all beat the classical EMA crossover and reject a frequency-matched random benchmark at $p \le 0.002$; the crossover itself does not ($p = 0.48$). We carry **S3 HP** into Phase 2 as the principled primary — its motivation is exactly the Harris–Yilmaz low-frequency argument, and $\lambda = 14{,}400$ is the standard quarterly-smoothing baseline — and **S4 LOWESS** as an empirical benchmark. On the 84 constituents, an equal-weight cross-sectional portfolio reaches **Sharpe 1.70 (HP) / 1.68 (LOWESS)** at ~26% CAGR and a −17% maximum drawdown, roughly tripling the mean single-name Sharpe through diversification. A BIST100 buy-and-hold over the same window earns a comparable Sharpe (~1.1) but with a deeper −32% drawdown — the strategy delivers similar risk-adjusted return at roughly half the drawdown. A BIST100-HP regime filter acts as a variance reducer (lower volatility and trade count, marginally lower Sharpe), and out-of-sample walk-forward Sharpe stays positive in about three-quarters of folds.

## 2. Background and Motivation

### 2.1 Why trend following works

Classical efficient-market theory predicts no exploitable serial dependence in returns. In practice, persistent return autocorrelation has been documented across asset classes and decades (Moskowitz, Ooi & Pedersen 2012; Asness, Moskowitz & Pedersen 2013). Behavioral explanations — anchoring, under-reaction to news, herding — and structural ones — slow capital flows, leverage cycles, central-bank reaction functions — both predict that prices adjust to information gradually rather than instantaneously. A strategy that is long when the slow trend is up and short when it is down harvests this drift.

### 2.2 Harris & Yilmaz (2008) — the anchor idea

Harris and Yilmaz study daily exchange rates and decompose each price series into a **low-frequency component** (extracted with the Hodrick–Prescott filter, the same filter used in business-cycle econometrics) and a **high-frequency residual**. Their key claims, which we operationalize for equities, are:

1. The low-frequency component is highly persistent — when its slope flips sign, it tends to keep the new sign for many periods.
2. Trading rules based on the **sign of the change in the low-frequency component** outperform classical moving-average crossover rules in out-of-sample tests across major currency pairs.
3. The economic case for a low-pass filter (rather than a plain moving average) is a smoother, more stable trend estimate with fewer whipsaws.

We take this directly into the equity setting: the **HP-direction** rule is the primary candidate, with EMA crossover, EMA direction, and LOWESS direction as benchmark trend extractors against which it is compared.

## 3. Strategy Design

### 3.1 Hypothesis

> The smoothed trend component of an equity price contains exploitable directional information. Going long when the trend is rising and short when it is falling, on a fixed-cash-per-trade basis, produces a Sharpe ratio statistically superior to a frequency-matched random strategy. A market-regime filter (the BIST100's own trend) reshapes the risk profile by avoiding trades that fight the index.

### 3.2 Signal variants

Each variant produces a binary trend state $T_t \in \{+1, -1\}$ on a daily bar.

| # | Name | Indicator | Tunable parameter(s) |
|---|------|-----------|----------------------|
| S1 | EMA Crossover | $\text{sign}(\text{EMA}_{\text{fast}} - \text{EMA}_{\text{slow}})$ | $(n_{\text{fast}}, n_{\text{slow}})$ |
| S2 | EMA Direction | $\text{sign}(\Delta \text{EMA}_n)$ | $n$ |
| S3 | HP Direction *(primary)* | $\text{sign}(\Delta\, \text{HP}_\lambda(P))$ | $\lambda$ |
| S4 | LOWESS Direction | $\text{sign}(\Delta\, \text{LOWESS}_{\text{frac}}(P))$ | $\text{frac}$ |

### 3.3 Trade and position rules

Same convention for all variants:

- Enter long when $T_t$ flips from $-1$ to $+1$.
- Close long and enter short when $T_t$ flips from $+1$ to $-1$.
- Always-in-the-market: no flat state in the base strategy; there is no separate stop-loss or take-profit, matching the brief.

Position sizing uses the course-mandated `FixedCashSizer` (100,000 TL notional per *order*); orders are `bt.Order.Market`; commission is 0. The strategy holds **multiple concurrent 100k lots** so total capital is fully deployed when in the market — on a signal flip it submits $\lfloor \text{equity} / 100\text{k} \rfloor$ orders in one bar (≈10 lots for a fresh 1M account), closing any opposite position first, and holds the stacked lots until the next flip. This respects the brief's per-order cap while avoiding the cash-drag of single-lot trend-following on one instrument.

### 3.4 Why HP is the primary candidate

- Its theoretical motivation matches Harris & Yilmaz (2008) directly.
- It has a single interpretable hyperparameter $\lambda$ controlling the trend/noise tradeoff (larger $\lambda$ → smoother, slower trend), with $\lambda = 14{,}400$ the standard quarterly-smoothing baseline.
- The HP filter implementation is given in the brief and is non-iterative (a single linear solve per refit).

S4 LOWESS is carried through Phase 2 as a head-to-head empirical benchmark and reported alongside HP throughout. On the final data the two are effectively tied: HP edges LOWESS on the base portfolio (Sharpe 1.70 vs 1.68) and on per-stock significance (13 vs 9 names at $p<0.01$), while LOWESS edges slightly on the mean single-name Sharpe and once the regime filter is on. Keeping HP as the primary is a deliberate, theoretically motivated choice rather than a benchmark swap.

### 3.5 Regime filter (Phase 2)

After selecting the primary strategy on the BIST100 Index, we deploy it on individual stocks **gated by the BIST100's own HP trend**:

- A long entry on stock $i$ requires the stock signal to be long **and** BIST100 close $>$ HP-filtered BIST100 close.
- A short entry on stock $i$ requires the stock signal to be short **and** BIST100 close $<$ HP-filtered BIST100 close.
- Exits ignore the regime filter, so positions are not trapped when the regime flips.

The economic story is to trade with the index, not against it — a standard top-down/bottom-up combination consistent with using a single dominant low-frequency factor.

## 4. Data

### 4.1 Universe and period

- **Phase 1 (tuning):** BIST100 Index daily close, ticker `XU100.IS` (single series, used for hyperparameter selection).
- **Phase 2 (deployment):** BIST100 constituents at current membership, no universe rebalancing. After ingest and cleaning, **84 names** clear the minimum-history floor.
- **Fields:** daily OHLCV, in TL. **Period:** 2015-01-01 to 2026, ≈2,860 daily bars per series.
- **Capital:** 1,000,000 TL initial; 100,000 TL per order; commission 0.

### 4.2 Cleaning and corporate-action adjustment

Per ticker we drop duplicate timestamps, keep the trading-day calendar (no calendar forward-fill), and drop names with fewer than 750 bars of history. A back-adjustment step (`_adjust_splits`) detects any single-day price ratio greater than $3\times$ (or below $1/3\times$) and rescales all prior OHLC by that ratio. This corrects two issues that yfinance does not back-adjust:

- The **2005-01-03 Turkish lira redenomination** (1,000,000 old TL → 1 new TL), a ~1000× single-day jump present in every Turkish equity series but not in the index.
- Isolated **corporate-action gaps** (e.g. a ~4× factor on MGROS, 2009-08-04) that otherwise produce extreme drawdowns on the short side.

The largest legitimate single-day move in the universe (HEKTS, +44%) is well below the $3\times$ threshold, so the adjustment is conservative.

## 5. Methodology

### 5.1 Pipeline

```
  ┌───────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │  raw vendor   │ →  │  ingestion   │ →  │  cleaning    │ →  │  indicators  │ →  │  backtest    │
  │  API / CSV    │    │  + schema    │    │  + alignment │    │  (causal)    │    │  Backtrader  │
  └───────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
        data/raw/ (immutable)   data/clean/ (parquet)   data/panel/ (wide panels)   results/ (parquet)
```

- **Ingestion** validates the `date, open, high, low, close, volume` schema and writes immutable per-ticker parquet.
- **Cleaning** dedupes, sorts, back-adjusts (§4.2), and applies the history floor.
- **Alignment** builds a wide `(date × ticker)` price panel plus the index series, feeding both the indicator stage and the cross-sectional portfolio aggregation.
- **Indicators** are one module per extractor (EMA, HP, LOWESS, regime flag), each producing a causal $+1/-1/0$ signal line.
- **Backtest** wraps each indicator in a Backtrader strategy that consumes the precomputed signal as a data line and acts on sign flips.

### 5.2 Avoiding look-ahead bias

The HP filter and LOWESS are **global smoothers**: applied to a whole series, the value at time $t$ depends on data after $t$. Used directly in a backtest this leaks the future into the past. We therefore wrap both in a **causal rolling refit** that, at each bar $t$, refits the smoother on the trailing window $[t-W+1,\, t]$ and returns only the rightmost (right-edge) trend value. The HP linear system is pre-factored once (Cholesky) and re-solved per bar. EMA is causal by construction (it uses only past observations), so only HP and LOWESS need the wrapper. The bare filter functions are used only for plotting and exploratory analysis.

Indicator values produced at bar $t$ inform orders submitted at bar $t+1$, which Backtrader's `next()` convention with market orders gives naturally.

### 5.3 Backtest mechanics

- Engine: Backtrader with `FixedCashSizer(cash_per_trade=100_000)`, `setcash(1_000_000)`, `setcommission(commission=0)`, market orders, daily bars.
- One Cerebro instance per (strategy, ticker) pair in Phase 2 for clean P&L attribution; aggregation is done in post-processing.

### 5.4 Performance metrics

Reported for the base strategy and after each enhancement: net P/L (TL and %), CAGR, annualized volatility, Sharpe, Sortino, maximum drawdown (% and duration), Calmar, win rate / average win / average loss, number of trades and average holding period, and turnover.

### 5.5 Statistical significance — Monte Carlo

Per the brief, significance is a Monte-Carlo $p$-value on the Sharpe ratio. For each strategy we generate $N = 1000$ random strategies that match the empirical (a) run-length distribution (≈ trade frequency × holding period) and (b) long/short proportion, and compute the fraction whose Sharpe exceeds the strategy's. A fast vectorized simulator scores both the strategy and the matched-random pool consistently, so the comparison is apples-to-apples. Random seeds are fixed for reproducibility.

### 5.6 Optimization and walk-forward protocol

- **Phase 1 — in-sample tuning on the BIST100 Index**, grid search per strategy:
  - S1: $n_{\text{fast}} \in \{5,10,20,30\}$, $n_{\text{slow}} \in \{50,100,200\}$, $n_{\text{fast}} < n_{\text{slow}}$.
  - S2: $n \in \{10,20,50,100,200\}$.
  - S3: $\lambda \in \{100, 1600, 14400, 129600\}$, rolling window $W \in \{252, 504, 1260\}$.
  - S4: $\text{frac} \in \{0.05, 0.1, 0.2, 0.3\}$, rolling window as S3.
  - **Selection rule:** highest Sharpe subject to a minimum trade-count floor (≥ 30 round trips) to reject degenerate configs.
- **Phase 2 — walk-forward** (primary strategy, per stock): 3-year training window, 1-year test window, 1-year step, re-selecting parameters on each training fold and trading the winner on the next year. The 5-year-warmup ($W = 1260$) configs are pruned from the walk-forward grid because they leave too little post-warmup room for fold-level retraining.

## 6. Results

All Phase-2 numbers below are over the 84-name cleaned universe, 2015–2026.

### 6.1 Phase 1 — index tuning

**Best configuration per strategy** (BIST100 Index, in-sample, with ≥ 30 trades):

| Strategy | Best params | Sharpe | CAGR | MDD |
|---|---|---|---|---|
| S4 LOWESS direction | frac = 0.05, W = 252 | **1.11** | 25.1% | −25.1% |
| S3 HP direction *(primary)* | $\lambda$ = 100, W = 252 | 1.04 | 23.6% | −30.3% |
| S2 EMA direction | n = 10 | 0.96 | 21.7% | −32.7% |
| S1 EMA crossover | 5 / 200 | 0.70 | 14.3% | −48.6% |

The slope-of-trend strategies (S3, S4) and EMA direction (S2) all beat the classical crossover (S1) — exactly the Harris–Yilmaz claim. The in-sample optimum for HP is $\lambda = 100$; we carry the theoretically motivated baseline $\lambda = 14{,}400$ (with $W = 504$) into Phase 2 as a smoother, less overfit setting. LOWESS is carried at frac = 0.2, $W = 252$. Reference: BIST100 buy-and-hold over the window returns ~16× total at CAGR ≈ 27.5%, Sharpe ≈ 1.1, MDD ≈ −32%.

**Monte-Carlo significance** (BIST100 Index, $N = 1000$):

| Strategy | Sharpe | $p$-value |
|---|---|---|
| S4 LOWESS | 1.17 | **0.000** |
| S3 HP | 1.15 | 0.000 |
| S2 EMA direction | 1.08 | 0.002 |
| S1 EMA crossover | 0.67 | 0.479 |

S2, S3 and S4 reject the matched-random benchmark at $p \le 0.002$. The classical crossover S1 does not ($p = 0.48$): its few long trades are indistinguishable from luck — the paper's central point.

### 6.2 Phase 2 — per stock

Full-history backtest per name, base and BIST100-HP regime-gated variants:

| | HP base | HP regime | LOWESS base | LOWESS regime |
|---|---|---|---|---|
| mean Sharpe | 0.539 | 0.422 | 0.551 | 0.450 |
| mean CAGR | 15.2% | 9.6% | 16.1% | 10.4% |
| mean MDD | −61.0% | −55.9% | −64.2% | −58.4% |
| total trades | 18,591 | 12,398 | 21,536 | 12,559 |
| Sharpe > 0 | 79/84 | 74/84 | 80/84 | 74/84 |

Single-name MDDs in the −50% to −65% range are the cost of always-in-market full-capital deployment on individual equities, where shorts bleed against nominal drift. The portfolio aggregation in §6.3 dilutes this to ~−17%.

**Per-stock Monte-Carlo** ($N = 1000$, base variant):

| | HP | LOWESS |
|---|---|---|
| $p < 0.01$ | 13/84 | 9/84 |
| $p < 0.05$ | 33/84 | 30/84 |
| mean strategy Sharpe | 0.603 | 0.580 |

### 6.3 Phase 2 — equal-weight portfolio

Cross-sectional mean of the per-ticker strategy returns (NaN-skipping for pre-IPO bars; flat-but-active names contribute 0):

| | HP base | HP regime | LOWESS base | LOWESS regime |
|---|---|---|---|---|
| Sharpe | **1.70** | 1.36 | 1.68 | 1.45 |
| CAGR | 26.2% | 17.1% | 24.9% | 17.3% |
| ann. vol | 14.3% | 12.2% | 13.8% | 11.4% |
| MDD | −17.5% | −16.9% | −18.4% | −18.9% |
| Calmar | 1.50 | 1.01 | 1.35 | 0.91 |
| total return | 14.8× | 6.2× | 13.1× | 6.3× |

The portfolio Sharpe (1.70) is roughly triple the mean single-name Sharpe (~0.54): the cross-sectional diversification benefit. Total return is below buy-and-hold's nominal upside, but at roughly half the drawdown (−17% vs −32%).

### 6.4 Walk-forward out-of-sample

Re-tuning the HP grid on each 3-year window and trading the winner on the next year — 397 folds over 68 names, all out-of-sample:

| | Base | Regime |
|---|---|---|
| mean test Sharpe | 0.68 | 0.55 |
| median test Sharpe | 0.71 | 0.58 |
| % folds Sharpe > 0 | 73.8% | 70.3% |
| mean test MDD | −34.7% | −30.3% |

Out-of-sample Sharpe is positive in about three-quarters of folds and below the Phase-1 in-sample index Sharpe — the honest cost of per-stock noise and fold-level retraining. Parameter selection almost never picks the longer $W = 504$ window, so it is dropped from future sweeps. The LOWESS walk-forward is similar (mean test Sharpe 0.62 base / 0.47 regime).

### 6.5 Regime filter

The BIST100-HP regime filter is a **variance reducer, not a Sharpe enhancer**: it cuts trade count by ~30% and lowers portfolio volatility (14.3% → 12.2% for HP), but the base portfolio still edges it on Sharpe (1.70 vs 1.36). It rescues the weakest base names and clips some of the strongest, helping 33 of 84 names in net. The same pattern holds at the per-stock and walk-forward levels.

### 6.6 HP vs LOWESS

On the 2015-start window the two are neck-and-neck. HP edges LOWESS on the base portfolio (1.70 vs 1.68) and on per-stock significance (13 vs 9 names at $p<0.01$); LOWESS edges slightly on the mean single-name Sharpe and once the regime filter is on. Both reject the random benchmark handily. This supports keeping HP as the principled primary with LOWESS as a close, well-behaved alternative.

### 6.7 Extension — conviction sizing and short-side overlay (beyond the brief)

As an exploration beyond the mandated setup, we tested an overlay on top of the equal-weight portfolio, reported strictly as a **delta** against the binary baseline. It replaces the binary ±1 signal with a vol-normalized trend-slope magnitude (conviction) and manages the short leg:

| Scheme | Sharpe | Δ vs. baseline |
|---|---|---|
| binary / both (baseline) | 1.67 | ref. |
| conviction (retvol) / both | 1.95 | +0.28 |
| binary / regime-gated shorts | 2.24 | +0.57 |
| conviction (retvol) / long-only | 2.00 | +0.33 |
| **conviction / regime-gated shorts** | **2.40** | **+0.73** |

Conviction adds ~+0.28 Sharpe (a within-name, time-series tilt: weak just-flipped trends are universally noisier), and keeping shorts only when the index regime confirms adds the most. The short leg alone is a money-loser against the positive-drift index but a crash hedge. We also tested and rejected inverse-vol, vol-target, risk-parity, fractional Kelly, and cross-sectional name selection — the universe is essentially one TL-inflation factor, so trailing-covariance methods are too noisy out-of-sample and cross-sectional rank does not persist (year-over-year per-name Sharpe rank correlation ≈ 0.07); this negative result is itself informative.

**Caveat.** Conviction sizing and short-dropping change per-name exposure away from the binary signal + `FixedCashSizer`, so this is framed as an overlay, not a replacement for the brief-compliant baseline. The absolute Sharpes are inflated by the unit-gross daily-rebalance reconstruction — only the deltas are meaningful, and the levers warrant re-validation through the full Backtrader engine before being treated as headline results.

## 7. Limitations

- **Survivorship bias.** The Phase-2 universe is a snapshot of current membership; delisted names are absent. We disclose this; correcting it is out of scope per the brief.
- **No transaction costs.** Commissions and slippage are zero per the brief. Turnover is high (the conviction overlay roughly doubles it), so real-world costs would erode returns.
- **Single dominant factor.** The BIST universe behaves largely as one TL-inflation factor, limiting cross-sectional diversification; the long/short signing already harvests most of what is available.
- **Deep single-name drawdowns.** Always-in-market full-capital shorts on positively drifting equities produce −50% to −65% single-name drawdowns; portfolio aggregation mitigates this to ~−17%.
- **In-sample vs. out-of-sample gap.** The index Phase-1 Sharpe (~1.0–1.1) overstates the realistic out-of-sample per-stock Sharpe (walk-forward mean ~0.68).

## 8. Implementation

```
ec581-project/
├── DESIGN.md / NB_Projects.pdf / requirements.txt
├── config.py                      # paths, seeds, capital constants, universe loader
├── main.py                        # centralized CLI (data, sweep, phase2, walkforward, ...)
├── data/  (gitignored)            # raw/ → clean/ → panel/
├── results/                       # parquet outputs per phase
├── notebooks/                     # 01_eda, 02_index_comparison, 03_walk_forward, 04_regime_filter
└── src/
    ├── data/      ingest, clean (back-adjustment), panel, build
    ├── features/  ema, hp, lowess (causal rolling wrappers), regime
    ├── backtest/  sizers (FixedCashSizer), feeds, strategies, runner, sweep
    └── eval/      metrics, montecarlo, walkforward, portfolio, sizing, and per-phase drivers
```

Each pipeline stage is a pure function of its inputs plus a parameter dict, the full chain rebuilds from `data/raw/` forward with a single command, and all Monte-Carlo seeds are fixed in `config.py`, so results are reproducible. The pipeline is parametrized over the market (a `Dataset` abstraction), so the same code runs on other universes (e.g. S&P 500) for future out-of-sample robustness, though parameters are not retuned there.

## 9. Conclusion

The Harris–Yilmaz thesis holds on Turkish equities: **momentum measured on the low-frequency trend component beats momentum on the raw price.** Every direction strategy (HP, LOWESS, EMA-direction) beats the classical moving-average crossover and rejects a frequency-matched random benchmark, while the crossover does not. Deployed on the BIST100 constituents and aggregated into an equal-weight portfolio, the strategy reaches a Sharpe near 1.70 at roughly half the drawdown of buy-and-hold, with diversification tripling the single-name Sharpe. The HP filter (principled) and LOWESS (empirical) finish effectively tied, the regime filter behaves as a variance reducer rather than a return enhancer, and the edge survives out-of-sample in a walk-forward test. A conviction-sizing and regime-gated-short overlay is a promising extension beyond the brief, pending validation through the full engine.
