# Trend Following Strategy — Design Document

**Project:** Section 4 of `NB_Projects.pdf` — Trend Following
**Reference article:** Harris, R. D. F. and Yilmaz, F. (2008/2009), *A Momentum Trading Strategy Based on the Low Frequency Component of the Exchange Rate*, Journal of Banking and Finance.

> **Status (2026-05-18).** Milestones M1–M5 are complete; M6 (deck) is the remaining deliverable. The strategy was updated 2026-05-17 to deploy full capital per signal flip (multiple concurrent 100k FixedCashSizer-sized lots, ≈10 lots per fresh 1M account) — the old single-position implementation left 90% of capital in cash and produced misleading single-digit CAGRs at the index level. All Phase-1, Phase-2, and walk-forward results in this document are the post-fix numbers. The headline finding is unchanged: HP-direction stays the *principled* primary strategy (theoretical Harris–Yilmaz motivation), and LOWESS-direction *empirically* dominates HP at every aggregation level (per-stock, per-stock MC, equal-weight portfolio). Section 11 captures concrete numbers; Section 10 lays out the remaining plan.

---

## 1. Executive Summary

We design a long/short trend-following strategy for Turkish equities (BIST100 constituents) inspired by Harris & Yilmaz (2008). The core idea is that asset prices contain a slow-moving, persistent **trend component** that can be isolated from short-term noise using low-pass filters; the **direction of that trend** is a tradable signal. We implement and compare four trend extractors — EMA crossover, EMA direction, Hodrick–Prescott (HP) filter, and LOWESS — on the BIST100 Index, select the best, and deploy it on individual stocks with a market-regime filter that gates entries by the BIST100's own HP-filtered trend.

**As-built outcome.** All four strategies were implemented and tuned on the BIST100 Index (Phase 1). Per the Phase-1 sweep, S3 HP (Sharpe 0.82 at λ=14400, window=504) was carried into Phase 2 as the principled primary; S4 LOWESS (Sharpe 0.88 at frac=0.2, window=252) was carried as the empirical benchmark. Phase 2 ran both strategies on 29 cleaned BIST100 constituents in base and regime-gated variants; we report Backtrader full-history metrics per stock, run-length-bootstrap Monte-Carlo Sharpe p-values per stock, an equal-weight cross-sectional portfolio aggregator (DESIGN.md M4), and a 6-config walk-forward driver (HP-only as of 2026-05-18). Headline portfolio numbers: HP base Sharpe 0.895 / CAGR 15.3% over 24y; LOWESS base Sharpe 1.05 / CAGR 18.2%. BIST100 buy-and-hold over the same window returns 23.5% CAGR at Sharpe 0.92 and a -63% MDD; the strategies give up ~5-8pp of nominal CAGR for roughly half the drawdown (-27% portfolio-level).

## 2. Background and Motivation

### 2.1 Why trend following works (briefly)

Classical efficient-market theory predicts no exploitable serial dependence in returns. In practice, persistent autocorrelation in returns has been documented across asset classes and decades (Moskowitz et al. 2012; Asness et al. 2013). The leading behavioral explanations — anchoring, under-reaction to news, herding — and structural explanations — slow flows of capital, leverage cycles, central-bank reaction functions — predict that prices adjust to information gradually rather than instantaneously. A strategy that is long when the slow trend is up, and short when it is down, harvests this drift.

### 2.2 Harris & Yilmaz (2008) — the anchor idea

Harris and Yilmaz study daily exchange rates and decompose each price series into a **low-frequency component** (extracted with the Hodrick–Prescott filter, the same filter used in business-cycle econometrics) and a **high-frequency residual**. Their key claims, which we operationalize for equities:

1. The low-frequency component is highly persistent — when its slope flips sign, it tends to keep the new sign for many periods.
2. Trading rules based on the **sign of the change in the low-frequency component** outperform classical moving-average crossover rules in out-of-sample tests across major currency pairs.
3. The economic case for using a low-pass filter (rather than just an MA) is that the filter is *non-causal* during fitting but causal at the right edge: it gives a smoother, more stable trend estimate with fewer whipsaws.

We take this directly into our equity setting: the **HP-direction** rule is the primary candidate, and EMA crossover, EMA direction, and LOWESS direction are benchmark trend extractors against which it is compared.

## 3. Trade Idea

### 3.1 Hypothesis

> The smoothed trend component of an equity price contains exploitable directional information. Going long when the trend is rising and short when it is falling, on a fixed-cash-per-trade basis, will produce a Sharpe ratio statistically superior to a same-frequency random trading strategy. Adding a market-regime filter (BIST100 own trend) further improves Sharpe by avoiding trades that fight the index.

### 3.2 Strategy variants — single-asset signal

Each variant produces a binary trend state $T_t \in \{+1, -1\}$ on a daily bar.

| # | Name | Indicator | Tunable parameter(s) |
|---|------|-----------|----------------------|
| S1 | EMA Crossover | $\text{EMA}_{\text{fast}} - \text{EMA}_{\text{slow}}$, sign | $(n_{\text{fast}}, n_{\text{slow}})$ |
| S2 | EMA Direction | $\text{sign}(\Delta \text{EMA}_n)$ | $n$ |
| S3 | HP Direction *(primary)* | $\text{sign}(\Delta \text{HP}_\lambda(P))$ | $\lambda$ |
| S4 | LOWESS Direction | $\text{sign}(\Delta \text{LOWESS}_{\text{frac}}(P))$ | $\text{frac}$ |

Trade rules (all variants, same convention):
- Enter long when $T_t$ flips from $-1$ to $+1$.
- Close long and enter short when $T_t$ flips from $+1$ to $-1$.
- Always-in-the-market: no flat state in the base strategy.

Position sizing uses the course-mandated `FixedCashSizer` (100,000 TL notional per *order*); orders are `bt.Order.Market`; commission = 0. **The strategy holds multiple concurrent 100k lots so total capital is fully deployed when in the market** — on a signal flip we fire `floor(broker_equity / cash_per_trade)` orders in one bar (≈10 orders for a fresh 1M account), close any opposite position first, and continue holding the stacked lots until the next flip. This matches the brief's per-trade cap while avoiding the cash-drag pathology of single-lot trend-following on a single instrument.

### 3.3 Why HP is the primary candidate

- It is the strategy whose *theoretical motivation* matches Harris & Yilmaz (2008) directly.
- It has a single, interpretable hyperparameter $\lambda$ that controls the trend/noise tradeoff (larger $\lambda$ → smoother, slower trend).
- Course constraint: the HP filter implementation is given in the brief and is non-iterative (single linear solve per refit).

A practical caveat we address explicitly: the HP filter as written is **applied to the entire series**, which leaks future information at any historical bar. For backtesting we re-fit HP on a rolling/expanding window so that $\text{HP}_t$ uses only data $\le t$. The same restriction applies to LOWESS. Both rolling wrappers (`rolling_hp_trend`, `rolling_lowess_trend`) live in `src/features/`.

**Honest reporting (post-implementation).** S4 LOWESS narrowly beat S3 HP on the Phase-1 BIST100 sweep (Sharpe 0.88 vs 0.82) and then beat it more clearly on Phase 2 (mean per-stock Sharpe 0.454 vs 0.419; portfolio Sharpe 1.05 vs 0.90). We keep HP as the *primary* strategy of the writeup because the choice is theoretically motivated by Harris–Yilmaz and the spread is small enough that a deliberate, single-criterion theoretical pick is defensible. LOWESS is carried through Phase 2 as a head-to-head benchmark and reported alongside HP — not silently swapped in.

### 3.4 Regime filter (Phase 2)

After picking the best of {S1..S4} on the BIST100 Index, we deploy it on individual stocks **gated by the BIST100's own HP trend**:

- Long entry on stock $i$ requires: stock signal long **AND** BIST100 close > HP-filtered BIST100 close.
- Short entry on stock $i$ requires: stock signal short **AND** BIST100 close < HP-filtered BIST100 close.
- Exits ignore the regime filter (we don't want to trap positions when the regime flips).

The economic story: trade with the index, not against it. This is a standard top-down/bottom-up combination, also present in the Harris–Yilmaz spirit of using a single dominant low-frequency factor.

## 4. Universe, Data, and Time Period

- **Universe (Phase 1):** BIST100 Index daily close (single series, used for hyperparameter selection). Ticker `XU100.IS` on Yahoo Finance.
- **Universe (Phase 2):** All BIST100 constituents — current membership taken at project start, no rebalancing of the universe (per course brief simplifications).
- **Data fields:** OHLCV daily, in TL.
- **Period:** All available history — adjusted to a common start once data is loaded; in practice ~2003–2026 for most names.
- **Initial capital:** 1,000,000 TL. **Per-trade cash:** 100,000 TL. **Commission:** 0.

**As-built data outcomes** (see CLAUDE.md for the full table):

- The Phase-2 universe in `config.BIST100_CONSTITUENTS` is a curated **31-name list** of well-known BIST blue chips (not the official 100). `data/clean/` contains **29 of 31** post-yfinance ingest; `SODA.IS`, `KOZAL.IS`, `KOZAA.IS`, `TRALT.IS`, `TRMET.IS` failed as "possibly delisted / quote not found" and need ticker remapping before they can be added back. The official current BIST100 membership should still be substituted before final submission, but is treated as out-of-scope for the model — the brief accepts survivorship bias on a snapshot of current membership.
- Cleaned data spans 2003-01-01 to 2026-05-11, panel shape 5997 × 29. PGSUS has the shortest history at 3273 bars (later IPO); all 29 still clear the `MIN_HISTORY_DAYS=750` floor in `clean.py`.
- **Currency redenomination fix + corporate-action gaps.** yfinance does not back-adjust for the 2005-01-03 Turkish TL 1M-to-1 redenomination, leaving a ~1000× single-day price jump in every Turkish equity series (but not the index). `src/data/clean.py::_adjust_splits` detects any single-day price ratio > 3× (or < 1/3×) and rescales all prior OHLC by that ratio, idempotently. The 3× threshold (lowered from 50× on 2026-05-17) also catches the MGROS 2009-08-04 corporate-action gap (factor 4.04×) that the old 50× threshold missed and that caused -14358% MDD blowups on the short side of the post-fix full-deployment strategy. The largest legitimate single-day move in our universe is HEKTS +44%, well below the 3× threshold. All `data/clean/*.parquet` were regenerated with this adjustment.

## 5. Data Pipeline

### 5.1 Pipeline stages

```
  ┌───────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │  raw vendor   │ →  │  ingestion   │ →  │  cleaning    │ →  │  indicators  │ →  │  backtest    │
  │  CSV / API    │    │  + schema    │    │  + alignment │    │  + caching   │    │  Backtrader  │
  └───────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                              │                    │                    │                    │
                              ▼                    ▼                    ▼                    ▼
                       data/raw/           data/clean/          data/features/        results/
                       (immutable)         (parquet)            (parquet)             (parquet + json)
```

### 5.2 Stage details

**Ingestion (`src/data/ingest.py`).** Read raw vendor files (CSV exports from the course data pack or a Yahoo/`yfinance` pull for `.IS` tickers if needed). Validate schema: `date, open, high, low, close, volume`. Persist as immutable parquet under `data/raw/{ticker}.parquet`.

**Cleaning (`src/data/clean.py`).** Per ticker: drop duplicate timestamps; keep the trading-day calendar (no calendar-forward-fill); apply `_adjust_splits` to back-adjust pre-corporate-action OHLC when a single-day ratio > 50× (or < 1/50×) is detected — this catches the 2005-01-03 TL redenomination and any unhandled real stock split. Drop tickers whose history is shorter than `MIN_HISTORY_DAYS=750`. Output to `data/clean/{ticker}.parquet`.

**Cross-sectional alignment (`src/data/panel.py`).** Build a wide panel `prices.parquet` of shape `(date × ticker)` for the BIST100 universe, plus the `bist100.parquet` index series. This panel feeds both the indicator stage and any cross-sectional analysis.

**Indicator computation (`src/features/`).** One module per trend extractor:
- `features/ema.py` — EMA series, `s1_crossover`, `s2_direction` (causal by construction).
- `features/hp.py` — bare `hp_filter`, causal `rolling_hp_trend(prices, lam, window)`, and `hp_direction_signal` for S3.
- `features/lowess.py` — bare `lowess_trend`, causal `rolling_lowess_trend(prices, frac, window)`, and `lowess_direction_signal` for S4.
- `features/regime.py` — `bist_regime_flag(bist_close, lam, window)` returning +1/-1/0 per bar.

**Deviation from plan.** Feature parquet caching keyed by parameter signature (mentioned in this section originally) was *not* implemented. In-script computation is fast enough for our grid sizes (Phase-1 sweep ≈17 min, walk-forward ≈9 min on HP), and adding a cache layer would mostly help LOWESS. Left as a follow-up if we extend the grids; see Section 10.

**Backtest (`src/backtest/`).** Backtrader strategies wrap each indicator and consume a precomputed trend series via a `PandasData` feed extension that adds a `trend` line (and `regime` line for the gated variant). The strategy reads `self.data.trend[0]` and acts on sign flips. `TrendStrategy` (always-in-market) and `TrendRegimeStrategy` (regime-gated entries, exits ignore the regime) are the two production classes. `src/backtest/runner.py::run_backtest` wires up Cerebro with `FixedCashSizer`, `commission=0`, and analyzers that record per-bar equity and per-trade pnl.

### 5.3 Look-ahead and survivorship

- HP and LOWESS are global smoothers. **Always** use the rolling-window wrappers in backtests; the bare functions are for plotting / EDA only.
- Universe is *current* BIST100, which introduces survivorship bias. We disclose this in the writeup; correcting it is out of scope per course assumptions.
- Indicator values produced at bar $t$ may only inform orders submitted on bar $t+1$ (Backtrader's `next()` convention with market orders gives this naturally).

### 5.4 Reproducibility

- Each pipeline stage is a pure function of its inputs + a parameter dict; a `params.json` is written next to every output.
- A single `make data` (or `python -m src.data.build`) command rebuilds the full chain from `data/raw/` forward.
- Random seeds for Monte Carlo are fixed in `config.py`.

## 6. Backtesting and Evaluation

### 6.1 Mechanics

- Engine: Backtrader with `FixedCashSizer(cash_per_trade=100_000)`, `setcash(1_000_000)`, `setcommission(commission=0)`, `bt.Order.Market`.
- Bar frequency: daily.
- One Cerebro instance per (strategy, ticker) pair in Phase 2 to keep P&L attribution clean; aggregation done in post-processing.

### 6.2 Performance metrics

Reported for both the base strategy and the strategy after each enhancement (parameter optimization, walk-forward, regime filter):

1. Net P/L (TL and %)
2. CAGR
3. Annualized volatility
4. Sharpe ratio
5. Sortino ratio
6. Max drawdown (% and duration)
7. Calmar ratio
8. Win rate / average win / average loss
9. Number of trades, average holding period
10. Turnover

### 6.3 Optimization plan

- **Phase 1 — in-sample tuning on BIST100 Index:** grid search per strategy via `src/backtest/sweep.py::default_specs()`.
  - S1: `n_fast ∈ {5,10,20,30}`, `n_slow ∈ {50,100,200}` with `n_fast < n_slow`.
  - S2: `n ∈ {10,20,50,100,200}`.
  - S3: `λ ∈ {100, 1600, 14400, 129600}`, rolling window `W ∈ {252, 504, 1260}`.
  - S4: `frac ∈ {0.05, 0.1, 0.2, 0.3}`, rolling window same as S3.
- **Selection rule:** highest Sharpe, with a minimum-trade-count floor (≥ 30 round trips) to avoid degenerate configs.
- **Phase 2 — walk-forward** (best strategy from Phase 1, applied per stock): training window 3y, test window 1y, step 1y; re-pick parameters on each training window. **Deviation:** the `window=1260` configs (5y warmup) are pruned in the walk-forward grid because they leave too little post-warmup room for fold-level retraining (`src/eval/run_walkforward.py::WF_HP_GRID_FULL`).

**Phase-1 winners (BIST100 Index, full history, 41 configs, ~17 min, full-deployment strategy):**

| Strategy | Best params | Sharpe | Trades | CAGR | MDD |
|---|---|---|---|---|---|
| S4 LOWESS direction | frac=0.2, window=252 | 0.876 | 560 | 19.89% | -46.4% |
| S3 HP direction *(primary)* | λ=14400, window=504 | 0.818 | 576 | 17.52% | -43.3% |
| S2 EMA direction | n=20 | 0.742 | 634 | 16.71% | -47.1% |
| S1 EMA crossover | n_fast=5, n_slow=100 | 0.719 | 126 | 15.60% | -52.8% |

(Reference: BIST100 buy-and-hold over the same 23.3-year window is **136.3× total, CAGR 23.46%, Sharpe 0.92, MDD -63.4%**. The strategies underperform B&H on absolute return but cut drawdown by ~15-20pp.)

### 6.4 Statistical significance

Per the course brief: Monte Carlo p-value on Sharpe ratio. For each strategy variant we generate $N=1000$ random strategies that match the empirical (a) run-length distribution (≈trade frequency × holding period) and (b) long/short proportion, and compute the fraction whose Sharpe exceeds ours. Histogram + one-sided p-value are reported. The fast vectorized simulator in `src/eval/montecarlo.py` is used for both the strategy and the matched-random pool so the comparison is apples-to-apples; this means the Sharpes here differ slightly from the Backtrader sweep above because the fast sim ignores fixed-shares scaling and warmup truncation.

**Phase-1 MC (BIST100 Index, N=1000):**

| Strategy | Sharpe | MC mean | MC q95 | p-value |
|---|---|---|---|---|
| S4 LOWESS | 0.87 | 0.16 | 0.50 | 0.000 |
| S2 EMA dir | 0.86 | 0.19 | 0.54 | 0.002 |
| S3 HP | 0.80 | 0.14 | 0.47 | 0.001 |
| S1 EMA xover | 0.61 | 0.29 | 0.67 | 0.086 |

S2/S3/S4 all reject random at p<0.005. S1 (classical MA crossover) is only marginal at 5% — consistent with the Harris–Yilmaz finding that LF-filter direction beats MA crossover.

### 6.5 Phase-2 results (29 stocks, redenomination-adjusted)

Three views per strategy (HP and LOWESS), both base and BIST100-HP regime-gated variants.

**(a) Single-config full-history per stock** (`src/eval/run_phase2.py`):

| | HP base | HP regime | LOWESS base | LOWESS regime |
|---|---|---|---|---|
| mean Sharpe | 0.419 | 0.372 | 0.454 | 0.421 |
| mean CAGR | 9.82% | 7.56% | 11.37% | 9.11% |
| mean MDD | -70.4% | -66.6% | -72.9% | -65.7% |
| total trades | 16,717 | 11,630 | 17,226 | 11,592 |
| Sharpe>0 | 26/29 | 28/29 | 27/29 | 28/29 |

(Per-stock MDDs in the -50% to -90% range are the cost of always-in-market full-capital deployment on individual equities, with shorts that can bleed against nominal drift. The portfolio aggregation in (c) dilutes this to ~-27%.)

**(b) Per-stock MC** (`src/eval/run_phase2_mc.py`, N=1000, base-only because the regime variant's sticky-flat path-dependence is not modelled by the fast simulator):

| | HP | LOWESS |
|---|---|---|
| p<0.01 | 8/29 | 10/29 |
| p<0.05 | 16/29 | 18/29 |
| mean strat Sharpe | 0.436 | 0.465 |

**(c) Equal-weight cross-sectional portfolio** (`src/eval/run_phase2_portfolio.py`, uses the path-aware `effective_position` state machine so the regime variant *is* reproduced correctly here):

| | HP base | HP regime | LOWESS base | LOWESS regime |
|---|---|---|---|---|
| Sharpe | 0.895 | 0.823 | 1.049 | 0.958 |
| CAGR | 15.29% | 11.94% | 18.15% | 13.74% |
| ann vol | 17.7% | 15.1% | 17.3% | 14.6% |
| MDD | -26.8% | -25.1% | -28.5% | -25.5% |
| Total return (24y) | 29.6× | 14.7× | 52.9× | 21.4× |

**(d) Walk-forward** (`src/eval/run_walkforward.py`, S3 HP only as of 2026-05-18; LOWESS walk-forward is planned in Section 10 but not run): 6-config grid (λ ∈ {1600, 14400, 129600} × window ∈ {252, 504}), 3y train / 1y test / 1y step → 477 fold rows per variant. Mean test Sharpe 0.358 (base) / 0.294 (regime); 66%/64% of folds Sharpe>0; mean test MDD ≈ -25%.

**Findings.** (1) Out-of-sample Sharpe is materially below the Phase-1 in-sample index Sharpe — the honest cost of per-stock noise and fold-level retraining, not a bug. (2) The regime filter acts as a *variance reducer*, not a *Sharpe enhancer*: it cuts trade count ~30%, tightens MDD, rescues the worst-of-base tickers (THYAO, BIMAS, ENKAI), but slightly underperforms on aggregate Sharpe. (3) LOWESS dominates HP at every aggregation level — per-stock, MC significance, and portfolio — but the gap is modest enough that keeping HP as the principled primary is honest reporting rather than benchmark-swapping.

## 7. Risk Management and Operational Constraints

The course brief asks us to ignore commissions and use a fixed cash sizer; we follow that. Beyond what the brief mandates, we add two sanity checks reported in the writeup but not changing trade decisions:

- **Single-name concentration:** at any time, a strategy holds at most $\lfloor \text{cash}_{\text{available}} / 100{,}000 \rfloor$ positions, so concentration is bounded by capital.
- **Regime-filter dead time:** report the fraction of bars during which the regime filter blocks new entries — if too high (> 70%), the filter is over-restrictive and we report this as a finding.

## 8. Project Structure

As-built layout (deviations from the originally-planned skeleton are flagged inline):

```
ec581-project/
├── DESIGN.md                  # this file
├── CLAUDE.md                  # development memo for Claude
├── NB_Projects.pdf            # course brief
├── requirements.txt           # scientific-Python + backtrader + yfinance + pyarrow
├── config.py                  # paths, seeds, capital constants, BIST100 universe
├── data/                      # gitignored
│   ├── raw/                   # yfinance ingest, immutable
│   ├── clean/                 # per-ticker cleaned parquet (redenomination-adjusted)
│   └── panel/                 # wide prices/volumes panel + bist100 OHLCV
├── results/                   # parquet outputs (Phase-1 sweep, Phase-2 base/regime/MC/portfolio/walk-forward)
└── src/
    ├── data/
    │   ├── ingest.py          # yfinance + CSV fallback → data/raw/{ticker}.parquet
    │   ├── clean.py           # dedupe, sort, _adjust_splits, MIN_HISTORY filter → data/clean/
    │   ├── panel.py           # wide prices/volumes panels + BIST100 OHLCV → data/panel/
    │   └── build.py           # CLI: `python -m src.data.build [--smoke] [--source ...]`
    ├── features/
    │   ├── ema.py             # EMA, S1 crossover, S2 direction (causal)
    │   ├── hp.py              # bare hp_filter + rolling_hp_trend + hp_direction_signal (S3)
    │   ├── lowess.py          # bare lowess_trend + rolling_lowess_trend + lowess_direction_signal (S4)
    │   └── regime.py          # BIST100 vs HP regime flag (§3.4)
    ├── backtest/
    │   ├── sizers.py          # FixedCashSizer (verbatim from brief p.3)
    │   ├── feeds.py           # PandasData + trend (+regime) line
    │   ├── strategies.py      # TrendStrategy + TrendRegimeStrategy
    │   ├── runner.py          # Cerebro orchestration + analyzers
    │   └── sweep.py           # Phase-1 S1..S4 grid search (DESIGN §6.3 grids)
    └── eval/
        ├── metrics.py             # CAGR, Sharpe, Sortino, MDD, Calmar, win rate, turnover
        ├── montecarlo.py          # run-length-bootstrap Sharpe MC (fast vectorized sim)
        ├── walkforward.py         # make_folds + walk_forward driver
        ├── portfolio.py           # effective_position state machine + equal-weight aggregator
        ├── strategies.py          # registry: maps "hp"/"lowess" → (signal_fn, trend_fn, params, stub)
        ├── sizing.py              # conviction-sizing / short-side overlay sub-model (§10.7)
        ├── run_walkforward.py     # CLI: per-stock walk-forward (HP + regime)
        ├── run_phase2.py          # CLI: per-stock single-config base/regime
        ├── run_phase2_mc.py       # CLI: per-stock Monte-Carlo Sharpe p-value
        ├── run_phase2_portfolio.py # CLI: equal-weight cross-sectional portfolio aggregator
        └── run_phase2_sizing.py   # CLI: conviction-sizing ablation scoreboard (§10.7)
```

**Planned but not yet built** (see Section 10):

```
src/plots/                    # shared figure helpers reused across notebooks
notebooks/
├── 01_eda.ipynb
├── 02_index_strategy_comparison.ipynb
├── 03_walk_forward.ipynb
└── 04_regime_filter.ipynb
data/features/                # parameter-keyed feature parquet cache (deferred — see §5.2 deviation)
```

## 9. Milestones and Deliverables

| # | Milestone | Output | Status |
|---|-----------|--------|--------|
| M1 | Data pipeline end-to-end on BIST100 + 5 stocks | `data/clean/`, `data/panel/` populated; smoke-test confirmed | **Done.** Index + 29 constituents cleaned and panelled, redenomination-adjusted. |
| M2 | Four index-level strategies + base backtests | Metrics tables; per-strategy results parquet | **Done.** `results/phase1_sweep.parquet`. |
| M3 | Hyperparameter optimization, single-strategy selection | Selected strategy + parameter justification | **Done.** S3 HP (λ=14400, window=504) primary; S4 LOWESS (frac=0.2, window=252) carried as benchmark. |
| M4 | Walk-forward on individual stocks; aggregated portfolio curve | Per-stock metrics; portfolio-level equity/Sharpe | **Done.** `results/phase2_walkforward_*` (HP); `results/phase2_portfolio_*` (HP + LOWESS). LOWESS walk-forward pending — see §10. |
| M5 | Regime filter and final comparison | Side-by-side base/regime metrics + Monte-Carlo p-values | **Done.** Phase-1 MC (`results/phase1_mc.parquet`) + Phase-2 per-stock MC (`results/phase2_mc_*`). Regime-variant and portfolio MC not yet run — see §10. |
| M6 | Presentation deck (30-min slot per brief) | Slides + final notebook | **Pending.** Notebooks 01–04 not yet built; deck not yet drafted. See §10. |

## 10. Status and Remaining Work

What's left, in roughly the order it should be tackled. Items are scoped so each is a self-contained piece of work; nothing here blocks the others, so they can be parallelized or reordered if useful.

### 10.1 Notebooks (M6 enablers) — **required for the writeup**

Four notebooks under `notebooks/`, all sourced from the existing parquets in `results/` (no recomputation). Each should produce 2–4 figures + 1–2 tables ready to lift into slides.

1. **`01_eda.ipynb`** — universe inventory: per-ticker history length, common date range, missing-bar counts, log-price overview, the 2005 redenomination jump before/after `_adjust_splits` (one figure on a single name like THYAO illustrating the fix). Loads from `data/clean/*.parquet`.
2. **`02_index_strategy_comparison.ipynb`** — Phase-1 on BIST100: equity curves of S1..S4 at their best params overlaid against buy-and-hold; the metrics table from `results/phase1_sweep.parquet`; the Monte-Carlo histogram for each strategy with the strategy Sharpe as a vertical line (`results/phase1_mc_distributions.parquet`). This is the "we picked HP" notebook.
3. **`03_walk_forward.ipynb`** — Phase-2 HP walk-forward: per-fold test Sharpe heatmap (ticker × fold), parameter selection frequency stacked bar (which λ wins where), aggregate metrics table by ticker. Loads `results/phase2_walkforward_{base,regime}.parquet`.
4. **`04_regime_filter.ipynb`** — Phase-2 final comparison: per-ticker side-by-side Sharpe (base vs regime); the portfolio equity curve (`results/phase2_portfolio_equity.parquet`) with HP base, HP regime, LOWESS base, LOWESS regime, and BIST100 buy-and-hold overlaid; the per-stock MC significance bar chart; the head-to-head HP vs LOWESS table.

### 10.2 `src/plots/` shared helpers — **light infrastructure for the notebooks**

A small module (one file is fine: `src/plots/figures.py`) factoring out the figures the four notebooks repeat. Suggested helpers, each ~20–30 lines:

- `equity_curve_overlay(curves: dict[str, pd.Series], log: bool = True) -> plt.Figure`
- `mc_distribution(strat_sharpe: float, mc_sharpes: np.ndarray, label: str)`
- `per_ticker_sharpe_bar(df, sig_p_values=None)` — bars with significance markers from a Phase-2 MC parquet.
- `walkforward_heatmap(df)` — ticker × fold heatmap of test Sharpe.
- `param_selection_stack(df)` — stacked bar over folds.
- `drawdown_curve(equity)` — underwater plot.

Keep matplotlib-only; no seaborn dependency. This module exists so notebooks stay short and the deck figures are visually consistent.

### 10.3 LOWESS walk-forward (M4 completeness) — **optional benchmark**

The HP walk-forward driver (`run_walkforward.py`) is parametrized over a signal function but currently hardcodes `hp_direction_signal`. Two ways to add LOWESS:

(a) Easy: copy `WF_HP_GRID_FULL` into a `WF_LOWESS_GRID_FULL` (frac ∈ {0.1, 0.2, 0.3} × window ∈ {252, 504}) and add a `--strategy {hp,lowess}` arg that dispatches through `src/eval/strategies.py`, mirroring the refactor already done for `run_phase2.py` / `run_phase2_mc.py` / `run_phase2_portfolio.py`.
(b) Run the LOWESS walk-forward — expect ~25–40 min on the full universe (LOWESS is ~5–10× slower per bar than HP because each rolling-window refit can't be pre-factored).

Worth doing if time permits: it closes the HP-vs-LOWESS head-to-head out-of-sample. Not strictly required by the brief.

### 10.4 Regime-variant MC and portfolio MC — **strengthens M5**

Two extensions to `src/eval/montecarlo.py`, both unblocked by the `effective_position` state machine added in `src/eval/portfolio.py`:

1. **Regime-variant per-stock MC.** Current `monte_carlo_sharpe` uses a path-agnostic fast simulator that can't model the regime variant's sticky-flat behaviour. Build `monte_carlo_sharpe_regime(prices, signal, regime, n_iter)` that draws matched-randomness signals from the same run-length pool, runs each through `effective_position(rand_sig, regime)`, and computes `strategy_returns(prices, eff_pos)`. Then per-stock MC tables can be reported for both base *and* regime.
2. **Portfolio-level MC.** The portfolio Sharpes (HP 0.86 / LOWESS 1.01) deserve their own significance test against a matched-random portfolio. Each MC iteration: for each ticker, draw an independent matched-random signal; build the effective-position panel; equal-weight average via `portfolio_returns`; compute portfolio Sharpe. This is the cleanest one-number significance statement for the deck.

Both are < 100 LOC each. The path-aware simulator is slower than the fast one (per-bar state loop in Python), so target N=500 for portfolio MC if N=1000 turns out costly; resolution at p<0.01 is fine with N=500.

### 10.5 Other deferred items (low-priority)

- **Official BIST100 membership.** `config.BIST100_CONSTITUENTS` is a curated 31-name list; the writeup should state this explicitly and ideally swap in the official current 100 membership (the 5 yfinance-unavailable tickers will likely still drop out). The model code is membership-agnostic — just edit the constant.
- **Feature parquet cache.** `data/features/` was planned for parameter-keyed indicator caching. Not built; the per-script computation is fast enough at our grid sizes. Becomes relevant if we extend sweeps materially or run many LOWESS configurations.
- **Sector grouping.** `config.py` has placeholder sector comments. A natural per-sector portfolio aggregation could be added with one extra column on the summary and a `groupby` in the portfolio aggregator. Defer unless asked by the prof.
- **Trim WF grid.** Walk-forward parameter selection picks `window=252` in 466/476 folds; `window=504` is essentially dead weight. Could drop it on the next WF run for ~30% speedup.

### 10.6 M6 — final deck

The 30-minute presentation slot per the brief. Outline:

1. Background / motivation (Harris–Yilmaz, low-frequency component) — 2 slides.
2. Strategy family + causal-wrapper caveat — 2 slides.
3. Phase-1 selection (table + MC histograms) — 3 slides.
4. Phase-2 architecture + walk-forward + regime — 4 slides.
5. Final HP vs LOWESS head-to-head + portfolio curve — 3 slides.
6. Conclusions + limitations (survivorship, no-cost assumption, etc.) — 2 slides.

Total ~16 slides → fits comfortably in 30 min. Each notebook in §10.1 produces 2–3 of these slides' figures.

### 10.7 Conviction-sizing / short-side overlay (extension beyond the brief) — **sub-model built, validation pending**

A sizing/selection layer that sits on top of the equal-weight cross-sectional portfolio (§6.5(c)), motivated by a pressure-test of the Part-4 sizing brainstorm (see `progress.md` Parts 4–5). The pressure-test killed the *risk-based* branch and found two *causal* levers that move portfolio Sharpe:

1. **Conviction sizing** — replace the binary ±1 signal with a vol-normalized trend-slope magnitude, so a just-flipped weak trend gets less capital than an established steep one. ~+0.25 Sharpe at neutral drawdown in the reconstruction, robust across four conviction definitions. It works because it is a *within-name, time-series* tilt (weak slopes are universally noisier), not a cross-sectional bet on which names are good (which does not persist — year-over-year per-name Sharpe rank correlation ≈ 0.07).
2. **Short-side policy** — the short leg is a standalone money-loser (Sharpe ≈ −0.80, fighting a positive-drift inflationary index) but a crash hedge. Dropping it (`long_only`) lifts Sharpe ~+0.16 at deeper MDD; gating shorts through the §3.4 BIST100 regime filter (`regime_short`) is the middle ground.

**Empirically dead, deliberately not built:** inverse-vol, vol-target, risk-parity, fractional Kelly (= `Σ⁻¹μ` tangency, inherits a noisy-Σ̂ *and* noisy-μ̂ problem), and cross-sectional name selection. The whole universe is essentially one TL-inflation factor (raw-return effective bets ≈ 9 of 83); the long/short *signing* already harvests the diversification, and trailing covariance is too noisy to invert out-of-sample (causal LOWESS risk-parity collapses to Sharpe 0.45). This negative result is itself reportable.

**As-built (`src/eval/sizing.py`, `src/eval/run_phase2_sizing.py`).** The sub-model operates on panels like `portfolio.py`: it builds a signed **unit-gross** weight panel from each name's held position × a non-negative conviction magnitude, then combines it with the next bar's raw return. `run_phase2_sizing.py` emits a 9-row ablation scoreboard (binary baseline + 4 conviction definitions + 2 short policies + stacks) with a `sharpe_delta` column, written to `results/phase2_sizing_{stub}_{scoreboard,equity}.parquet`. CLI: `python main.py phase2-sizing [--smoke] [--strategy hp|lowess]`. `StrategySpec` was extended with a `trend_fn` (rolling HP/LOWESS *level*) so the slope magnitude is available.

**Course-constraint caveat.** Conviction sizing and short-dropping both change per-name exposure away from the binary signal + `FixedCashSizer` + long/short mandate (CLAUDE.md "do not change without asking"). This is framed strictly as an **overlay** reported as a delta against the brief-compliant equal-weight baseline; it needs Dr. Yuksel's sign-off before becoming a headline result.

**Validation status.** Smoke (5 names) reproduces the directional finding (conviction +0.16–0.21 Sharpe; `long_only` / `regime_short` lift further; turnover ~doubles, free at `commission=0`). Still pending: full-universe scoreboard generation, re-validation through the real `TrendStrategy` Backtrader engine (the reconstruction's absolute Sharpes are inflated by unit-gross daily rebalance — only deltas are meaningful), and a walk-forward / sub-period split (2018–2026 is largely one regime).

## 11. Open Questions

1. ~~Do we have adjusted-close vendor data for the full BIST100 history?~~ **Resolved.** yfinance gives auto-adjusted data for normal splits/dividends but does *not* backfill the 2005-01-03 TL redenomination; we apply our own back-adjustment in `clean.py::_adjust_splits` (threshold 50×).
2. ~~How frequently do we re-fit HP/LOWESS in the rolling wrapper?~~ **Resolved.** Every bar. Phase-1 sweep in ~17 min on a single core; walk-forward ~9 min for HP. LOWESS portfolio takes ~25 min — slow but tractable.
3. ~~Walk-forward window sizes?~~ **Resolved.** 3y train / 1y test / 1y step with the `window=1260` configs pruned (5y warmup too long for fold-level retraining). 17 folds per ticker on the longest histories.
4. **Open.** Should `--strategy lowess` walk-forward be run before the deck? (§10.3.) Cost ~30 min; not required by the brief.
5. **Open.** Regime-variant per-stock MC and portfolio-level MC (§10.4) — worth the ~1–2h to implement before the deck for the cleanest statistical statement, or skip and lean on the per-stock base MC + portfolio CAGR?
6. **Open.** Conviction-sizing / short-side overlay (§10.7) — the sub-model is built but only smoke-tested. Run the full-universe scoreboard and re-validate through the Backtrader engine before the deck (as an "extension beyond the brief" slide), or leave it as a documented design with the negative risk-based result? Needs the constraint conversation with Dr. Yuksel either way.

---

*Prepared as a course-project design document. Sections 1–9 describe the intended design; §10 plans remaining work; §11 tracks open questions. Numbers in tables (parameter grids, thresholds) were starting points — final selections came out of Phase-1 optimization (§6.3) and are recorded throughout.*
