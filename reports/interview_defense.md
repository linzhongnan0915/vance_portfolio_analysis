# Interview Defense Guide

Companion to [methodology.md](methodology.md) and [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md).
This repo is an **educational** Vance-style ETF risk research project, not a live product.

**Final policy recommendation:** Fixed + SHY Version A (QQQ 25%, SPY 25%, DIA 25%, GLD 15%, TLT 5%, SHY 5%).

---

## 1. 60-Second Project Pitch

"I built a staged portfolio risk management research pipeline on a publicly discussed ETF-style allocation. The work is not about picking the highest backtest Sharpe in isolation. It asks whether a fixed policy portfolio can survive out-of-sample walk-forward testing, how it behaves in COVID and 2022 stress windows, and whether a small SHY sleeve fixes a specific failure mode without breaking wealth-management constraints.

The engine lives in modular Python under `src/`: daily returns, monthly rebalance, three weight rules (fixed, inverse-volatility, minimum-variance), and a walk-forward loop with a strict twelve-month training window and one-month test step. Later stages add mandate scoring, reconciliation, and overfitting checks. A Streamlit dashboard under `dashboard/` only reads precomputed CSVs from `output/` so presentation stays separate from research logic. The conclusion is a defendable risk policy choice, not a single metric leaderboard."

---

## 2. Research Question

**Core question:** Can an approximate Vance-style ETF mix be operated as a **documented, repeatable risk policy**, and does a **5% SHY** allocation improve resilience in 2022-style regimes without violating mandate limits?

**Why this is risk management, not return chasing:**

| Angle | What the project tests |
|-------|-------------------------|
| Policy stability | Fixed weights vs rules that drift into cash-like allocations |
| Out-of-sample discipline | Walk-forward months from 2010 onward, not one in-sample fit |
| Regime risk | COVID liquidity shock vs 2022 rate shock and bond-equity correlation |
| Mandate fit | Equity budget, SHY cap, diversification after optimization |
| Governance | Stage 6 audit reconciles weights, turnover, and metrics across stages |

A return-chasing framing would stop at "highest Sharpe wins." This repo explicitly rejects that when mandate analysis shows the winner is ~73% SHY under uncapped inverse-volatility + SHY.

---

## 3. Why This Portfolio / Universe

Educational approximate replication of a multi-asset ETF discussion (not verified disclosed holdings).

| Ticker | Role in the research design |
|--------|------------------------------|
| **QQQ** | Growth / Nasdaq equity risk |
| **SPY** | Core US large-cap beta; also the stated benchmark |
| **DIA** | Broader large-cap diversification vs single-factor QQQ |
| **GLD** | Inflation / crisis diversifier (imperfect, but explicit sleeve) |
| **TLT** | Long-duration Treasury hedge (tested; fails in 2022 sample) |
| **SHY** | Short-duration Treasury **defensive sleeve** (Stage 5B onward) |

Baseline fixed weights before SHY: 25% / 25% / 25% / 15% / 10% on QQQ / SPY / DIA / GLD / TLT.

The universe is **fixed ex ante** (no dynamic ETF listing filter). That is a simplification you must disclose in interviews.

---

## 4. Why SPY Benchmark

**Why SPY is reasonable here:**

- It is the largest liquid US equity proxy and easy for interviewers to interpret.
- Much of the portfolio risk budget is US equity (QQQ + SPY + DIA = 75% in Version A).
- SPY enables standard relative metrics: beta, tracking error, information ratio.

**Limitations you should state proactively:**

| Issue | Interview line |
|-------|----------------|
| Mixed-asset portfolio vs equity benchmark | SPY ignores gold and bond sleeves; low beta does not mean "low risk" in absolute terms |
| Double-counting | SPY is both a **holding** and the **benchmark** in Version A (25% weight) |
| Policy vs peer benchmark | A custom policy benchmark (e.g. 60/40 or fixed-policy blend) may be more fair for mandate reviews |
| Total return only | Benchmark is price/total-return proxy, not net of fees, taxes, or cash drag |

If challenged, say: "SPY is a communication benchmark for equity-centric risk metrics; for production I would add a policy benchmark and possibly a 60/40 or inflation-aware peer."

---

## 5. Backtesting Design

| Parameter | Setting in this repo |
|-----------|----------------------|
| Return | Daily simple return on adjusted close |
| Rebalance | First trading day of each month; weights drift intra-month |
| Analysis start | 2009-01-01 (`ANALYSIS_START`) |
| Training window | 12 months of daily returns ending **before** each test month |
| Test window | One calendar month of realized returns per step |
| Strategies | `fixed_baseline`, `inverse_volatility`, `minimum_variance` |
| Min-var constraint | Long-only, 40% max weight per asset |

**Walk-forward logic (spoken):**

"For each test month, I only estimate volatilities or covariance using data strictly before that month's first trading day. I set weights for that month, simulate daily portfolio returns until month-end, then roll forward. The decision log records `training_end` and `testing_start`; tests assert `training_end < testing_start` for every row."

**Look-ahead controls:**

- Parameter estimation window ends at the last train day before `testing_start`.
- In-sample baseline stages (Stage 1) are labeled separately from walk-forward headlines.
- `tests/test_no_lookahead.py` enforces timing on synthetic data (CI does not need network or CSV cache).

---

## 6. Why SHY

**2022 problem in this project:** Equities and **long-duration Treasuries (TLT)** fell together as rates rose. The usual "bonds hedge stocks" story broke in that regime.

**Why SHY (short-duration Treasuries):**

- Lower duration than TLT -> less rate sensitivity in a hiking cycle.
- Acts as a **liquidity and volatility dampener**, not a return engine.
- Version A only allocates **5%** SHY, funded by cutting TLT from 10% to 5%, preserving the equity sleeve.

**What SHY is not:**

- Not a replacement for strategic equity risk in a growth mandate.
- Not a guarantee against future correlation spikes.
- Not a substitute for a full liability-driven or ALM framework.

---

## 7. Why Not Highest Sharpe

Walk-forward headline metrics (approximate, 2010 through latest):

| Strategy | Ann. return | Ann. vol | Sharpe | Max DD |
|----------|-------------|----------|--------|--------|
| Fixed | ~13.7% | ~13.1% | ~1.04 | ~-24% |
| Inverse-vol | ~11.7% | ~10.3% | ~1.14 | ~-23% |
| Min-variance | ~12.1% | ~10.9% | ~1.12 | ~-24% |
| SPY | ~14.4% | ~17.1% | ~0.84 | ~-34% |

**Rejected variant:** Uncapped inverse-volatility + SHY (~73% SHY in mandate stage analysis).

| Concern | Why it matters |
|---------|----------------|
| Cash-like allocation | ~73% SHY behaves like a short-duration cash bucket, not a diversified growth policy |
| Equity budget | Mandate scoring targets minimum equity exposure; optimizer drifts away from client intent |
| Information Ratio vs SPY | High Sharpe alone does not mean acceptable active risk relative to a simple SPY mandate |
| Policy narrative | Wealth management needs a explainable static policy, not a black box that piles into SHY in stress |

**Interview line:** "I treat Sharpe as one input. The final pick is Fixed + SHY Version A because it passes walk-forward risk metrics **and** mandate constraints. The highest-Sharpe rule fails the policy story."

---

## 8. Key Risk Metrics

Short definitions you can say aloud (this repo uses risk-free rate = 0 unless noted).

| Metric | One-line interview explanation |
|--------|--------------------------------|
| **Annualized return** | Compound daily simple returns, then scale to a per-year figure over the sample length |
| **Annualized volatility** | Daily return standard deviation times sqrt(252) |
| **Sharpe ratio** | Annualized return divided by annualized volatility (excess vs 0 rf here) |
| **Max drawdown** | Worst peak-to-trough loss on the cumulative wealth curve |
| **VaR 95% (daily)** | Historical 5th percentile loss on daily portfolio returns (implemented in `src/metrics.py`) |
| **CVaR 95% (daily)** | Average daily loss on days at or beyond the VaR threshold (tail mean) |
| **Beta vs SPY** | Regression slope of portfolio daily returns on SPY daily returns |
| **Tracking error** | Annualized volatility of active return (portfolio minus SPY) |
| **Information ratio** | Mean active return divided by tracking error (relative consistency vs SPY) |

**How to use them together:** Low vol and high Sharpe on inverse-vol do not automatically win if tracking error vs SPY, mandate caps, and weight concentration tell a different policy story.

---

## 9. Model Risk / Validation View

How a model validation or risk analyst might review this work:

| Pillar | Review focus in this repo |
|--------|---------------------------|
| **Conceptual soundness** | Is the question policy-based? Does SHY solve a stated 2022 failure mode without abandoning equity risk? |
| **Data quality** | Adjusted closes aligned across tickers; Yahoo Finance or similar; no survivorship-free dynamic universe |
| **Implementation** | `src/` modules separated from `scripts/stages/`; pytest on weights, returns, look-ahead timing |
| **Benchmark choice** | SPY documented with known limitations; consider policy benchmark as challenger |
| **Backtesting assumptions** | Monthly rebalance, 0 bps baseline costs, long-only, walk-forward train/test split |
| **Limitations** | Educational weights, single historical path, no live slippage or tax model |
| **Monitoring triggers** | Max drawdown breach, rolling vol spike, weight drift vs policy, SHY sleeve above cap, IR collapse vs SPY |

Stage 6 **audit** outputs under `output/stage6/audit/` are your evidence that you thought about reconciliation, not only headline metrics.

---

## 10. Limitations

State these without apology; they show judgment.

1. **Educational replication** - Not an exact copy of any disclosed portfolio.
2. **Public data vendor** - Yahoo Finance-style feeds; corporate actions may differ from production.
3. **Zero transaction cost baseline** - Overstates net returns; Stage 5 runs sensitivity but default is 0 bps.
4. **Survivorship / universe** - ETF list fixed in advance; no delisted fund history.
5. **No live execution** - No slippage, market impact, borrow, or fund-level fees unless you extend the engine.
6. **No tax modeling** - Total return only.
7. **Single historical sample** - One walk-forward path does not prove future correlation structure.
8. **SHY / TLT as ETFs** - Duration and roll risk differ from theoretical bond indices.

---

## 11. Likely Interview Questions And Strong Answers

### Q1. How did you avoid look-ahead bias?

**Answer:** Walk-forward design: for each test month, parameters use only the prior twelve months ending before `testing_start`. The backtest logs `training_end` and `testing_start`, and unit tests assert `training_end < testing_start`. In-sample Stage 1 results are not marketed as out-of-sample performance.

### Q2. Why did you choose SPY as benchmark?

**Answer:** SPY is a liquid US equity reference that matches much of the risk budget and supports beta, tracking error, and information ratio. I also acknowledge limitations: the portfolio holds gold and bonds, and SPY is itself a 25% holding in Version A. In production I would add a policy benchmark.

### Q3. Why did you add SHY?

**Answer:** The 2022 stress window showed equity and long-duration bonds falling together; TLT was a weak hedge. SHY adds a small short-duration Treasury sleeve (5% in Version A) to reduce rate sensitivity without replacing equity risk.

### Q4. Why not choose the strategy with the highest Sharpe?

**Answer:** Inverse-volatility achieves a higher walk-forward Sharpe (~1.14 vs ~1.04 fixed), but uncapped inverse-vol + SHY concentrates about 73% in SHY, which is cash-like and fails mandate constraints. The project selects Fixed + SHY Version A for policy fit, not a single metric.

### Q5. What happened in 2022?

**Answer:** Rates rose, growth and bonds sold off together, and long-duration Treasuries did not hedge equities. The repo stress-tests that window explicitly and uses it to motivate a modest SHY sleeve rather than doubling down on TLT.

### Q6. What would you improve next?

**Answer:** Transaction cost and slippage model, production-grade data feed, policy benchmark challenger, factor exposures, rolling regime flags, and hosted Streamlit with pinned sample `output/` for reviewers. See Section 12.

### Q7. How would you add transaction costs?

**Answer:** The engine already accepts `tx_cost_bps` in `run_walk_forward` and applies turnover on rebalance days. I would default to realistic bps by asset class, rerun Stage 5 sensitivity, and report net Sharpe and net max drawdown alongside gross results.

### Q8. How would this differ in production?

**Answer:** Production needs entitlements data, corporate action QA, mandate system integration, approval workflow, daily risk reporting, and model governance documentation. The dashboard would read governed datasets, not a local `output/` folder. Monitoring would include limit breaches, drift from policy weights, and exception-based reviews when IR or CVaR deteriorates.

### Q9. What does VaR / CVaR tell you here?

**Answer:** They summarize historical daily tail loss on the simulated policy return series. They are not forward-looking regulatory VaR unless you fit a separate model. I use them as descriptive tail metrics alongside max drawdown.

### Q10. How do you know the code is trustworthy?

**Answer:** Modular `src/` package, deterministic pytest fixtures (no network in CI), GitHub Actions running `python scripts/run_tests.py`, and a Stage 6 audit reconciling metrics and weights. I separate dashboard display from research logic.

---

## 12. Next Improvements

Realistic upgrades that strengthen the repo without changing the core story:

| Priority | Upgrade | Why |
|----------|---------|-----|
| 1 | Transaction cost + turnover model | Net performance and capacity realism |
| 2 | Bundled sample `output/` or release artifact | Reviewers can run dashboard without full pipeline |
| 3 | Streamlit Cloud / container deploy | Public demo for recruiters |
| 4 | Production data feed (e.g. vendor API with QA) | Data lineage and restatement control |
| 5 | Factor exposure (equity, rates, gold beta) | Explain returns, not only labels |
| 6 | Rolling regime detection | Flag when 2022-style correlation returns |
| 7 | Policy benchmark challenger | Fair relative performance vs 60/40 or fixed blend |
| 8 | Dashboard modularization | Split pages; keep zero backtest logic in UI layer |

---

## Quick Reference

| Item | Value |
|------|--------|
| Pipeline entry | `python scripts/run_all_stages.py` |
| Dashboard | `streamlit run dashboard/app.py` |
| Tests | `python scripts/run_tests.py` |
| Train / test | 12 months train, 1 month test, monthly rebalance |
| Recommended policy | Fixed + SHY Version A (25/25/25/15/5/5) |

**Disclaimer:** Educational research project. Not investment advice.
