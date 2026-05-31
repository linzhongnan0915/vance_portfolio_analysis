# Methodology and Risk Interpretation

This memo documents backtest design, assumptions, limitations, and how a risk manager
or analyst should read the results. Generated tables also appear under
`output/backtest_summary/`.

---

## Objective

Evaluate whether a **fixed, rules-based ETF allocation** (Vance-style educational
replication) behaves acceptably as a wealth-management risk policy, and whether a
**small SHY sleeve** improves tail behavior in 2022-style regimes without breaking
diversification or mandate limits.

---

## What is being backtested?

| Dimension | Definition |
|-----------|------------|
| Universe | QQQ, SPY, DIA, GLD, TLT; SHY added in extension stages |
| Benchmark | SPY buy-and-hold on the same calendar |
| Sample | Walk-forward test months from January 2010 through latest available data |
| Estimation window | Prior 12 months of daily returns (inverse-vol and min-variance only) |
| Rebalance | First trading day of each month; weights drift intra-month |
| Context | Publicly discussed ETF weights; **not** a verified disclosure copy |

---

## Strategies

### 1. Fixed monthly rebalance

Static weights 25% / 25% / 25% / 15% / 10% (QQQ / SPY / DIA / GLD / TLT). Tests
whether a **policy portfolio** without optimization survives multiple macro regimes.

### 2. Inverse-volatility

Weights proportional to inverse of trailing 12-month volatility, renormalized monthly.
Tests whether **volatility targeting** lowers portfolio vol without full optimization.

### 3. Minimum-variance

Long-only weights from a 12-month sample covariance matrix, capped at 40% per asset.
Tests whether **mean-variance optimization** improves risk-adjusted outcomes out of sample.

### 4. Fixed + SHY (mandate stage)

Reduces TLT to 5% and adds 5% SHY (Version A). Motivated by **2022 bond-equity
correlation breakdown**, where long-duration Treasuries failed as an equity hedge.

---

## Methodological assumptions

| Topic | Convention |
|-------|------------|
| Price input | Adjusted close from public market data |
| Return | Daily simple return: (P_t / P_{t-1}) - 1 |
| Annualized volatility | Daily std * sqrt(252) |
| Cumulative return | Compound daily simple returns |
| Risk-free rate | 0 for excess-return metrics |
| Look-ahead | Walk-forward: `training_end < testing_start` for each test month |
| Transaction costs | 0 bps baseline; sensitivity in robustness stage |
| Survivorship | ETF list fixed ex ante (no dynamic listing bias correction) |
| Leverage | Long-only, fully invested unless stated otherwise in a stage |

---

## Backtesting design

| Element | Design choice |
|---------|----------------|
| Engine | Monthly rebalance on first trading day; weights drift between rebalance dates |
| Parameter estimation | Rolling 12-month window ending before each test month |
| Strategies | Fixed, inverse-volatility, minimum-variance; SHY variants in later stages |
| Benchmark | SPY total return on the same calendar |
| OOS discipline | Walk-forward test months only for policy comparison headlines |
| Costs | 0 bps default; robustness stage varies assumptions |

---

## Walk-forward protocol

For each test month:

1. Estimate parameters using only the prior 12 months (where applicable).
2. Set portfolio weights for that month.
3. Record realized daily portfolio returns until the next rebalance.
4. Advance one month; repeat.

This design supports **out-of-sample** claims. In-sample baseline stages are labeled
separately and should not be quoted as OOS performance.

---

## Stress and robustness

- **COVID-19 window:** Liquidity shock and rapid equity drawdown; tests tail risk and
  recovery path versus SPY.
- **2022 window:** Rate hikes and simultaneous equity and long-bond weakness; tests
  whether TLT-heavy hedges fail and whether SHY adds stability.
- **Robustness stage:** Explores cost assumptions and stability of rankings across
  subperiods; overfitting diagnostics flag fragile parameter choices.

---

## Headline walk-forward results (indicative)

Approximate full-sample walk-forward metrics (2010 through latest):

| Strategy | Ann. return | Ann. vol | Sharpe | Max drawdown |
|----------|-------------|----------|--------|--------------|
| Fixed | ~13.7% | ~13.1% | ~1.04 | ~-24% |
| Inverse-vol | ~11.7% | ~10.3% | ~1.14 | ~-23% |
| Min-variance | ~12.1% | ~10.9% | ~1.12 | ~-24% |
| SPY | ~14.4% | ~17.1% | ~0.84 | ~-34% |

Higher Sharpe on inverse-vol or min-var does **not** automatically imply mandate
approval: uncapped inverse-vol with SHY can concentrate in cash-like weights and fail
client diversification norms.

**Policy choice:** Fixed + SHY Version A (25/25/25/15/5/5) for mandate consistency.

---

## Risk interpretation (how to read outputs)

- **Volatility and max drawdown** describe historical pain; they are not forward VaR
  unless you fit a separate risk model.
- **Sharpe ratio** assumes zero risk-free rate here; compare strategies under the same
  convention only.
- **Walk-forward metrics** are more defensible for policy selection than a single
  in-sample fit.
- **2022 episode:** Interpret TLT underperformance as **hedge failure in a rising-rate
  regime**, not as proof that bonds never hedge equities.
- **SHY role:** Short-duration Treasuries reduce rate sensitivity; they are not a
  substitute for strategic equity risk in a growth mandate.

---

## Limitations

1. **Data vendor risk:** Yahoo Finance (or similar) may have splits, dividends, and
   corporate actions handled differently than a production data feed.
2. **Replication gap:** Weights are educational approximations, not audited holdings.
3. **Cost model:** Zero-cost baseline overstates net returns; use Stage 5 sensitivity.
4. **Regime sample:** One walk-forward path does not cover all future correlation structures.
5. **No live execution:** Slippage, liquidity, and tax effects are not modeled unless
   explicitly added in a stage.
6. **SHY / TLT mapping:** ETFs are proxies; duration and credit exposure differ from
   idealized portfolio theory assumptions.

---

## Model validation checklist (interview framing)

| Area | Question |
|------|----------|
| Conceptual soundness | Does the policy match the stated WM objective? |
| Data quality | Are returns aligned, adjusted, and free of future data? |
| Implementation | Do tests confirm weights sum to 1 and dates do not leak? |
| Benchmarking | Is SPY the right comparator for this equity-heavy mix? |
| Monitoring | What would trigger a policy review (drawdown, drift, mandate breach)? |

---

## Disclaimer

Educational approximate replication for research and career portfolio use. Not
investment advice. Past simulated performance does not guarantee future results.
