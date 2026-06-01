# Project Summary (Interview One-Pager)

## Elevator pitch

Built a **staged portfolio risk research pipeline** on a Vance-style ETF allocation:
walk-forward backtests, COVID/2022 stress tests, robustness and overfitting checks,
mandate scoring, and a Streamlit dashboard that reads only precomputed outputs.

---

## Problem and question

**Problem:** Wealth managers need a **documented, repeatable risk policy**, not a
black-box optimizer that drifts into cash in stress years.

**Question:** Can a fixed ETF mix be defended out of sample, and does a **5% SHY**
sleeve improve 2022-style resilience without violating diversification mandates?

---

## Current policy candidate (for review)

| Status | Policy |
|--------|--------|
| **Policy candidate** | Fixed + SHY Version A: QQQ 25%, SPY 25%, DIA 25%, GLD 15%, TLT 5%, SHY 5% |
| **Reject** | Uncapped inverse-volatility + SHY (~73% SHY; cash-like, mandate-inconsistent) |

**Why this candidate:** Selected for **interpretability and mandate consistency**, not the
highest walk-forward Sharpe. Version A is a simple defensive adjustment that modestly improves
2022 stress behavior but is **not full-period dominant**. Uncapped inverse-vol + SHY abandons
the intended equity-risk budget. **Next step:** Test constrained dynamic SHY strategies before
calling any allocation optimal.

---

## Technical stack

- Python, pandas, numpy
- Modular engine in `src/backtest.py`, signals in `src/signals.py`, metrics in `src/metrics.py`
- Orchestration: `scripts/run_all_stages.py`
- UI: Streamlit under `dashboard/` (display only)
- Quality: `tests/` plus `python scripts/run_tests.py`

---

## What to emphasize in interviews

1. **Walk-forward design:** 12-month train, 1-month test; strict `training_end < testing_start`.
2. **Regime narrative:** COVID (liquidity shock) vs 2022 (rates up, TLT hedge failed) motivates SHY.
3. **Mandate layer:** Risk metrics alone did not drive the selection; mandate scoring surfaced a policy candidate for review.
4. **Audit stage:** Reconciliation across stages before sign-off.
5. **Separation of concerns:** `src/` = logic, `output/` = artifacts, `dashboard/` = presentation.

---

## Indicative results (walk-forward, full sample)

- Fixed policy: ~13.7% return, ~13.1% vol, Sharpe ~1.04, max DD ~-24%
- SPY benchmark: ~14.4% return, ~17.1% vol, Sharpe ~0.84, max DD ~-34%
- Inverse-vol / min-var: higher Sharpe, lower vol, but not advanced as the policy candidate after mandate review

---

## How to reproduce

```text
pip install -r requirements.txt
python scripts/run_all_stages.py
streamlit run dashboard/app.py
python scripts/run_tests.py
```

---

## Follow-up questions you should expect

- How did you prevent look-ahead bias?
- Why not pick the highest Sharpe strategy?
- What breaks if 2022-style correlation persists?
- How would you add transaction costs or a 2% cash buffer in production?
- What would you monitor live (drawdown, tracking error, weight drift)?

---

## References in repo

- [docs/GUIDE.md](../docs/GUIDE.md) - runbook and folder map
- [methodology.md](methodology.md) - assumptions, limitations, risk reading
- [docs/STRUCTURE.md](../docs/STRUCTURE.md) - layout conventions

**Disclaimer:** Educational project; not investment advice.
