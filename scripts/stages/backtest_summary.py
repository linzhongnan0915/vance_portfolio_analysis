"""
backtest_summary.py
Generates Backtest Summary table + Walk-Forward Decision Log + educational narrative.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_core import BENCHMARK, TICKERS, load_adjusted_prices, prices_to_returns
from walk_forward_engine import STRATEGIES, run_walk_forward
from src.config import DATA_DIR, ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.display.float_format = "{:.4f}".format

OUTPUT_DIR = ROOT / "output" / "backtest_summary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_LABELS = {
    "fixed_baseline": "Fixed monthly rebalance",
    "inverse_volatility": "Inverse-volatility",
    "minimum_variance": "Minimum-variance",
}

STRATEGY_RULES = {
    "fixed_baseline": (
        "Hold fixed target weights (QQQ/SPY/DIA 25% each, GLD 15%, TLT 10%). "
        "Reset to targets on the first trading day of each month."
    ),
    "inverse_volatility": (
        "Each month, estimate trailing 12-month daily vol per ETF. "
        "Set weights proportional to 1/volatility, normalize to 100%, long-only."
    ),
    "minimum_variance": (
        "Each month, estimate trailing 12-month covariance matrix. "
        "Solve long-only minimum-variance weights (sum=100%, max 40% per asset)."
    ),
}

DATA_USED = (
    "Trailing 12 calendar months of daily adjusted-close returns for QQQ, SPY, DIA, GLD, TLT "
    "(Yahoo Finance). Benchmark: SPY."
)

WHEN_UPDATED = "First trading day of each month (walk-forward rebalance date)."
USES_FUTURE = "No — each rebalance uses only data available through the prior trading day."

REGIME_PERIODS = {
    "2009-2019 In-sample": ("2009-01-01", "2019-12-31"),
    "2020 COVID year": ("2020-01-01", "2020-12-31"),
    "2021 Recovery": ("2021-01-01", "2021-12-31"),
    "2022 Rate hikes": ("2022-01-01", "2022-12-31"),
    "2023-2024 Recovery": ("2023-01-01", "2024-12-31"),
    "2025 YTD": ("2025-01-01", "2099-12-31"),
}


def regime_returns(log: pd.DataFrame, strategy: str) -> dict[str, float]:
    s = log[log["strategy"] == strategy].copy()
    s["testing_start"] = pd.to_datetime(s["testing_start"])
    out = {}
    for name, (start, end) in REGIME_PERIODS.items():
        sub = s[(s["testing_start"] >= start) & (s["testing_start"] <= end)]
        if len(sub) == 0:
            continue
        out[name] = (1 + sub["monthly_portfolio_return"]).prod() - 1
    return out


def fmt_pct(x: float) -> str:
    return f"{x:.2%}" if pd.notna(x) else "N/A"


def fmt_num(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}" if pd.notna(x) else "N/A"


def build_summary_table(wf, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in STRATEGIES:
        dw = wf.daily_weights[key]
        equity_exp = dw[["QQQ", "SPY", "DIA"]].sum(axis=1).mean()
        gld_exp = dw["GLD"].mean()
        tlt_exp = dw["TLT"].mean()

        regimes = regime_returns(wf.decision_log, key)
        worst = min(regimes, key=regimes.get) if regimes else "N/A"
        best = max(regimes, key=regimes.get) if regimes else "N/A"

        if key == "fixed_baseline":
            takeaway = (
                "Simplest policy; competitive Sharpe and lower drawdown than SPY. "
                "Robust in COVID; hurt in 2022 when bonds stopped hedging equities."
            )
        elif key == "inverse_volatility":
            takeaway = (
                "Lower vol and drawdown than fixed, but lower return and higher tracking error. "
                "Helped in COVID (overweighted TLT); hurt in 2022 (overweighted losing TLT/GLD)."
            )
        else:
            takeaway = (
                "Similar risk reduction to inverse-vol; covariance optimization did not avoid "
                "2022 regime failure. Sensitive to estimation error in short windows."
            )

        rows.append(
            {
                "Strategy": STRATEGY_LABELS[key],
                "Strategy rule": STRATEGY_RULES[key],
                "Data used for decisions": DATA_USED,
                "When weights are updated": WHEN_UPDATED,
                "Uses future data?": USES_FUTURE,
                "Average equity exposure (QQQ+SPY+DIA)": equity_exp,
                "Average GLD exposure": gld_exp,
                "Average TLT exposure": tlt_exp,
                "Annualized return": metrics.loc["Annualized Return", key],
                "Annualized volatility": metrics.loc["Annualized Volatility", key],
                "Sharpe ratio": metrics.loc["Sharpe Ratio", key],
                "Max drawdown": metrics.loc["Maximum Drawdown", key],
                "Calmar ratio": metrics.loc["Calmar Ratio", key],
                "Beta vs SPY": metrics.loc["Beta vs SPY", key],
                "Tracking error vs SPY": metrics.loc["Tracking Error vs SPY", key],
                "Information ratio vs SPY": metrics.loc["Information Ratio vs SPY", key],
                "Average monthly turnover": metrics.loc["Average Monthly Turnover", key],
                "Worst stress period": f"{worst} ({fmt_pct(regimes.get(worst, np.nan))})",
                "Best stress period": f"{best} ({fmt_pct(regimes.get(best, np.nan))})",
                "Key takeaway": takeaway,
            }
        )
    return pd.DataFrame(rows)


def build_decision_log_table(log: pd.DataFrame) -> pd.DataFrame:
    df = log.copy()
    df["rebalance_date"] = pd.to_datetime(df["rebalance_date"])
    df["training_window"] = (
        pd.to_datetime(df["training_start"]).dt.strftime("%Y-%m-%d")
        + " to "
        + pd.to_datetime(df["training_end"]).dt.strftime("%Y-%m-%d")
    )
    df["strategy"] = df["strategy"].map(STRATEGY_LABELS)
    df["weights_chosen"] = df.apply(
        lambda r: (
            f"QQQ {r['weight_QQQ']:.1%}, SPY {r['weight_SPY']:.1%}, DIA {r['weight_DIA']:.1%}, "
            f"GLD {r['weight_GLD']:.1%}, TLT {r['weight_TLT']:.1%}"
        ),
        axis=1,
    )
    df["next_month_portfolio_return"] = df["monthly_portfolio_return"]
    df["next_month_SPY_return"] = df["monthly_benchmark_return"]
    df["excess_return_vs_SPY"] = df["monthly_portfolio_return"] - df["monthly_benchmark_return"]

    return df[
        [
            "rebalance_date",
            "training_window",
            "strategy",
            "weights_chosen",
            "next_month_portfolio_return",
            "next_month_SPY_return",
            "excess_return_vs_SPY",
            "turnover",
        ]
    ].rename(
        columns={
            "next_month_portfolio_return": "next_month_portfolio_return",
            "next_month_SPY_return": "next_month_SPY_return",
            "excess_return_vs_SPY": "excess_return",
        }
    )


EDUCATIONAL = """
# Backtest Summary — Educational Overview

## What is a backtest?

A **backtest** is a historical simulation of a rule-based investment process. We pretend we
are running a portfolio policy month by month using only information that would have been
available at the time — then we record returns, risk, drawdowns, and trading (turnover).

This is **not** a guarantee of future results. It is a structured way to ask: *if we had
followed this exact rule in the past, what would have happened?*

---

## What exactly was backtested?

**Universe:** QQQ, SPY, DIA, GLD, TLT (daily adjusted-close total return proxies).

**Benchmark:** SPY.

**Period:** Walk-forward test months from January 2010 through latest available data (~197 months
per strategy). Each month uses the prior **12 months** of daily returns for parameter estimation
(inverse-vol and min-variance only).

**Rebalancing:** Monthly on the first trading day of each month. Weights drift within the month;
they reset at the next rebalance.

**Approximate educational context:** Fixed-weight mix inspired by publicly discussed ETF-style
allocations (Vance-style replication). **Not** an exact copy of any disclosed portfolio.

---

## Three strategies tested

### 1. Fixed monthly rebalance
Tests whether a **static allocation** (25/25/25/15/10) would have been robust across regimes
without any optimization. This is the baseline "policy portfolio."

### 2. Inverse-volatility
Tests whether **reducing exposure to historically volatile assets** (weight ∝ 1/vol) improves
risk control. Lower-vol assets (often GLD, TLT) receive higher weight.

### 3. Minimum-variance
Tests whether **covariance-based optimization** can reduce portfolio volatility using a 12-month
sample covariance matrix, subject to long-only constraints and a 40% max weight per asset.

---

## What results were produced?

The backtest output is **not just cumulative return**. For each strategy we produced:

| Category | Metrics / artifacts |
|----------|---------------------|
| Return & risk | Ann. return, vol, Sharpe, Sortino, max drawdown, Calmar, VaR/CVaR |
| vs benchmark | Beta, tracking error, information ratio vs SPY |
| Implementability | Average and maximum monthly turnover |
| Transparency | Full walk-forward decision log (rebalance date, training window, weights, next-month return) |
| Regime analysis | In-sample vs out-of-sample, COVID vs 2022 stress diagnostics (Stages 2–4) |
| Overfitting checks | IS/OOS degradation, cost sensitivity, training-window sensitivity |

---

## Headline results (walk-forward, 2010–latest)

See `strategy_summary.csv` for the full table. In brief:

- **Fixed baseline:** ~13.7% ann. return, ~13.1% vol, Sharpe ~1.04, max DD ~-24%.
- **Inverse-volatility:** ~11.7% return, **~10.3% vol**, Sharpe ~1.14, max DD ~-23%.
- **Minimum-variance:** ~12.1% return, ~10.9% vol, Sharpe ~1.12, max DD ~-24%.
- **SPY:** ~14.4% return, ~17.1% vol, Sharpe ~0.84, max DD ~-34%.

All three portfolio policies **beat SPY on risk-adjusted metrics** (Sharpe, drawdown) but
**underperformed SPY on raw return** over this period.

**Stress behavior:**
- **2020 (COVID):** All strategies benefited from TLT/gold diversifier behavior; dynamic rules helped most.
- **2022 (rate hikes):** All strategies lost ~18%; diversification failed (stocks and long bonds fell together).

---

## Files in this folder

| File | Description |
|------|-------------|
| `strategy_summary.csv` | One row per strategy — master backtest summary table |
| `walk_forward_decision_log.csv` | Every rebalance decision (591 rows = 197 months × 3 strategies) |
| `backtest_summary.xlsx` | Excel workbook with both tables |
| `BACKTEST_SUMMARY.md` | This document |

---

*Disclaimer: Educational approximate replication — not investment advice.*
"""


def main() -> None:
    print("Running walk-forward backtest for summary...")
    prices = load_adjusted_prices(DATA_DIR)
    daily = prices_to_returns(prices)[TICKERS]
    bench = prices_to_returns(prices)[BENCHMARK]
    wf = run_walk_forward(daily, bench)

    # Metrics from daily walk-forward series
    from portfolio_core import compute_metrics, metrics_to_series

    metrics_cols = {}
    for s in STRATEGIES:
        m = compute_metrics(wf.daily_returns[s], bench.loc[wf.daily_returns[s].index])
        ms = metrics_to_series(m)
        ms["Average Monthly Turnover"] = wf.turnover_history[s].mean()
        metrics_cols[s] = ms
    metrics = pd.DataFrame(metrics_cols)

    summary = build_summary_table(wf, metrics)
    decision = build_decision_log_table(wf.decision_log)

    # Display-friendly summary (formatted percentages)
    summary_display = summary.copy()
    pct_cols = [
        "Average equity exposure (QQQ+SPY+DIA)",
        "Average GLD exposure",
        "Average TLT exposure",
        "Annualized return",
        "Annualized volatility",
        "Max drawdown",
        "Average monthly turnover",
    ]
    for c in pct_cols:
        if c in summary_display.columns:
            summary_display[c] = summary_display[c].apply(lambda x: fmt_pct(x) if isinstance(x, (int, float)) else x)
    for c in ["Sharpe ratio", "Calmar ratio", "Beta vs SPY", "Information ratio vs SPY"]:
        summary_display[c] = summary_display[c].apply(lambda x: fmt_num(x, 2))
    summary_display["Tracking error vs SPY"] = summary_display["Tracking error vs SPY"].apply(
        lambda x: fmt_pct(x)
    )

    summary.to_csv(OUTPUT_DIR / "strategy_summary.csv", index=False)
    summary_display.to_csv(OUTPUT_DIR / "strategy_summary_display.csv", index=False)
    decision.to_csv(OUTPUT_DIR / "walk_forward_decision_log.csv", index=False)

    with pd.ExcelWriter(OUTPUT_DIR / "backtest_summary.xlsx", engine="openpyxl") as writer:
        summary_display.to_excel(writer, sheet_name="Strategy Summary", index=False)
        decision.to_excel(writer, sheet_name="Walk-Forward Decision Log", index=False)

    (OUTPUT_DIR / "BACKTEST_SUMMARY.md").write_text(EDUCATIONAL.strip(), encoding="utf-8")

    print("\n=== STRATEGY SUMMARY ===")
    print(summary_display.to_string(index=False))
    print(f"\nDecision log rows: {len(decision)}")
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
