# %% [markdown]
# # Stage 3 — Walk-Forward Rolling-Window Backtest
#
# Educational framework: fixed vs inverse-vol vs min-variance.
# Training = 12 months, test = 1 month, roll forward monthly. No look-ahead.

# %% Setup
"""
Vance_portfolio_analysis_stage3.py
Walk-forward backtest comparing three monthly-rebalanced strategies.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize

from portfolio_core import (
    BENCHMARK,
    RISK_FREE_RATE,
    TARGET_WEIGHTS,
    TICKERS,
    TRADING_DAYS,
    compute_drawdown,
    compute_metrics,
    load_adjusted_prices,
    metrics_to_series,
    prices_to_returns,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", palette="muted")
pd.options.display.float_format = "{:.4f}".format

from src.config import ROOT
OUTPUT_DIR = ROOT / "output" / "stage3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANALYSIS_START = "2009-01-01"
TRAIN_MONTHS = 12
MAX_WEIGHT = 0.40
MIN_WEIGHT = 0.0

STRATEGIES = ["fixed_baseline", "inverse_volatility", "minimum_variance"]


def explain(title: str, text: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print(text.strip())
    print()


# --- Weight engines (use TRAINING data only) ---

def estimate_volatilities(train: pd.DataFrame) -> pd.Series:
    """Annualized vol from daily returns in training window."""
    return train.std(ddof=1) * np.sqrt(TRADING_DAYS)


def fixed_weights(_train: pd.DataFrame) -> tuple[pd.Series, str]:
    w = TARGET_WEIGHTS.copy()
    return w, ""


def inverse_volatility_weights(train: pd.DataFrame) -> tuple[pd.Series, str]:
    vol = estimate_volatilities(train)
    if (vol <= 0).any() or vol.isna().any():
        w = pd.Series(1.0 / len(TICKERS), index=TICKERS)
        return w, "Zero/NaN vol in training; fallback to equal weight"
    inv = 1.0 / vol
    w = inv / inv.sum()
    return w, ""


def minimum_variance_weights(
    train: pd.DataFrame,
    max_w: float = MAX_WEIGHT,
    min_w: float = MIN_WEIGHT,
) -> tuple[pd.Series, str]:
    if len(train) < 30:
        w = pd.Series(1.0 / len(TICKERS), index=TICKERS)
        return w, "Insufficient training rows; equal weight fallback"

    cov = train.cov().values
    n = len(TICKERS)
    x0 = np.ones(n) / n
    bounds = [(min_w, max_w) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

    notes = []
    if not result.success:
        notes.append(f"Optimizer: {result.message}; equal-weight fallback")
        w_arr = x0
    else:
        w_arr = result.x
        w_arr = np.clip(w_arr, min_w, max_w)
        w_arr = w_arr / w_arr.sum()
        at_max = [TICKERS[i] for i, v in enumerate(w_arr) if v >= max_w - 1e-4]
        at_min = [TICKERS[i] for i, v in enumerate(w_arr) if v <= min_w + 1e-4]
        if at_max:
            notes.append(f"Max weight ({max_w:.0%}) binding: {', '.join(at_max)}")
        if at_min:
            notes.append(f"Min weight binding: {', '.join(at_min)}")

    w = pd.Series(w_arr, index=TICKERS)
    return w, "; ".join(notes)


WEIGHT_FN = {
    "fixed_baseline": fixed_weights,
    "inverse_volatility": inverse_volatility_weights,
    "minimum_variance": minimum_variance_weights,
}


# --- Walk-forward engine ---

def first_trading_day_on_or_after(daily_index: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp | None:
    sub = daily_index[daily_index >= ts]
    return sub[0] if len(sub) else None


def month_period_bounds(
    daily_index: pd.DatetimeIndex,
    year: int,
    month: int,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start = pd.Timestamp(year=year, month=month, day=1)
    if month == 12:
        end = pd.Timestamp(year=year + 1, month=1, day=1) - pd.Timedelta(days=1)
    else:
        end = pd.Timestamp(year=year, month=month + 1, day=1) - pd.Timedelta(days=1)
    test_start = first_trading_day_on_or_after(daily_index, start)
    test_end_candidates = daily_index[(daily_index >= start) & (daily_index <= end)]
    test_end = test_end_candidates[-1] if len(test_end_candidates) else None
    return test_start, test_end


def simulate_month_returns(
    test_returns: pd.DataFrame,
    start_weights: np.ndarray,
) -> tuple[pd.Series, np.ndarray]:
    """Daily portfolio returns within test month; weights drift; return end weights."""
    w = start_weights.copy()
    tickers = list(test_returns.columns)
    port = []
    for dt, row in test_returns.iterrows():
        r = row[tickers].values
        port.append(float(np.dot(w, r)))
        w = w * (1.0 + r)
        if w.sum() > 0:
            w = w / w.sum()
    return pd.Series(port, index=test_returns.index), w


def turnover(w_new: np.ndarray, w_old: np.ndarray) -> float:
    return float(0.5 * np.abs(w_new - w_old).sum())


@dataclass
class WalkForwardResult:
    decision_log: pd.DataFrame
    daily_returns: dict[str, pd.Series]
    weight_history: dict[str, pd.DataFrame]
    turnover_history: dict[str, pd.Series]


def run_walk_forward(
    asset_returns: pd.DataFrame,
    bench_returns: pd.Series,
    analysis_start: str = ANALYSIS_START,
    train_months: int = TRAIN_MONTHS,
) -> WalkForwardResult:
    asset_returns = asset_returns.sort_index()
    bench_returns = bench_returns.sort_index()
    idx = asset_returns.index

    # First test month: need train_months of history ending before test
    start_ts = pd.Timestamp(analysis_start)
    first_test = start_ts + pd.DateOffset(months=train_months)
    first_test = pd.Timestamp(year=first_test.year, month=first_test.month, day=1)

    months: list[tuple[int, int]] = []
    cursor = first_test
    last_date = idx[-1]
    while cursor <= last_date:
        months.append((cursor.year, cursor.month))
        cursor = cursor + pd.DateOffset(months=1)

    log_rows = []
    daily_by_strategy: dict[str, list[pd.Series]] = {s: [] for s in STRATEGIES}
    weight_hist: dict[str, list[dict]] = {s: [] for s in STRATEGIES}
    turnover_hist: dict[str, list[tuple[pd.Timestamp, float]]] = {s: [] for s in STRATEGIES}
    end_weights: dict[str, np.ndarray | None] = {s: None for s in STRATEGIES}

    for year, month in months:
        test_start, test_end = month_period_bounds(idx, year, month)
        if test_start is None or test_end is None:
            continue

        train_end = idx[idx < test_start][-1]
        train_start_target = test_start - pd.DateOffset(months=train_months)
        train_candidates = idx[(idx >= train_start_target) & (idx <= train_end)]
        if len(train_candidates) < 60:
            continue
        train_start = train_candidates[0]

        train = asset_returns.loc[train_start:train_end]
        test = asset_returns.loc[test_start:test_end]
        bench_month = bench_returns.loc[test_start:test_end]
        est_vol = estimate_volatilities(train)
        monthly_bench = (1 + bench_month).prod() - 1

        for strategy in STRATEGIES:
            w_series, opt_notes = WEIGHT_FN[strategy](train)
            w_new = w_series.values.astype(float)
            w_new = w_new / w_new.sum()

            w_old = end_weights[strategy]
            if w_old is None:
                to = 0.0
                w_at_rebal = w_new
            else:
                to = turnover(w_new, w_old)
                w_at_rebal = w_new

            port_daily, w_end = simulate_month_returns(test, w_at_rebal)
            end_weights[strategy] = w_end
            daily_by_strategy[strategy].append(port_daily)
            turnover_hist[strategy].append((test_start, to))

            wh = {"rebalance_date": test_start, **{f"weight_{t}": w_new[i] for i, t in enumerate(TICKERS)}}
            weight_hist[strategy].append(wh)

            monthly_port = (1 + port_daily).prod() - 1
            log_rows.append(
                {
                    "rebalance_date": test_start.date(),
                    "training_start": train_start.date(),
                    "training_end": train_end.date(),
                    "testing_start": test_start.date(),
                    "testing_end": test_end.date(),
                    "strategy": strategy,
                    **{f"est_vol_{t}": est_vol[t] for t in TICKERS},
                    **{f"weight_{t}": w_new[i] for i, t in enumerate(TICKERS)},
                    "monthly_portfolio_return": monthly_port,
                    "monthly_benchmark_return": monthly_bench,
                    "turnover": to,
                    "notes": opt_notes,
                }
            )

    decision_log = pd.DataFrame(log_rows)
    daily_returns = {
        s: pd.concat(daily_by_strategy[s]).sort_index() for s in STRATEGIES if daily_by_strategy[s]
    }
    weight_history = {
        s: pd.DataFrame(weight_hist[s]).set_index("rebalance_date") if weight_hist[s] else pd.DataFrame()
        for s in STRATEGIES
    }
    turnover_history = {
        s: pd.Series(dict(turnover_hist[s]), name="turnover") if turnover_hist[s] else pd.Series(dtype=float)
        for s in STRATEGIES
    }
    return WalkForwardResult(decision_log, daily_returns, weight_history, turnover_history)


def extended_metrics(
    port: pd.Series,
    bench: pd.Series,
    turnover: pd.Series,
) -> pd.Series:
    base = metrics_to_series(compute_metrics(port, bench))
    monthly = port.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    base["Average Monthly Turnover"] = turnover.mean() if len(turnover) else np.nan
    base["Maximum Monthly Turnover"] = turnover.max() if len(turnover) else np.nan
    base["Worst Monthly Return"] = monthly.min() if len(monthly) else np.nan
    base["Best Monthly Return"] = monthly.max() if len(monthly) else np.nan
    return base


def rolling_vol(r: pd.Series, w: int = 252) -> pd.Series:
    return r.rolling(w).std() * np.sqrt(TRADING_DAYS)


def rolling_sharpe(r: pd.Series, w: int = 252) -> pd.Series:
    m = r.rolling(w).mean() * TRADING_DAYS
    s = r.rolling(w).std() * np.sqrt(TRADING_DAYS)
    return (m - RISK_FREE_RATE) / s


def rolling_beta(port: pd.Series, bench: pd.Series, w: int = 252) -> pd.Series:
    return port.rolling(w).cov(bench) / bench.rolling(w).var()


def save_fig(fig: plt.Figure, name: str) -> None:
    p = OUTPUT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart -> {p}")


# =============================================================================
explain(
    "STAGE 3 — Walk-forward backtesting (what and why)",
    """
What is walk-forward backtesting?
  Each month you PRETEND you only know the past. You train on the prior 12 months,
  pick weights, then test on the NEXT month. Then you roll forward one month and repeat.
  Every test month is out-of-sample relative to its training window.

Why is each test month out-of-sample?
  Weights for March 2020 use data through February 2020 only — not March returns.
  This mimics real portfolio management and reduces overfitting.

Inverse-volatility weighting (risk-based allocation):
  Lower-volatility assets get higher weight. It is simple, intuitive, and often
  stabilizes the portfolio — but can overweight bonds/gold and underweight equities.

Minimum-variance optimization:
  Uses the covariance matrix to find the lowest-risk mix. Can overfit because
  covariance estimates are noisy in 12-month windows — small samples, unstable correlations.
  We cap each asset at 40% to avoid concentration from estimation error.

Turnover and implementability:
  Turnover = trading needed to move from old weights to new weights. High turnover
  means higher costs and slippage in live trading — paper backtests often ignore this.

Why compare to fixed baseline AND SPY?
  Fixed baseline = our Stage 1 policy (did dynamic rules help vs doing nothing?).
  SPY = market standard (did any strategy beat a simple index?).
""",
)

# Load data
prices = load_adjusted_prices(ROOT / "data")
daily = prices_to_returns(prices)[TICKERS]
bench = prices_to_returns(prices)[BENCHMARK]

explain(
    "Walk-forward setup",
    f"""
Training window: {TRAIN_MONTHS} calendar months (daily returns)
Testing window: 1 calendar month
Rebalance: first trading day of each test month
Analysis start: {ANALYSIS_START} (first test month after 12 months of training)
No look-ahead: weights use training data only; applied to subsequent month.
""",
)

wf = run_walk_forward(daily, bench)
decision_log = wf.decision_log
decision_log.to_csv(OUTPUT_DIR / "walk_forward_decision_log.csv", index=False)
print(f"Decision log: {len(decision_log)} rows ({len(decision_log) // 3} months x 3 strategies)")
print(decision_log.head(6).to_string())

# Align benchmark to walk-forward daily window
all_starts = [wf.daily_returns[s].index.min() for s in STRATEGIES if s in wf.daily_returns]
wf_start = min(all_starts)
bench_wf = bench.loc[wf_start:]

# --- Performance metrics ---
metrics_table = pd.DataFrame(
    {
        s: extended_metrics(wf.daily_returns[s], bench_wf, wf.turnover_history[s])
        for s in STRATEGIES
    }
)
metrics_table["SPY_benchmark"] = extended_metrics(
    bench_wf, bench_wf, pd.Series(0.0, index=wf.turnover_history["fixed_baseline"].index)
)
print("\n--- STRATEGY COMPARISON ---")
print(metrics_table.T.to_string())
metrics_table.to_csv(OUTPUT_DIR / "strategy_metrics.csv")

# --- Rolling diagnostics ---
rolling = {}
for s in STRATEGIES:
    r = wf.daily_returns[s]
    rolling[s] = pd.DataFrame(
        {
            "vol_252": rolling_vol(r),
            "sharpe_252": rolling_sharpe(r),
            "drawdown": compute_drawdown((1 + r).cumprod()),
            "beta_252": rolling_beta(r, bench_wf),
        }
    )
    rolling[s].to_csv(OUTPUT_DIR / f"rolling_{s}.csv")

# --- Visualizations ---
# 1. Cumulative return
fig, ax = plt.subplots(figsize=(12, 5))
for s, label in [
    ("fixed_baseline", "Fixed baseline"),
    ("inverse_volatility", "Inverse vol"),
    ("minimum_variance", "Min variance"),
]:
    (1 + wf.daily_returns[s]).cumprod().plot(ax=ax, label=label)
(1 + bench_wf).cumprod().plot(ax=ax, label="SPY", ls="--", color="black", alpha=0.7)
ax.set_title("Walk-Forward Cumulative Return Comparison")
ax.set_ylabel("Growth of $1")
ax.legend()
save_fig(fig, "01_cumulative_return.png")

# 2. Drawdown
fig, ax = plt.subplots(figsize=(12, 4))
for s, label in [
    ("fixed_baseline", "Fixed"),
    ("inverse_volatility", "Inv vol"),
    ("minimum_variance", "Min var"),
]:
    compute_drawdown((1 + wf.daily_returns[s]).cumprod()).plot(ax=ax, label=label)
ax.set_title("Drawdown Comparison")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend()
save_fig(fig, "02_drawdown.png")

# 3. Weights over time
for s, title in [
    ("inverse_volatility", "Inverse-Volatility Weights Over Time"),
    ("minimum_variance", "Minimum-Variance Weights Over Time"),
]:
    wh = wf.weight_history[s]
    if wh.empty:
        continue
    fig, ax = plt.subplots(figsize=(12, 5))
    cols = [c for c in wh.columns if c.startswith("weight_")]
    wh[cols].plot(ax=ax, stacked=True)
    ax.set_title(title)
    ax.set_ylabel("Weight")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend([c.replace("weight_", "") for c in cols], loc="upper left", bbox_to_anchor=(1, 1))
    save_fig(fig, f"03_weights_{s}.png")

# 4. Turnover
fig, ax = plt.subplots(figsize=(12, 4))
for s, label in [
    ("fixed_baseline", "Fixed"),
    ("inverse_volatility", "Inv vol"),
    ("minimum_variance", "Min var"),
]:
    wf.turnover_history[s].plot(ax=ax, label=label, alpha=0.85)
ax.set_title("Monthly Turnover at Rebalance")
ax.set_ylabel("Turnover")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend()
save_fig(fig, "04_turnover.png")

# 5. Rolling vol comparison
fig, ax = plt.subplots(figsize=(12, 4))
for s in STRATEGIES:
    rolling[s]["vol_252"].plot(ax=ax, label=s)
ax.set_title("Rolling 252-Day Volatility")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend()
save_fig(fig, "05_rolling_volatility.png")

# 6. Rolling Sharpe
fig, ax = plt.subplots(figsize=(12, 4))
for s in STRATEGIES:
    rolling[s]["sharpe_252"].plot(ax=ax, label=s)
ax.set_title("Rolling 252-Day Sharpe Ratio")
ax.legend()
save_fig(fig, "06_rolling_sharpe.png")

# 7. Monthly return heatmap (fixed baseline)
monthly = wf.daily_returns["fixed_baseline"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
hm = monthly.to_frame("ret")
hm["year"] = hm.index.year
hm["month"] = hm.index.month
pivot = hm.pivot(index="year", columns="month", values="ret")
fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.35)))
sns.heatmap(pivot, cmap="RdYlGn", center=0, ax=ax, cbar_kws={"label": "Monthly return"})
ax.set_title("Monthly Return Heatmap — Fixed Baseline (Walk-Forward)")
save_fig(fig, "07_monthly_heatmap_fixed.png")

# --- Excel ---
with pd.ExcelWriter(OUTPUT_DIR / "stage3_report.xlsx", engine="openpyxl") as writer:
    decision_log.to_excel(writer, sheet_name="Decision Log", index=False)
    metrics_table.T.to_excel(writer, sheet_name="Strategy Metrics")
    for s in STRATEGIES:
        wf.weight_history[s].to_excel(writer, sheet_name=f"Weights_{s[:20]}")
        rolling[s].to_excel(writer, sheet_name=f"Rolling_{s[:20]}")

print(f"\nExcel -> {OUTPUT_DIR / 'stage3_report.xlsx'}")

# --- Interpretation ---
m = metrics_table
interp = f"""
STAGE 3 INTERPRETATION — Walk-Forward Backtest
{'=' * 72}

Setup: 12-month training, 1-month test, monthly roll-forward from {wf_start.date()}.

STRATEGY SUMMARY (full walk-forward period)
  Fixed baseline:     ann return {m.loc['Annualized Return','fixed_baseline']:.2%}, vol {m.loc['Annualized Volatility','fixed_baseline']:.2%}, Sharpe {m.loc['Sharpe Ratio','fixed_baseline']:.2f}, max DD {m.loc['Maximum Drawdown','fixed_baseline']:.2%}
  Inverse volatility: ann return {m.loc['Annualized Return','inverse_volatility']:.2%}, vol {m.loc['Annualized Volatility','inverse_volatility']:.2%}, Sharpe {m.loc['Sharpe Ratio','inverse_volatility']:.2f}, max DD {m.loc['Maximum Drawdown','inverse_volatility']:.2%}
  Minimum variance:   ann return {m.loc['Annualized Return','minimum_variance']:.2%}, vol {m.loc['Annualized Volatility','minimum_variance']:.2%}, Sharpe {m.loc['Sharpe Ratio','minimum_variance']:.2f}, max DD {m.loc['Maximum Drawdown','minimum_variance']:.2%}
  SPY benchmark:      ann return {m.loc['Annualized Return','SPY_benchmark']:.2%}, vol {m.loc['Annualized Volatility','SPY_benchmark']:.2%}, Sharpe {m.loc['Sharpe Ratio','SPY_benchmark']:.2f}

TURNOVER (implementability)
  Fixed avg turnover:   {m.loc['Average Monthly Turnover','fixed_baseline']:.2%}
  Inv-vol avg turnover: {m.loc['Average Monthly Turnover','inverse_volatility']:.2%}
  Min-var avg turnover: {m.loc['Average Monthly Turnover','minimum_variance']:.2%}

LEARNING POINTS
  1. Walk-forward makes every test month genuinely out-of-sample vs its training window.
  2. Inverse-vol tilts toward lower-risk assets (often GLD/TLT); check weight charts.
  3. Min-variance can look good in-sample to training but may shift weights aggressively
     when covariance estimates are noisy — watch turnover and weight stability.
  4. Higher turnover strategies need cost assumptions before live implementation.
  5. Beating fixed baseline in a walk-forward test is stronger evidence than a single
     full-sample backtest (Stages 1-2).

DISCLAIMER: Educational approximate replication — not investment advice.
Outputs: {OUTPUT_DIR}
"""
print(interp)
(OUTPUT_DIR / "interpretation_stage3.txt").write_text(interp, encoding="utf-8")
print("Stage 3 complete.")
