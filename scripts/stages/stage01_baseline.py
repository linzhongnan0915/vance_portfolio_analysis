# %% [markdown]
# # Vance-Style Replicated Portfolio Analysis (Educational)
#
# **Disclaimer:** This is an *approximate*, *educational* replication based on publicly
# discussed ETF-style exposures. It is **not** an exact copy of any disclosed portfolio.
# Public disclosures often show asset **ranges**, not precise weights.
#
# **Stage 1 scope:** Fixed weights, monthly rebalancing, in-sample / out-of-sample evaluation.
# **Not included yet:** Rolling-window backtest, walk-forward optimization, dynamic strategies.

# %% Setup
"""
Vance_portfolio_analysis.py
Educational fixed-weight, monthly-rebalanced portfolio workflow.

Project folder: Global_Ai/vance_portfolio_analysis/
  data/    - cached ETF prices
  output/  - charts, tables, Excel report, interpretation

Run section-by-section in Cursor/VS Code (each # %% block) or run the full script.
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
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", palette="muted")
pd.options.display.float_format = "{:.4f}".format

# --- Project paths (all relative to this folder) ---
from src.config import ROOT
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Portfolio definition (approximate educational weights) ---
TICKERS = ["QQQ", "SPY", "DIA", "GLD", "TLT"]
BENCHMARK = "SPY"
TARGET_WEIGHTS = pd.Series(
    {"QQQ": 0.25, "SPY": 0.25, "DIA": 0.25, "GLD": 0.15, "TLT": 0.10},
    name="target_weight",
)

IN_SAMPLE_START = "2009-01-01"
IN_SAMPLE_END = "2019-12-31"
OUT_SAMPLE_START = "2020-01-01"

TRADING_DAYS = 252
RISK_FREE_RATE = 0.0  # Assumption: excess-return focus; document in report


def explain(title: str, text: str) -> None:
    """Print a formatted learning note."""
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)
    safe = text.strip().replace("\u2022", "-")
    try:
        print(safe)
    except UnicodeEncodeError:
        print(safe.encode("ascii", errors="replace").decode("ascii"))
    print()


# %% Step 1 — Load data
explain(
    "STEP 1 — Load adjusted close prices",
    """
What is adjusted close?
  Stock/ETF prices are 'adjusted' for splits and dividends so historical levels
  are comparable. Without adjustment, a split looks like a sudden crash and
  dividend payments are ignored — both would distort return calculations.

Why adjusted close?
  Portfolio risk analysis needs *total return* (price change + distributions).
  Adjusted close is the standard input for return-based analytics.

What we do here:
  1. Download daily adjusted close for QQQ, SPY, DIA, GLD, TLT (and SPY benchmark).
  2. Align all series to common trading dates (inner join).
  3. Forward-fill is NOT used for prices (gaps mean missing trading data).
  4. Drop rows where any asset is missing to keep a clean panel.
""",
)


def load_adjusted_prices(
    tickers: list[str],
    start: str = "2008-01-01",
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Download or load cached adjusted close prices."""
    cache_path = cache_path or DATA_DIR / "vance_etf_prices.csv"

    if cache_path.exists():
        print(f"Loading cached prices from {cache_path}")
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return prices.sort_index()

    print(f"Downloading adjusted close from Yahoo Finance: {tickers}")
    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers

    prices = prices.sort_index()
    prices = prices.dropna(how="any")
    prices.to_csv(cache_path)
    print(f"Saved cache to {cache_path}")
    return prices


all_tickers = sorted(set(TICKERS + [BENCHMARK]))
prices = load_adjusted_prices(all_tickers, start="2008-01-01")

print(f"Price panel shape: {prices.shape}")
print(f"Date range: {prices.index.min().date()} → {prices.index.max().date()}")
print("\nFirst 3 rows:")
print(prices.head(3))
print("\nMissing values per column:")
print(prices.isna().sum())


# %% Step 2 — Convert prices to returns
explain(
    "STEP 2 — Convert prices to daily returns",
    """
Why returns instead of raw prices?
  - Prices are non-stationary (trend upward over time); returns are more stable.
  - Portfolio math is additive in *log* returns and linear in *simple* weights x returns.
  - Risk metrics (volatility, VaR, drawdown) are defined on return series.
  - Comparing $100 vs $400 price levels is meaningless; % changes are comparable.

Simple daily return formula:
  r_t = (P_t / P_{t-1}) - 1

We use simple returns (standard for daily equity/ETF work).
""",
)


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="any")


daily_returns = prices_to_returns(prices)
print(f"Daily return panel: {daily_returns.shape}")
print("\nSample daily returns:")
print(daily_returns.head(3))


# %% Step 3 — Define fixed portfolio weights
explain(
    "STEP 3 — Fixed target weights (approximate educational portfolio)",
    """
Each ETF represents a different risk exposure:

  QQQ (25%) — Nasdaq-100; growth / tech-heavy US equity
  SPY (25%) — S&P 500; broad large-cap US equity core
  DIA (25%) — Dow Jones; 30 large blue-chip industrials
  GLD (15%) — Gold; inflation / crisis diversifier, non-equity
  TLT (10%) — Long-term US Treasuries; duration / rate sensitivity, flight-to-quality

Three equity ETFs overlap but are not identical — together they tilt equity-heavy
with meaningful gold and bond diversifiers.

Risk management check: weights must sum to 100%.
""",
)

weight_sum = TARGET_WEIGHTS.sum()
print("Target weights:")
print(TARGET_WEIGHTS.to_frame("Weight"))
print(f"\nWeight sum: {weight_sum:.4f} ({weight_sum * 100:.1f}%)")
assert np.isclose(weight_sum, 1.0), "Weights must sum to 1.0"

weights_table = TARGET_WEIGHTS.to_frame("Weight")
weights_table["Weight %"] = (weights_table["Weight"] * 100).round(1)
weights_table.to_csv(OUTPUT_DIR / "target_weights.csv")
print(f"\nSaved → {OUTPUT_DIR / 'target_weights.csv'}")


# %% Step 4 — Monthly rebalanced portfolio returns
explain(
    "STEP 4 — Monthly rebalanced portfolio returns",
    """
Rebalancing logic (transparent, no look-ahead bias):

  1. On the FIRST trading day of each calendar month, reset holdings to TARGET weights.
  2. Between rebalances, weights DRIFT as assets move at different speeds.
  3. Each day's portfolio return = dot(current_weights, asset_daily_returns).
  4. After each day, update drifted weights: w_i ← w_i × (1 + r_i), then normalize.

Why monthly rebalancing? (Portfolio risk management perspective)
  - Keeps actual allocations close to policy targets - prevents one asset dominating.
  - Controls concentration risk that builds passively in a buy-and-hold portfolio.
  - Reflects a realistic policy many wealth managers follow (review monthly).
  - Too-frequent rebalancing adds turnover cost; too-rare allows drift.

Monthly rebalance vs buy-and-hold:
  - Buy-and-hold: invest once, never trade - weights drift permanently.
  - Monthly rebalance: sell winners / buy losers to restore policy mix.
  - Rebalancing can reduce risk but may trim momentum effects; we implement monthly
    rebalance as the MAIN policy for this analysis.

Look-ahead bias avoided: rebalance uses only information available on that date.
""",
)


def is_new_month(date: pd.Timestamp, prev_date: pd.Timestamp) -> bool:
    return (date.year, date.month) != (prev_date.year, prev_date.month)


def monthly_rebalanced_returns(
    asset_returns: pd.DataFrame,
    target_weights: pd.Series,
) -> pd.Series:
    """Compute daily portfolio returns with monthly rebalancing to target weights."""
    tickers = list(target_weights.index)
    w = target_weights.values.astype(float)
    w = w / w.sum()

    port_rets = []
    dates = asset_returns.index

    for i, dt in enumerate(dates):
        if i > 0 and is_new_month(dt, dates[i - 1]):
            w = target_weights.values.astype(float)
            w = w / w.sum()

        r = asset_returns.loc[dt, tickers].values
        port_r = float(np.dot(w, r))
        port_rets.append(port_r)

        w = w * (1.0 + r)
        if w.sum() > 0:
            w = w / w.sum()

    return pd.Series(port_rets, index=dates, name="portfolio_return")


def buy_and_hold_returns(
    asset_returns: pd.DataFrame,
    target_weights: pd.Series,
) -> pd.Series:
    """Single initial allocation, no rebalancing (for comparison only)."""
    w = target_weights.values.astype(float)
    w = w / w.sum()
    tickers = list(target_weights.index)
    return (asset_returns[tickers] * w).sum(axis=1).rename("buy_hold_return")


port_tickers = [t for t in TARGET_WEIGHTS.index if t in daily_returns.columns]
port_returns = monthly_rebalanced_returns(daily_returns[port_tickers], TARGET_WEIGHTS)
bh_returns = buy_and_hold_returns(daily_returns[port_tickers], TARGET_WEIGHTS)
bench_returns = daily_returns[BENCHMARK].rename("benchmark_return")

print(f"Portfolio return series: {len(port_returns)} days")
print("\nComparison — cumulative return (full history):")
cum_port = (1 + port_returns).prod() - 1
cum_bh = (1 + bh_returns).prod() - 1
print(f"  Monthly rebalanced: {cum_port:.2%}")
print(f"  Buy-and-hold:       {cum_bh:.2%}")
print("\nFirst 5 portfolio returns:")
print(port_returns.head())


# %% Step 5 — In-sample / out-of-sample split
explain(
    "STEP 5 — In-sample vs out-of-sample split",
    """
In-sample (2009–2019):
  Period used to *describe* how the portfolio behaved historically under the policy.
  Think of it as the "training / characterization" window.

Out-of-sample (2020–latest):
  Period NOT used to design the strategy (weights are fixed in advance).
  Tests whether the same policy still behaves reasonably in new market regimes
  (COVID, inflation shock, rate hikes, etc.).

Why out-of-sample matters:
  - Reduces overfitting illusion - good in-sample stats can flatter a portfolio.
  - Real clients experience future data, not past data.
  - Risk managers want to know if risk characteristics (vol, drawdown) persist.

Note: We are NOT optimizing parameters on in-sample data here (weights are fixed).
      The split still teaches how to report and compare regimes transparently.
""",
)


def split_sample(
    series: pd.Series,
    in_start: str,
    in_end: str,
    out_start: str,
) -> tuple[pd.Series, pd.Series]:
    """Split return series into in-sample and out-of-sample windows."""
    in_sample = series.loc[in_start:in_end]
    out_sample = series.loc[out_start:]
    return in_sample, out_sample


port_is, port_oos = split_sample(
    port_returns, IN_SAMPLE_START, IN_SAMPLE_END, OUT_SAMPLE_START
)
bench_is, bench_oos = split_sample(
    bench_returns, IN_SAMPLE_START, IN_SAMPLE_END, OUT_SAMPLE_START
)

print("In-sample portfolio:", port_is.index.min().date(), "→", port_is.index.max().date(), f"({len(port_is)} days)")
print("Out-of-sample portfolio:", port_oos.index.min().date(), "→", port_oos.index.max().date(), f"({len(port_oos)} days)")


# %% Step 6 — Performance and risk metrics
@dataclass
class MetricResult:
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_95: float
    cvar_95: float
    beta: float
    tracking_error: float
    information_ratio: float


METRIC_EXPLANATIONS = {
    "cumulative_return": "Total growth over the period: (1+r1)(1+r2)... - 1.",
    "annualized_return": "Average return scaled to a 1-year basis (252 trading days).",
    "annualized_volatility": "Std dev of daily returns x sqrt(252); main risk gauge.",
    "sharpe_ratio": "Return per unit of total volatility (higher = better risk-adjusted).",
    "sortino_ratio": "Like Sharpe but penalizes only downside volatility.",
    "max_drawdown": "Worst peak-to-trough loss; pain metric for clients.",
    "calmar_ratio": "Annualized return / |max drawdown|; return vs worst loss.",
    "var_95": "95% VaR: on a typical bad day (5th percentile), loss is about this much.",
    "cvar_95": "Expected shortfall: average loss on days worse than VaR (tail risk).",
    "beta": "Sensitivity to SPY: beta=1 moves with market; beta<1 defensive; beta>1 aggressive.",
    "tracking_error": "Std dev of (portfolio - benchmark) return; active risk vs SPY.",
    "information_ratio": "Active return / tracking error; skill per unit active risk.",
}


def compute_drawdown(cum_wealth: pd.Series) -> pd.Series:
    return cum_wealth / cum_wealth.cummax() - 1.0


def compute_metrics(
    port: pd.Series,
    bench: pd.Series,
    rf: float = RISK_FREE_RATE,
    periods: int = TRADING_DAYS,
) -> MetricResult:
    aligned = pd.concat([port, bench], axis=1, join="inner").dropna()
    rp = aligned.iloc[:, 0]
    rb = aligned.iloc[:, 1]
    active = rp - rb

    n = len(rp)
    cum_ret = (1 + rp).prod() - 1
    ann_ret = (1 + cum_ret) ** (periods / n) - 1 if n > 0 else np.nan
    ann_vol = rp.std(ddof=1) * np.sqrt(periods)

    wealth = (1 + rp).cumprod()
    dd = compute_drawdown(wealth)
    max_dd = dd.min()

    downside = rp[rp < 0]
    downside_std = downside.std(ddof=1) * np.sqrt(periods) if len(downside) > 1 else np.nan

    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    sortino = (ann_ret - rf) / downside_std if downside_std and downside_std > 0 else np.nan
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

    var_95 = -np.quantile(rp, 0.05)
    cvar_95 = -rp[rp <= -var_95].mean() if (rp <= -var_95).any() else np.nan

    cov = np.cov(rp, rb, ddof=1)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan
    te = active.std(ddof=1) * np.sqrt(periods)
    ir = (active.mean() * periods) / te if te > 0 else np.nan

    return MetricResult(
        cumulative_return=cum_ret,
        annualized_return=ann_ret,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        var_95=var_95,
        cvar_95=cvar_95,
        beta=beta,
        tracking_error=te,
        information_ratio=ir,
    )


def metrics_to_series(m: MetricResult) -> pd.Series:
    return pd.Series(
        {
            "Cumulative Return": m.cumulative_return,
            "Annualized Return": m.annualized_return,
            "Annualized Volatility": m.annualized_volatility,
            "Sharpe Ratio": m.sharpe_ratio,
            "Sortino Ratio": m.sortino_ratio,
            "Maximum Drawdown": m.max_drawdown,
            "Calmar Ratio": m.calmar_ratio,
            "VaR 95% (daily)": m.var_95,
            "CVaR 95% (daily)": m.cvar_95,
            "Beta vs SPY": m.beta,
            "Tracking Error vs SPY": m.tracking_error,
            "Information Ratio vs SPY": m.information_ratio,
        }
    )


explain(
    "STEP 6 — Performance & risk metrics",
    "Below we compute metrics for portfolio and benchmark, in-sample and out-of-sample.\n"
    + "\n".join(f"  - {k}: {v}" for k, v in METRIC_EXPLANATIONS.items()),
)

metrics_is_port = compute_metrics(port_is, bench_is)
metrics_oos_port = compute_metrics(port_oos, bench_oos)
metrics_is_bench = compute_metrics(bench_is, bench_is)
metrics_oos_bench = compute_metrics(bench_oos, bench_oos)

for m in (metrics_is_bench, metrics_oos_bench):
    m.beta = 1.0
    m.tracking_error = 0.0
    m.information_ratio = np.nan

summary_table = pd.DataFrame(
    {
        "In-Sample Portfolio": metrics_to_series(metrics_is_port),
        "Out-of-Sample Portfolio": metrics_to_series(metrics_oos_port),
        "In-Sample SPY": metrics_to_series(metrics_is_bench),
        "Out-of-Sample SPY": metrics_to_series(metrics_oos_bench),
    }
)

print("\n--- METRICS TABLE ---")
print(summary_table.to_string())

comparison_table = pd.DataFrame(
    {
        "Portfolio In-Sample": metrics_to_series(metrics_is_port),
        "Portfolio Out-of-Sample": metrics_to_series(metrics_oos_port),
        "SPY In-Sample": metrics_to_series(metrics_is_bench),
        "SPY Out-of-Sample": metrics_to_series(metrics_oos_bench),
    }
)
comparison_table.to_csv(OUTPUT_DIR / "metrics_comparison.csv")
summary_table.to_csv(OUTPUT_DIR / "metrics_summary.csv")
print(f"\nSaved tables → {OUTPUT_DIR}")


# %% Step 7 — Rolling risk diagnostics
explain(
    "STEP 7 — Rolling risk diagnostics (NOT walk-forward backtesting)",
    """
These rolling charts are DIAGNOSTIC tools:
  - Rolling 63-day (~3 month) volatility: short-term risk regime shifts.
  - Rolling 252-day (~1 year) volatility: longer structural risk level.
  - Rolling 252-day Sharpe: how risk-adjusted performance evolves.
  - Drawdown curve: live 'pain' experience over time.

Important: This is NOT rolling-window strategy optimization.
  We are not re-estimating weights or tuning parameters in each window.
  Stage 2 will cover walk-forward / rolling backtests separately.
""",
)


def rolling_vol(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).std() * np.sqrt(TRADING_DAYS)


def rolling_sharpe(series: pd.Series, window: int, rf: float = RISK_FREE_RATE) -> pd.Series:
    roll_mean = series.rolling(window).mean() * TRADING_DAYS
    roll_std = series.rolling(window).std() * np.sqrt(TRADING_DAYS)
    return (roll_mean - rf) / roll_std


full_wealth_port = (1 + port_returns.loc[IN_SAMPLE_START:]).cumprod()
full_wealth_bench = (1 + bench_returns.loc[IN_SAMPLE_START:]).cumprod()
dd_port = compute_drawdown(full_wealth_port)
dd_bench = compute_drawdown(full_wealth_bench)

roll_vol_63_port = rolling_vol(port_returns.loc[IN_SAMPLE_START:], 63)
roll_vol_252_port = rolling_vol(port_returns.loc[IN_SAMPLE_START:], 252)
roll_sharpe_252_port = rolling_sharpe(port_returns.loc[IN_SAMPLE_START:], 252)


# %% Step 8 — Visualizations
def save_fig(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved chart → {path}")
    plt.close(fig)


fig, ax = plt.subplots(figsize=(12, 5))
(1 + port_returns.loc[IN_SAMPLE_START:]).cumprod().plot(ax=ax, label="Vance-style portfolio (monthly rebal)")
(1 + bench_returns.loc[IN_SAMPLE_START:]).cumprod().plot(ax=ax, label="SPY benchmark", alpha=0.85)
ax.axvline(pd.Timestamp(OUT_SAMPLE_START), color="gray", ls="--", lw=1, label="Out-of-sample start")
ax.set_title("Cumulative Return: Approximate Vance-Style Portfolio vs SPY")
ax.set_ylabel("Growth of $1")
ax.legend()
save_fig(fig, "01_cumulative_return.png")

fig, ax = plt.subplots(figsize=(12, 4))
dd_port.plot(ax=ax, label="Portfolio drawdown", color="crimson")
dd_bench.plot(ax=ax, label="SPY drawdown", alpha=0.7)
ax.axvline(pd.Timestamp(OUT_SAMPLE_START), color="gray", ls="--", lw=1)
ax.set_title("Drawdown Curve: Portfolio vs SPY")
ax.set_ylabel("Drawdown")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend()
save_fig(fig, "02_drawdown.png")

fig, ax = plt.subplots(figsize=(12, 5))
roll_vol_63_port.plot(ax=ax, label="63-day rolling vol (portfolio)")
roll_vol_252_port.plot(ax=ax, label="252-day rolling vol (portfolio)")
ax.set_title("Rolling Annualized Volatility (Diagnostic)")
ax.set_ylabel("Volatility")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend()
save_fig(fig, "03_rolling_volatility.png")

fig, ax = plt.subplots(figsize=(12, 4))
roll_sharpe_252_port.plot(ax=ax, color="darkgreen")
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Rolling 252-Day Sharpe Ratio (Diagnostic)")
ax.set_ylabel("Sharpe")
save_fig(fig, "04_rolling_sharpe.png")

monthly_rets = port_returns.loc[IN_SAMPLE_START:].resample("ME").apply(lambda x: (1 + x).prod() - 1)
heatmap_data = monthly_rets.to_frame("ret")
heatmap_data["year"] = heatmap_data.index.year
heatmap_data["month"] = heatmap_data.index.month
pivot = heatmap_data.pivot(index="year", columns="month", values="ret")
fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.35)))
sns.heatmap(
    pivot,
    annot=False,
    cmap="RdYlGn",
    center=0,
    ax=ax,
    cbar_kws={"label": "Monthly return"},
)
ax.set_title("Monthly Return Heatmap — Portfolio")
ax.set_xlabel("Month")
ax.set_ylabel("Year")
save_fig(fig, "05_monthly_heatmap.png")

bar_metrics = ["Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Maximum Drawdown"]
bar_data = summary_table.loc[bar_metrics, ["In-Sample Portfolio", "Out-of-Sample Portfolio"]]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, metric in zip(axes.flat, bar_metrics):
    bar_data.loc[metric].plot(kind="bar", ax=ax, color=["steelblue", "coral"], rot=0)
    ax.set_title(metric)
    if "Drawdown" in metric or "Return" in metric or "Volatility" in metric:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}" if abs(y) < 5 else f"{y:.2f}"))
fig.suptitle("In-Sample vs Out-of-Sample: Key Metrics", y=1.02)
fig.tight_layout()
save_fig(fig, "06_is_oos_comparison.png")

print("\nAll charts saved to:", OUTPUT_DIR)


# %% Step 9 — Output tables (Excel)
with pd.ExcelWriter(OUTPUT_DIR / "vance_portfolio_report.xlsx", engine="openpyxl") as writer:
    weights_table.to_excel(writer, sheet_name="Target Weights")
    summary_table.to_excel(writer, sheet_name="Metrics Summary")
    comparison_table.to_excel(writer, sheet_name="Portfolio vs Benchmark")
    pd.DataFrame({"portfolio": port_returns, "benchmark": bench_returns}).to_excel(
        writer, sheet_name="Daily Returns"
    )

print(f"Excel report → {OUTPUT_DIR / 'vance_portfolio_report.xlsx'}")


# %% Step 10 — Learning-focused interpretation
def interpret_results() -> str:
    is_port = metrics_to_series(metrics_is_port)
    oos_port = metrics_to_series(metrics_oos_port)
    is_spy = metrics_to_series(metrics_is_bench)
    oos_spy = metrics_to_series(metrics_oos_bench)

    text = f"""
LEARNING SUMMARY - Approximate Vance-Style Portfolio (Stage 1)
{'=' * 72}

1) WHAT IS THIS PORTFOLIO EXPOSED TO?
   - ~75% US equity across QQQ (growth/tech), SPY (broad market), DIA (blue chips).
   - ~15% gold (GLD) - diversifier, often low correlation with equities.
   - ~10% long Treasuries (TLT) - duration risk, potential equity crisis buffer.
   This is an equity-centric allocation with modest defensive sleeves.

2) IN-SAMPLE vs OUT-OF-SAMPLE PERFORMANCE
   In-sample (2009-2019):
     - Portfolio ann. return: {is_port['Annualized Return']:.2%}  |  SPY: {is_spy['Annualized Return']:.2%}
     - Portfolio ann. vol:    {is_port['Annualized Volatility']:.2%}  |  SPY: {is_spy['Annualized Volatility']:.2%}
     - Max drawdown:          {is_port['Maximum Drawdown']:.2%}  |  SPY: {is_spy['Maximum Drawdown']:.2%}

   Out-of-sample (2020-latest):
     - Portfolio ann. return: {oos_port['Annualized Return']:.2%}  |  SPY: {oos_spy['Annualized Return']:.2%}
     - Portfolio ann. vol:    {oos_port['Annualized Volatility']:.2%}  |  SPY: {oos_spy['Annualized Volatility']:.2%}
     - Max drawdown:          {oos_port['Maximum Drawdown']:.2%}  |  SPY: {oos_spy['Maximum Drawdown']:.2%}

   Compare whether risk/return characteristics stayed in the same ballpark or shifted
   sharply - that is the core OOS validation question.

3) DID DIVERSIFICATION HELP vs SPY ALONE?
   - Portfolio beta (OOS): {oos_port['Beta vs SPY']:.2f} - {'lower' if oos_port['Beta vs SPY'] < 1 else 'similar/higher'} market sensitivity.
   - Tracking error (OOS): {oos_port['Tracking Error vs SPY']:.2%} - how different daily paths are from SPY.
   - If vol or max drawdown is lower than SPY with similar return, diversification helped.
   - Gold/TLT help most when equities sell off; in rising-rate regimes TLT can hurt.

4) MOST IMPORTANT RISK METRICS (portfolio risk management lens)
   Priority for interns / managers:
     a) Maximum drawdown - client pain and suitability conversations
     b) Volatility & VaR/CVaR - daily/monthly loss expectations
     c) Beta & tracking error - how "market-like" the portfolio behaves
     d) Sharpe/Sortino - efficiency, but sensitive to assumptions
   Always report WITH the time period and rebalancing policy.

5) LIMITATIONS OF THIS STAGE
   - Approximate weights - not verified against any official disclosure.
   - Monthly rebalance at close - no transaction costs, taxes, or slippage.
   - ETF total return via adjusted close - small deviation from live fund NAV.
   - Fixed weights - no dynamic risk budgeting, no regime switching.
   - Single benchmark (SPY) - equity-heavy portfolio may need multi-benchmark view.
   - No statistical significance testing on OOS differences yet.

6) STAGE 2 - WHAT COMES NEXT
   - Rolling-window backtest: recompute metrics on moving windows (e.g. 3Y windows).
   - Walk-forward analysis: simulate what a risk manager would see in real time.
   - Sensitivity analysis: rebalance frequency, weight bands, cost assumptions.
   - Optional: compare to minimum-variance, risk parity, or momentum overlays
     (only after understanding this fixed-policy baseline).

{'=' * 72}
DISCLAIMER: Educational approximate replication - not investment advice.
"""
    return text


interpretation = interpret_results()
print(interpretation)

with open(OUTPUT_DIR / "interpretation.txt", "w", encoding="utf-8") as f:
    f.write(interpretation)

print(f"Interpretation saved → {OUTPUT_DIR / 'interpretation.txt'}")
print("\nDone. Review outputs in:", OUTPUT_DIR)
