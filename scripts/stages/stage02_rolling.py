# %% [markdown]
# # Stage 2 — Rolling Diagnostics & Regime Analysis
#
# **Same policy as Stage 1:** Fixed weights (QQQ/SPY/DIA 25% each, GLD 15%, TLT 10%),
# monthly rebalance. **No weight optimization.**
#
# **Goal:** Does this fixed policy behave consistently across market regimes?

# %% Setup
"""
Vance_portfolio_analysis_stage2.py
Rolling risk diagnostics + subperiod regime analysis.

Outputs: vance_portfolio_analysis/output/stage2/
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from portfolio_core import (
    BENCHMARK,
    IN_SAMPLE_START,
    RISK_FREE_RATE,
    TARGET_WEIGHTS,
    TRADING_DAYS,
    build_portfolio_series,
    compute_drawdown,
    compute_metrics,
    subperiod_metrics_to_series,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", palette="muted")
pd.options.display.float_format = "{:.4f}".format

from src.config import ROOT
OUTPUT_DIR = ROOT / "output" / "stage2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Subperiod definitions (regime lens for learning)
SUBPERIODS: dict[str, tuple[str, str | None]] = {
    "2009-2019 In-Sample": ("2009-01-01", "2019-12-31"),
    "2020 COVID Crash": ("2020-01-01", "2020-12-31"),
    "2021 Recovery / Liquidity": ("2021-01-01", "2021-12-31"),
    "2022 Rate Hike / Bond Drawdown": ("2022-01-01", "2022-12-31"),
    "2023-2024 Market Recovery": ("2023-01-01", "2024-12-31"),
    "2025 to Latest": ("2025-01-01", None),
}

ROLLING_EXPLANATIONS = {
    "volatility_63": (
        "Rolling 63-day (~3 month) volatility",
        "Short-term risk gauge. Spikes flag sudden stress (e.g. COVID March 2020). "
        "Risk managers watch this for near-term limit breaches and client updates.",
    ),
    "volatility_252": (
        "Rolling 252-day (~1 year) volatility",
        "Structural risk level. Smoother than 63-day; shows sustained high/low risk regimes. "
        "Useful for strategic asset allocation reviews.",
    ),
    "sharpe_252": (
        "Rolling 252-day Sharpe ratio",
        "Return per unit of risk over the past year. Shows when the policy 'paid' for risk taken. "
        "Falling Sharpe in bad regimes signals deteriorating risk-adjusted performance.",
    ),
    "max_drawdown_252": (
        "Rolling 252-day maximum drawdown",
        "Worst peak-to-trough loss within each trailing year. "
        "Directly tied to client pain and suitability — how bad could the last 12 months have felt?",
    ),
    "beta_252": (
        "Rolling 252-day beta vs SPY",
        "Market sensitivity over the past year. Beta > 1 = more aggressive than SPY; < 1 = defensive. "
        "Stable beta suggests predictable equity exposure; jumps flag regime change.",
    ),
    "correlation_252": (
        "Rolling 252-day correlation vs SPY",
        "How closely portfolio daily moves track SPY. High correlation = equity-dominated behavior; "
        "lower correlation suggests diversifiers (GLD, TLT) are working.",
    ),
    "tracking_error_252": (
        "Rolling 252-day tracking error vs SPY",
        "Annualized std dev of (portfolio - SPY) return. Active risk vs benchmark. "
        "Higher TE = portfolio path differs more from SPY.",
    ),
    "var_95_252": (
        "Rolling 252-day VaR 95% (daily)",
        "On a bad day (5th percentile), loss was about this much, using only the past year of data. "
        "Regulatory and internal limit monitoring often reference rolling VaR.",
    ),
    "cvar_95_252": (
        "Rolling 252-day CVaR 95% (daily)",
        "Average loss on days worse than VaR — tail risk. "
        "More conservative than VaR; captures crash severity beyond the threshold.",
    ),
}


def explain(title: str, text: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print(text.strip())
    print()


def slice_period(series: pd.Series, start: str, end: str | None) -> pd.Series:
    if end is None:
        return series.loc[start:]
    return series.loc[start:end]


def rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def rolling_sharpe(returns: pd.Series, window: int, rf: float = RISK_FREE_RATE) -> pd.Series:
    roll_mean = returns.rolling(window).mean() * TRADING_DAYS
    roll_std = returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    return (roll_mean - rf) / roll_std


def rolling_max_drawdown(returns: pd.Series, window: int) -> pd.Series:
    def max_dd(window_returns: np.ndarray) -> float:
        wealth = np.cumprod(1.0 + window_returns)
        peak = np.maximum.accumulate(wealth)
        dd = wealth / peak - 1.0
        return float(dd.min())

    return returns.rolling(window).apply(max_dd, raw=True)


def rolling_beta(port: pd.Series, bench: pd.Series, window: int) -> pd.Series:
    return port.rolling(window).cov(bench) / bench.rolling(window).var()


def rolling_correlation(port: pd.Series, bench: pd.Series, window: int) -> pd.Series:
    return port.rolling(window).corr(bench)


def rolling_tracking_error(port: pd.Series, bench: pd.Series, window: int) -> pd.Series:
    active = port - bench
    return active.rolling(window).std() * np.sqrt(TRADING_DAYS)


def rolling_var(returns: pd.Series, window: int, alpha: float = 0.05) -> pd.Series:
    return returns.rolling(window).apply(lambda x: -np.quantile(x, alpha), raw=True)


def rolling_cvar(returns: pd.Series, window: int, alpha: float = 0.05) -> pd.Series:
    def cvar(x: np.ndarray) -> float:
        var = -np.quantile(x, alpha)
        tail = x[x <= -var]
        return float(-tail.mean()) if len(tail) > 0 else np.nan

    return returns.rolling(window).apply(cvar, raw=True)


def save_line_chart(
    series: pd.Series | pd.DataFrame,
    title: str,
    ylabel: str,
    filename: str,
    pct_format: bool = False,
    hline: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    if isinstance(series, pd.DataFrame):
        series.plot(ax=ax)
    else:
        series.plot(ax=ax, color="steelblue")
    if hline is not None:
        ax.axhline(hline, color="black", lw=0.8, ls="--")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if pct_format:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.axvline(pd.Timestamp("2020-01-01"), color="gray", ls=":", lw=1, alpha=0.7)
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart -> {path}")


# %% Load portfolio returns (same Stage 1 policy)
explain(
    "STAGE 2 — Setup",
    """
We reuse the Stage 1 fixed-weight, monthly-rebalanced portfolio.
No weights are changed. Stage 2 asks: does RISK BEHAVIOR stay understandable
across different market environments?

Two tools:
  1. Rolling diagnostics — how metrics evolve day-by-day using trailing windows.
  2. Subperiod tables — snapshot metrics for named regimes (COVID, rate hikes, etc.).

This is NOT optimization and NOT walk-forward weight tuning.
""",
)

port_returns, bench_returns, weight_drift = build_portfolio_series(ROOT / "data")
aligned = pd.concat([port_returns, bench_returns], axis=1, join="inner").dropna()
port_returns = aligned.iloc[:, 0]
bench_returns = aligned.iloc[:, 1]

print(f"Analysis period: {port_returns.index.min().date()} to {port_returns.index.max().date()}")
print(f"Observations: {len(port_returns)}")


# %% Rolling diagnostics
explain(
    "Rolling diagnostics — what and why",
    """
Each rolling metric uses ONLY past data in its window (no look-ahead).
Default long window = 252 trading days (~1 year); short vol = 63 days (~3 months).

These charts answer: 'If I were a risk manager watching this portfolio live,
would risk stay stable or surprise me in certain years?'
""",
)

W_LONG = 252
W_SHORT = 63

rolling_metrics = {
    "volatility_63": rolling_volatility(port_returns, W_SHORT),
    "volatility_252": rolling_volatility(port_returns, W_LONG),
    "sharpe_252": rolling_sharpe(port_returns, W_LONG),
    "max_drawdown_252": rolling_max_drawdown(port_returns, W_LONG),
    "beta_252": rolling_beta(port_returns, bench_returns, W_LONG),
    "correlation_252": rolling_correlation(port_returns, bench_returns, W_LONG),
    "tracking_error_252": rolling_tracking_error(port_returns, bench_returns, W_LONG),
    "var_95_252": rolling_var(port_returns, W_LONG),
    "cvar_95_252": rolling_cvar(port_returns, W_LONG),
}

rolling_df = pd.DataFrame(rolling_metrics)
rolling_df.to_csv(OUTPUT_DIR / "rolling_metrics_daily.csv")
print(f"Saved daily rolling metrics -> {OUTPUT_DIR / 'rolling_metrics_daily.csv'}")

chart_specs = [
    ("volatility_63", "03_rolling_vol_63.png", True, None),
    ("volatility_252", "04_rolling_vol_252.png", True, None),
    ("sharpe_252", "05_rolling_sharpe_252.png", False, 0.0),
    ("max_drawdown_252", "06_rolling_max_drawdown_252.png", True, None),
    ("beta_252", "07_rolling_beta_252.png", False, 1.0),
    ("correlation_252", "08_rolling_correlation_252.png", False, None),
    ("tracking_error_252", "09_rolling_tracking_error_252.png", True, None),
    ("var_95_252", "10_rolling_var_95_252.png", True, None),
    ("cvar_95_252", "11_rolling_cvar_95_252.png", True, None),
]

for key, fname, pct, hline in chart_specs:
    title, desc = ROLLING_EXPLANATIONS[key]
    print(f"\nChart: {title}")
    print(f"  Risk meaning: {desc}")
    save_line_chart(rolling_metrics[key], title, title.split("(")[0].strip(), fname, pct, hline)

# Combined vol chart (Stage 1 style, kept for comparison)
fig, ax = plt.subplots(figsize=(12, 5))
rolling_metrics["volatility_63"].plot(ax=ax, label="63-day vol")
rolling_metrics["volatility_252"].plot(ax=ax, label="252-day vol")
ax.set_title("Rolling Volatility: 63-day vs 252-day")
ax.set_ylabel("Annualized volatility")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_rolling_vol_combined.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved chart -> {OUTPUT_DIR / '02_rolling_vol_combined.png'}")


# %% Subperiod analysis
explain(
    "Subperiod regime analysis",
    """
We slice history into economically meaningful windows.
Each subperiod gets a full risk snapshot (return, vol, Sharpe, drawdown, VaR, beta, etc.).

This complements rolling charts: rolling = continuous timeline; subperiods = named stories
(COVID, rate hikes) that managers and clients actually remember.
""",
)

subperiod_rows = {}
for name, (start, end) in SUBPERIODS.items():
    p = slice_period(port_returns, start, end)
    b = slice_period(bench_returns, start, end)
    if len(p) < 20:
        print(f"Skipping {name}: insufficient data ({len(p)} days)")
        continue
    m = compute_metrics(p, b)
    subperiod_rows[name] = subperiod_metrics_to_series(m)
    print(f"{name}: {p.index.min().date()} to {p.index.max().date()} ({len(p)} days)")

subperiod_table = pd.DataFrame(subperiod_rows).T
print("\n--- SUBPERIOD METRICS (Portfolio) ---")
print(subperiod_table.to_string())
subperiod_table.to_csv(OUTPUT_DIR / "subperiod_metrics.csv")

# Benchmark subperiod table for comparison
bench_subperiod_rows = {}
for name, (start, end) in SUBPERIODS.items():
    b = slice_period(bench_returns, start, end)
    if len(b) < 20:
        continue
    m = compute_metrics(b, b)
    m.beta = 1.0
    m.tracking_error = 0.0
    m.information_ratio = np.nan
    bench_subperiod_rows[name] = subperiod_metrics_to_series(m)

bench_subperiod_table = pd.DataFrame(bench_subperiod_rows).T
bench_subperiod_table.to_csv(OUTPUT_DIR / "subperiod_metrics_spy.csv")


# Subperiod bar chart — max drawdown comparison
if not subperiod_table.empty:
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(subperiod_table))
    w = 0.35
    ax.bar(x - w / 2, subperiod_table["Maximum Drawdown"], w, label="Portfolio", color="steelblue")
    ax.bar(x + w / 2, bench_subperiod_table.loc[subperiod_table.index, "Maximum Drawdown"], w, label="SPY", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(subperiod_table.index, rotation=25, ha="right")
    ax.set_title("Maximum Drawdown by Subperiod: Portfolio vs SPY")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "12_subperiod_max_drawdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart -> {OUTPUT_DIR / '12_subperiod_max_drawdown.png'}")


# %% Weight drift / rebalancing evidence
explain(
    "Monthly rebalancing & concentration control",
    """
We track how far drifted weights deviate from policy targets each day.
Without rebalancing, equity winners (e.g. QQQ) would dominate over time.
Monthly reset caps max drift — evidence that rebalancing controlled concentration.
""",
)

max_drift = weight_drift["max_drift_from_target"]
print(f"Max single-asset drift from target (any day): {max_drift.max():.2%}")
print(f"Median daily max drift: {max_drift.median():.2%}")

# Drift at month-end (day before rebalance) vs after rebalance
month_end_drift = max_drift.groupby([max_drift.index.year, max_drift.index.month]).max()
print(f"Worst month-end drift within a month (max): {month_end_drift.max():.2%}")

fig, ax = plt.subplots(figsize=(12, 4))
max_drift.plot(ax=ax, color="purple", alpha=0.8)
ax.set_title("Daily Max Weight Drift from Policy Targets (Monthly Rebalance Policy)")
ax.set_ylabel("Max |actual - target| weight")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "13_weight_drift.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved chart -> {OUTPUT_DIR / '13_weight_drift.png'}")


# %% Diversifier contribution (GLD, TLT vs equities in stress subperiods)
gld_rets = pd.read_csv(ROOT / "data" / "vance_etf_prices.csv", index_col=0, parse_dates=True)["GLD"].pct_change()
tlt_rets = pd.read_csv(ROOT / "data" / "vance_etf_prices.csv", index_col=0, parse_dates=True)["TLT"].pct_change()
spy_rets = bench_returns

stress_periods = {
    "2020 COVID Crash": ("2020-02-01", "2020-03-31"),
    "2022 Rate Hike / Bond Drawdown": ("2022-01-01", "2022-12-31"),
}
print("\n--- Diversifier behavior in stress windows (cumulative return) ---")
for label, (s, e) in stress_periods.items():
    g = (1 + gld_rets.loc[s:e]).prod() - 1
    t = (1 + tlt_rets.loc[s:e]).prod() - 1
    s_ret = (1 + spy_rets.loc[s:e]).prod() - 1
    p = (1 + port_returns.loc[s:e]).prod() - 1
    print(f"{label}:")
    print(f"  SPY: {s_ret:.2%}  |  GLD: {g:.2%}  |  TLT: {t:.2%}  |  Portfolio: {p:.2%}")


# %% Excel export
with pd.ExcelWriter(OUTPUT_DIR / "stage2_report.xlsx", engine="openpyxl") as writer:
    rolling_df.to_excel(writer, sheet_name="Rolling Metrics")
    subperiod_table.to_excel(writer, sheet_name="Subperiod Portfolio")
    bench_subperiod_table.to_excel(writer, sheet_name="Subperiod SPY")
    weight_drift.to_excel(writer, sheet_name="Weight Drift")

print(f"\nExcel report -> {OUTPUT_DIR / 'stage2_report.xlsx'}")


# %% Interpretation
def build_interpretation(
    subperiod: pd.DataFrame,
    bench_sub: pd.DataFrame,
    roll: pd.DataFrame,
) -> str:
    # Identify worst subperiods by drawdown and vol
    worst_dd = subperiod["Maximum Drawdown"].idxmin()
    worst_vol = subperiod["Annualized Volatility"].idxmax()
    best_sharpe = subperiod["Sharpe Ratio"].idxmax()

    vol_252_recent = roll["volatility_252"].iloc[-1]
    vol_252_median = roll["volatility_252"].median()
    beta_recent = roll["beta_252"].iloc[-1]
    corr_recent = roll["correlation_252"].iloc[-1]

    is_row = subperiod.loc["2009-2019 In-Sample"] if "2009-2019 In-Sample" in subperiod.index else None
    covid = subperiod.loc["2020 COVID Crash"] if "2020 COVID Crash" in subperiod.index else None
    rate22 = subperiod.loc["2022 Rate Hike / Bond Drawdown"] if "2022 Rate Hike / Bond Drawdown" in subperiod.index else None

    text = f"""
STAGE 2 INTERPRETATION - Rolling Diagnostics & Regime Analysis
{'=' * 72}

DISCLAIMER: Approximate educational Vance-style portfolio. Not investment advice.

1) IS RISK BEHAVIOR STABLE OVER TIME?
   Rolling 252-day volatility median: {vol_252_median:.2%}
   Latest rolling 252-day volatility: {vol_252_median:.2%} -> recent: {vol_252_recent:.2%}
   Worst subperiod volatility: {worst_vol} ({subperiod.loc[worst_vol, 'Annualized Volatility']:.2%})
   Best subperiod Sharpe: {best_sharpe} ({subperiod.loc[best_sharpe, 'Sharpe Ratio']:.2f})

   Reading: Risk is NOT constant. Volatility spikes in crisis years and calms in
   steady bull markets. A fixed-weight policy still experiences regime-dependent risk
   even without changing weights. Rolling charts make that visible in real time.

2) WHICH MARKET ENVIRONMENTS HURT MOST?
   Worst subperiod max drawdown: {worst_dd} ({subperiod.loc[worst_dd, 'Maximum Drawdown']:.2%})
   SPY drawdown same period: {bench_sub.loc[worst_dd, 'Maximum Drawdown']:.2%}

   COVID 2020: portfolio vol {covid['Annualized Volatility']:.2%} vs SPY {bench_sub.loc['2020 COVID Crash', 'Annualized Volatility']:.2%} (if available)
   2022 rate hikes: portfolio return {rate22['Annualized Return']:.2%}, max DD {rate22['Maximum Drawdown']:.2%}

   2022 is often painful for diversified portfolios because BOTH equities and long
   bonds (TLT) sold off — diversification failed temporarily. COVID was equity-heavy
   stress where bonds/gold helped more.

3) DID GLD AND TLT HELP DIVERSIFY EQUITY RISK?
   Latest rolling correlation vs SPY: {corr_recent:.2f} (1.0 = moves like SPY)
   Latest rolling beta vs SPY: {beta_recent:.2f}

   When correlation drops and beta < 1, diversifiers are doing their job.
   In 2022, TLT often moved WITH equities (both down) — correlation rose and
   diversification benefit disappeared. GLD helps more in inflation/geopolitical shocks.

4) DID MONTHLY REBALANCING CONTROL CONCENTRATION?
   Max weight drift observed: {max_drift.max():.2%}
   Median daily max drift: {max_drift.median():.2%}

   Without monthly reset, equity ETFs (especially QQQ in strong tech cycles) would
   dominate weights. Rebalancing forces sell-high/buy-low vs policy — controlling
   concentration but not eliminating regime risk.

5) IMPLICATIONS BEFORE STAGE 3 (OPTIMIZATION)
   - The policy is understandable and relatively defensive vs SPY (lower beta, lower DD
     in several subperiods), but it is NOT risk-stable across all regimes.
   - Optimization (Stage 3) should not assume one static vol/beta from 2009-2019.
   - Any weight change should be stress-tested in 2020 and 2022-like windows.
   - Consider whether 10% TLT is enough duration hedge when rates rise.
   - Stage 3 can explore weights, but Stage 2 shows WHERE the current policy fails
     and succeeds — optimization must improve weak regimes, not just average return.

{'=' * 72}
Outputs: {OUTPUT_DIR}
"""
    # Fix typo in template - vol recent line
    text = text.replace(
        f"Latest rolling 252-day volatility: {vol_252_median:.2%} -> recent: {vol_252_recent:.2%}",
        f"Latest rolling 252-day volatility: {vol_252_recent:.2%} (median: {vol_252_median:.2%})",
    )
    if is_row is not None:
        pass
    return text


interpretation = build_interpretation(subperiod_table, bench_subperiod_table, rolling_df)
print(interpretation)

with open(OUTPUT_DIR / "interpretation_stage2.txt", "w", encoding="utf-8") as f:
    f.write(interpretation)

print(f"Interpretation saved -> {OUTPUT_DIR / 'interpretation_stage2.txt'}")
print("\nStage 2 complete.")
