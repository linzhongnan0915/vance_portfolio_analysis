# %% [markdown]
# # Overfitting Diagnostics — Structured IS vs OOS Evaluation
#
# Overfitting is NOT one number. It is a **pattern** diagnosed across multiple signals.

# %% Setup
"""
overfitting_diagnostics.py
Structured overfitting diagnostics for walk-forward strategies.
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

from portfolio_core import (
    BENCHMARK,
    IN_SAMPLE_END,
    OUT_SAMPLE_START,
    TICKERS,
    compute_drawdown,
    compute_metrics,
    load_adjusted_prices,
    prices_to_returns,
)
from walk_forward_engine import STRATEGIES, run_walk_forward

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.display.float_format = "{:.4f}".format

from src.config import ROOT
OUTPUT_DIR = ROOT / "output" / "overfitting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IS_END = pd.Timestamp(IN_SAMPLE_END)
OOS_START = pd.Timestamp(OUT_SAMPLE_START)
TRAIN_WINDOWS = [6, 9, 12, 18, 24]
TX_COST_BPS = [0, 10, 25]  # basis points per unit turnover (one-way)


def explain(title: str, text: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print(text.strip())
    print()


def risk_flag_sharpe_degradation(ratio: float) -> tuple[str, str]:
    if np.isnan(ratio):
        return "N/A", "medium"
    if ratio >= 0.85:
        return ">= 0.85: OOS Sharpe largely preserved", "low"
    if ratio >= 0.70:
        return "0.70-0.85: moderate degradation", "medium"
    return "< 0.70: material Sharpe collapse OOS", "high"


def risk_flag_return_degradation(ratio: float) -> tuple[str, str]:
    if np.isnan(ratio):
        return "N/A", "medium"
    if ratio >= 0.80:
        return ">= 0.80: return largely preserved", "low"
    if ratio >= 0.60:
        return "0.60-0.80: moderate return fade", "medium"
    return "< 0.60: strong return fade OOS", "high"


def risk_flag_dd_deterioration(delta: float) -> tuple[str, str]:
    """delta = OOS max DD - IS max DD (more negative = worse)."""
    if np.isnan(delta):
        return "N/A", "medium"
    if delta >= -0.03:
        return "<= 3pp worse: stable drawdown profile", "low"
    if delta >= -0.08:
        return "3-8pp worse: moderate tail deterioration", "medium"
    return "> 8pp worse: severe drawdown deterioration OOS", "high"


def risk_flag_diff(metric_diff: float, low: float, high: float, higher_is_better: bool) -> tuple[str, str]:
    if np.isnan(metric_diff):
        return "N/A", "medium"
    if higher_is_better:
        if metric_diff >= low:
            return f"OOS - IS >= {low}: improvement or stable", "low"
        if metric_diff >= high:
            return f"OOS - IS between {high} and {low}: mild fade", "medium"
        return f"OOS - IS < {high}: material fade", "high"
    else:
        if metric_diff <= low:
            return f"OOS - IS <= {low}: stable", "low"
        if metric_diff <= high:
            return "moderate deterioration", "medium"
        return "material deterioration", "high"


def risk_flag_pct_beat(pct: float, benchmark_label: str) -> tuple[str, str]:
    if np.isnan(pct):
        return "N/A", "medium"
    if pct >= 0.55:
        return f">= 55% months beat {benchmark_label}", "low"
    if pct >= 0.45:
        return f"45-55%: near coin-flip vs {benchmark_label}", "medium"
    return f"< 45%: rarely beats {benchmark_label} OOS", "high"


def risk_flag_turnover(avg_to: float) -> tuple[str, str]:
    if np.isnan(avg_to):
        return "N/A", "medium"
    if avg_to <= 0.015:
        return "<= 1.5%/mo: low turnover", "low"
    if avg_to <= 0.03:
        return "1.5-3%/mo: moderate turnover", "medium"
    return "> 3%/mo: high turnover / cost sensitive", "high"


def risk_flag_cost_impact(sharpe_drop: float) -> tuple[str, str]:
    if np.isnan(sharpe_drop):
        return "N/A", "medium"
    if sharpe_drop <= 0.05:
        return "<= 0.05 Sharpe lost at 25bps: robust after costs", "low"
    if sharpe_drop <= 0.15:
        return "0.05-0.15 Sharpe lost: moderately cost-sensitive", "medium"
    return "> 0.15 Sharpe lost: fragile after costs", "high"


def risk_flag_train_sensitivity(sharpe_range: float) -> tuple[str, str]:
    if np.isnan(sharpe_range):
        return "N/A", "medium"
    if sharpe_range <= 0.15:
        return "<= 0.15 Sharpe range across windows: stable", "low"
    if sharpe_range <= 0.30:
        return "0.15-0.30: moderate parameter sensitivity", "medium"
    return "> 0.30: high sensitivity to training window", "high"


def add_row(
    rows: list[dict],
    strategy: str,
    name: str,
    value: float | str,
    interpretation: str,
    flag: str,
) -> None:
    rows.append(
        {
            "strategy": strategy,
            "diagnostic_name": name,
            "value": value,
            "threshold_or_interpretation": interpretation,
            "risk_flag": flag,
        }
    )


def split_is_oos_daily(daily: pd.Series) -> tuple[pd.Series, pd.Series]:
    is_s = daily.loc[:IS_END]
    oos_s = daily.loc[OOS_START:]
    return is_s, oos_s


def split_log_is_oos(log: pd.DataFrame, strategy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = log[log["strategy"] == strategy].copy()
    s["testing_start"] = pd.to_datetime(s["testing_start"])
    is_df = s[s["testing_start"] <= IS_END]
    oos_df = s[s["testing_start"] >= OOS_START]
    return is_df, oos_df


def weight_instability(weight_hist: pd.DataFrame) -> float:
    """Mean L1 weight change month-to-month."""
    if weight_hist.empty or len(weight_hist) < 2:
        return np.nan
    wcols = [c for c in weight_hist.columns if c.startswith("weight_")]
    diffs = weight_hist[wcols].diff().abs().sum(axis=1)
    return diffs.iloc[1:].mean()


def apply_transaction_costs(log: pd.DataFrame, cost_bps: float) -> pd.Series:
    cost = cost_bps / 10_000.0
    net = log["monthly_portfolio_return"] - log["turnover"] * cost
    return pd.Series(net.values, index=pd.to_datetime(log["testing_start"]))


def metrics_from_monthly(port_m: pd.Series, bench_m: pd.Series) -> dict:
    """Approximate metrics from monthly return series (OOS cost analysis)."""
    aligned = pd.concat([port_m, bench_m], axis=1, join="inner").dropna()
    rp = aligned.iloc[:, 0]
    rb = aligned.iloc[:, 1]
    n = len(rp)
    if n < 2:
        return {"sharpe": np.nan, "ann_return": np.nan, "max_dd": np.nan}
    ann_ret = (1 + rp).prod() ** (12 / n) - 1
    ann_vol = rp.std(ddof=1) * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    max_dd = compute_drawdown((1 + rp).cumprod()).min()
    return {"sharpe": sharpe, "ann_return": ann_ret, "max_dd": max_dd}


def build_strategy_diagnostics(
    strategy: str,
    daily: pd.Series,
    bench: pd.Series,
    log: pd.DataFrame,
    weight_hist: pd.DataFrame,
    fixed_oos_log: pd.DataFrame,
) -> list[dict]:
    rows: list[dict] = []
    is_d, oos_d = split_is_oos_daily(daily)
    is_b, oos_b = split_is_oos_daily(bench)
    is_log, oos_log = split_log_is_oos(log, strategy)

    m_is = compute_metrics(is_d, is_b)
    m_oos = compute_metrics(oos_d, oos_b)

    sharpe_deg = m_oos.sharpe_ratio / m_is.sharpe_ratio if m_is.sharpe_ratio not in (0, np.nan) else np.nan
    ret_deg = (
        m_oos.annualized_return / m_is.annualized_return
        if m_is.annualized_return not in (0, np.nan)
        else np.nan
    )
    dd_det = m_oos.max_drawdown - m_is.max_drawdown
    calmar_diff = m_oos.calmar_ratio - m_is.calmar_ratio
    ir_diff = m_oos.information_ratio - m_is.information_ratio

    interp, flag = risk_flag_sharpe_degradation(sharpe_deg)
    add_row(rows, strategy, "Sharpe degradation ratio (OOS/IS)", f"{sharpe_deg:.3f}", interp, flag)

    interp, flag = risk_flag_return_degradation(ret_deg)
    add_row(rows, strategy, "Return degradation ratio (OOS/IS)", f"{ret_deg:.3f}", interp, flag)

    interp, flag = risk_flag_dd_deterioration(dd_det)
    add_row(
        rows,
        strategy,
        "Max drawdown deterioration (OOS DD - IS DD)",
        f"{dd_det:.2%}",
        interp,
        flag,
    )

    interp, flag = risk_flag_diff(calmar_diff, -0.10, -0.30, higher_is_better=True)
    add_row(rows, strategy, "Calmar ratio difference (OOS - IS)", f"{calmar_diff:.3f}", interp, flag)

    interp, flag = risk_flag_diff(ir_diff, -0.05, -0.20, higher_is_better=True)
    add_row(rows, strategy, "Information ratio difference (OOS - IS)", f"{ir_diff:.3f}", interp, flag)

    # Walk-forward monthly win rates (OOS)
    if len(oos_log):
        pct_spy = (oos_log["monthly_portfolio_return"] > oos_log["monthly_benchmark_return"]).mean()
        interp, flag = risk_flag_pct_beat(pct_spy, "SPY")
        add_row(
            rows,
            strategy,
            "% walk-forward OOS months beating SPY",
            f"{pct_spy:.1%}",
            interp,
            flag,
        )

        if strategy != "fixed_baseline" and len(fixed_oos_log):
            merged = oos_log.merge(
                fixed_oos_log[["testing_start", "monthly_portfolio_return"]],
                on="testing_start",
                suffixes=("", "_fixed"),
            )
            pct_fixed = (merged["monthly_portfolio_return"] > merged["monthly_portfolio_return_fixed"]).mean()
            interp, flag = risk_flag_pct_beat(pct_fixed, "fixed baseline")
            add_row(
                rows,
                strategy,
                "% walk-forward OOS months beating fixed baseline",
                f"{pct_fixed:.1%}",
                interp,
                flag,
            )

    avg_to = oos_log["turnover"].mean() if len(oos_log) else np.nan
    max_to = oos_log["turnover"].max() if len(oos_log) else np.nan
    interp, flag = risk_flag_turnover(avg_to)
    add_row(rows, strategy, "Average monthly turnover (OOS)", f"{avg_to:.2%}", interp, flag)
    add_row(rows, strategy, "Maximum monthly turnover (OOS)", f"{max_to:.2%}", "Peak rebalance intensity", flag)

    # Transaction costs (OOS monthly — compare gross vs net at same frequency)
    if len(oos_log):
        bench_m = pd.Series(
            oos_log["monthly_benchmark_return"].values,
            index=pd.to_datetime(oos_log["testing_start"]),
        )
        gross_m = pd.Series(
            oos_log["monthly_portfolio_return"].values,
            index=pd.to_datetime(oos_log["testing_start"]),
        )
        gross_sharpe_m = metrics_from_monthly(gross_m, bench_m)["sharpe"]
        for bps in TX_COST_BPS:
            if bps == 0:
                continue
            net_m = apply_transaction_costs(oos_log, bps)
            m_net = metrics_from_monthly(net_m, bench_m)
            sharpe_drop = gross_sharpe_m - m_net["sharpe"]
            interp, flag = risk_flag_cost_impact(sharpe_drop)
            add_row(
                rows,
                strategy,
                f"Sharpe drop at {bps}bps turnover cost (OOS)",
                f"{sharpe_drop:.3f}",
                interp,
                flag,
            )
            add_row(
                rows,
                strategy,
                f"OOS Sharpe after {bps}bps costs",
                f"{m_net['sharpe']:.3f}",
                f"Monthly Sharpe: net vs gross {gross_sharpe_m:.3f}",
                flag,
            )

    # Weight instability (dynamic strategies)
    if strategy != "fixed_baseline":
        instab = weight_instability(weight_hist)
        if instab <= 0.02:
            i_interp, i_flag = "<= 2pp avg weight change: stable weights", "low"
        elif instab <= 0.05:
            i_interp, i_flag = "2-5pp: moderate weight churn", "medium"
        else:
            i_interp, i_flag = "> 5pp: unstable weights / overfitting risk", "high"
        add_row(rows, strategy, "Avg month-to-month weight change (L1)", f"{instab:.2%}", i_interp, i_flag)

    return rows


def training_window_sensitivity(
    asset_returns: pd.DataFrame,
    bench_returns: pd.Series,
    strategy: str,
) -> pd.DataFrame:
    records = []
    for tw in TRAIN_WINDOWS:
        wf = run_walk_forward(asset_returns, bench_returns, train_months=tw, strategies=[strategy])
        oos_d = wf.daily_returns[strategy].loc[OOS_START:]
        oos_b = bench_returns.loc[oos_d.index]
        m = compute_metrics(oos_d, oos_b)
        records.append(
            {
                "strategy": strategy,
                "train_months": tw,
                "oos_sharpe": m.sharpe_ratio,
                "oos_max_drawdown": m.max_drawdown,
                "oos_ann_return": m.annualized_return,
            }
        )
    return pd.DataFrame(records)


def overall_overfitting_score(flags: list[str]) -> str:
    high = flags.count("high")
    med = flags.count("medium")
    if high >= 3:
        return "HIGH overfitting concern"
    if high >= 1 or med >= 4:
        return "MEDIUM overfitting concern"
    return "LOW overfitting concern"


# =============================================================================
explain(
    "OVERFITTING DIAGNOSTICS — How to read this report",
    """
Overfitting in portfolio backtesting is NOT proven by one number.

It is diagnosed through a PATTERN:
  - Strong in-sample (IS) performance that fades out-of-sample (OOS)
  - Unstable weights or parameters month-to-month
  - High turnover that erodes results after transaction costs
  - Poor robustness when training window or assumptions change slightly

This report computes structured diagnostics with risk flags (low / medium / high)
for each strategy from the Stage 3 walk-forward framework.

IS period: walk-forward test months with testing_start <= 2019-12-31
OOS period: walk-forward test months with testing_start >= 2020-01-01
""",
)

prices = load_adjusted_prices(ROOT / "data")
daily = prices_to_returns(prices)[TICKERS]
bench = prices_to_returns(prices)[BENCHMARK]

wf = run_walk_forward(daily, bench)
log = wf.decision_log
fixed_oos = split_log_is_oos(log, "fixed_baseline")[1]

all_rows: list[dict] = []
for strategy in STRATEGIES:
    all_rows.extend(
        build_strategy_diagnostics(
            strategy,
            wf.daily_returns[strategy],
            bench.loc[wf.daily_returns[strategy].index],
            log,
            wf.weight_history[strategy],
            fixed_oos,
        )
    )

diag_df = pd.DataFrame(all_rows)

# Training-window sensitivity (dynamic strategies)
sens_frames = []
for strat in ["inverse_volatility", "minimum_variance"]:
    sens_frames.append(training_window_sensitivity(daily, bench, strat))
sensitivity_df = pd.concat(sens_frames, ignore_index=True)
sensitivity_df.to_csv(OUTPUT_DIR / "training_window_sensitivity.csv", index=False)

for strat in ["inverse_volatility", "minimum_variance"]:
    sub = sensitivity_df[sensitivity_df["strategy"] == strat]
    sharpe_range = sub["oos_sharpe"].max() - sub["oos_sharpe"].min()
    dd_range = sub["oos_max_drawdown"].max() - sub["oos_max_drawdown"].min()
    interp, flag = risk_flag_train_sensitivity(sharpe_range)
    diag_df = pd.concat(
        [
            diag_df,
            pd.DataFrame(
                [
                    {
                        "strategy": strat,
                        "diagnostic_name": "OOS Sharpe range across training windows (6-24mo)",
                        "value": f"{sharpe_range:.3f}",
                        "threshold_or_interpretation": interp,
                        "risk_flag": flag,
                    },
                    {
                        "strategy": strat,
                        "diagnostic_name": "OOS max drawdown range across training windows",
                        "value": f"{dd_range:.2%}",
                        "threshold_or_interpretation": "Wide range = drawdown sensitive to window choice",
                        "risk_flag": "medium" if abs(dd_range) > 0.05 else "low",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

# Overall score per strategy
summary_rows = []
for strat in STRATEGIES:
    flags = diag_df.loc[diag_df["strategy"] == strat, "risk_flag"].tolist()
    summary_rows.append(
        {
            "strategy": strat,
            "diagnostic_name": "OVERALL OVERFITTING ASSESSMENT",
            "value": overall_overfitting_score(flags),
            "threshold_or_interpretation": f"Based on {flags.count('high')} high, {flags.count('medium')} medium flags",
            "risk_flag": overall_overfitting_score(flags).split()[0].lower(),
        }
    )
diag_df = pd.concat([diag_df, pd.DataFrame(summary_rows)], ignore_index=True)

diag_df.to_csv(OUTPUT_DIR / "overfitting_diagnostics.csv", index=False)
print("\n--- OVERFITTING DIAGNOSTICS TABLE ---")
print(diag_df.to_string(index=False))

# Summary pivot: IS vs OOS metrics
is_oos_summary = []
for strat in STRATEGIES:
    d = wf.daily_returns[strat]
    b = bench.loc[d.index]
    is_d, oos_d = split_is_oos_daily(d)
    is_b, oos_b = split_is_oos_daily(b)
    for label, p, bm in [("IS", is_d, is_b), ("OOS", oos_d, oos_b)]:
        m = compute_metrics(p, bm)
        is_oos_summary.append(
            {
                "strategy": strat,
                "period": label,
                "ann_return": m.annualized_return,
                "ann_vol": m.annualized_volatility,
                "sharpe": m.sharpe_ratio,
                "max_dd": m.max_drawdown,
                "calmar": m.calmar_ratio,
                "info_ratio": m.information_ratio,
            }
        )
is_oos_df = pd.DataFrame(is_oos_summary)
is_oos_df.to_csv(OUTPUT_DIR / "is_oos_summary.csv", index=False)

# Chart: IS vs OOS Sharpe by strategy
pivot = is_oos_df.pivot(index="strategy", columns="period", values="sharpe")
fig, ax = plt.subplots(figsize=(8, 4))
pivot.plot(kind="bar", ax=ax, rot=0)
ax.set_title("Sharpe Ratio: In-Sample vs Out-of-Sample")
ax.axhline(0, color="black", lw=0.5)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "sharpe_is_vs_oos.png", dpi=150)
plt.close(fig)

# Chart: training window sensitivity
fig, ax = plt.subplots(figsize=(9, 4))
for strat in ["inverse_volatility", "minimum_variance"]:
    sub = sensitivity_df[sensitivity_df["strategy"] == strat]
    ax.plot(sub["train_months"], sub["oos_sharpe"], marker="o", label=strat)
ax.set_xlabel("Training window (months)")
ax.set_ylabel("OOS Sharpe (2020+)")
ax.set_title("Training Window Sensitivity — OOS Sharpe")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "training_window_sharpe.png", dpi=150)
plt.close(fig)

with pd.ExcelWriter(OUTPUT_DIR / "overfitting_report.xlsx", engine="openpyxl") as writer:
    diag_df.to_excel(writer, sheet_name="Diagnostics", index=False)
    is_oos_df.to_excel(writer, sheet_name="IS vs OOS Summary", index=False)
    sensitivity_df.to_excel(writer, sheet_name="Train Window Sens", index=False)

narrative = f"""
OVERFITTING DIAGNOSTICS NARRATIVE
{'=' * 72}

Pattern-based diagnosis (not a single score):
  Overfitting appears when IS looks strong but OOS weakens, weights jump around,
  turnover is high, and results vanish after costs or small parameter changes.

KEY FINDINGS (walk-forward, IS <= 2019, OOS >= 2020):
{is_oos_df.to_string(index=False)}

See overfitting_diagnostics.csv for full flagged table.

Files: {OUTPUT_DIR}
"""
print(narrative)
(OUTPUT_DIR / "interpretation_overfitting.txt").write_text(narrative, encoding="utf-8")
print("Overfitting diagnostics complete.")
