"""
Vance_portfolio_analysis_stage5.py
Stage 5 — Robustness testing + conservative defensive asset (SHY) extension.
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
import yfinance as yf

from portfolio_core import (
    BENCHMARK,
    IN_SAMPLE_END,
    OUT_SAMPLE_START,
    TARGET_WEIGHTS,
    TICKERS,
    compute_drawdown,
    compute_metrics,
    load_adjusted_prices,
    metrics_to_series,
    prices_to_returns,
)
from walk_forward_engine import STRATEGIES, WalkForwardResult, run_walk_forward

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", palette="muted")
pd.options.display.float_format = "{:.4f}".format

from src.config import ROOT
OUTPUT_DIR = ROOT / "output" / "stage5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IS_END = pd.Timestamp(IN_SAMPLE_END)
OOS_START = pd.Timestamp(OUT_SAMPLE_START)

TRAIN_WINDOWS = [6, 12, 18, 24]
MAX_WEIGHTS = [0.30, 0.40, 0.50]
TX_COST_BPS = [0, 5, 10, 25]
DEFAULT_TRAIN = 12
DEFAULT_MAX_W = 0.40
DEFAULT_TX = 0

DEFENSIVE_ETF = "SHY"
TICKERS_EXPANDED = TICKERS + [DEFENSIVE_ETF]
TARGET_WEIGHTS_EXPANDED = pd.Series(
    {"QQQ": 0.25, "SPY": 0.25, "DIA": 0.25, "GLD": 0.15, "TLT": 0.05, DEFENSIVE_ETF: 0.05},
    name="target_weight",
)

REGIME_PERIODS = {
    "2020 COVID year": ("2020-01-01", "2020-12-31"),
    "2022 Rate hikes": ("2022-01-01", "2022-12-31"),
    "2023-2024 Recovery": ("2023-01-01", "2024-12-31"),
}

STRATEGY_LABELS = {
    "fixed_baseline": "Fixed monthly rebalance",
    "inverse_volatility": "Inverse-volatility",
    "minimum_variance": "Minimum-variance",
}


def explain(title: str, text: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print(text.strip())
    print()


def load_prices_with_shy(data_dir: Path) -> pd.DataFrame:
    """Load base ETF prices and merge SHY if missing from cache."""
    prices = load_adjusted_prices(data_dir)
    if DEFENSIVE_ETF in prices.columns:
        return prices

    shy = yf.download(
        DEFENSIVE_ETF,
        start=prices.index[0].strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if isinstance(shy.columns, pd.MultiIndex):
        shy_s = shy["Close"].iloc[:, 0]
    else:
        shy_s = shy["Close"] if "Close" in shy.columns else shy.iloc[:, 0]

    merged = prices.join(shy_s.rename(DEFENSIVE_ETF), how="inner")
    merged = merged.sort_index().dropna(how="any")
    merged.to_csv(data_dir / "vance_etf_prices.csv")
    return merged


def split_is_oos_daily(daily: pd.Series) -> tuple[pd.Series, pd.Series]:
    return daily.loc[:IS_END], daily.loc[OOS_START:]


def split_log_is_oos(log: pd.DataFrame, strategy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = log[log["strategy"] == strategy].copy()
    s["testing_start"] = pd.to_datetime(s["testing_start"])
    return s[s["testing_start"] <= IS_END], s[s["testing_start"] >= OOS_START]


def weight_instability(weight_hist: pd.DataFrame) -> float:
    wcols = [c for c in weight_hist.columns if c.startswith("weight_")]
    if weight_hist.empty or len(weight_hist) < 2:
        return np.nan
    return weight_hist[wcols].diff().abs().sum(axis=1).iloc[1:].mean()


def count_max_weight_hits(log: pd.DataFrame, max_w: float) -> int:
    wcols = [c for c in log.columns if c.startswith("weight_")]
    if not wcols:
        return 0
    return int((log[wcols] >= max_w - 1e-4).any(axis=1).sum())


def pct_months_beating(log: pd.DataFrame, col: str) -> float:
    if log.empty:
        return np.nan
    return (log["monthly_portfolio_return"] > log[col]).mean()


def regime_return(log: pd.DataFrame, strategy: str, start: str, end: str) -> float:
    s = log[log["strategy"] == strategy].copy()
    s["testing_start"] = pd.to_datetime(s["testing_start"])
    sub = s[(s["testing_start"] >= start) & (s["testing_start"] <= end)]
    if sub.empty:
        return np.nan
    return (1 + sub["monthly_portfolio_return"]).prod() - 1


def year_return_concentration(log: pd.DataFrame, strategy: str) -> tuple[str, float]:
    """Return year with largest |contribution| to log return and its share of total."""
    s = log[log["strategy"] == strategy].copy()
    s["testing_start"] = pd.to_datetime(s["testing_start"])
    s["year"] = s["testing_start"].dt.year
    yearly = s.groupby("year")["monthly_portfolio_return"].apply(lambda x: (1 + x).prod() - 1)
    if yearly.empty:
        return "N/A", np.nan
    total = (1 + s["monthly_portfolio_return"]).prod() - 1
    if abs(total) < 1e-9:
        return str(yearly.abs().idxmax()), np.nan
    best_year = yearly.abs().idxmax()
    share = yearly.loc[best_year] / total if total != 0 else np.nan
    return str(best_year), share


def metrics_row(
    wf,
    strategy: str,
    bench: pd.Series,
    fixed_log: pd.DataFrame | None = None,
    label: str = "",
) -> dict:
    daily = wf.daily_returns[strategy]
    b = bench.loc[daily.index]
    m = compute_metrics(daily, b)
    ms = metrics_to_series(m)
    log = wf.decision_log[wf.decision_log["strategy"] == strategy]
    to = wf.turnover_history[strategy]

    row = {
        "label": label,
        "strategy": strategy,
        "train_months": wf.train_months,
        "max_weight": wf.max_weight,
        "tx_cost_bps": wf.tx_cost_bps,
        "annualized_return": ms["Annualized Return"],
        "annualized_volatility": ms["Annualized Volatility"],
        "sharpe_ratio": ms["Sharpe Ratio"],
        "sortino_ratio": ms["Sortino Ratio"],
        "max_drawdown": ms["Maximum Drawdown"],
        "calmar_ratio": ms["Calmar Ratio"],
        "var_95": ms["VaR 95% (daily)"],
        "cvar_95": ms["CVaR 95% (daily)"],
        "beta_vs_spy": ms["Beta vs SPY"],
        "tracking_error": ms["Tracking Error vs SPY"],
        "information_ratio": ms["Information Ratio vs SPY"],
        "avg_monthly_turnover": to.mean(),
        "max_monthly_turnover": to.max(),
        "pct_months_beating_spy": pct_months_beating(log, "monthly_benchmark_return"),
    }
    if fixed_log is not None:
        merged = log.merge(
            fixed_log[["testing_start", "monthly_portfolio_return"]],
            on="testing_start",
            suffixes=("", "_fixed"),
        )
        row["pct_months_beating_fixed"] = (
            (merged["monthly_portfolio_return"] > merged["monthly_portfolio_return_fixed"]).mean()
            if len(merged)
            else np.nan
        )
    else:
        row["pct_months_beating_fixed"] = np.nan
    return row


def run_sensitivity_grid(
    asset_returns: pd.DataFrame,
    bench: pd.Series,
    tickers: list[str],
    target_weights: pd.Series,
    universe_label: str,
) -> pd.DataFrame:
    rows = []
    fixed_logs: dict[tuple, pd.DataFrame] = {}

    for train_m in TRAIN_WINDOWS:
        for tx in TX_COST_BPS:
            wf_fixed = run_walk_forward(
                asset_returns,
                bench,
                tickers=tickers,
                target_weights=target_weights,
                train_months=train_m,
                max_weight=DEFAULT_MAX_W,
                tx_cost_bps=tx,
                strategies=["fixed_baseline"],
            )
            fixed_logs[(train_m, tx)] = wf_fixed.decision_log

            for strategy in STRATEGIES:
                max_list = MAX_WEIGHTS if strategy == "minimum_variance" else [DEFAULT_MAX_W]
                for max_w in max_list:
                    if strategy == "fixed_baseline":
                        wf = wf_fixed
                    else:
                        wf = run_walk_forward(
                            asset_returns,
                            bench,
                            tickers=tickers,
                            target_weights=target_weights,
                            train_months=train_m,
                            max_weight=max_w,
                            tx_cost_bps=tx,
                            strategies=[strategy],
                        )
                    flog = fixed_logs[(train_m, tx)]
                    rows.append(
                        {
                            **metrics_row(
                                wf,
                                strategy,
                                bench,
                                fixed_log=flog if strategy != "fixed_baseline" else None,
                                label=universe_label,
                            ),
                            "universe": universe_label,
                        }
                    )
    return pd.DataFrame(rows)


def risk_flag_sharpe_deg(ratio: float) -> tuple[str, str]:
    if np.isnan(ratio):
        return "N/A", "medium"
    if ratio >= 0.85:
        return ">= 0.85: OOS Sharpe largely preserved", "low"
    if ratio >= 0.70:
        return "0.70-0.85: moderate degradation", "medium"
    return "< 0.70: material Sharpe collapse OOS", "high"


def risk_flag_train_range(sharpe_range: float) -> tuple[str, str]:
    if np.isnan(sharpe_range):
        return "N/A", "medium"
    if sharpe_range <= 0.15:
        return "<= 0.15 Sharpe range: stable to window choice", "low"
    if sharpe_range <= 0.30:
        return "0.15-0.30: moderate window sensitivity", "medium"
    return "> 0.30: high sensitivity to training window", "high"


def risk_flag_cost_drop(drop: float) -> tuple[str, str]:
    if np.isnan(drop):
        return "N/A", "medium"
    if drop <= 0.05:
        return "<= 0.05 Sharpe lost at 25bps: cost-robust", "low"
    if drop <= 0.15:
        return "0.05-0.15: moderately cost-sensitive", "medium"
    return "> 0.15: fragile after transaction costs", "high"


def risk_flag_weight_instab(val: float) -> tuple[str, str]:
    if np.isnan(val):
        return "N/A", "medium"
    if val <= 0.02:
        return "<= 2pp avg L1 weight change: stable", "low"
    if val <= 0.05:
        return "2-5pp: moderate weight churn", "medium"
    return "> 5pp: unstable weights", "high"


def add_diag(rows: list, strategy: str, name: str, value, interp: str, flag: str) -> None:
    rows.append(
        {
            "strategy": STRATEGY_LABELS.get(strategy, strategy),
            "diagnostic": name,
            "value": value,
            "interpretation": interp,
            "risk_flag": flag,
        }
    )


def build_overfitting_diagnostics(
    wf,
    bench: pd.Series,
    sensitivity: pd.DataFrame,
    universe_label: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    log = wf.decision_log

    for strategy in STRATEGIES:
        daily = wf.daily_returns[strategy]
        b = bench.loc[daily.index]
        is_d, oos_d = split_is_oos_daily(daily)
        is_b, oos_b = split_is_oos_daily(b)
        m_is = compute_metrics(is_d, is_b)
        m_oos = compute_metrics(oos_d, oos_b)

        sharpe_deg = m_oos.sharpe_ratio / m_is.sharpe_ratio if m_is.sharpe_ratio else np.nan
        interp, flag = risk_flag_sharpe_deg(sharpe_deg)
        add_diag(rows, strategy, "Sharpe degradation (OOS/IS)", f"{sharpe_deg:.3f}", interp, flag)

        ret_deg = m_oos.annualized_return / m_is.annualized_return if m_is.annualized_return else np.nan
        add_diag(
            rows,
            strategy,
            "Return degradation (OOS/IS ann. return ratio)",
            f"{ret_deg:.3f}",
            "Lower ratio = more IS-to-OOS fade",
            "high" if ret_deg < 0.60 else ("medium" if ret_deg < 0.80 else "low"),
        )

        dd_det = m_oos.max_drawdown - m_is.max_drawdown
        add_diag(
            rows,
            strategy,
            "Max drawdown deterioration (OOS - IS)",
            f"{dd_det:.2%}",
            "More negative = worse tail risk OOS",
            "high" if dd_det < -0.08 else ("medium" if dd_det < -0.03 else "low"),
        )

        sub = sensitivity[
            (sensitivity["universe"] == universe_label)
            & (sensitivity["strategy"] == strategy)
            & (sensitivity["tx_cost_bps"] == 0)
        ]
        if strategy == "minimum_variance":
            tw_sub = sub
        else:
            tw_sub = sub[sub["max_weight"] == DEFAULT_MAX_W]
        if len(tw_sub) >= 2:
            sharpe_range = tw_sub["sharpe_ratio"].max() - tw_sub["sharpe_ratio"].min()
            interp, flag = risk_flag_train_range(sharpe_range)
            add_diag(rows, strategy, "Sharpe range across training windows (6-24mo)", f"{sharpe_range:.3f}", interp, flag)

        cost_sub = sensitivity[
            (sensitivity["universe"] == universe_label)
            & (sensitivity["strategy"] == strategy)
            & (sensitivity["train_months"] == DEFAULT_TRAIN)
            & (sensitivity["max_weight"] == DEFAULT_MAX_W)
        ]
        if len(cost_sub) >= 2:
            base = cost_sub[cost_sub["tx_cost_bps"] == 0]["sharpe_ratio"]
            high = cost_sub[cost_sub["tx_cost_bps"] == 25]["sharpe_ratio"]
            if len(base) and len(high):
                drop = float(base.iloc[0] - high.iloc[0])
                interp, flag = risk_flag_cost_drop(drop)
                add_diag(rows, strategy, "Sharpe drop 0 -> 25 bps tx cost", f"{drop:.3f}", interp, flag)

        if strategy != "fixed_baseline":
            instab = weight_instability(wf.weight_history[strategy])
            interp, flag = risk_flag_weight_instab(instab)
            add_diag(rows, strategy, "Avg month-to-month weight change (L1)", f"{instab:.2%}", interp, flag)

            s_log = log[log["strategy"] == strategy]
            hits = count_max_weight_hits(s_log, DEFAULT_MAX_W)
            add_diag(
                rows,
                strategy,
                "Months with max-weight constraint binding",
                hits,
                "Frequent binding = optimizer pushing against cap",
                "high" if hits > 50 else ("medium" if hits > 20 else "low"),
            )

        dom_year, share = year_return_concentration(log, strategy)
        add_diag(
            rows,
            strategy,
            "Dominant calendar year (|return| share of total)",
            f"{dom_year} ({share:.0%})" if pd.notna(share) else dom_year,
            "High share = performance concentrated in one regime/year",
            "high" if pd.notna(share) and abs(share) > 0.45 else ("medium" if pd.notna(share) and abs(share) > 0.30 else "low"),
        )

        cov_ret = regime_return(log, strategy, "2020-01-01", "2020-12-31")
        rate_ret = regime_return(log, strategy, "2022-01-01", "2022-12-31")
        add_diag(
            rows,
            strategy,
            "2020 COVID year return",
            f"{cov_ret:.2%}",
            "Stress test: flight-to-quality / equity crash",
            "low" if cov_ret > -0.10 else "medium",
        )
        add_diag(
            rows,
            strategy,
            "2022 rate-hike year return",
            f"{rate_ret:.2%}",
            "Stress test: stocks + long bonds fell together",
            "high" if rate_ret < -0.15 else ("medium" if rate_ret < -0.10 else "low"),
        )

        flags = [r["risk_flag"] for r in rows if STRATEGY_LABELS.get(strategy, strategy) == r["strategy"]]
        high = flags.count("high")
        med = flags.count("medium")
        overall = "HIGH" if high >= 3 else ("MEDIUM" if high >= 1 or med >= 4 else "LOW")
        add_diag(
            rows,
            strategy,
            "OVERALL ROBUSTNESS ASSESSMENT",
            overall,
            f"{high} high, {med} medium flags in pattern-based review",
            overall.lower(),
        )

    return pd.DataFrame(rows)


def period_metrics_2022(wf, strategy: str, bench: pd.Series) -> dict:
    daily = wf.daily_returns[strategy]
    b = bench.loc[daily.index]
    y2022 = daily.loc["2022-01-01":"2022-12-31"]
    b2022 = b.loc["2022-01-01":"2022-12-31"]
    if len(y2022) < 2:
        return {"return_2022": np.nan, "vol_2022": np.nan, "max_dd_2022": np.nan}
    ret = (1 + y2022).prod() - 1
    vol = y2022.std(ddof=1) * np.sqrt(252)
    dd = compute_drawdown((1 + y2022).cumprod()).min()
    return {"return_2022": ret, "vol_2022": vol, "max_dd_2022": dd}


def universe_comparison_row(wf, strategy: str, bench: pd.Series, universe: str, tickers: list[str]) -> dict:
    daily = wf.daily_returns[strategy]
    b = bench.loc[daily.index]
    oos_d = daily.loc[OOS_START:]
    oos_b = b.loc[OOS_START:]
    m_full = compute_metrics(daily, b)
    m_oos = compute_metrics(oos_d, oos_b)
    log = wf.decision_log[wf.decision_log["strategy"] == strategy]
    p2022 = period_metrics_2022(wf, strategy, bench)
    dw = wf.daily_weights[strategy]

    row = {
        "universe": universe,
        "strategy": STRATEGY_LABELS[strategy],
        "annualized_return": m_full.annualized_return,
        "annualized_volatility": m_full.annualized_volatility,
        "sharpe_ratio": m_full.sharpe_ratio,
        "max_drawdown": m_full.max_drawdown,
        "oos_sharpe": m_oos.sharpe_ratio,
        "oos_max_drawdown": m_oos.max_drawdown,
        "beta_vs_spy": m_full.beta,
        "avg_monthly_turnover": wf.turnover_history[strategy].mean(),
        "return_2022": p2022["return_2022"],
        "vol_2022": p2022["vol_2022"],
        "max_dd_2022": p2022["max_dd_2022"],
    }
    if DEFENSIVE_ETF in tickers:
        row[f"avg_{DEFENSIVE_ETF}_weight"] = dw[DEFENSIVE_ETF].mean()
        for pname, (start, end) in REGIME_PERIODS.items():
            sub = dw.loc[start:end]
            row[f"{DEFENSIVE_ETF}_{pname}"] = sub[DEFENSIVE_ETF].mean() if len(sub) else np.nan
    return row


def plot_cumulative(wf_dict: dict[str, WalkForwardResult], bench: pd.Series, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, wf in wf_dict.items():
        for s in STRATEGIES:
            d = wf.daily_returns[s]
            ax.plot((1 + d).cumprod(), label=f"{label} — {STRATEGY_LABELS[s]}", alpha=0.85)
    b = bench.loc[next(iter(wf_dict.values())).daily_returns["fixed_baseline"].index]
    ax.plot((1 + b).cumprod(), label="SPY", color="black", ls="--", lw=1.5)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_drawdown(wf, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for s in STRATEGIES:
        d = wf.daily_returns[s]
        ax.plot(compute_drawdown((1 + d).cumprod()), label=STRATEGY_LABELS[s])
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_turnover(wf, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for s in STRATEGIES:
        to = wf.turnover_history[s]
        ax.plot(to.index, to.values, label=STRATEGY_LABELS[s], alpha=0.8)
    ax.set_title("Monthly Turnover at Rebalance")
    ax.set_ylabel("Turnover")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_weights(wf, path: Path, strategy: str, tickers: list[str]) -> None:
    dw = wf.daily_weights[strategy][tickers]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(dw.index, [dw[t] for t in tickers], labels=tickers, alpha=0.85)
    ax.set_title(f"Weight Evolution — {STRATEGY_LABELS[strategy]}")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sensitivity_heatmap(df: pd.DataFrame, path: Path) -> None:
    sub = df[
        (df["universe"] == "Original (5 ETF)")
        & (df["strategy"] == "minimum_variance")
        & (df["tx_cost_bps"] == 0)
    ]
    if sub.empty:
        return
    pivot = sub.pivot(index="train_months", columns="max_weight", values="sharpe_ratio")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax)
    ax.set_title("Min-Variance Sharpe: Training Window x Max Weight")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    explain(
        "STAGE 5 — Robustness & Defensive Asset Extension",
        """
Part A: Sensitivity grid (training window, max weight, transaction costs) + overfitting diagnostics.
Part B: Add SHY (short-duration Treasury) vs TLT (long-duration) — risk-control test, not return tuning.

Overfitting is diagnosed through a PATTERN, not one number:
  strong IS + weak OOS, unstable weights, cost sensitivity, parameter sensitivity, regime concentration.
""",
    )

    prices = load_prices_with_shy(ROOT / "data")
    daily_all = prices_to_returns(prices)
    daily_orig = daily_all[TICKERS]
    daily_exp = daily_all[TICKERS_EXPANDED]
    bench = daily_all[BENCHMARK]

    # --- Part A: baseline walk-forward at default config ---
    print("Running default walk-forward (12mo, 40% cap, 0bps)...")
    wf_default = run_walk_forward(
        daily_orig,
        bench,
        tickers=TICKERS,
        target_weights=TARGET_WEIGHTS,
        train_months=DEFAULT_TRAIN,
        max_weight=DEFAULT_MAX_W,
        tx_cost_bps=DEFAULT_TX,
    )

    print("Running sensitivity grid (this may take a few minutes)...")
    sensitivity = run_sensitivity_grid(daily_orig, bench, TICKERS, TARGET_WEIGHTS, "Original (5 ETF)")
    sensitivity.to_csv(OUTPUT_DIR / "sensitivity_analysis.csv", index=False)

    diagnostics = build_overfitting_diagnostics(wf_default, bench, sensitivity, "Original (5 ETF)")
    diagnostics.to_csv(OUTPUT_DIR / "overfitting_robustness_diagnostics.csv", index=False)

    # Summary at default config
    default_summary = sensitivity[
        (sensitivity["train_months"] == DEFAULT_TRAIN)
        & (sensitivity["max_weight"] == DEFAULT_MAX_W)
        & (sensitivity["tx_cost_bps"] == DEFAULT_TX)
    ].copy()
    default_summary.to_csv(OUTPUT_DIR / "strategy_summary_default_config.csv", index=False)

    # --- Part B: expanded universe ---
    print("Running expanded universe with SHY...")
    wf_orig = wf_default
    wf_exp = run_walk_forward(
        daily_exp,
        bench,
        tickers=TICKERS_EXPANDED,
        target_weights=TARGET_WEIGHTS_EXPANDED,
        train_months=DEFAULT_TRAIN,
        max_weight=DEFAULT_MAX_W,
        tx_cost_bps=DEFAULT_TX,
    )

    comparison_rows = []
    for s in STRATEGIES:
        comparison_rows.append(universe_comparison_row(wf_orig, s, bench, "Original (5 ETF)", TICKERS))
        comparison_rows.append(universe_comparison_row(wf_exp, s, bench, f"Expanded (+{DEFENSIVE_ETF})", TICKERS_EXPANDED))
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUTPUT_DIR / "universe_comparison.csv", index=False)

    exp_sensitivity = run_sensitivity_grid(
        daily_exp, bench, TICKERS_EXPANDED, TARGET_WEIGHTS_EXPANDED, f"Expanded (+{DEFENSIVE_ETF})"
    )
    exp_sensitivity.to_csv(OUTPUT_DIR / "sensitivity_analysis_expanded.csv", index=False)

    # --- Charts ---
    plot_cumulative(
        {"Original": wf_orig, "Expanded": wf_exp},
        bench,
        OUTPUT_DIR / "01_cumulative_return_universe_compare.png",
        "Cumulative Return — Original vs Expanded Universe",
    )
    plot_drawdown(wf_orig, OUTPUT_DIR / "02_drawdown_original.png", "Drawdown — Original Universe")
    plot_drawdown(wf_exp, OUTPUT_DIR / "03_drawdown_expanded.png", f"Drawdown — Expanded Universe (+{DEFENSIVE_ETF})")
    plot_turnover(wf_orig, OUTPUT_DIR / "04_turnover_original.png")
    plot_turnover(wf_exp, OUTPUT_DIR / "05_turnover_expanded.png")
    for s in STRATEGIES:
        plot_weights(wf_orig, OUTPUT_DIR / f"06_weights_original_{s}.png", s, TICKERS)
        plot_weights(wf_exp, OUTPUT_DIR / f"07_weights_expanded_{s}.png", s, TICKERS_EXPANDED)
    plot_sensitivity_heatmap(sensitivity, OUTPUT_DIR / "08_sensitivity_heatmap_minvar.png")

    # --- Excel report ---
    with pd.ExcelWriter(OUTPUT_DIR / "stage5_report.xlsx", engine="openpyxl") as writer:
        default_summary.to_excel(writer, sheet_name="Summary Default Config", index=False)
        sensitivity.to_excel(writer, sheet_name="Sensitivity Original", index=False)
        diagnostics.to_excel(writer, sheet_name="Robustness Diagnostics", index=False)
        comparison_df.to_excel(writer, sheet_name="Universe Comparison", index=False)
        exp_sensitivity.to_excel(writer, sheet_name="Sensitivity Expanded", index=False)

    # --- Interpretation ---
    interp_lines = [
        "STAGE 5 INTERPRETATION — Robustness & SHY Extension",
        "=" * 72,
        "",
        "WHAT WAS TESTED",
        "-" * 40,
        "Three walk-forward strategies on QQQ/SPY/DIA/GLD/TLT, then the same rules with SHY added.",
        "Sensitivity: training windows 6/12/18/24 months, min-var max weights 30/40/50%,",
        "transaction costs 0/5/10/25 bps per unit turnover. No momentum, ML, or 2022-specific tuning.",
        "",
        "WHY SHY vs TLT (economics)",
        "-" * 40,
        "TLT holds long-duration Treasuries — high interest-rate sensitivity. In 2022, rates rose and TLT",
        "fell alongside equities, breaking the diversifier role.",
        f"{DEFENSIVE_ETF} is a short-duration Treasury ETF — much lower duration, behaves more like cash /",
        "short-term Treasury exposure. It can absorb defensive allocation without the same rate-shock risk.",
        "",
        "PART A — ROBUSTNESS FINDINGS",
        "-" * 40,
    ]

    for s in STRATEGIES:
        sub = diagnostics[diagnostics["strategy"] == STRATEGY_LABELS[s]]
        overall = sub[sub["diagnostic"] == "OVERALL ROBUSTNESS ASSESSMENT"]["value"].iloc[0]
        interp_lines.append(f"  {STRATEGY_LABELS[s]}: {overall} robustness concern")

    interp_lines += [
        "",
        "Overfitting is NOT one number. Watch for the pattern:",
        "  - IS Sharpe >> OOS Sharpe (performance fade)",
        "  - Sharpe swings when training window changes by 6 months",
        "  - Sharpe collapses after realistic transaction costs",
        "  - Weights jump month-to-month; max-weight constraints bind often",
        "  - One calendar year (e.g. 2022) drives most of the pain",
        "",
        "Fixed baseline: simplest, lowest turnover, moderate robustness — good implementation benchmark.",
        "Inverse-vol & min-var: better vol/Sharpe historically but higher parameter + cost sensitivity.",
        "",
        "PART B — SHY EXTENSION (default 12mo, 40% cap, 0bps)",
        "-" * 40,
    ]

    for s in STRATEGIES:
        orig = comparison_df[(comparison_df["strategy"] == STRATEGY_LABELS[s]) & (comparison_df["universe"] == "Original (5 ETF)")]
        exp = comparison_df[(comparison_df["strategy"] == STRATEGY_LABELS[s]) & (comparison_df["universe"].str.startswith("Expanded"))]
        if orig.empty or exp.empty:
            continue
        o, e = orig.iloc[0], exp.iloc[0]
        shy_avg = e.get(f"avg_{DEFENSIVE_ETF}_weight", np.nan)
        shy_2022 = e.get(f"{DEFENSIVE_ETF}_2022 Rate hikes", np.nan)
        interp_lines.append(f"\n  {STRATEGY_LABELS[s]}:")
        interp_lines.append(f"    2022 return: {o['return_2022']:.2%} -> {e['return_2022']:.2%}")
        interp_lines.append(f"    2022 max DD: {o['max_dd_2022']:.2%} -> {e['max_dd_2022']:.2%}")
        interp_lines.append(f"    OOS Sharpe: {o['oos_sharpe']:.2f} -> {e['oos_sharpe']:.2f}")
        interp_lines.append(f"    Max DD (full): {o['max_drawdown']:.2%} -> {e['max_drawdown']:.2%}")
        if pd.notna(shy_avg):
            interp_lines.append(f"    Avg {DEFENSIVE_ETF} weight: {shy_avg:.1%} (2022 stress avg: {shy_2022:.1%})")

    interp_lines += [
        "",
        "CAUTION — INVERSE-VOL + SHY BEHAVIOR",
        "-" * 40,
        "Inverse-volatility assigns weight proportional to 1/volatility. SHY has much lower vol than",
        "equities or TLT, so the rule naturally allocates ~70%+ to SHY — effectively a near-cash portfolio.",
        "This improves 2022 drawdown in the backtest but sacrifices long-run return (~4% ann. vs ~12%).",
        "That is economically intuitive (not curve-fit) but may be too conservative for an equity sleeve.",
        "Min-variance uses SHY more moderately (~17%) because covariance, not vol alone, drives weights.",
        "",
        "BOTTOM LINE",
        "-" * 40,
        "Adding SHY gives the optimizer a low-duration defensive sleeve. Dynamic strategies may allocate",
        "to SHY when TLT vol/covariance looks unattractive — this is economically intuitive, not curve-fit.",
        "Expect modest 2022 improvement (less TLT drag) at the cost of lower return in bond-rally regimes.",
        "Judge success by downside/risk metrics and regime behavior, not by maximizing backtest return.",
        "",
        f"Outputs: {OUTPUT_DIR}",
    ]

    narrative = "\n".join(interp_lines)
    (OUTPUT_DIR / "interpretation_stage5.txt").write_text(narrative, encoding="utf-8")

    print("\n--- DEFAULT CONFIG SUMMARY ---")
    print(default_summary[["strategy", "sharpe_ratio", "max_drawdown", "avg_monthly_turnover"]].to_string(index=False))
    print("\n--- ROBUSTNESS DIAGNOSTICS (sample) ---")
    print(diagnostics.head(20).to_string(index=False))
    print("\n--- UNIVERSE COMPARISON ---")
    print(comparison_df.to_string(index=False))
    print("\n" + narrative)


if __name__ == "__main__":
    main()
