"""
Vance_portfolio_analysis_stage7.py
Stage 7 — Mandate-constrained strategy comparison and revised final recommendation.
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

from mandate_constraints import (
    DEFENSIVE_ETF,
    EQUITY_TICKERS,
    MandateConstraints,
    classify_mandate,
    fixed_shy_weights,
    make_mandate_weight_functions,
)
from portfolio_core import (
    BENCHMARK,
    TARGET_WEIGHTS,
    TICKERS,
    compute_drawdown,
    compute_metrics,
    load_adjusted_prices,
    metrics_to_series,
    prices_to_returns,
)
from walk_forward_engine import make_weight_functions, run_walk_forward

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", palette="muted")

from src.config import ROOT
OUTPUT_DIR = ROOT / "output" / "stage7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS_EXPANDED = TICKERS + [DEFENSIVE_ETF]
COVID_START, COVID_END = "2020-02-19", "2020-03-23"
TRAIN_MONTHS, MAX_WEIGHT, TX = 12, 0.40, 0

BASE_MANDATE = MandateConstraints(
    shy_cap=None,
    max_equity_single=0.40,
    min_equity_total=0.60,
    max_single_asset=0.40,
)


def load_prices_with_shy(data_dir: Path) -> pd.DataFrame:
    prices = load_adjusted_prices(data_dir)
    if DEFENSIVE_ETF in prices.columns:
        return prices
    import yfinance as yf

    shy = yf.download(DEFENSIVE_ETF, start=prices.index[0].strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    shy_s = shy["Close"].iloc[:, 0] if isinstance(shy.columns, pd.MultiIndex) else shy["Close"]
    merged = prices.join(shy_s.rename(DEFENSIVE_ETF), how="inner").dropna(how="any")
    merged.to_csv(data_dir / "vance_etf_prices.csv")
    return merged


def period_return(daily: pd.Series, start: str, end: str) -> float:
    sub = daily.loc[start:end]
    return float((1 + sub).prod() - 1) if len(sub) else np.nan


def exposure_stats(daily_w: pd.DataFrame, tickers: list[str]) -> dict:
    eq = [t for t in EQUITY_TICKERS if t in daily_w.columns]
    eq_exp = daily_w[eq].sum(axis=1) if eq else pd.Series(0.0, index=daily_w.index)
    shy = daily_w[DEFENSIVE_ETF] if DEFENSIVE_ETF in daily_w.columns else pd.Series(0.0, index=daily_w.index)
    return {
        "avg_equity_exposure": float(eq_exp.mean()),
        "min_equity_exposure": float(eq_exp.min()),
        "avg_shy_allocation": float(shy.mean()),
        "max_shy_allocation": float(shy.max()),
    }


def build_policy_row(
    policy_id: str,
    policy_name: str,
    wf,
    strategy: str,
    bench: pd.Series,
    mandate_class: str,
) -> dict:
    daily = wf.daily_returns[strategy]
    b = bench.loc[daily.index]
    m = compute_metrics(daily, b)
    ms = metrics_to_series(m)
    to = wf.turnover_history[strategy]
    dw = wf.daily_weights[strategy]
    exp = exposure_stats(dw, list(dw.columns))

    y2022 = daily.loc["2022-01-01":"2022-12-31"]
    dd2022 = float(compute_drawdown((1 + y2022).cumprod()).min()) if len(y2022) > 1 else np.nan

    return {
        "policy_id": policy_id,
        "policy_name": policy_name,
        "mandate_consistency": mandate_class,
        **{k: ms[k] for k in [
            "Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Sortino Ratio",
            "Maximum Drawdown", "Calmar Ratio", "Beta vs SPY", "Tracking Error vs SPY",
            "Information Ratio vs SPY",
        ]},
        "return_2022": period_return(daily, "2022-01-01", "2022-12-31"),
        "max_dd_2022": dd2022,
        "covid_return": period_return(daily, COVID_START, COVID_END),
        "avg_monthly_turnover": float(to.mean()),
        "max_monthly_turnover": float(to.max()),
        **exp,
    }


def run_policy(
    label: str,
    strategy: str,
    tickers: list[str],
    target_weights: pd.Series,
    mandate: MandateConstraints,
    daily_all: pd.DataFrame,
    bench: pd.Series,
):
    wfn = make_mandate_weight_functions(tickers, target_weights, mandate, max_weight_optimizer=MAX_WEIGHT)
    wf = run_walk_forward(
        daily_all[tickers],
        bench,
        tickers=tickers,
        target_weights=target_weights,
        train_months=TRAIN_MONTHS,
        max_weight=MAX_WEIGHT,
        tx_cost_bps=TX,
        strategies=[strategy],
        custom_weight_fn={strategy: wfn[strategy]},
    )
    dw = wf.daily_weights[strategy]
    exp = exposure_stats(dw, tickers)
    mclass = classify_mandate(
        exp["avg_equity_exposure"],
        exp["min_equity_exposure"],
        exp["avg_shy_allocation"],
        exp["max_shy_allocation"],
        label,
    )
    return build_policy_row(label, label, wf, strategy, bench, mclass)


def main() -> None:
    print("Stage 7 — Mandate-constrained strategy comparison")
    prices = load_prices_with_shy(ROOT / "data")
    daily_all = prices_to_returns(prices)
    bench = daily_all[BENCHMARK]

    rows = []

    # Reference: original fixed (no SHY)
    mandate_ref = MandateConstraints(shy_cap=0.0, min_equity_total=0.60)
    wfn = make_mandate_weight_functions(TICKERS, TARGET_WEIGHTS, mandate_ref)
    wf = run_walk_forward(
        daily_all[TICKERS], bench, tickers=TICKERS, target_weights=TARGET_WEIGHTS,
        strategies=["fixed_baseline"], custom_weight_fn={"fixed_baseline": wfn["fixed_baseline"]},
    )
    dw = wf.daily_weights["fixed_baseline"]
    exp = exposure_stats(dw, TICKERS)
    rows.append(build_policy_row(
        "REF_fixed_original", "Original fixed (no SHY)", wf, "fixed_baseline", bench,
        classify_mandate(exp["avg_equity_exposure"], exp["min_equity_exposure"], 0, 0, "equity"),
    ))

    # 1. Fixed + SHY at 5%, 10%, 15%
    for shy_pct in [0.05, 0.10, 0.15]:
        tw = fixed_shy_weights(shy_pct)
        mandate = MandateConstraints(shy_cap=shy_pct, min_equity_total=0.60)
        pid = f"FIX_shy_{int(shy_pct*100)}"
        rows.append(run_policy(
            pid,
            "fixed_baseline",
            TICKERS_EXPANDED,
            tw,
            mandate,
            daily_all,
            bench,
        ))

    # 2. Inverse-vol: uncapped (true baseline) vs SHY cap 10/15/20%
    tw_fb = fixed_shy_weights(0.05)
    wfn_unc = make_weight_functions(TICKERS_EXPANDED, tw_fb, max_weight=MAX_WEIGHT)
    wf_u = run_walk_forward(
        daily_all[TICKERS_EXPANDED], bench, tickers=TICKERS_EXPANDED, target_weights=tw_fb,
        strategies=["inverse_volatility"],
        custom_weight_fn={"inverse_volatility": wfn_unc["inverse_volatility"]},
    )
    dw_u = wf_u.daily_weights["inverse_volatility"]
    exp_u = exposure_stats(dw_u, TICKERS_EXPANDED)
    rows.append(build_policy_row(
        "INV_uncapped",
        "Inverse-vol + SHY (uncapped — mandate-inconsistent)",
        wf_u, "inverse_volatility", bench,
        "mandate-inconsistent",
    ))

    for shy_cap, pid in [(0.10, "INV_shy10"), (0.15, "INV_shy15"), (0.20, "INV_shy20")]:
        mandate = MandateConstraints(
            shy_cap=shy_cap,
            max_equity_single=0.40,
            min_equity_total=0.60,
        )
        tw = fixed_shy_weights(0.05)
        rows.append(run_policy(
            pid,
            "inverse_volatility",
            TICKERS_EXPANDED,
            tw,
            mandate,
            daily_all,
            bench,
        ))
        rows[-1]["policy_name"] = f"Inverse-vol + SHY (cap {shy_cap:.0%})"

    # 3. Min-var: 5 ETF (prior) + optional SHY capped
    mandate_mv = MandateConstraints(shy_cap=0.0, min_equity_total=0.60)
    wfn = make_mandate_weight_functions(TICKERS, TARGET_WEIGHTS, mandate_mv)
    wf = run_walk_forward(
        daily_all[TICKERS], bench, tickers=TICKERS, target_weights=TARGET_WEIGHTS,
        strategies=["minimum_variance"], custom_weight_fn={"minimum_variance": wfn["minimum_variance"]},
    )
    dw = wf.daily_weights["minimum_variance"]
    exp = exposure_stats(dw, TICKERS)
    rows.append(build_policy_row(
        "MINVAR_5etf",
        "Min-variance 5 ETF (equal-weight in practice)",
        wf, "minimum_variance", bench,
        classify_mandate(exp["avg_equity_exposure"], exp["min_equity_exposure"], 0, 0, "balanced"),
    ))

    for shy_cap in [0.10, 0.15, 0.20]:
        tw = fixed_shy_weights(0.05)
        mandate = MandateConstraints(shy_cap=shy_cap, min_equity_total=0.60)
        pid = f"MINVAR_shy{int(shy_cap*100)}"
        r = run_policy(pid, "minimum_variance", TICKERS_EXPANDED, tw, mandate, daily_all, bench)
        r["policy_name"] = f"Min-var + SHY (cap {shy_cap:.0%})"
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "mandate_constrained_comparison.csv", index=False)

    # Charts: SHY cap vs 2022 return for inverse-vol
    inv = df[df["policy_id"].str.startswith("INV")]
    fig, ax = plt.subplots(figsize=(8, 4))
    xlabels = inv["policy_id"].str.replace("INV_", "")
    ax.bar(xlabels, inv["return_2022"], color="#4878CF", alpha=0.8)
    ref2022 = df[df["policy_id"] == "REF_fixed_original"]["return_2022"].iloc[0]
    ax.axhline(ref2022, ls="--", color="gray", label="Fixed original 2022")
    ax.set_title("Inverse-Vol: 2022 Return by SHY Cap")
    ax.set_ylabel("2022 return")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "invvol_2022_by_shy_cap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(inv["avg_shy_allocation"], inv["Annualized Return"], s=80)
    for _, r in inv.iterrows():
        ax.annotate(r["policy_id"].replace("INV_", ""), (r["avg_shy_allocation"], r["Annualized Return"]), fontsize=8)
    ax.set_xlabel("Avg SHY allocation")
    ax.set_ylabel("Annualized return")
    ax.set_title("Inv-Vol: Return vs SHY Exposure")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "invvol_return_vs_shy.png", dpi=150)
    plt.close(fig)

    # Excel
    fmt = df.copy()
    pct_cols = [
        "Annualized Return", "Annualized Volatility", "Maximum Drawdown", "Tracking Error vs SPY",
        "return_2022", "max_dd_2022", "covid_return", "avg_monthly_turnover", "max_monthly_turnover",
        "avg_shy_allocation", "max_shy_allocation", "avg_equity_exposure", "min_equity_exposure",
    ]
    for c in pct_cols:
        if c in fmt.columns:
            fmt[c] = fmt[c].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    for c in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Beta vs SPY", "Information Ratio vs SPY"]:
        fmt[c] = fmt[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

    with pd.ExcelWriter(OUTPUT_DIR / "stage7_report.xlsx", engine="openpyxl") as writer:
        fmt.to_excel(writer, sheet_name="Mandate Comparison", index=False)

    narrative = build_narrative(df)
    (OUTPUT_DIR / "interpretation_stage7.txt").write_text(narrative, encoding="utf-8")

    print("\n--- MANDATE-CONSTRAINED COMPARISON ---")
    show = [
        "policy_id", "policy_name", "mandate_consistency", "Annualized Return", "Sharpe Ratio",
        "return_2022", "covid_return", "avg_shy_allocation", "avg_equity_exposure",
    ]
    print(df[show].to_string(index=False))
    print("\n" + narrative)


def build_narrative(df: pd.DataFrame) -> str:
    fix5 = df[df["policy_id"] == "FIX_shy_5"].iloc[0]
    fix10 = df[df["policy_id"] == "FIX_shy_10"].iloc[0]
    inv_u = df[df["policy_id"] == "INV_uncapped"].iloc[0]
    inv15 = df[df["policy_id"] == "INV_shy15"].iloc[0]
    min5 = df[df["policy_id"] == "MINVAR_5etf"].iloc[0]

    return f"""
STAGE 7 — MANDATE-CONSTRAINED STRATEGY COMPARISON
{'=' * 72}

POLICY CONSTRAINTS APPLIED
- SHY is a defensive sleeve (caps at 10%, 15%, 20% for dynamic strategies)
- Max single equity ETF: 40%; min total equity (QQQ+SPY+DIA): 60%
- Long-only, weights sum to 100%, monthly rebalance
- Inverse-vol SHY redistribution: when SHY exceeds cap, excess is redistributed
  to other assets proportional to their current (uncapped) weights

WHY UNCAPPED INVERSE-VOL+SHY IS NOT RECOMMENDED
- Uncapped avg SHY: {inv_u['avg_shy_allocation']:.1%}; equity: {inv_u['avg_equity_exposure']:.1%}
- Mandate class: {inv_u['mandate_consistency']}
- High Sharpe ({inv_u['Sharpe Ratio']:.2f}) comes from ~{inv_u['avg_shy_allocation']:.0%} cash-like SHY, not equity risk management
- Return only {inv_u['Annualized Return']:.1%} — inappropriate for growth-oriented WM mandates

WHY A SHY CAP IS NECESSARY
- Without cap, 1/vol rule treats SHY as the dominant asset (lowest vol)
- Wealth-management mandates require meaningful equity exposure (>=60%)
- Capping SHY forces defensive benefit without turning portfolio into cash

FIXED + SHY (transparent allocations)
- 5% SHY (25/25/25/15/5/5): 2022 {fix5['return_2022']:.1%}, equity {fix5['avg_equity_exposure']:.1%} — {fix5['mandate_consistency']}
- 10% SHY: 2022 {fix10['return_2022']:.1%}, equity {fix10['avg_equity_exposure']:.1%}
- Higher SHY improves 2022 modestly; COVID slightly worse (less TLT)

INVERSE-VOL + SHY CAPPED
- Cap 15% example: return {inv15['Annualized Return']:.1%}, Sharpe {inv15['Sharpe Ratio']:.2f},
  2022 {inv15['return_2022']:.1%}, avg SHY {inv15['avg_shy_allocation']:.1%}
- Capped versions remain equity-oriented but add complexity vs fixed for modest benefit

MIN-VARIANCE
- 5-ETF min-var collapses to 20% equal weight each month (Stage 6 audit)
- Sharpe {min5['Sharpe Ratio']:.2f}, 2022 {min5['return_2022']:.1%} — does not solve 2022 regime
- SHY cap versions add limited 2022 improvement at return cost

TRADE-OFF: CASH PROTECTION VS GROWTH
- More SHY -> better 2022/rate-shock, lower long-run return, less COVID TLT rally participation
- For WM: prefer transparent fixed sleeve (5-10% SHY) over dynamic cash-like inverse-vol

REVISED FINAL RECOMMENDATION (portfolio risk management context)
PRIMARY: Fixed + SHY 5% (25/25/25/15/5/5) — equity-oriented, mandate-consistent, explainable
ALTERNATIVE: Fixed + SHY 10% if client prioritizes rate-shock over return
NOT RECOMMENDED: Uncapped inverse-vol+SHY (mandate-inconsistent despite high Sharpe)
NOT PRIMARY: Min-var (equal-weight in practice; optimizer adds complexity without clear benefit)

Outputs: {OUTPUT_DIR}
""".strip()


if __name__ == "__main__":
    main()
