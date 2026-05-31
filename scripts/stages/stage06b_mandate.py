"""
Vance_portfolio_analysis_stage6b.py
Stage 6B — Mandate-constrained strategy test (realistic WM policy constraints).

Prioritizes mandate consistency and explainability over Sharpe maximization.
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
import pandas as pd
import seaborn as sns

from mandate_constraints import (
    DEFENSIVE_ETF,
    EQUITY_TICKERS,
    MandateConstraints,
    apply_mandate_constraints,
    classify_mandate,
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
OUTPUT_DIR = ROOT / "output" / "stage6b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS_EXPANDED = TICKERS + [DEFENSIVE_ETF]
COVID_START, COVID_END = "2020-02-19", "2020-03-23"
TRAIN_MONTHS, MAX_WEIGHT, TX = 12, 0.40, 0

# Fixed + SHY versions (user-specified; all satisfy mandate pre-check)
FIXED_VERSIONS = {
    "FIX_A": {
        "name": "Fixed + SHY Version A (25/25/25/15/5/5)",
        "weights": pd.Series(
            {"QQQ": 0.25, "SPY": 0.25, "DIA": 0.25, "GLD": 0.15, "TLT": 0.05, DEFENSIVE_ETF: 0.05}
        ),
    },
    "FIX_B": {
        "name": "Fixed + SHY Version B (25/25/25/15/0/10)",
        "weights": pd.Series(
            {"QQQ": 0.25, "SPY": 0.25, "DIA": 0.25, "GLD": 0.15, "TLT": 0.0, DEFENSIVE_ETF: 0.10}
        ),
    },
    "FIX_C": {
        "name": "Fixed + SHY Version C (24/24/24/13/5/10)",
        "weights": pd.Series(
            {"QQQ": 0.24, "SPY": 0.24, "DIA": 0.24, "GLD": 0.13, "TLT": 0.05, DEFENSIVE_ETF: 0.10}
        ),
    },
}

MANDATE_RULES = MandateConstraints(
    shy_cap=0.20,
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


def validate_fixed_weights(tw: pd.Series, version_id: str) -> str:
    w, notes = apply_mandate_constraints(tw.copy(), MANDATE_RULES)
    eq = sum(w[t] for t in EQUITY_TICKERS if t in w.index)
    shy = w.get(DEFENSIVE_ETF, 0.0)
    mx = w.max()
    if abs(w.sum() - 1.0) > 1e-6:
        return f"FAIL sum={w.sum():.4f}"
    if eq < 0.60 - 1e-6:
        return f"FAIL equity={eq:.1%} < 60%"
    if mx > 0.40 + 1e-6:
        return f"FAIL max weight={mx:.1%} > 40%"
    if shy > 0.20 + 1e-6:
        return f"FAIL SHY={shy:.1%} > 20%"
    return f"PASS (equity {eq:.1%}, SHY {shy:.1%}, max {mx:.1%})"


def period_return(daily: pd.Series, start: str, end: str) -> float:
    sub = daily.loc[start:end]
    return float((1 + sub).prod() - 1) if len(sub) else float("nan")


def exposure_stats(daily_w: pd.DataFrame) -> dict:
    eq_cols = [t for t in EQUITY_TICKERS if t in daily_w.columns]
    eq_exp = daily_w[eq_cols].sum(axis=1)
    shy = daily_w[DEFENSIVE_ETF] if DEFENSIVE_ETF in daily_w.columns else pd.Series(0.0, index=daily_w.index)
    return {
        "avg_equity_exposure": float(eq_exp.mean()),
        "min_equity_exposure": float(eq_exp.min()),
        "avg_shy_allocation": float(shy.mean()),
        "max_shy_allocation": float(shy.max()),
    }


def build_row(policy_id: str, policy_name: str, wf, strategy: str, bench: pd.Series, mclass: str) -> dict:
    daily = wf.daily_returns[strategy]
    b = bench.loc[daily.index]
    m = compute_metrics(daily, b)
    ms = metrics_to_series(m)
    to = wf.turnover_history[strategy]
    exp = exposure_stats(wf.daily_weights[strategy])
    y2022 = daily.loc["2022-01-01":"2022-12-31"]
    dd2022 = float(compute_drawdown((1 + y2022).cumprod()).min()) if len(y2022) > 1 else float("nan")

    return {
        "policy_id": policy_id,
        "policy_name": policy_name,
        "mandate_consistency": mclass,
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
        "return_2022": period_return(daily, "2022-01-01", "2022-12-31"),
        "max_dd_2022": dd2022,
        "covid_return": period_return(daily, COVID_START, COVID_END),
        "avg_monthly_turnover": float(to.mean()),
        "max_monthly_turnover": float(to.max()),
        **exp,
    }


def run_mandate_policy(
    policy_id: str,
    policy_name: str,
    strategy: str,
    tickers: list[str],
    target_weights: pd.Series,
    mandate: MandateConstraints,
    daily_all: pd.DataFrame,
    bench: pd.Series,
    mclass_override: str | None = None,
) -> dict:
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
    exp = exposure_stats(wf.daily_weights[strategy])
    mclass = mclass_override or classify_mandate(
        exp["avg_equity_exposure"],
        exp["min_equity_exposure"],
        exp["avg_shy_allocation"],
        exp["max_shy_allocation"],
        policy_name,
    )
    return build_row(policy_id, policy_name, wf, strategy, bench, mclass)


def build_narrative(df: pd.DataFrame) -> str:
    ref = df[df["policy_id"] == "REF_original"].iloc[0]
    fix_a = df[df["policy_id"] == "FIX_A"].iloc[0]
    fix_b = df[df["policy_id"] == "FIX_B"].iloc[0]
    inv_u = df[df["policy_id"] == "INV_uncapped"].iloc[0]
    inv10 = df[df["policy_id"] == "INV_cap10"].iloc[0]
    min_ref = df[df["policy_id"] == "MINVAR_ref"].iloc[0]

    ret_sac_a = fix_a["annualized_return"] - ref["annualized_return"]
    ret_sac_b = fix_b["annualized_return"] - ref["annualized_return"]

    return f"""
STAGE 6B — MANDATE-CONSTRAINED STRATEGY TEST
{'=' * 72}

MANDATE CONSTRAINTS
- SHY defensive sleeve: max 10% / 15% / 20% (dynamic strategies)
- Min equity (QQQ+SPY+DIA): 60%; max any ETF: 40%; long-only; sum 100%; monthly rebalance
- SHY excess redistribution: cap SHY, then redistribute excess pro-rata to non-SHY assets
- No 2022-specific tuning

FIXED VERSION CONSTRAINT CHECK
- Version A (5% SHY): {validate_fixed_weights(FIXED_VERSIONS['FIX_A']['weights'], 'A')}
- Version B (10% SHY): {validate_fixed_weights(FIXED_VERSIONS['FIX_B']['weights'], 'B')}
- Version C (10% SHY, trim equity): {validate_fixed_weights(FIXED_VERSIONS['FIX_C']['weights'], 'C')}

WHY UNCAPPED INVERSE-VOL+SHY IS NOT RECOMMENDED (despite Sharpe {inv_u['sharpe_ratio']:.2f})
- Avg SHY {inv_u['avg_shy_allocation']:.1%}, avg equity {inv_u['avg_equity_exposure']:.1%}
- Return only {inv_u['annualized_return']:.1%} — cash-like, mandate-inconsistent
- High Sharpe reflects low vol, not equity risk management

DOES CAPPED SHY IMPROVE 2022?
- Fixed A vs original: {fix_a['return_2022']:.1%} vs {ref['return_2022']:.1%} ({fix_a['return_2022']-ref['return_2022']:+.1%})
- Fixed B (10% SHY): {fix_b['return_2022']:.1%} ({fix_b['return_2022']-ref['return_2022']:+.1%} vs original)
- Inv-vol cap 10%: {inv10['return_2022']:.1%} — similar 2022 help but equity pinned at ~60%

RETURN SACRIFICED FOR SHY
- Fixed A vs original: {ret_sac_a:+.2%}
- Fixed B vs original: {ret_sac_b:+.2%}
- Uncapped inv-vol: {inv_u['annualized_return']-ref['annualized_return']:+.2%} (extreme)

EQUITY EXPOSURE MAINTAINED?
- Fixed A/B/C: ~72-75% equity — equity-oriented
- Capped inv-vol/min-var: ~60% (mandate floor) — balanced, not equity-oriented

FINAL RECOMMENDATION
PRIMARY: Fixed + SHY Version A (25/25/25/15/5/5)
  - Best mix of explainability, equity exposure (75%), modest 2022 help, lowest turnover
ALTERNATIVE: Version B if client accepts slightly lower return for more rate-shock buffer
NOT RECOMMENDED: Uncapped inverse-vol+SHY
SECONDARY (not primary): Capped inverse-vol — adds optimizer complexity; equity stuck at 60%;
  2022 benefit similar to Fixed B but worse IR and higher turnover
MIN-VAR REFERENCE: Equal-weight 20% each in practice; 2022 {min_ref['return_2022']:.1%} unchanged vs adding SHY meaningfully

Fixed + SHY beats capped inverse-vol for WM/portfolio-risk context because:
explainability, higher equity sleeve, lower turnover, no monthly estimation risk.

Outputs: {OUTPUT_DIR}
""".strip()


def main() -> None:
    print("Stage 6B — Mandate-constrained strategy test")
    prices = load_prices_with_shy(ROOT / "data")
    daily_all = prices_to_returns(prices)
    bench = daily_all[BENCHMARK]
    rows: list[dict] = []

    # Reference: original fixed (no SHY)
    mandate0 = MandateConstraints(shy_cap=0.0, min_equity_total=0.60)
    wfn0 = make_mandate_weight_functions(TICKERS, TARGET_WEIGHTS, mandate0)
    wf0 = run_walk_forward(
        daily_all[TICKERS], bench, tickers=TICKERS, target_weights=TARGET_WEIGHTS,
        strategies=["fixed_baseline"], custom_weight_fn={"fixed_baseline": wfn0["fixed_baseline"]},
    )
    exp0 = exposure_stats(wf0.daily_weights["fixed_baseline"])
    rows.append(build_row(
        "REF_original", "Original fixed (no SHY)", wf0, "fixed_baseline", bench,
        classify_mandate(exp0["avg_equity_exposure"], exp0["min_equity_exposure"], 0, 0, "equity"),
    ))

    # 1. Fixed Versions A, B, C
    for pid, spec in FIXED_VERSIONS.items():
        tw = spec["weights"]
        shy_cap = float(tw[DEFENSIVE_ETF])
        mandate = MandateConstraints(shy_cap=shy_cap, min_equity_total=0.60)
        rows.append(run_mandate_policy(
            pid, spec["name"], "fixed_baseline", TICKERS_EXPANDED, tw, mandate, daily_all, bench,
        ))

    # 2. Inverse-vol uncapped + capped
    tw_fb = FIXED_VERSIONS["FIX_A"]["weights"]
    wfn_unc = make_weight_functions(TICKERS_EXPANDED, tw_fb, max_weight=MAX_WEIGHT)
    wf_u = run_walk_forward(
        daily_all[TICKERS_EXPANDED], bench, tickers=TICKERS_EXPANDED, target_weights=tw_fb,
        strategies=["inverse_volatility"],
        custom_weight_fn={"inverse_volatility": wfn_unc["inverse_volatility"]},
    )
    rows.append(build_row(
        "INV_uncapped", "Inverse-vol + SHY (uncapped)", wf_u, "inverse_volatility", bench,
        "mandate-inconsistent",
    ))

    for cap, pid in [(0.10, "INV_cap10"), (0.15, "INV_cap15"), (0.20, "INV_cap20")]:
        mandate = MandateConstraints(shy_cap=cap, min_equity_total=0.60, max_equity_single=0.40)
        rows.append(run_mandate_policy(
            pid,
            f"Inverse-vol + SHY (cap {cap:.0%})",
            "inverse_volatility",
            TICKERS_EXPANDED,
            tw_fb,
            mandate,
            daily_all,
            bench,
        ))

    # 3. Min-var reference + SHY capped
    mandate_mv = MandateConstraints(shy_cap=0.0, min_equity_total=0.60)
    wfn_mv = make_mandate_weight_functions(TICKERS, TARGET_WEIGHTS, mandate_mv)
    wf_mv = run_walk_forward(
        daily_all[TICKERS], bench, tickers=TICKERS, target_weights=TARGET_WEIGHTS,
        strategies=["minimum_variance"],
        custom_weight_fn={"minimum_variance": wfn_mv["minimum_variance"]},
    )
    exp_mv = exposure_stats(wf_mv.daily_weights["minimum_variance"])
    rows.append(build_row(
        "MINVAR_ref",
        "Min-variance 5 ETF (equal-weight in practice)",
        wf_mv, "minimum_variance", bench,
        classify_mandate(exp_mv["avg_equity_exposure"], exp_mv["min_equity_exposure"], 0, 0, "balanced"),
    ))

    for cap, pid in [(0.10, "MINVAR_cap10"), (0.15, "MINVAR_cap15"), (0.20, "MINVAR_cap20")]:
        mandate = MandateConstraints(shy_cap=cap, min_equity_total=0.60)
        rows.append(run_mandate_policy(
            pid, f"Min-var + SHY (cap {cap:.0%})", "minimum_variance",
            TICKERS_EXPANDED, tw_fb, mandate, daily_all, bench,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "stage6b_comparison.csv", index=False)

    # Charts
    fig, ax = plt.subplots(figsize=(9, 4))
    fixed = df[df["policy_id"].str.startswith("FIX") | (df["policy_id"] == "REF_original")]
    ax.barh(fixed["policy_id"], fixed["return_2022"], color="#6ACC65", alpha=0.85)
    ax.set_xlabel("2022 return")
    ax.set_title("Fixed Policies — 2022 Return")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fixed_2022_returns.png", dpi=150)
    plt.close(fig)

    inv = df[df["policy_id"].str.startswith("INV")]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(inv["avg_shy_allocation"], inv["annualized_return"], s=90)
    for _, r in inv.iterrows():
        ax.annotate(r["policy_id"].replace("INV_", ""), (r["avg_shy_allocation"], r["annualized_return"]), fontsize=8)
    ax.set_xlabel("Avg SHY allocation")
    ax.set_ylabel("Annualized return")
    ax.set_title("Inverse-vol: Return vs SHY (uncapped vs capped)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "invvol_return_vs_shy.png", dpi=150)
    plt.close(fig)

    fmt = df.copy()
    pct_cols = [
        "annualized_return", "annualized_volatility", "max_drawdown", "tracking_error",
        "return_2022", "max_dd_2022", "covid_return", "avg_monthly_turnover", "max_monthly_turnover",
        "avg_shy_allocation", "max_shy_allocation", "avg_equity_exposure", "min_equity_exposure",
        "var_95", "cvar_95",
    ]
    for c in pct_cols:
        fmt[c] = fmt[c].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    for c in ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "beta_vs_spy", "information_ratio"]:
        fmt[c] = fmt[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

    with pd.ExcelWriter(OUTPUT_DIR / "stage6b_report.xlsx", engine="openpyxl") as writer:
        fmt.to_excel(writer, sheet_name="Comparison", index=False)

    narrative = build_narrative(df)
    (OUTPUT_DIR / "interpretation_stage6b.txt").write_text(narrative, encoding="utf-8")

    print("\n--- CONSTRAINT PRE-CHECK (Fixed versions) ---")
    for pid, spec in FIXED_VERSIONS.items():
        print(f"  {pid}: {validate_fixed_weights(spec['weights'], pid)}")

    print("\n--- STAGE 6B COMPARISON ---")
    show = [
        "policy_id", "mandate_consistency", "annualized_return", "sharpe_ratio",
        "return_2022", "covid_return", "avg_shy_allocation", "avg_equity_exposure", "avg_monthly_turnover",
    ]
    print(df[show].to_string(index=False))
    print("\n" + narrative)


if __name__ == "__main__":
    main()
