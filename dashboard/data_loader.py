"""Dashboard CSV loaders — read-only; no backtest logic."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DASHBOARD_OUTPUT_DIR, OUTPUT_DIR, ROOT
from src.metrics import compute_drawdown, compute_metrics, metrics_to_series

OUTPUT = OUTPUT_DIR
DASHBOARD = DASHBOARD_OUTPUT_DIR

PIPELINE_HINT = "Run: python scripts/run_all_stages.py"


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, **kwargs)
    return pd.DataFrame()


def _read_required(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} at {path}. {PIPELINE_HINT}")
    return pd.read_csv(path, index_col=0, parse_dates=True)


# --- Stage outputs ---


def load_target_weights() -> pd.DataFrame:
    return _read_csv(OUTPUT / "target_weights.csv")


def load_metrics_comparison() -> pd.DataFrame:
    return _read_csv(OUTPUT / "metrics_comparison.csv", index_col=0)


def load_stage5b_universe_full() -> pd.DataFrame:
    return _read_csv(OUTPUT / "stage5b" / "universe_comparison_full.csv")


def load_stage5b_universe_delta() -> pd.DataFrame:
    return _read_csv(OUTPUT / "stage5b" / "universe_comparison_delta.csv")


def load_audit_reconciliation() -> pd.DataFrame:
    return _read_csv(OUTPUT / "stage6" / "audit" / "suspicious_items_reconciliation.csv")


def load_audit_metric_reconciliation() -> pd.DataFrame:
    return _read_csv(OUTPUT / "stage6" / "audit" / "metric_reconciliation.csv")


def load_audit_summary() -> str:
    p = OUTPUT / "stage6" / "audit" / "audit_summary.txt"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def load_stage3_metrics() -> pd.DataFrame:
    return _read_csv(OUTPUT / "stage3" / "strategy_metrics.csv", index_col=0)


def load_stage3_decision_log() -> pd.DataFrame:
    df = _read_csv(OUTPUT / "stage3" / "walk_forward_decision_log.csv")
    if not df.empty:
        for c in ["rebalance_date", "training_start", "training_end", "testing_start", "testing_end"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def load_stage6b_comparison() -> pd.DataFrame:
    return _read_csv(OUTPUT / "stage6b" / "stage6b_comparison.csv")


def load_stage6_scorecard() -> pd.DataFrame:
    return _read_csv(OUTPUT / "stage6" / "policy_scorecard.csv")


def load_overfitting_diagnostics() -> pd.DataFrame:
    p = OUTPUT / "stage5" / "overfitting_robustness_diagnostics.csv"
    if p.exists():
        return pd.read_csv(p)
    return _read_csv(OUTPUT / "overfitting" / "overfitting_diagnostics.csv")


def load_training_window_sensitivity() -> pd.DataFrame:
    return _read_csv(OUTPUT / "overfitting" / "training_window_sensitivity.csv")


def load_sensitivity_analysis() -> pd.DataFrame:
    p = OUTPUT / "stage5" / "sensitivity_analysis.csv"
    return _read_csv(p) if p.exists() else pd.DataFrame()


def load_stress_contributions() -> pd.DataFrame:
    return _read_csv(OUTPUT / "stage4" / "asset_return_contributions.csv")


def load_stress_audit() -> pd.DataFrame:
    return _read_csv(OUTPUT / "stage6" / "audit" / "stress_period_audit.csv")


def load_corr(when: str, period: str) -> pd.DataFrame:
    return _read_csv(OUTPUT / "stage4" / f"corr_{when}_{period}.csv", index_col=0)


def load_weight_regime(strategy: str, tag: str) -> pd.DataFrame:
    return _read_csv(OUTPUT / "stage4" / f"weight_regime_{strategy}_{tag}.csv")


def load_rolling(strategy: str) -> pd.DataFrame:
    files = {
        "fixed_baseline": "rolling_fixed_baseline.csv",
        "inverse_volatility": "rolling_inverse_volatility.csv",
        "minimum_variance": "rolling_minimum_variance.csv",
    }
    f = files.get(strategy)
    if not f:
        return pd.DataFrame()
    return _read_csv(OUTPUT / "stage3" / f, index_col=0, parse_dates=True)


def load_stage2_rolling() -> pd.DataFrame:
    return _read_csv(OUTPUT / "stage2" / "rolling_metrics_daily.csv", index_col=0, parse_dates=True)


def load_is_oos_summary() -> pd.DataFrame:
    return _read_csv(OUTPUT / "overfitting" / "is_oos_summary.csv")


def load_policy_metrics_comparison() -> pd.DataFrame:
    return _read_csv(OUTPUT / "stage6" / "policy_metrics_comparison.csv")


def load_turnover_by_phase() -> pd.DataFrame:
    return _read_csv(OUTPUT / "stage4" / "turnover_by_phase.csv")


# --- Dashboard precomputed series (from pipeline) ---


def load_key_return_series() -> pd.DataFrame:
    return _read_required(DASHBOARD / "key_return_series.csv", "key_return_series.csv")


def load_baseline_stage1_returns() -> pd.DataFrame:
    return _read_required(DASHBOARD / "baseline_stage1_returns.csv", "baseline_stage1_returns.csv")


def load_baseline_is_oos_metrics() -> pd.DataFrame:
    p = DASHBOARD / "baseline_is_oos_metrics.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def rebalance_vs_buyhold_summary() -> pd.DataFrame:
    p = DASHBOARD / "rebalance_vs_buyhold_summary.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def metrics_from_returns(rets: pd.DataFrame) -> pd.DataFrame:
    p = DASHBOARD / "strategy_metrics_from_key_returns.csv"
    if p.exists() and not rets.empty:
        cached = pd.read_csv(p)
        if set(cached["Strategy"]) >= set(rets.columns):
            return cached[cached["Strategy"].isin(rets.columns)]
    bench = rets["SPY"]
    rows = []
    for col in rets.columns:
        aligned = rets[[col, "SPY"]].dropna()
        if len(aligned) < 2:
            continue
        m = compute_metrics(aligned[col], aligned["SPY"])
        rows.append({"Strategy": col, **metrics_to_series(m).to_dict()})
    return pd.DataFrame(rows)


# --- Display helpers (light transforms only) ---


def cumulative_wealth(rets: pd.DataFrame) -> pd.DataFrame:
    return (1 + rets.dropna(how="all")).cumprod()


def drawdown_series(rets: pd.DataFrame) -> pd.DataFrame:
    wealth = (1 + rets).cumprod()
    return wealth.apply(compute_drawdown)


TURNOVER_LOOKUP = {
    "Original fixed": ("REF_original", "fixed_baseline"),
    "Fixed + SHY (5%)": ("FIX_A", None),
    "Inverse-vol (uncapped)": ("INV_uncapped", None),
    "Inverse-vol (SHY cap 10%)": ("INV_cap10", None),
    "Minimum-variance": ("MINVAR_ref", "minimum_variance"),
    "SPY": (None, None),
}


def turnover_from_decision_log() -> pd.DataFrame:
    log = load_stage3_decision_log()
    if log.empty or "turnover" not in log.columns:
        return pd.DataFrame()
    return (
        log.groupby("strategy")["turnover"]
        .agg(avg_monthly_turnover="mean", max_monthly_turnover="max")
        .reset_index()
    )


def turnover_for_strategies(strategies: list[str]) -> pd.DataFrame:
    s6b = load_stage6b_comparison()
    log_t = turnover_from_decision_log()
    rows = []
    for strat in strategies:
        pid, log_key = TURNOVER_LOOKUP.get(strat, (None, None))
        avg_t = max_t = None
        if pid and not s6b.empty and "policy_id" in s6b.columns:
            hit = s6b[s6b["policy_id"] == pid]
            if not hit.empty:
                avg_t = hit.iloc[0]["avg_monthly_turnover"]
                max_t = hit.iloc[0]["max_monthly_turnover"]
        if avg_t is None and log_key and not log_t.empty:
            hit = log_t[log_t["strategy"] == log_key]
            if not hit.empty:
                avg_t = hit.iloc[0]["avg_monthly_turnover"]
                max_t = hit.iloc[0]["max_monthly_turnover"]
        if strat == "SPY":
            avg_t = max_t = 0.0
        rows.append({"Strategy": strat, "Avg Monthly Turnover": avg_t, "Max Monthly Turnover": max_t})
    return pd.DataFrame(rows)


def get_assumptions_table() -> pd.DataFrame:
    rows = [
        ("ETF universe (base)", "QQQ, SPY, DIA, GLD, TLT"),
        ("ETF universe (expanded)", "QQQ, SPY, DIA, GLD, TLT, SHY"),
        ("Original target weights", "QQQ 25%, SPY 25%, DIA 25%, GLD 15%, TLT 10%"),
        ("Recommended policy (Fix A)", "QQQ 25%, SPY 25%, DIA 25%, GLD 15%, TLT 5%, SHY 5%"),
        ("Benchmark", "SPY"),
        ("Data source", "Yahoo Finance (adjusted close)"),
        ("Return calculation", "Daily pct_change on adjusted close"),
        ("Rebalance frequency", "Monthly (first trading day)"),
        ("Walk-forward training window", "12 calendar months"),
        ("Walk-forward test window", "1 calendar month"),
        ("In-sample period", "2009-01-01 to 2019-12-31"),
        ("Out-of-sample period", "2020-01-01 to latest"),
        ("Max asset weight (min-var)", "40%"),
        ("SHY cap (mandate test)", "10%, 15%, or 20%"),
        ("Minimum equity exposure", "60% (QQQ + SPY + DIA)"),
        ("Constraints", "Long-only; weights sum to 100%"),
        ("Transaction costs (sensitivity)", "0, 5, 10, 25 bps per turnover (Stage 5)"),
        ("Default backtest costs", "0 bps"),
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Value"])


def get_etf_descriptions() -> pd.DataFrame:
    rows = [
        ("QQQ", "Nasdaq-100 growth / tech equity"),
        ("SPY", "Broad US large-cap equity (benchmark)"),
        ("DIA", "Dow Jones industrial equity"),
        ("GLD", "Gold — inflation / crisis diversifier"),
        ("TLT", "Long-duration Treasuries — rate-sensitive"),
        ("SHY", "Short-duration Treasuries — cash-like defensive sleeve"),
    ]
    return pd.DataFrame(rows, columns=["ETF", "Role"])


def get_research_flow() -> pd.DataFrame:
    rows = [
        (
            "Research question",
            "Can an approximate Vance-style ETF allocation be managed as a credible risk policy — "
            "and which defensive extension fits a wealth-management mandate?",
        ),
        (
            "Stage 1 — Baseline IS/OOS",
            "Fixed 25/25/25/15/10 portfolio vs SPY. IS 2009–2019, OOS 2020+. "
            "Finding: lower drawdown than SPY OOS; modest negative IR vs benchmark.",
        ),
        (
            "Stage 2 — Rolling diagnostics",
            "Track time-varying vol, Sharpe, beta, tracking error. "
            "Finding: risk profile stable until 2020+ regime shifts.",
        ),
        (
            "Stage 3 — Walk-forward backtest",
            "12-month train / 1-month test for fixed, inverse-vol, min-var. "
            "Finding: dynamic rules add complexity; min-var collapses to equal-weight.",
        ),
        (
            "Stage 4 — Stress diagnosis",
            "COVID (2020) vs 2022 rate hikes. "
            "Finding: TLT hedged in 2020 but failed in 2022; diversification broke down.",
        ),
        (
            "Stage 5 — Robustness",
            "IS/OOS fade, training-window & cost sensitivity. "
            "Finding: more regime vulnerability than classic severe overfitting.",
        ),
        (
            "Stage 5B — SHY extension",
            "Add short-duration Treasury sleeve. "
            "Finding: uncapped inverse-vol + SHY → ~73% SHY (cash-like); fixed + SHY modest 2022 help.",
        ),
        (
            "Stage 6/6B — Mandate selection",
            "Score policies on downside, explainability, mandate fit. "
            "Finding: Fixed + SHY Version A best balance for WM presentation.",
        ),
        (
            "Final decision",
            "Primary: Fixed + SHY A (25/25/25/15/5/5). Conservative: Fixed + SHY B (10% SHY). "
            "Not recommended: uncapped inverse-vol + SHY.",
        ),
    ]
    return pd.DataFrame(rows, columns=["Step", "Content"])
