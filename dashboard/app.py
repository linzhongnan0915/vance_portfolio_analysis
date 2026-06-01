"""
dashboard/app.py — Vance Portfolio Risk Dashboard

Run: streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import dashboard.charts as charts
import dashboard.data_loader as dd
import dashboard.formatting as fmt

st.set_page_config(
    page_title="Vance-Style ETF Portfolio Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISCLAIMER = (
    "**Disclaimer:** Educational approximate ETF-style replication for portfolio risk analysis. "
    "Not an exact copy of any disclosed portfolio. Not investment advice or a trading signal."
)


@st.cache_data(show_spinner="Loading return series (first run may take ~15s)...")
def cached_returns() -> pd.DataFrame:
    return dd.load_key_return_series()




# --- Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Section",
    [
        "Overview",
        "Baseline IS/OOS",
        "Assumptions & Constraints",
        "Portfolio Construction",
        "Backtest Results",
        "SHY Extension",
        "Walk-Forward Log",
        "Rolling Diagnostics",
        "Stress Analysis",
        "Robustness",
        "Mandate Selection",
        "Validation / Reconciliation",
        "Policy Candidate",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Data: `vance_portfolio_analysis/output/`")
st.sidebar.caption("Run: `streamlit run dashboard/app.py`")

st.title("Vance-Style ETF Portfolio Risk Dashboard")
st.markdown(DISCLAIMER)
st.markdown("---")

# =============================================================================
if page == "Overview":
    st.header("Executive Summary")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Project Objective")
        st.markdown(
            """
Evaluate an **approximate Vance-style ETF allocation** using a structured risk-management workflow:
baseline policy → walk-forward backtest → stress diagnosis → robustness checks → mandate-constrained selection.
"""
        )
        st.subheader("Portfolio Definition")
        st.markdown(
            """
- **Universe:** QQQ, SPY, DIA, GLD, TLT (+ SHY for defensive extension)
- **Original weights:** 25% / 25% / 25% / 15% / 10%
- **Benchmark:** SPY | **Rebalance:** Monthly
- **In-sample:** 2009–2019 | **Out-of-sample:** 2020–latest
"""
        )
    with c2:
        st.subheader("Main Findings")
        st.info(
            "Primary issue is **2022 regime vulnerability** (stocks + long bonds fell together), "
            "not classic severe overfitting. Uncapped inverse-vol + SHY allocated ~73% to SHY — cash-like and mandate-inconsistent."
        )
        st.success(
            "Current policy candidate: Fixed + SHY Version A. This is a simple, "
            "mandate-consistent defensive adjustment that modestly improves 2022 stress "
            "behavior, but it is not a full-period dominant strategy. Constrained dynamic "
            "SHY strategies should be tested before calling it optimal."
        )

    st.subheader("Workflow Completed")
    stages = pd.DataFrame([
        ("Stage 1", "Fixed-weight baseline IS/OOS"),
        ("Stage 2", "Rolling diagnostics & subperiods"),
        ("Stage 3", "Walk-forward backtest (3 strategies)"),
        ("Stage 4", "COVID vs 2022 stress diagnosis"),
        ("Stage 5", "Robustness & overfitting sensitivity"),
        ("Stage 5B", "SHY universe extension"),
        ("Stage 6/6B", "Final selection & mandate constraints"),
    ], columns=["Stage", "Description"])
    st.dataframe(stages, width="stretch", hide_index=True)

    st.header("Project Map / Research Flow")
    flow = dd.get_research_flow()
    st.markdown(
        """
**Research question:** Can an approximate Vance-style ETF allocation be managed as a credible
risk policy — and which defensive extension fits a wealth-management mandate?
"""
    )
    st.subheader("Methodology")
    st.markdown(
        "Baseline validation → rolling risk monitoring → walk-forward backtest → "
        "stress diagnosis → robustness checks → SHY extension → mandate-constrained selection."
    )
    st.subheader("Key finding by stage")
    findings = flow[~flow["Step"].isin(["Research question", "Final decision"])]
    st.dataframe(findings.rename(columns={"Step": "Stage", "Content": "Key Finding"}),
                 width="stretch", hide_index=True)
    decision = flow[flow["Step"] == "Final decision"]
    if not decision.empty:
        st.info(
            f"**Policy candidate / next research step:** {decision.iloc[0]['Content']} "
            "Interpret as a mandate-consistent adjustment, not a full-period dominant strategy."
        )

# =============================================================================
elif page == "Baseline IS/OOS":
    st.header("Baseline In-Sample / Out-of-Sample")
    st.markdown(
        """
The **Stage 1 baseline** anchors the entire project. Before testing dynamic rules or defensive
extensions, we ask: does the original fixed allocation behave as expected vs SPY, and does that
hold **out-of-sample** (2020+) when COVID and 2022 regimes arrive?
"""
    )

    comp = dd.load_metrics_comparison()
    if not comp.empty:
        st.subheader("Portfolio vs SPY — IS/OOS Metrics")
        show = comp.copy()
        pct_rows = [
            "Cumulative Return", "Annualized Return", "Annualized Volatility", "Maximum Drawdown",
            "VaR 95% (daily)", "CVaR 95% (daily)", "Tracking Error vs SPY",
        ]
        for idx in show.index:
            if idx in pct_rows:
                show.loc[idx] = show.loc[idx].map(lambda x: fmt.pct(x) if isinstance(x, (int, float)) else x)
            elif idx in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Beta vs SPY", "Information Ratio vs SPY"]:
                show.loc[idx] = show.loc[idx].map(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
        st.dataframe(show, width="stretch")
        st.caption("Source: Stage 1 `output/metrics_comparison.csv`. Monthly rebalanced fixed portfolio.")

    rb_bh = dd.rebalance_vs_buyhold_summary()
    if not rb_bh.empty:
        st.subheader("Monthly Rebalance vs Buy-and-Hold")
        rb_show = rb_bh.assign(
            **{c: rb_bh[c].map(fmt.pct) for c in ["Monthly Rebalance", "Buy-and-Hold", "Difference (Rebal − BH)"]}
        )
        st.dataframe(rb_show, width="stretch", hide_index=True)
        st.caption(
            "Monthly rebalance resets weights to target — controlling drift and concentration. "
            "Buy-and-hold lets weights drift; small cumulative difference here confirms rebalance discipline matters modestly."
        )

    baseline = dd.load_baseline_stage1_returns()
    if not baseline.empty:
        wealth = dd.cumulative_wealth(baseline)
        dd_s = dd.drawdown_series(baseline)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(wealth, title="Cumulative Return — Baseline Policies")
            fig.update_layout(legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig2 = px.line(dd_s, title="Drawdown — Baseline Policies")
            st.plotly_chart(fig2, width="stretch")
        st.caption(
            "Original fixed portfolio (monthly rebalance) vs buy-and-hold same weights vs SPY. "
            "OOS period (2020+) shows wider drawdowns — motivating later stress and robustness work."
        )

    st.info(
        "**Why this baseline matters:** Every later stage compares against this policy. "
        "If the fixed portfolio fails mandate or stress tests, we adjust defensively (SHY) — "
        "not chase the highest backtest Sharpe."
    )

# =============================================================================
elif page == "Assumptions & Constraints":
    st.header("Assumptions & Constraints")
    st.dataframe(dd.get_assumptions_table(), width="stretch", hide_index=True)

    st.subheader("Why Mandate Constraints Matter")
    st.markdown(
        """
- **SHY is a defensive sleeve**, not the main portfolio. Without caps, inverse-volatility assigns ~70%+ to SHY
  because it has the lowest trailing volatility — turning the strategy into a **cash-like portfolio**.
- **Minimum 60% equity** ensures the policy remains appropriate for wealth-management / growth mandates.
- **Uncapped inverse-vol + SHY is not recommended** despite high Sharpe (1.24): return ~4%, equity ~15%, IR −0.73 vs SPY.
"""
    )
    mandate6b = dd.load_stage6b_comparison()
    if not mandate6b.empty:
        st.subheader("Mandate-Constrained Results (Stage 6B)")
        show = [
            "policy_id", "mandate_consistency", "annualized_return", "sharpe_ratio",
            "return_2022", "avg_shy_allocation", "avg_equity_exposure",
        ]
        st.dataframe(
            mandate6b[show].assign(
                annualized_return=mandate6b["annualized_return"].map(lambda x: fmt.pct(x)),
                return_2022=mandate6b["return_2022"].map(lambda x: fmt.pct(x)),
                avg_shy_allocation=mandate6b["avg_shy_allocation"].map(lambda x: fmt.pct(x)),
                avg_equity_exposure=mandate6b["avg_equity_exposure"].map(lambda x: fmt.pct(x)),
                sharpe_ratio=mandate6b["sharpe_ratio"].map(lambda x: f"{x:.2f}"),
            ),
            width="stretch",
            hide_index=True,
        )

# =============================================================================
elif page == "Portfolio Construction":
    st.header("Portfolio Construction")
    tw = dd.load_target_weights()
    if not tw.empty:
        st.subheader("Original Target Weights")
        tw = tw.rename(columns={"Unnamed: 0": "ETF"})
        st.dataframe(tw, width="stretch", hide_index=True)
    st.subheader("ETF Roles")
    st.dataframe(dd.get_etf_descriptions(), width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    orig = {"QQQ": 0.25, "SPY": 0.25, "DIA": 0.25, "GLD": 0.15, "TLT": 0.10}
    rec = {"QQQ": 0.25, "SPY": 0.25, "DIA": 0.25, "GLD": 0.15, "TLT": 0.05, "SHY": 0.05}
    with c1:
        charts.pie_weights(orig, "Original Fixed Portfolio")
    with c2:
        charts.pie_weights(rec, "Policy candidate: Fixed + SHY (Version A)")

    st.caption(
        "Version A shifts 5% from TLT to SHY — a simple defensive adjustment; not a full-period dominant strategy."
    )

# =============================================================================
elif page == "Backtest Results":
    st.header("Backtest Results")
    rets = cached_returns()
    wealth = dd.cumulative_wealth(rets)
    dd_s = dd.drawdown_series(rets)

    fig = px.line(wealth, title="Cumulative Return (Growth of $1)", labels={"value": "Wealth", "index": "Date"})
    fig.update_layout(legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    st.caption("Walk-forward daily returns from 2010 test start. SHY policies use expanded universe.")

    fig2 = px.line(dd_s, title="Drawdown", labels={"value": "Drawdown", "index": "Date"})
    st.plotly_chart(fig2, width="stretch")
    st.caption("Drawdown = current wealth / peak wealth − 1. Shows tail-risk differences across policies.")

    metrics = dd.metrics_from_returns(rets)
    if not metrics.empty:
        st.subheader("Performance Metrics")
        turnover = dd.turnover_for_strategies(metrics["Strategy"].tolist())
        metrics = metrics.merge(turnover, on="Strategy", how="left")
        display_cols = [
            "Strategy", "Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Sortino Ratio",
            "Maximum Drawdown", "Calmar Ratio", "VaR 95% (daily)", "CVaR 95% (daily)",
            "Beta vs SPY", "Tracking Error vs SPY", "Information Ratio vs SPY",
            "Avg Monthly Turnover", "Max Monthly Turnover",
        ]
        avail = [c for c in display_cols if c in metrics.columns]
        metrics_display = fmt.fmt_metrics_table(metrics[avail])
        for c in ["Avg Monthly Turnover", "Max Monthly Turnover"]:
            if c in metrics_display.columns:
                metrics_display[c] = metrics_display[c].astype(object)
                metrics_display[c] = metrics_display[c].apply(
                    lambda x: fmt.pct(x) if isinstance(x, (int, float)) else x
                )
        st.dataframe(metrics_display, width="stretch", hide_index=True)
        st.caption("Turnover from walk-forward rebalance logs (Stage 3) and mandate comparison (Stage 6B). SPY = buy-and-hold.")

# =============================================================================
elif page == "SHY Extension":
    st.header("SHY Defensive Extension (Stage 5B)")
    st.markdown(
        """
Adding **SHY** (short-duration Treasuries) tests whether a cash-like defensive sleeve improves
2022-style rate-shock resilience without turning the portfolio into a capital-preservation product.
"""
    )

    uni = dd.load_stage5b_universe_full()
    if not uni.empty:
        st.subheader("Original (5 ETF) vs Expanded (+SHY) Universe")
        show_cols = [
            "universe", "strategy", "annualized_return", "sharpe_ratio", "max_drawdown",
            "return_2022", "max_dd_2022", "avg_shy_weight", "avg_monthly_turnover",
        ]
        avail = [c for c in show_cols if c in uni.columns]
        display = uni[avail].copy()
        for c in ["annualized_return", "max_drawdown", "return_2022", "max_dd_2022",
                  "avg_shy_weight", "avg_monthly_turnover"]:
            if c in display.columns:
                display[c] = display[c].map(lambda x: fmt.pct(x) if pd.notna(x) else "—")
        if "sharpe_ratio" in display.columns:
            display["sharpe_ratio"] = uni["sharpe_ratio"].map(lambda x: f"{x:.2f}")
        st.dataframe(display, width="stretch", hide_index=True)

    delta = dd.load_stage5b_universe_delta()
    if not delta.empty:
        st.subheader("Impact of Adding SHY (Expanded − Original)")
        dshow = delta.copy()
        for c in dshow.columns:
            if c == "strategy":
                continue
            if "return" in c or "drawdown" in c or "volatility" in c or "turnover" in c or "weight" in c or "beta" in c or "tracking" in c:
                dshow[c] = dshow[c].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "—")
            elif "sharpe" in c or "sortino" in c or "calmar" in c or "information" in c:
                dshow[c] = dshow[c].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
        st.dataframe(dshow, width="stretch", hide_index=True)

    s6b = dd.load_stage6b_comparison()
    if not s6b.empty:
        st.subheader("Key SHY Policies — 2022 & Allocation")
        key_ids = ["REF_original", "FIX_A", "FIX_B", "INV_uncapped", "INV_cap10"]
        sub = s6b[s6b["policy_id"].isin(key_ids)]
        if not sub.empty:
            shy_show = sub[[
                "policy_id", "policy_name", "mandate_consistency",
                "annualized_return", "return_2022", "max_dd_2022",
                "avg_shy_allocation", "max_shy_allocation", "avg_equity_exposure",
            ]].assign(
                annualized_return=sub["annualized_return"].map(fmt.pct),
                return_2022=sub["return_2022"].map(fmt.pct),
                max_dd_2022=sub["max_dd_2022"].map(fmt.pct),
                avg_shy_allocation=sub["avg_shy_allocation"].map(fmt.pct),
                max_shy_allocation=sub["max_shy_allocation"].map(fmt.pct),
                avg_equity_exposure=sub["avg_equity_exposure"].map(fmt.pct),
            )
            st.dataframe(shy_show, width="stretch", hide_index=True)

    st.markdown(
        """
**Trade-off: defensive exposure vs return sacrifice**

| Policy | 2022 return | Avg SHY | Verdict |
|--------|-------------|---------|---------|
| Original fixed | ~−18.0% | 0% | Baseline — full TLT duration risk |
| Fixed + SHY (A) | ~−16.6% | ~5% | **Policy candidate** — modest 2022 buffer, small full-period return/Sharpe cost |
| Fixed + SHY (B) | ~−15.2% | ~10% | Conservative candidate — more buffer, −0.2pp return cost |
| Inv-vol + SHY uncapped | ~−7.3% | ~73% | **Not recommended** — cash-like, ~4% return |

- **Fixed + SHY** trades ~0.1–0.2pp annual return for modestly better 2022 stress behavior — mandate-consistent, not dominant on every metric.
- Further constrained dynamic SHY strategies should be tested before calling any variant optimal.
- **Uncapped inverse-vol + SHY** mechanically overweight SHY (lowest vol) — high Sharpe from low absolute risk, not equity risk management.
"""
    )

# =============================================================================
elif page == "Walk-Forward Log":
    st.header("Walk-Forward Decision Log")
    log = dd.load_stage3_decision_log()
    if log.empty:
        st.warning("Decision log not found. Run Stage 3 first.")
    else:
        strategies = ["All"] + sorted(log["strategy"].unique().tolist())
        sel_strat = st.selectbox("Strategy", strategies)
        stress = st.selectbox("Quick filter", ["All dates", "COVID 2020", "2022 rate hikes", "2020–2024 OOS"])
        dmin = st.date_input("From", value=log["testing_start"].min().date())
        dmax = st.date_input("To", value=log["testing_start"].max().date())

        filt = log.copy()
        if sel_strat != "All":
            filt = filt[filt["strategy"] == sel_strat]
        filt = filt[(filt["testing_start"] >= pd.Timestamp(dmin)) & (filt["testing_start"] <= pd.Timestamp(dmax))]
        if stress == "COVID 2020":
            filt = filt[(filt["testing_start"] >= "2020-01-01") & (filt["testing_start"] <= "2020-12-31")]
        elif stress == "2022 rate hikes":
            filt = filt[(filt["testing_start"] >= "2022-01-01") & (filt["testing_start"] <= "2022-12-31")]
        elif stress == "2020–2024 OOS":
            filt = filt[filt["testing_start"] >= "2020-01-01"]

        show_cols = [
            "rebalance_date", "strategy", "training_start", "training_end", "testing_start", "testing_end",
            "weight_QQQ", "weight_SPY", "weight_DIA", "weight_GLD", "weight_TLT",
            "monthly_portfolio_return", "monthly_benchmark_return", "turnover",
        ]
        show_cols = [c for c in show_cols if c in filt.columns]
        display = filt[show_cols].sort_values("testing_start", ascending=False).copy()
        pct_cols = [x for x in show_cols if x.startswith("weight_") or "return" in x or x == "turnover"]
        for c in pct_cols:
            if c in display.columns and display[c].dtype in ("float64", "float32"):
                display[c] = display[c].map(lambda x: fmt.pct(x) if pd.notna(x) else "")
        st.dataframe(display, width="stretch", hide_index=True)
        st.caption(
            f"Showing {len(filt)} rows. Stage 3 log covers 5-ETF strategies only (no SHY column). "
            "Full log: `output/stage3/walk_forward_decision_log.csv`"
        )

# =============================================================================
elif page == "Rolling Diagnostics":
    st.header("Rolling Diagnostics")
    strat = st.selectbox(
        "Strategy (Stage 3 walk-forward)",
        ["fixed_baseline", "inverse_volatility", "minimum_variance"],
        format_func=lambda x: x.replace("_", " ").title(),
    )
    roll = dd.load_rolling(strat)
    s2 = dd.load_stage2_rolling()
    if roll.empty and s2.empty:
        st.warning("Rolling data not found. Run Stage 2 and Stage 3 first.")
    else:
        tabs = st.tabs(["Volatility", "Sharpe", "Drawdown", "Beta & Tracking Error"])
        with tabs[0]:
            if not s2.empty and "volatility_63" in s2.columns:
                fig = px.line(s2.dropna(subset=["volatility_63"]), y="volatility_63",
                              title="Rolling 63-Day Volatility — Fixed Baseline")
                st.plotly_chart(fig, width="stretch")
                st.caption("Short-horizon (≈3 month) vol. Spikes flag recent turbulence; uses trailing data only.")
            if not s2.empty and "volatility_252" in s2.columns:
                fig = px.line(s2.dropna(subset=["volatility_252"]), y="volatility_252",
                              title="Rolling 252-Day Volatility — Fixed Baseline")
                st.plotly_chart(fig, width="stretch")
                st.caption("One-year annualized vol. Smoother view of regime shifts (e.g., 2020, 2022).")
            elif not roll.empty and "vol_252" in roll.columns:
                fig = px.line(roll.dropna(subset=["vol_252"]), y="vol_252",
                              title=f"Rolling 252-Day Volatility — {strat.replace('_', ' ').title()}")
                st.plotly_chart(fig, width="stretch")
        with tabs[1]:
            sharpe_col = "sharpe_252" if "sharpe_252" in (roll.columns if not roll.empty else []) else None
            if sharpe_col and not roll.empty:
                fig = px.line(roll.dropna(subset=[sharpe_col]), y=sharpe_col, title="Rolling 252-Day Sharpe Ratio")
                st.plotly_chart(fig, width="stretch")
                st.caption("Risk-adjusted return over trailing year. Declining Sharpe may signal regime change.")
            elif not s2.empty and "sharpe_252" in s2.columns:
                fig = px.line(s2.dropna(subset=["sharpe_252"]), y="sharpe_252", title="Rolling 252-Day Sharpe — Fixed Baseline")
                st.plotly_chart(fig, width="stretch")
        with tabs[2]:
            if not roll.empty and "drawdown" in roll.columns:
                fig = px.line(roll, y="drawdown", title="Rolling Drawdown")
                st.plotly_chart(fig, width="stretch")
                st.caption("Underwater chart from running peak. Depth and duration show tail-risk episodes.")
            elif not s2.empty and "max_drawdown_252" in s2.columns:
                fig = px.line(s2.dropna(subset=["max_drawdown_252"]), y="max_drawdown_252",
                              title="Rolling 252-Day Max Drawdown — Fixed Baseline")
                st.plotly_chart(fig, width="stretch")
        with tabs[3]:
            if not roll.empty and "beta_252" in roll.columns:
                fig = px.line(roll.dropna(subset=["beta_252"]), y="beta_252", title="Rolling 252-Day Beta vs SPY")
                st.plotly_chart(fig, width="stretch")
                st.caption("Market sensitivity. Beta near 0.7 reflects balanced multi-asset mix vs SPY.")
            if not s2.empty and "tracking_error_252" in s2.columns:
                fig = px.line(s2.dropna(subset=["tracking_error_252"]), y="tracking_error_252",
                              title="Rolling 252-Day Tracking Error vs SPY")
                st.plotly_chart(fig, width="stretch")
                st.caption("Active risk vs benchmark. Higher TE = more deviation from SPY path.")

# =============================================================================
elif page == "Stress Analysis":
    st.header("Stress Period Analysis")
    period = st.radio("Stress period", ["COVID shock (Feb–Mar 2020)", "2022 rate hikes (full year)"])
    tag = "COVI" if "COVID" in period else "2022"
    period_key = "COVID" if tag == "COVI" else "2022_full_year"
    period_label = "COVID shock (Feb–Mar 2020)" if tag == "COVI" else "2022 rate hikes"

    audit = dd.load_stress_audit()
    if not audit.empty:
        sub = audit[audit["period"].str.contains("COVID" if tag == "COVI" else "2022", regex=False, na=False)]
        st.subheader(f"Strategy Returns — {period_label}")
        show = sub[["policy", "portfolio_return", "spy_return"]].copy()
        show.columns = ["Policy", "Portfolio Return", "SPY Return"]
        show["Portfolio Return"] = show["Portfolio Return"].map(fmt.pct)
        show["SPY Return"] = show["SPY Return"].map(fmt.pct)
        st.dataframe(show, width="stretch", hide_index=True)

        if tag == "2022":
            st.subheader("Max Drawdown During Period")
            s6b = dd.load_stage6b_comparison()
            if not s6b.empty and "max_dd_2022" in s6b.columns:
                dd_show = s6b[["policy_id", "policy_name", "max_dd_2022"]].copy()
                dd_show.columns = ["Policy ID", "Policy", "Max Drawdown 2022"]
                dd_show["Max Drawdown 2022"] = dd_show["Max Drawdown 2022"].map(fmt.pct)
                st.dataframe(dd_show, width="stretch", hide_index=True)
        else:
            st.caption("COVID window return shown above reflects Feb 19 – Mar 23, 2020 peak-to-trough crash dates.")

        st.subheader("Average Weights During Stress")
        wcols = [c for c in sub.columns if c.startswith("avg_w_")]
        if wcols:
            wshow = sub[["policy"] + wcols].copy()
            wshow.columns = ["Policy"] + [c.replace("avg_w_", "").upper() for c in wcols]
            for c in wshow.columns[1:]:
                wshow[c] = wshow[c].map(lambda x: fmt.pct(x) if pd.notna(x) else "—")
            st.dataframe(wshow, width="stretch", hide_index=True)

    contrib = dd.load_stress_contributions()
    if not contrib.empty:
        st.subheader(f"Asset Return Contributions — {period_label}")
        pmatch = "COVID" if tag == "COVI" else "2022"
        subc = contrib[contrib["period"].str.contains(pmatch, na=False)]
        st.dataframe(subc, width="stretch", hide_index=True)

    wr = dd.load_weight_regime("fixed_baseline", tag)
    if not wr.empty:
        st.subheader("Weight Regime — Fixed Baseline (before / during / after)")
        st.dataframe(wr, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Correlation Before Stress")
        cm = dd.load_corr("before", tag)
        if not cm.empty:
            fig = px.imshow(cm.astype(float), text_auto=".2f", title=f"Before {period_label}")
            st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Correlation During Stress")
        cm2 = dd.load_corr("during", tag)
        if not cm2.empty:
            fig = px.imshow(cm2.astype(float), text_auto=".2f", title=f"During {period_label}")
            st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
**Why COVID and 2022 differed**
- **COVID (2020):** Flight to quality — TLT rallied (+14% in crash window), negative stock-bond correlation. Diversifiers worked.
- **2022:** Rising rates — TLT fell ~31% alongside equities. Stock-bond correlation turned **positive**. TLT failed as hedger.
- **Why SHY helps:** Short duration ≈ cash-like. Less rate sensitivity than TLT; absorbs defensive allocation without 2022-style duration risk.
"""
    )

# =============================================================================
elif page == "Robustness":
    st.header("Robustness / Overfitting Diagnostics")
    st.markdown(
        "Overfitting is **not one number** — it is a pattern: IS/OOS fade, unstable weights, "
        "cost sensitivity, parameter sensitivity. This project found **more regime vulnerability than classic severe overfitting**."
    )

    diag = dd.load_overfitting_diagnostics()
    if not diag.empty:
        st.subheader("Diagnostics Table")
        strat_filter = st.multiselect(
            "Filter strategy",
            diag["strategy"].unique().tolist() if "strategy" in diag.columns else [],
            default=diag["strategy"].unique().tolist()[:3] if "strategy" in diag.columns else [],
        )
        dsub = diag[diag["strategy"].isin(strat_filter)] if strat_filter else diag
        st.dataframe(dsub, width="stretch", hide_index=True)

    tw = dd.load_training_window_sensitivity()
    if not tw.empty:
        st.subheader("Training Window Sensitivity")
        fig = px.line(tw, x="train_months", y="oos_sharpe", color="strategy", markers=True,
                      title="OOS Sharpe by Training Window (months)")
        st.plotly_chart(fig, width="stretch")
        st.caption("Low sensitivity across 6–24 month windows suggests results are not overfit to one training length.")

    sens = dd.load_sensitivity_analysis()
    if not sens.empty and "tx_cost_bps" in sens.columns:
        st.subheader("Transaction Cost Sensitivity")
        cost = sens.groupby(["strategy", "tx_cost_bps"])["sharpe_ratio"].mean().reset_index()
        fig = px.line(cost, x="tx_cost_bps", y="sharpe_ratio", color="strategy", markers=True,
                      title="Sharpe vs Transaction Cost (bps)")
        st.plotly_chart(fig, width="stretch")
        st.caption("All strategies show low cost sensitivity at 0–25 bps — turnover is modest.")

    pm = dd.load_policy_metrics_comparison()
    if not pm.empty:
        st.subheader("Turnover & Weight Stability")
        stab = pm[["policy_id", "policy_name", "avg_monthly_turnover", "max_monthly_turnover", "weight_stability_l1"]]
        stab = stab.assign(
            avg_monthly_turnover=stab["avg_monthly_turnover"].map(fmt.pct),
            max_monthly_turnover=stab["max_monthly_turnover"].map(fmt.pct),
            weight_stability_l1=stab["weight_stability_l1"].map(lambda x: fmt.pct(x) if pd.notna(x) and x > 0 else "0 (static)"),
        )
        st.dataframe(stab, width="stretch", hide_index=True)
        st.caption("Weight stability L1 = average month-to-month weight change. Zero = constant weights (fixed or min-var collapse).")

    bind = dd.load_overfitting_diagnostics()
    if not bind.empty:
        hits = bind[bind["diagnostic"].str.contains("constraint binding|weight stability", case=False, na=False)]
        if not hits.empty:
            st.subheader("Max Constraint Hits")
            st.dataframe(hits, width="stretch", hide_index=True)

    isoos = dd.load_is_oos_summary()
    if not isoos.empty:
        st.subheader("In-Sample vs Out-of-Sample")
        st.dataframe(isoos, width="stretch", hide_index=True)

# =============================================================================
elif page == "Mandate Selection":
    st.header("Mandate-Constrained Strategy Selection")
    m6b = dd.load_stage6b_comparison()
    if not m6b.empty:
        st.dataframe(
            m6b[[
                "policy_id", "policy_name", "mandate_consistency",
                "annualized_return", "sharpe_ratio", "return_2022",
                "avg_shy_allocation", "avg_equity_exposure", "avg_monthly_turnover",
            ]].assign(
                annualized_return=m6b["annualized_return"].map(fmt.pct),
                return_2022=m6b["return_2022"].map(fmt.pct),
                avg_shy_allocation=m6b["avg_shy_allocation"].map(fmt.pct),
                avg_equity_exposure=m6b["avg_equity_exposure"].map(fmt.pct),
                avg_monthly_turnover=m6b["avg_monthly_turnover"].map(fmt.pct),
                sharpe_ratio=m6b["sharpe_ratio"].map(lambda x: f"{x:.2f}"),
            ),
            width="stretch",
            hide_index=True,
        )

    score = dd.load_stage6_scorecard()
    if not score.empty:
        st.subheader("Qualitative Scorecard (Stage 6)")
        score_display = score.rename(columns={
            "downside_protection": "Downside Protection",
            "risk_adjusted_return": "Risk-Adjusted Performance",
            "regime_robustness": "2022 Robustness",
            "turnover_cost_efficiency": "Turnover Efficiency",
            "interpretability": "Explainability",
            "implementation_simplicity": "Implementation Simplicity",
            "wm_suitability": "Mandate Consistency",
            "composite_score": "Composite Score",
        })
        st.dataframe(score_display, width="stretch", hide_index=True)
        st.caption("Scores 1–5 (higher = better). Mandate consistency reflects wealth-management suitability, not raw Sharpe.")

    st.subheader("Classification Legend")
    st.markdown(
        """
| Class | Meaning |
|-------|---------|
| **Equity-oriented** | ≥70% equity, SHY ≤10% — growth WM mandates |
| **Balanced risk-managed** | ~60% equity floor, moderate SHY |
| **Capital-preservation-like** | Low equity or high SHY by design |
| **Mandate-inconsistent** | Uncapped inv-vol + SHY (~73% SHY) — **not recommended** |
"""
    )
    st.error("**Not recommended:** Inverse-vol + SHY uncapped — high Sharpe from cash exposure, not equity risk management.")
    st.warning(
        "**Policy candidates for review:** Fixed + SHY (A/B) as simple defensive adjustments; "
        "capped inverse-vol only if a dynamic policy is required. Not full-period dominant strategies."
    )

# =============================================================================
elif page == "Validation / Reconciliation":
    st.header("Validation / Reconciliation (Stage 6 Audit)")
    st.markdown(
        """
Independent checks confirm that metrics, walk-forward timing, and weight logic reconcile.
This section documents **what was verified** and **how to interpret surprising-looking results**.
"""
    )

    summary = dd.load_audit_summary()
    if summary:
        st.subheader("Audit Summary")
        st.code(summary.strip(), language=None)

    recon = dd.load_audit_reconciliation()
    if not recon.empty:
        st.subheader("Key Items Reconciled")
        st.dataframe(recon, width="stretch", hide_index=True)

    mrec = dd.load_audit_metric_reconciliation()
    if not mrec.empty:
        st.subheader("Formula Reconciliation")
        show = mrec[["policy", "return_match", "vol_match", "lib_sharpe", "lib_ir",
                     "instability_rebalance_l1", "unique_weight_rows", "n_rebalances"]].copy()
        show.columns = ["Policy", "Return OK", "Vol OK", "Sharpe", "IR",
                        "Weight L1 (rebal)", "Unique weight rows", "Rebalances"]
        show["Weight L1 (rebal)"] = mrec["instability_rebalance_l1"].map(
            lambda x: fmt.pct(x) if pd.notna(x) and x > 0 else "0 (static / rounded)"
        )
        show["Sharpe"] = mrec["lib_sharpe"].map(lambda x: f"{x:.2f}")
        show["IR"] = mrec["lib_ir"].map(lambda x: f"{x:.2f}")
        st.dataframe(show, width="stretch", hide_index=True)
        st.caption("Library metrics match manual recomputation. Walk-forward: training_end < testing_start for all rows.")

    st.subheader("Interpretation Notes")
    st.markdown(
        """
**Min-variance behaved like equal-weight in practice**
- Optimizer re-runs monthly but consistently lands near 20% per asset (5 ETFs).
- Reported weight stability of 0% reflects **rounded display**, not a bug — weights drift at 4th decimal.
- Conclusion: adds complexity without clear benefit vs fixed weights.

**Uncapped inverse-vol + SHY became cash-like**
- 1/vol rule assigns ~73% avg to SHY (lowest trailing volatility).
- ~4% annual return, ~3% vol, Sharpe 1.24 — but beta ~0.14 and IR −0.73 vs SPY.
- High Sharpe here means low-risk cash proxy, not superior equity risk management.

**Metric labels clarified**
- *Weight stability L1* at rebalance dates vs daily drift — min-var shows 0% at rebalance but non-zero daily drift.
- *Information ratio* can be strongly negative even when Sharpe is high if absolute return trails SPY.
- *Max-weight constraint* applies to min-var (40% cap), not inverse-vol — uncapped SHY allocation is by design.
"""
    )

# =============================================================================
elif page == "Policy Candidate":
    st.header("Policy Candidate / Next Research Step")
    st.markdown(
        "The selection is based on interpretability and mandate consistency, not the highest Sharpe ratio."
    )

    st.subheader("Why not highest Sharpe?")
    st.warning(
        """
**Uncapped inverse-vol + SHY has the highest Sharpe (~1.24) but is not recommended.**

Sharpe measures return per unit of *portfolio* volatility. When ~73% is in SHY, volatility collapses
to ~3% and Sharpe rises — but absolute return is only ~4% vs ~14% for SPY-linked policies.
Information ratio vs SPY is **−0.73**. Equity exposure averages ~15%.

For wealth-management mandates requiring ~60%+ equity, this policy is **mandate-inconsistent**
and behaves like a cash allocation, not a diversified growth portfolio.
"""
    )

    c1, c2 = st.columns(2)
    with c1:
        st.success(
            """
### Policy candidate: Fixed + SHY Version A
**QQQ 25% | SPY 25% | DIA 25% | GLD 15% | TLT 5% | SHY 5%**

- Simple, mandate-consistent defensive adjustment (~75% equity)
- 2022 stress: modest improvement vs original (−16.6% vs −18.0%); small full-period return/Sharpe cost
- Lowest turnover (~1%/mo), fully explainable — not dominant on every walk-forward metric
- Further constrained dynamic SHY strategies should be tested before calling it optimal
"""
        )
    with c2:
        st.info(
            """
### Conservative: Fixed + SHY Version B
**QQQ 25% | SPY 25% | DIA 25% | GLD 15% | TLT 0% | SHY 10%**

- More rate-shock buffer: 2022 −15.2%
- Slightly lower return (−0.23pp vs original)
- Still equity-oriented (~75%)
"""
        )

    st.error(
        """
### Not recommended
| Strategy | Why |
|----------|-----|
| **Uncapped inverse-vol + SHY** | ~73% SHY, ~4% return, mandate-inconsistent |
| **Min-variance as primary** | Collapses to equal-weight; no clear edge over fixed |
| **Any policy chosen on Sharpe alone** | Cash-like allocations inflate Sharpe without meeting WM mandate |
"""
    )

    st.subheader("Remaining Risks")
    st.markdown(
        """
1. **2022-style regime** — stocks and long bonds can fall together; no static rule fully prevents this
2. **US equity concentration** — QQQ/SPY/DIA only; no international diversification
3. **Educational replication** — approximate weights, not a disclosed live portfolio
4. **Estimation risk** — dynamic strategies depend on 12-month trailing windows
"""
    )

    st.subheader("Next Steps")
    st.markdown(
        """
- Test **constrained dynamic SHY strategies** before treating Version A as more than a policy candidate
- Monitor rolling beta, drawdown, and stock–bond correlation (2022-style regime flag)
- Assume ~5–10 bps transaction costs in live implementation
- Re-run walk-forward quarterly: `python scripts/stages/stage03_walkforward.py`
- Further tests (vol-targeting, TLT trend filter) only as **robustness checks**, not return-chasing
"""
    )

st.markdown("---")
st.caption("Vance-Style ETF Portfolio Risk Dashboard | Risk management analysis — not a trading signal")
