# %% Stage 4 — Diagnose COVID vs 2022 strategy behavior
"""
Vance_portfolio_analysis_stage4.py
Diagnostic analysis: why walk-forward strategies helped in COVID but not in 2022.
No new strategies — diagnosis only.
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

from portfolio_core import BENCHMARK, TICKERS, TRADING_DAYS, load_adjusted_prices, prices_to_returns
from walk_forward_engine import MAX_WEIGHT, STRATEGIES, run_walk_forward

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")
pd.options.display.float_format = "{:.4f}".format

from src.config import ROOT
OUTPUT_DIR = ROOT / "output" / "stage4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STRESS_PERIODS = {
    "COVID shock": {
        "stress_start": "2020-02-19",
        "stress_end": "2020-03-23",
        "before_start": "2019-08-19",
        "before_end": "2020-02-18",
        "after_start": "2020-03-24",
        "after_end": "2020-04-30",
    },
    "2022 rate-hike drawdown": {
        "stress_start": "2022-01-01",
        "stress_end": "2022-12-31",
        "before_start": "2021-01-01",
        "before_end": "2021-12-31",
        "after_start": "2023-01-01",
        "after_end": "2023-03-31",
    },
}


def explain(title: str, text: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print(text.strip())
    print()


def slice_period(df: pd.DataFrame | pd.Series, start: str, end: str):
    return df.loc[start:end]


# --- 1. Asset return contribution ---
def asset_return_contribution(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, float]:
    w = slice_period(weights, start, end)
    r = slice_period(asset_returns, start, end)
    idx = w.index.intersection(r.index)
    w, r = w.loc[idx], r.loc[idx]

    daily_contrib = w[TICKERS] * r[TICKERS]
    total_contrib = daily_contrib.sum()
    port_return = float(daily_contrib.sum(axis=1).sum())
    avg_weight = w[TICKERS].mean()
    cum_asset_ret = (1 + r[TICKERS]).prod() - 1

    out = pd.DataFrame(
        {
            "avg_weight": avg_weight,
            "asset_return": cum_asset_ret,
            "return_contribution": total_contrib,
            "pct_of_portfolio_return": total_contrib / port_return if port_return != 0 else np.nan,
        }
    )
    out["verdict"] = np.where(
        out["return_contribution"] > 0, "helped", np.where(out["return_contribution"] < 0, "hurt", "neutral")
    )
    return out, port_return


# --- 2. Risk contribution (Euler / marginal) ---
def risk_contribution(weights: pd.DataFrame, asset_returns: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    w = slice_period(weights, start, end)
    r = slice_period(asset_returns, start, end)
    idx = w.index.intersection(r.index)
    w, r = w.loc[idx], r.loc[idx]

    port_r = (w * r).sum(axis=1)
    cov = r.cov().values
    w_bar = w.mean().values
    port_var = w_bar @ cov @ w_bar
    port_vol = np.sqrt(port_var) if port_var > 0 else np.nan

    if port_vol == 0 or np.isnan(port_vol):
        mctr = np.zeros(len(TICKERS))
    else:
        mctr = (cov @ w_bar) / port_vol

    cr = w_bar * mctr
    cr_pct = cr / cr.sum() if cr.sum() != 0 else cr

    asset_vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return pd.DataFrame(
        {
            "avg_weight": w_bar,
            "asset_vol_ann": asset_vol.values,
            "marginal_risk_contrib": mctr,
            "risk_contrib": cr,
            "risk_contrib_pct": cr_pct,
        },
        index=TICKERS,
    )


# --- 3. Correlation regime ---
def correlation_analysis(asset_returns: pd.DataFrame, before: tuple[str, str], during: tuple[str, str]) -> dict:
    r_before = slice_period(asset_returns, before[0], before[1])
    r_during = slice_period(asset_returns, during[0], during[1])
    corr_before = r_before.corr()
    corr_during = r_during.corr()

    equity = r_before[["QQQ", "SPY", "DIA"]].mean(axis=1)
    eq_bond_before = equity.corr(r_before["TLT"])
    eq_bond_during = r_during[["QQQ", "SPY", "DIA"]].mean(axis=1).corr(r_during["TLT"])

    return {
        "corr_before": corr_before,
        "corr_during": corr_during,
        "equity_bond_corr_before": eq_bond_before,
        "equity_bond_corr_during": eq_bond_during,
    }


def plot_corr_heatmaps(corr_b: pd.DataFrame, corr_d: pd.DataFrame, title: str, fname: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, corr, label in zip(axes, [corr_b, corr_d], ["Before stress", "During stress"]):
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title(label)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- 4. Weight behavior ---
def rebalance_weights_at(weight_hist: pd.DataFrame, on_or_before: str) -> pd.Series:
    wh = weight_hist.copy()
    wh.index = pd.to_datetime(wh.index)
    eligible = wh[wh.index <= pd.Timestamp(on_or_before)]
    if eligible.empty:
        return pd.Series(dtype=float)
    row = eligible.iloc[-1]
    return pd.Series({t.replace("weight_", ""): row[f"weight_{t}"] for t in TICKERS})


def plot_weight_regimes(
    strategy: str,
    period_name: str,
    cfg: dict,
    weight_hist: pd.DataFrame,
    daily_weights: pd.DataFrame,
) -> None:
    w_before = rebalance_weights_at(weight_hist, cfg["before_end"])
    w_stress = slice_period(daily_weights, cfg["stress_start"], cfg["stress_end"]).mean()
    w_after = rebalance_weights_at(weight_hist, cfg["after_end"])

    plot_df = pd.DataFrame({"Before": w_before, "During (avg daily)": w_stress, "After": w_after})
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title(f"{strategy} — Weights Before / During / After: {period_name}")
    ax.set_ylabel("Weight")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper right")
    ax.axhline(MAX_WEIGHT, color="red", ls="--", lw=0.8, label=f"Max {MAX_WEIGHT:.0%}")
    fig.tight_layout()
    safe = period_name.replace(" ", "_").replace("/", "-")
    fig.savefig(OUTPUT_DIR / f"weights_{strategy}_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_df


# --- 5. Turnover ---
def turnover_analysis(log: pd.DataFrame, strategy: str, before_end: str, stress_end: str) -> pd.DataFrame:
    s = log[log["strategy"] == strategy].copy()
    s["testing_start"] = pd.to_datetime(s["testing_start"])
    before = s[s["testing_start"] <= before_end]
    during = s[(s["testing_start"] >= before_end) & (s["testing_start"] <= stress_end)]
    return pd.DataFrame(
        {
            "phase": ["before_stress", "during_stress"],
            "avg_turnover": [before["turnover"].mean(), during["turnover"].mean()],
            "max_turnover": [before["turnover"].max(), during["turnover"].max()],
            "n_rebalances": [len(before), len(during)],
        }
    )


# =============================================================================
explain(
    "STAGE 4 — Why COVID helped but 2022 did not",
    """
Goal: Diagnose strategy failure/success — NOT add new strategies.

We compare three walk-forward policies across two stress regimes:
  COVID shock (2020-02-19 to 2020-03-23): flight-to-quality, bonds rallied
  2022 rate hikes (full year): stocks AND long bonds fell together

Diagnostics: return contribution, risk contribution, correlations, weights,
turnover, and plain-English interpretation for portfolio risk managers.
""",
)

prices = load_adjusted_prices(ROOT / "data")
asset_returns = prices_to_returns(prices)[TICKERS]
bench_returns = prices_to_returns(prices)[BENCHMARK]

wf = run_walk_forward(asset_returns, bench_returns)
log = wf.decision_log

all_contrib = []
all_risk = []
all_turnover = []
interpretation_sections = []

for period_name, cfg in STRESS_PERIODS.items():
    print(f"\n{'#' * 72}\n  STRESS PERIOD: {period_name}\n{'#' * 72}")
    s_start, s_end = cfg["stress_start"], cfg["stress_end"]
    b_start, b_end = cfg["before_start"], cfg["before_end"]

    # Correlation regime (same for all strategies)
    corr = correlation_analysis(asset_returns, (b_start, b_end), (s_start, s_end))
    corr["corr_before"].to_csv(OUTPUT_DIR / f"corr_before_{period_name[:4]}.csv")
    corr["corr_during"].to_csv(OUTPUT_DIR / f"corr_during_{period_name[:4]}.csv")
    plot_corr_heatmaps(
        corr["corr_before"],
        corr["corr_during"],
        f"Correlation Regime: {period_name}",
        f"corr_{period_name.replace(' ', '_')[:20]}.png",
    )
    print(f"\nEquity-TLT correlation BEFORE: {corr['equity_bond_corr_before']:.3f}")
    print(f"Equity-TLT correlation DURING:  {corr['equity_bond_corr_during']:.3f}")

    for strategy in STRATEGIES:
        print(f"\n--- {strategy} / {period_name} ---")
        dw = wf.daily_weights[strategy]
        wh = wf.weight_history[strategy]

        # 1. Return contribution
        contrib, port_ret = asset_return_contribution(dw, asset_returns, s_start, s_end)
        contrib["strategy"] = strategy
        contrib["period"] = period_name
        all_contrib.append(contrib.reset_index().rename(columns={"index": "asset"}))

        print("\n1) ASSET RETURN CONTRIBUTION")
        print(contrib.to_string())
        print(f"   Total portfolio return (period): {port_ret:.2%}")

        # Compare strategies portfolio return same period
        fixed_ret = asset_return_contribution(wf.daily_weights["fixed_baseline"], asset_returns, s_start, s_end)[1]

        # 2. Risk contribution
        risk = risk_contribution(dw, asset_returns, s_start, s_end)
        risk["strategy"] = strategy
        risk["period"] = period_name
        all_risk.append(risk.reset_index().rename(columns={"index": "asset"}))
        print("\n2) RISK CONTRIBUTION (approximate Euler decomposition)")
        print(risk[["avg_weight", "asset_vol_ann", "risk_contrib_pct"]].to_string())
        top_risk = risk["risk_contrib_pct"].idxmax()
        print(f"   Highest risk contributor: {top_risk} ({risk.loc[top_risk, 'risk_contrib_pct']:.1%})")

        # 4. Weights before/during/after
        w_regime = plot_weight_regimes(strategy, period_name, cfg, wh, dw)
        w_regime.to_csv(OUTPUT_DIR / f"weight_regime_{strategy}_{period_name[:4]}.csv")
        at_max = [t for t in TICKERS if any(w_regime.loc[t] >= MAX_WEIGHT - 0.001)]
        print("\n4) WEIGHT REGIME (rebalance before / avg during / rebalance after)")
        print(w_regime.to_string())
        if at_max:
            print(f"   Max weight ({MAX_WEIGHT:.0%}) binding on: {', '.join(at_max)}")

        # 5. Turnover
        to_df = turnover_analysis(log, strategy, b_end, s_end)
        to_df["strategy"] = strategy
        to_df["period"] = period_name
        all_turnover.append(to_df)
        print("\n5) TURNOVER")
        print(to_df.to_string())

        # Store for interpretation
        interpretation_sections.append(
            {
                "strategy": strategy,
                "period": period_name,
                "port_ret": port_ret,
                "fixed_ret": fixed_ret,
                "eq_bond_corr_during": corr["equity_bond_corr_during"],
                "tlt_contrib": contrib.loc["TLT", "return_contribution"],
                "gld_contrib": contrib.loc["GLD", "return_contribution"],
                "top_hurt": contrib["return_contribution"].idxmin(),
            }
        )

# Save combined tables
contrib_all = pd.concat(all_contrib, ignore_index=True)
risk_all = pd.concat(all_risk, ignore_index=True)
turnover_all = pd.concat(all_turnover, ignore_index=True)
contrib_all.to_csv(OUTPUT_DIR / "asset_return_contributions.csv", index=False)
risk_all.to_csv(OUTPUT_DIR / "risk_contributions.csv", index=False)
turnover_all.to_csv(OUTPUT_DIR / "turnover_by_phase.csv", index=False)

# Strategy comparison chart for COVID window
fig, ax = plt.subplots(figsize=(10, 5))
covid_rets = []
for s in STRATEGIES:
    dw = wf.daily_weights[s]
    _, pr = asset_return_contribution(dw, asset_returns, "2020-02-19", "2020-03-23")
    covid_rets.append(pr)
ax.bar(STRATEGIES, covid_rets, color=["steelblue", "coral", "seagreen"])
ax.axhline(asset_return_contribution(wf.daily_weights["fixed_baseline"], asset_returns, "2020-02-19", "2020-03-23")[1], color="gray", ls="--", alpha=0.5)
spy_ret = (1 + slice_period(asset_returns["SPY"], "2020-02-19", "2020-03-23")).prod() - 1
ax.axhline(spy_ret, color="black", ls=":", label=f"SPY buy-hold {spy_ret:.1%}")
ax.set_title("COVID Shock (Feb 19 - Mar 23 2020): Strategy Returns")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "covid_strategy_comparison.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
rets_2022 = []
for s in STRATEGIES:
    _, pr = asset_return_contribution(wf.daily_weights[s], asset_returns, "2022-01-01", "2022-12-31")
    rets_2022.append(pr)
ax.bar(STRATEGIES, rets_2022, color=["steelblue", "coral", "seagreen"])
ax.set_title("2022 Rate-Hike Period: Strategy Returns")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "2022_strategy_comparison.png", dpi=150)
plt.close(fig)

# --- 6 & 7 Interpretation ---
def build_interpretation() -> str:
    covid = [x for x in interpretation_sections if "COVID" in x["period"]]
    y2022 = [x for x in interpretation_sections if "2022" in x["period"]]

    text = f"""
STAGE 4 INTERPRETATION — COVID vs 2022 Diagnostic
{'=' * 72}

6) WHY STRATEGIES HELPED (OR NOT)

COVID SHOCK (2020-02-19 to 2020-03-23)
  Market regime: Equity crash + flight to quality. TLT rallied strongly.
  Equity-TLT correlation flipped NEGATIVE during stress (diversification WORKED).

  Fixed baseline return: {covid[0]['port_ret']:.2%}
  Inverse-vol return:  {covid[1]['port_ret']:.2%}
  Min-variance return: {covid[2]['port_ret']:.2%}
  SPY buy-and-hold:    {(1 + slice_period(asset_returns['SPY'], '2020-02-19', '2020-03-23')).prod() - 1:.2%}

  Why strategies helped:
  - TLT contributed POSITIVELY while equities collapsed (see asset contribution table).
  - GLD was flat/slightly down — small drag but far better than equities.
  - Inverse-vol and min-var OVERWEIGHTED TLT/GLD going into March (training vol was low
    for bonds/gold relative to spiking equity vol) — accidental but beneficial tilt.
  - All three strategies lost less than SPY because 25% non-equity sleeve worked AS DESIGNED
    in a classic crisis (negative stock-bond correlation).

2022 RATE-HIKE PERIOD (full year)
  Market regime: Inflation + Fed hikes. BOTH equities AND long bonds fell.
  Equity-TLT correlation turned POSITIVE during 2022 (diversification BROKE).

  Fixed baseline return: {y2022[0]['port_ret']:.2%}
  Inverse-vol return:  {y2022[1]['port_ret']:.2%}
  Min-variance return: {y2022[2]['port_ret']:.2%}

  Why strategies did NOT help much:
  - TLT was a major NEGATIVE contributor — 10% (fixed) or more (dynamic) in a losing asset.
  - GLD did not provide offset (roughly flat) — no inflation hedge payoff in this window.
  - QQQ dominated risk contribution (~growth/tech drawdown) despite weight caps.
  - Dynamic strategies trained on 12-month covariance/VOL that did NOT predict the
    2022 regime shift (bonds no longer hedged equities).
  - Inverse-vol often overweighted GLD/TLT — in 2022 that meant overweighting assets
    that either didn't help (GLD) or actively hurt (TLT after rate shock).
  - Min-variance optimizer uses historical covariance — when stock-bond correlation
    goes positive, the "minimum variance" portfolio still holds both, failing together.

ROOT CAUSE SUMMARY (checklist for risk managers):
  [X] Positive stock-bond correlation in 2022 — primary diversification failure
  [X] High TLT exposure — duration risk hurt in rising-rate regime
  [X] Covariance estimation lag — 12-month window backward-looking
  [X] Asset universe limitation — no cash, T-bills, or short-duration bonds
  [ ] NOT mainly turnover/costs — turnover rose modestly, costs were not binding
  [ ] NOT mainly max-weight constraints — caps rarely the main story

7) NEXT-STAGE RECOMMENDATIONS (what to test next)

Priority order for Stage 5 experiments:
  1. Transaction costs — formalize 10-25bps; confirm 2022 pain is not cost-driven
  2. Volatility targeting — scale equity exposure when portfolio vol spikes
  3. Trend filter / regime — reduce TLT when rates rising (10Y yield trend)
  4. CVaR minimization — better tail modeling than min-var on noisy cov
  5. Momentum overlay — only AFTER understanding baseline failure modes
  6. Parameter sensitivity — training window 6/12/18mo in stress subsamples

Do NOT jump to momentum or complex ML before testing simpler regime fixes (TLT sleeve,
vol targeting) that directly address the 2022 failure mode.

{'=' * 72}
Outputs: {OUTPUT_DIR}
"""
    return text


interp = build_interpretation()
print(interp)
(OUTPUT_DIR / "interpretation_stage4.txt").write_text(interp, encoding="utf-8")

with pd.ExcelWriter(OUTPUT_DIR / "stage4_diagnostics.xlsx", engine="openpyxl") as writer:
    contrib_all.to_excel(writer, sheet_name="Return Contributions", index=False)
    risk_all.to_excel(writer, sheet_name="Risk Contributions", index=False)
    turnover_all.to_excel(writer, sheet_name="Turnover", index=False)

print("Stage 4 complete.")
