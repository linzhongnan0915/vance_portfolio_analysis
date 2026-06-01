"""Dashboard display formatting."""

from __future__ import annotations

import pandas as pd


def pct(x, d: int = 2) -> str:
    return f"{x:.2%}" if pd.notna(x) else "N/A"


def fmt_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pct_cols = [
        "Cumulative Return",
        "Annualized Return",
        "Annualized Volatility",
        "Maximum Drawdown",
        "VaR 95% (daily)",
        "CVaR 95% (daily)",
        "Tracking Error vs SPY",
    ]
    for c in pct_cols:
        if c in out.columns:
            out[c] = out[c].astype(object)
            out[c] = out[c].apply(lambda x: pct(x) if isinstance(x, (int, float)) else x)
    for c in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Beta vs SPY", "Information Ratio vs SPY"]:
        if c in out.columns:
            out[c] = out[c].astype(object)
            out[c] = out[c].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    return out
