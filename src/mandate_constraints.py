"""Investment mandate constraints for portfolio weight construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.signals import estimate_volatilities

EQUITY_TICKERS = ["QQQ", "SPY", "DIA"]
DEFENSIVE_ETF = "SHY"


@dataclass
class MandateConstraints:
    shy_cap: float | None = None
    max_equity_single: float = 0.40
    min_equity_total: float = 0.60
    max_single_asset: float = 0.40


def _normalize(w: pd.Series) -> pd.Series:
    w = w.clip(lower=0.0)
    s = w.sum()
    return w / s if s > 0 else pd.Series(1.0 / len(w), index=w.index)


def _redistribute_proportional(w: pd.Series, from_idx: list[str], to_idx: list[str], amount: float) -> pd.Series:
    if amount <= 0 or not to_idx:
        return w
    out = w.copy()
    base = out[to_idx].sum()
    if base <= 0:
        share = amount / len(to_idx)
        for t in to_idx:
            out[t] += share
    else:
        for t in to_idx:
            out[t] += amount * (out[t] / base)
    return out


def apply_mandate_constraints(w: pd.Series, mandate: MandateConstraints) -> tuple[pd.Series, str]:
    notes: list[str] = []
    w = _normalize(w)

    if mandate.shy_cap is not None and DEFENSIVE_ETF in w.index:
        if w[DEFENSIVE_ETF] > mandate.shy_cap + 1e-10:
            excess = w[DEFENSIVE_ETF] - mandate.shy_cap
            w[DEFENSIVE_ETF] = mandate.shy_cap
            others = [t for t in w.index if t != DEFENSIVE_ETF]
            w = _redistribute_proportional(w, [DEFENSIVE_ETF], others, excess)
            notes.append(f"SHY capped at {mandate.shy_cap:.0%}; excess redistributed pro-rata")

    for t in EQUITY_TICKERS:
        if t in w.index and w[t] > mandate.max_equity_single + 1e-10:
            excess = w[t] - mandate.max_equity_single
            w[t] = mandate.max_equity_single
            others = [x for x in w.index if x != t]
            w = _redistribute_proportional(w, [t], others, excess)
            notes.append(f"{t} capped at {mandate.max_equity_single:.0%}")

    for t in ["GLD", "TLT"]:
        if t in w.index and w[t] > mandate.max_single_asset + 1e-10:
            excess = w[t] - mandate.max_single_asset
            w[t] = mandate.max_single_asset
            others = [x for x in w.index if x != t]
            w = _redistribute_proportional(w, [t], others, excess)

    w = _normalize(w)

    eq = [t for t in EQUITY_TICKERS if t in w.index]
    eq_sum = w[eq].sum()
    if eq and eq_sum < mandate.min_equity_total - 1e-10:
        deficit = mandate.min_equity_total - eq_sum
        non_eq = [t for t in w.index if t not in eq]
        non_sum = w[non_eq].sum()
        if non_sum >= deficit:
            w[eq] = w[eq] * (mandate.min_equity_total / eq_sum)
            take = w[non_eq] * (deficit / non_sum)
            w[non_eq] = w[non_eq] - take
            notes.append(f"Equity floor {mandate.min_equity_total:.0%} enforced")
        else:
            notes.append(f"WARNING: cannot reach equity floor {mandate.min_equity_total:.0%}")

    w = _normalize(w)
    return w, "; ".join(notes)


def make_mandate_weight_functions(
    tickers: list[str],
    target_weights: pd.Series,
    mandate: MandateConstraints,
    max_weight_optimizer: float = 0.40,
) -> dict:
    tw = target_weights.reindex(tickers).astype(float)
    tw = tw / tw.sum()

    def fixed_weights(_train: pd.DataFrame) -> tuple[pd.Series, str]:
        w, notes = apply_mandate_constraints(tw.copy(), mandate)
        return w, notes

    def inverse_volatility_weights(train: pd.DataFrame) -> tuple[pd.Series, str]:
        vol = estimate_volatilities(train[tickers])
        if (vol <= 0).any() or vol.isna().any():
            raw = pd.Series(1.0 / len(tickers), index=tickers)
        else:
            raw = (1.0 / vol) / (1.0 / vol).sum()
        w, notes = apply_mandate_constraints(raw, mandate)
        return w, notes

    def minimum_variance_weights(train: pd.DataFrame) -> tuple[pd.Series, str]:
        if len(train) < 30:
            raw = pd.Series(1.0 / len(tickers), index=tickers)
            w, notes = apply_mandate_constraints(raw, mandate)
            return w, f"Insufficient rows; equal-weight fallback. {notes}"

        sub = train[tickers]
        cov = sub.cov().values
        n = len(tickers)
        x0 = np.ones(n) / n
        bounds = [(0.0, max_weight_optimizer) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        result = minimize(lambda w: float(w @ cov @ w), x0, method="SLSQP", bounds=bounds, constraints=constraints)

        if not result.success:
            raw = pd.Series(1.0 / len(tickers), index=tickers)
            opt_note = "Optimizer failed; equal-weight fallback"
        else:
            arr = np.clip(result.x, 0.0, max_weight_optimizer)
            arr = arr / arr.sum()
            raw = pd.Series(arr, index=tickers)
            opt_note = ""
        w, notes = apply_mandate_constraints(raw, mandate)
        return w, f"{opt_note}; {notes}".strip("; ")

    return {
        "fixed_baseline": fixed_weights,
        "inverse_volatility": inverse_volatility_weights,
        "minimum_variance": minimum_variance_weights,
    }


def fixed_shy_weights(shy_pct: float) -> pd.Series:
    eq = 0.25
    if shy_pct <= 0.05 + 1e-9:
        return pd.Series({"QQQ": eq, "SPY": eq, "DIA": eq, "GLD": 0.15, "TLT": 0.05, DEFENSIVE_ETF: 0.05})
    if shy_pct <= 0.10 + 1e-9:
        return pd.Series({"QQQ": eq, "SPY": eq, "DIA": eq, "GLD": 0.15, "TLT": 0.0, DEFENSIVE_ETF: 0.10})
    return pd.Series({"QQQ": eq, "SPY": eq, "DIA": eq, "GLD": 0.10, "TLT": 0.0, DEFENSIVE_ETF: 0.15})


def classify_mandate(avg_equity: float, min_equity: float, avg_shy: float, max_shy: float, policy_label: str) -> str:
    if "uncapped" in policy_label.lower() or max_shy > 0.25 or avg_equity < 0.55:
        if avg_equity < 0.55 or avg_shy > 0.25:
            return "mandate-inconsistent"
    if avg_equity < 0.60 and "capital-preservation" not in policy_label.lower():
        return "mandate-inconsistent"
    if avg_shy > 0.20 and "capital-preservation" not in policy_label.lower():
        return "mandate-inconsistent"
    if avg_equity >= 0.70 and avg_shy <= 0.10:
        return "equity-oriented"
    if avg_equity >= 0.60 and avg_shy <= 0.15:
        return "balanced risk-managed"
    if avg_equity < 0.60 or avg_shy >= 0.15:
        return "capital-preservation-like"
    return "balanced risk-managed"
