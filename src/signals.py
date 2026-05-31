"""Signal / weight generation for walk-forward strategies."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import MAX_WEIGHT, MIN_WEIGHT, TRADING_DAYS


def estimate_volatilities(train: pd.DataFrame) -> pd.Series:
    return train.std(ddof=1) * np.sqrt(TRADING_DAYS)


def make_weight_functions(
    tickers: list[str],
    target_weights: pd.Series,
    max_weight: float = MAX_WEIGHT,
    min_weight: float = MIN_WEIGHT,
) -> dict[str, Callable[[pd.DataFrame], tuple[pd.Series, str]]]:
    tw = target_weights.reindex(tickers).astype(float)
    tw = tw / tw.sum()

    def fixed_weights(_train: pd.DataFrame) -> tuple[pd.Series, str]:
        return tw.copy(), ""

    def inverse_volatility_weights(train: pd.DataFrame) -> tuple[pd.Series, str]:
        vol = estimate_volatilities(train[tickers])
        if (vol <= 0).any() or vol.isna().any():
            w = pd.Series(1.0 / len(tickers), index=tickers)
            return w, "Zero/NaN vol; equal-weight fallback"
        inv = 1.0 / vol
        return inv / inv.sum(), ""

    def minimum_variance_weights(train: pd.DataFrame) -> tuple[pd.Series, str]:
        if len(train) < 30:
            w = pd.Series(1.0 / len(tickers), index=tickers)
            return w, "Insufficient training rows; equal-weight fallback"

        sub = train[tickers]
        cov = sub.cov().values
        n = len(tickers)
        x0 = np.ones(n) / n
        bounds = [(min_weight, max_weight) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(lambda w: float(w @ cov @ w), x0, method="SLSQP", bounds=bounds, constraints=constraints)

        notes = []
        if not result.success:
            notes.append("Optimizer failed; equal-weight fallback")
            w_arr = x0
        else:
            w_arr = np.clip(result.x, min_weight, max_weight)
            w_arr = w_arr / w_arr.sum()
            at_max = [tickers[i] for i, v in enumerate(w_arr) if v >= max_weight - 1e-4]
            if at_max:
                notes.append(f"Max weight binding: {', '.join(at_max)}")

        return pd.Series(w_arr, index=tickers), "; ".join(notes)

    return {
        "fixed_baseline": fixed_weights,
        "inverse_volatility": inverse_volatility_weights,
        "minimum_variance": minimum_variance_weights,
    }
