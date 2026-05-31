"""Portfolio weights, drift, and monthly simulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BENCHMARK, IN_SAMPLE_START, TARGET_WEIGHTS
from src.data_loader import load_adjusted_prices
from src.preprocessing import is_new_month, prices_to_returns


def monthly_rebalanced_returns(
    asset_returns: pd.DataFrame,
    target_weights: pd.Series,
) -> pd.Series:
    tickers = list(target_weights.index)
    w = target_weights.values.astype(float)
    w = w / w.sum()
    port_rets = []
    dates = asset_returns.index

    for i, dt in enumerate(dates):
        if i > 0 and is_new_month(dt, dates[i - 1]):
            w = target_weights.values.astype(float)
            w = w / w.sum()

        r = asset_returns.loc[dt, tickers].values
        port_rets.append(float(np.dot(w, r)))

        w = w * (1.0 + r)
        if w.sum() > 0:
            w = w / w.sum()

    return pd.Series(port_rets, index=dates, name="portfolio_return")


def weight_drift_history(
    asset_returns: pd.DataFrame,
    target_weights: pd.Series,
) -> pd.DataFrame:
    tickers = list(target_weights.index)
    w = target_weights.values.astype(float)
    w = w / w.sum()
    rows = []
    dates = asset_returns.index

    for i, dt in enumerate(dates):
        if i > 0 and is_new_month(dt, dates[i - 1]):
            w = target_weights.values.astype(float)
            w = w / w.sum()

        row = {t: w[j] for j, t in enumerate(tickers)}
        row["date"] = dt
        row["max_drift_from_target"] = max(abs(w[j] - target_weights[t]) for j, t in enumerate(tickers))
        rows.append(row)

        r = asset_returns.loc[dt, tickers].values
        w = w * (1.0 + r)
        if w.sum() > 0:
            w = w / w.sum()

    return pd.DataFrame(rows).set_index("date")


def simulate_month_returns(
    test_returns: pd.DataFrame,
    start_weights: np.ndarray,
    tickers: list[str],
) -> tuple[pd.Series, np.ndarray, pd.DataFrame]:
    w = start_weights.copy()
    port = []
    weight_rows = []
    for _dt, row in test_returns.iterrows():
        weight_rows.append({t: w[i] for i, t in enumerate(tickers)})
        r = row[tickers].values
        port.append(float(np.dot(w, r)))
        w = w * (1.0 + r)
        if w.sum() > 0:
            w = w / w.sum()
    weights_df = pd.DataFrame(weight_rows, index=test_returns.index)
    return pd.Series(port, index=test_returns.index), w, weights_df


def calc_turnover(w_new: np.ndarray, w_old: np.ndarray) -> float:
    return float(0.5 * np.abs(w_new - w_old).sum())


def build_portfolio_series(data_dir: Path) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    prices = load_adjusted_prices(data_dir)
    daily_returns = prices_to_returns(prices)
    port_tickers = [t for t in TARGET_WEIGHTS.index if t in daily_returns.columns]
    port_returns = monthly_rebalanced_returns(daily_returns[port_tickers], TARGET_WEIGHTS)
    bench_returns = daily_returns[BENCHMARK].rename("benchmark_return")
    drift = weight_drift_history(daily_returns[port_tickers], TARGET_WEIGHTS)
    return (
        port_returns.loc[IN_SAMPLE_START:],
        bench_returns.loc[IN_SAMPLE_START:],
        drift.loc[IN_SAMPLE_START:],
    )
