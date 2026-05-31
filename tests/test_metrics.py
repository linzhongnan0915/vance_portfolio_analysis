"""Metrics output shape and annualization convention."""

from __future__ import annotations

import pandas as pd

from src.config import BENCHMARK, TICKERS, TRADING_DAYS
from src.metrics import compute_metrics, metrics_to_series
from tests.fixtures import make_synthetic_daily_returns


def test_metrics_to_series_has_expected_keys(synthetic_daily_returns):
    daily = synthetic_daily_returns
    port = daily[TICKERS].mean(axis=1)
    bench = daily[BENCHMARK]
    m = compute_metrics(port, bench, periods=TRADING_DAYS)
    s = metrics_to_series(m)
    for key in ["Annualized Return", "Sharpe Ratio", "Maximum Drawdown", "Beta vs SPY"]:
        assert key in s.index


def test_metrics_output_is_scalar_series(synthetic_daily_returns):
    daily = synthetic_daily_returns
    port = daily[TICKERS].mean(axis=1)
    bench = daily[BENCHMARK]
    s = metrics_to_series(compute_metrics(port, bench))
    assert isinstance(s, pd.Series)
    assert len(s) >= 10
