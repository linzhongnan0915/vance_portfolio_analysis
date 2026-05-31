"""Backward-compatible shim — use ``src`` modules directly in new code."""

from src.config import (
    BENCHMARK,
    IN_SAMPLE_END,
    IN_SAMPLE_START,
    OUT_SAMPLE_START,
    RISK_FREE_RATE,
    TARGET_WEIGHTS,
    TICKERS,
    TRADING_DAYS,
)
from src.data_loader import load_adjusted_prices
from src.metrics import (
    MetricResult,
    compute_drawdown,
    compute_metrics,
    metrics_to_series,
    subperiod_metrics_to_series,
)
from src.portfolio import (
    build_portfolio_series,
    monthly_rebalanced_returns,
    weight_drift_history,
)
from src.preprocessing import is_new_month, prices_to_returns

__all__ = [
    "BENCHMARK",
    "IN_SAMPLE_END",
    "IN_SAMPLE_START",
    "MetricResult",
    "OUT_SAMPLE_START",
    "RISK_FREE_RATE",
    "TARGET_WEIGHTS",
    "TICKERS",
    "TRADING_DAYS",
    "build_portfolio_series",
    "compute_drawdown",
    "compute_metrics",
    "is_new_month",
    "load_adjusted_prices",
    "metrics_to_series",
    "monthly_rebalanced_returns",
    "prices_to_returns",
    "subperiod_metrics_to_series",
    "weight_drift_history",
]
