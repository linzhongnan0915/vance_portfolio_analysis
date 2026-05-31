"""Vance portfolio analysis — core library."""

from src.backtest import WalkForwardResult, run_walk_forward
from src.config import (
    ANALYSIS_START,
    BENCHMARK,
    DATA_DIR,
    IN_SAMPLE_END,
    IN_SAMPLE_START,
    MAX_WEIGHT,
    MIN_WEIGHT,
    OUT_SAMPLE_START,
    OUTPUT_DIR,
    RISK_FREE_RATE,
    ROOT,
    STRATEGIES,
    TARGET_WEIGHTS,
    TICKERS,
    TRAIN_MONTHS,
    TRADING_DAYS,
)
from src.data_loader import load_adjusted_prices
from src.metrics import MetricResult, compute_drawdown, compute_metrics, metrics_to_series, subperiod_metrics_to_series
from src.portfolio import (
    build_portfolio_series,
    calc_turnover,
    monthly_rebalanced_returns,
    simulate_month_returns,
    weight_drift_history,
)
from src.preprocessing import is_new_month, prices_to_returns
from src.signals import estimate_volatilities, make_weight_functions

__all__ = [
    "ANALYSIS_START",
    "BENCHMARK",
    "DATA_DIR",
    "IN_SAMPLE_END",
    "IN_SAMPLE_START",
    "MAX_WEIGHT",
    "MIN_WEIGHT",
    "MetricResult",
    "OUT_SAMPLE_START",
    "OUTPUT_DIR",
    "RISK_FREE_RATE",
    "ROOT",
    "STRATEGIES",
    "TARGET_WEIGHTS",
    "TICKERS",
    "TRAIN_MONTHS",
    "TRADING_DAYS",
    "WalkForwardResult",
    "build_portfolio_series",
    "calc_turnover",
    "compute_drawdown",
    "compute_metrics",
    "estimate_volatilities",
    "is_new_month",
    "load_adjusted_prices",
    "make_weight_functions",
    "metrics_to_series",
    "monthly_rebalanced_returns",
    "prices_to_returns",
    "run_walk_forward",
    "simulate_month_returns",
    "subperiod_metrics_to_series",
    "weight_drift_history",
]
