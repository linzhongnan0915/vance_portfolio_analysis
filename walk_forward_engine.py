"""Backward-compatible shim — use ``src.backtest`` / ``src.signals`` in new code."""

from src.backtest import WalkForwardResult, run_walk_forward
from src.config import ANALYSIS_START, MAX_WEIGHT, MIN_WEIGHT, STRATEGIES, TRAIN_MONTHS
from src.portfolio import calc_turnover, simulate_month_returns
from src.signals import estimate_volatilities, make_weight_functions

__all__ = [
    "ANALYSIS_START",
    "MAX_WEIGHT",
    "MIN_WEIGHT",
    "STRATEGIES",
    "TRAIN_MONTHS",
    "WalkForwardResult",
    "calc_turnover",
    "estimate_volatilities",
    "make_weight_functions",
    "run_walk_forward",
    "simulate_month_returns",
]
