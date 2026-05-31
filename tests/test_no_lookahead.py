"""Look-ahead and walk-forward timing checks."""

from __future__ import annotations

import pandas as pd

from src.backtest import run_walk_forward
from src.config import ANALYSIS_START, BENCHMARK, TARGET_WEIGHTS, TICKERS
from tests.fixtures import make_synthetic_daily_returns


def test_walk_forward_training_ends_before_test_start():
    daily = make_synthetic_daily_returns()
    bench = daily[BENCHMARK]
    result = run_walk_forward(
        daily[TICKERS],
        bench,
        tickers=TICKERS,
        target_weights=TARGET_WEIGHTS,
        strategies=["fixed_baseline"],
        analysis_start=ANALYSIS_START,
    )
    log = result.decision_log
    assert not log.empty, "Synthetic panel should yield at least one walk-forward month"
    for _, row in log.iterrows():
        train_end = pd.Timestamp(row["training_end"])
        test_start = pd.Timestamp(row["testing_start"])
        assert train_end < test_start, "Look-ahead: training must end before test month"
