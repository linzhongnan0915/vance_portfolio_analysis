"""Walk-forward backtest engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.config import (
    ANALYSIS_START,
    MAX_WEIGHT,
    STRATEGIES,
    TARGET_WEIGHTS,
    TICKERS,
    TRAIN_MONTHS,
)
from src.portfolio import calc_turnover, simulate_month_returns
from src.signals import estimate_volatilities, make_weight_functions


def first_trading_day_on_or_after(daily_index: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp | None:
    sub = daily_index[daily_index >= ts]
    return sub[0] if len(sub) else None


def month_period_bounds(
    daily_index: pd.DatetimeIndex,
    year: int,
    month: int,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start = pd.Timestamp(year=year, month=month, day=1)
    if month == 12:
        end = pd.Timestamp(year=year + 1, month=1, day=1) - pd.Timedelta(days=1)
    else:
        end = pd.Timestamp(year=year, month=month + 1, day=1) - pd.Timedelta(days=1)
    test_start = first_trading_day_on_or_after(daily_index, start)
    test_end_candidates = daily_index[(daily_index >= start) & (daily_index <= end)]
    test_end = test_end_candidates[-1] if len(test_end_candidates) else None
    return test_start, test_end


@dataclass
class WalkForwardResult:
    decision_log: pd.DataFrame
    daily_returns: dict[str, pd.Series]
    weight_history: dict[str, pd.DataFrame]
    turnover_history: dict[str, pd.Series]
    daily_weights: dict[str, pd.DataFrame]
    tickers: list[str] = field(default_factory=lambda: list(TICKERS))
    train_months: int = TRAIN_MONTHS
    max_weight: float = MAX_WEIGHT
    tx_cost_bps: float = 0.0


def run_walk_forward(
    asset_returns: pd.DataFrame,
    bench_returns: pd.Series,
    tickers: list[str] | None = None,
    target_weights: pd.Series | None = None,
    analysis_start: str = ANALYSIS_START,
    train_months: int = TRAIN_MONTHS,
    max_weight: float = MAX_WEIGHT,
    tx_cost_bps: float = 0.0,
    strategies: list[str] | None = None,
    custom_weight_fn: dict[str, Callable[[pd.DataFrame], tuple[pd.Series, str]]] | None = None,
) -> WalkForwardResult:
    tickers = tickers or list(TICKERS)
    target_weights = target_weights if target_weights is not None else TARGET_WEIGHTS.reindex(tickers)
    strategies = strategies or STRATEGIES

    asset_returns = asset_returns[tickers].sort_index()
    bench_returns = bench_returns.sort_index()
    idx = asset_returns.index

    weight_fn = custom_weight_fn if custom_weight_fn else make_weight_functions(tickers, target_weights, max_weight=max_weight)
    cost_rate = tx_cost_bps / 10_000.0

    start_ts = pd.Timestamp(analysis_start)
    first_test = start_ts + pd.DateOffset(months=train_months)
    first_test = pd.Timestamp(year=first_test.year, month=first_test.month, day=1)

    months: list[tuple[int, int]] = []
    cursor = first_test
    while cursor <= idx[-1]:
        months.append((cursor.year, cursor.month))
        cursor = cursor + pd.DateOffset(months=1)

    log_rows = []
    daily_by_strategy: dict[str, list[pd.Series]] = {s: [] for s in strategies}
    daily_w_by_strategy: dict[str, list[pd.DataFrame]] = {s: [] for s in strategies}
    weight_hist: dict[str, list[dict]] = {s: [] for s in strategies}
    turnover_hist: dict[str, list[tuple[pd.Timestamp, float]]] = {s: [] for s in strategies}
    end_weights: dict[str, np.ndarray | None] = {s: None for s in strategies}

    for year, month in months:
        test_start, test_end = month_period_bounds(idx, year, month)
        if test_start is None or test_end is None:
            continue

        train_end = idx[idx < test_start][-1]
        train_start_target = test_start - pd.DateOffset(months=train_months)
        train_candidates = idx[(idx >= train_start_target) & (idx <= train_end)]
        if len(train_candidates) < 60:
            continue
        train_start = train_candidates[0]

        train = asset_returns.loc[train_start:train_end]
        test = asset_returns.loc[test_start:test_end]
        bench_month = bench_returns.loc[test_start:test_end]
        est_vol = estimate_volatilities(train)
        monthly_bench = (1 + bench_month).prod() - 1

        for strategy in strategies:
            w_series, opt_notes = weight_fn[strategy](train)
            w_new = w_series.values.astype(float)
            w_new = w_new / w_new.sum()

            w_old = end_weights[strategy]
            to = 0.0 if w_old is None else calc_turnover(w_new, w_old)

            port_daily, w_end, w_daily = simulate_month_returns(test, w_new, tickers)
            if cost_rate > 0 and to > 0:
                port_daily.iloc[0] -= to * cost_rate

            end_weights[strategy] = w_end
            daily_by_strategy[strategy].append(port_daily)
            daily_w_by_strategy[strategy].append(w_daily)
            turnover_hist[strategy].append((test_start, to))
            weight_hist[strategy].append(
                {"rebalance_date": test_start, **{f"weight_{t}": w_new[i] for i, t in enumerate(tickers)}}
            )

            log_rows.append(
                {
                    "rebalance_date": test_start.date(),
                    "training_start": train_start.date(),
                    "training_end": train_end.date(),
                    "testing_start": test_start.date(),
                    "testing_end": test_end.date(),
                    "strategy": strategy,
                    **{f"est_vol_{t}": est_vol[t] for t in tickers},
                    **{f"weight_{t}": w_new[i] for i, t in enumerate(tickers)},
                    "monthly_portfolio_return": (1 + port_daily).prod() - 1,
                    "monthly_benchmark_return": monthly_bench,
                    "turnover": to,
                    "tx_cost_bps": tx_cost_bps,
                    "train_months": train_months,
                    "max_weight": max_weight,
                    "notes": opt_notes,
                }
            )

    decision_log = pd.DataFrame(log_rows)
    daily_returns = {s: pd.concat(daily_by_strategy[s]).sort_index() for s in strategies if daily_by_strategy[s]}
    weight_history = {
        s: pd.DataFrame(weight_hist[s]).set_index("rebalance_date") if weight_hist[s] else pd.DataFrame()
        for s in strategies
    }
    turnover_history = {
        s: pd.Series(dict(turnover_hist[s]), name="turnover") if turnover_hist[s] else pd.Series(dtype=float)
        for s in strategies
    }
    daily_weights = {
        s: pd.concat(daily_w_by_strategy[s]).sort_index() for s in strategies if daily_w_by_strategy[s]
    }
    return WalkForwardResult(
        decision_log,
        daily_returns,
        weight_history,
        turnover_history,
        daily_weights,
        tickers=tickers,
        train_months=train_months,
        max_weight=max_weight,
        tx_cost_bps=tx_cost_bps,
    )
