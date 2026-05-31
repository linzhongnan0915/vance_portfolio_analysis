"""Deterministic synthetic market data for unit tests (no network, no CSV cache)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BENCHMARK, TICKERS

# Small constant daily simple returns per ticker (deterministic, well below 50% cap).
_DAILY_DRIFT: dict[str, float] = {
    "QQQ": 0.00040,
    "SPY": 0.00030,
    "DIA": 0.00025,
    "GLD": 0.00010,
    "TLT": 0.00005,
}


def _all_tickers(tickers: list[str] | None) -> list[str]:
    base = list(tickers or TICKERS)
    if BENCHMARK not in base:
        base.append(BENCHMARK)
    return base


def make_synthetic_daily_returns(
    start: str = "2008-01-01",
    end: str = "2022-12-31",
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Business-day return panel with enough history for 12-month walk-forward training
    and multiple out-of-sample test months.

    SPY is included for benchmark-aligned metrics and backtests.
    """
    cols = _all_tickers(tickers)
    idx = pd.bdate_range(start=start, end=end, freq="B")
    data = {_t: np.full(len(idx), _DAILY_DRIFT.get(_t, 0.00020)) for _t in cols}
    return pd.DataFrame(data, index=idx)


def make_synthetic_prices(
    start: str = "2008-01-01",
    end: str = "2022-12-31",
    tickers: list[str] | None = None,
    base_level: float = 100.0,
) -> pd.DataFrame:
    """Adjusted-close style levels built from synthetic daily simple returns."""
    daily = make_synthetic_daily_returns(start=start, end=end, tickers=tickers)
    return (1.0 + daily).cumprod() * base_level
