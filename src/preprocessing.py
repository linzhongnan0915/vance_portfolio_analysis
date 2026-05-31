"""Return series construction and calendar helpers."""

from __future__ import annotations

import pandas as pd


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="any")


def is_new_month(date: pd.Timestamp, prev_date: pd.Timestamp) -> bool:
    return (date.year, date.month) != (prev_date.year, prev_date.month)
