"""Load and cache adjusted ETF prices."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import (
    BENCHMARK,
    DATA_DIR,
    DATA_PROCESSED_DIR,
    DEFAULT_PRICE_START,
    PRICE_CACHE_FILENAME,
    TICKERS,
)


def _price_cache_path(data_dir: Path | None = None) -> Path:
    base = data_dir or DATA_DIR
    processed = base / "processed" / PRICE_CACHE_FILENAME
    if processed.exists():
        return processed
    legacy = base / PRICE_CACHE_FILENAME
    return processed if (base / "processed").exists() else legacy


def load_adjusted_prices(
    data_dir: Path | None = None,
    tickers: list[str] | None = None,
    start: str = DEFAULT_PRICE_START,
) -> pd.DataFrame:
    data_dir = data_dir or DATA_DIR
    tickers = tickers or sorted(set(TICKERS + [BENCHMARK]))
    cache_path = _price_cache_path(data_dir)

    if cache_path.exists():
        return pd.read_csv(cache_path, index_col=0, parse_dates=True).sort_index()

    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers

    prices = prices.sort_index().dropna(how="any")
    save_path = data_dir / "processed" / PRICE_CACHE_FILENAME
    save_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(save_path)
    return prices
