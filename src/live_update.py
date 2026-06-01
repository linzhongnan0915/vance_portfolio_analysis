"""Daily live monitor artifacts (v3 MVP): prices, risk metrics, rebalance targets.

No broker connectivity, no order execution, no intraday or real-time claims.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.backtest import first_trading_day_on_or_after
from src.config import (
    BENCHMARK,
    DATA_DIR,
    LIVE_DAILY_SIGNAL_MODE,
    LIVE_DISCLAIMER,
    LIVE_POLICY_TICKERS,
    LIVE_POLICY_WEIGHTS,
    LIVE_STRATEGY_ID,
    LIVE_TIMEZONE,
    OUTPUT_LIVE_DIR,
    TRADING_DAYS,
)
from src.data_loader import _price_cache_path
from src.metrics import MetricResult, compute_metrics, metrics_to_series
from src.preprocessing import prices_to_returns

NY_TZ = ZoneInfo(LIVE_TIMEZONE)

LIVE_PRICE_COLUMNS = ["ticker", "adj_close", "price_date_used", "data_as_of"]
LIVE_METRICS_COLUMNS = [
    "strategy_id",
    "data_as_of",
    "price_date_used",
    "lookback_days",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "var_95_daily",
    "cvar_95_daily",
    "beta_vs_spy",
    "tracking_error",
    "information_ratio",
]


@dataclass
class LiveUpdateResult:
    data_as_of: date
    price_date_used: date
    signal_effective_date: date | None
    next_rebalance_date: date | None
    rebalance_due: bool
    prices_df: pd.DataFrame
    metrics_df: pd.DataFrame
    weights_df: pd.DataFrame
    signal: dict[str, Any]
    output_dir: Path
    dry_run: bool


def resolve_data_as_of(
    as_of: date | str | None = None,
    *,
    panel: pd.DataFrame | None = None,
) -> date:
    """
    Resolve data_as_of.

    When as_of is omitted and a price panel is provided, use the latest trading
    date in that panel (typical CLI path with local cache). Otherwise parse the
    explicit date; if omitted with no panel, fall back to today in America/New_York.
    """
    if as_of is None:
        if panel is not None and len(panel.index):
            return panel.index.max().date()
        return datetime.now(NY_TZ).date()
    if isinstance(as_of, str):
        return pd.Timestamp(as_of).date()
    return as_of


def load_prices_panel(
    data_dir: Path | None = None,
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Load adjusted closes from an injected panel (tests) or local processed cache (CLI).

    Does not call yfinance when cache is missing.
    """
    if prices is not None:
        out = prices.sort_index().copy()
        out.index = pd.to_datetime(out.index)
        return out

    cache = _price_cache_path(data_dir or DATA_DIR)
    if not cache.exists():
        raise FileNotFoundError(
            f"Price cache not found at {cache}. Run the research pipeline first or pass "
            "a prices DataFrame in tests."
        )
    out = pd.read_csv(cache, index_col=0, parse_dates=True).sort_index()
    out.index = pd.to_datetime(out.index)
    return out


def last_trading_day_on_or_before(index: pd.DatetimeIndex, as_of: date) -> pd.Timestamp:
    cutoff = pd.Timestamp(as_of)
    sub = index[index <= cutoff]
    if len(sub) == 0:
        raise ValueError(f"No trading days on or before {as_of}")
    return sub[-1]


def next_trading_day_after(index: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp | None:
    """Next index date strictly after ts, or None if the calendar ends at ts."""
    sub = index[index > ts]
    return sub[0] if len(sub) else None


def first_trading_day_of_month(index: pd.DatetimeIndex, year: int, month: int) -> pd.Timestamp | None:
    return first_trading_day_on_or_after(index, pd.Timestamp(year=year, month=month, day=1))


def next_scheduled_rebalance_date(index: pd.DatetimeIndex, price_date: pd.Timestamp) -> pd.Timestamp | None:
    """First trading day of the next calendar month strictly after price_date's month."""
    if price_date.month == 12:
        target = pd.Timestamp(year=price_date.year + 1, month=1, day=1)
    else:
        target = pd.Timestamp(year=price_date.year, month=price_date.month + 1, day=1)
    return first_trading_day_on_or_after(index, target)


def is_rebalance_due(index: pd.DatetimeIndex, price_date: pd.Timestamp) -> bool:
    """True only when price_date is the first trading day of its calendar month."""
    first = first_trading_day_of_month(index, price_date.year, price_date.month)
    return first is not None and price_date.normalize() == first.normalize()


def policy_weights() -> pd.Series:
    w = LIVE_POLICY_WEIGHTS.reindex(LIVE_POLICY_TICKERS).astype(float)
    w = w / w.sum()
    return w


def portfolio_returns(
    daily_returns: pd.DataFrame,
    weights: pd.Series,
    tickers: list[str] | None = None,
) -> pd.Series:
    tickers = tickers or list(weights.index)
    missing = [t for t in tickers if t not in daily_returns.columns]
    if missing:
        raise ValueError(f"Missing return columns for portfolio: {missing}")
    w = weights.reindex(tickers).fillna(0.0)
    return daily_returns[tickers].mul(w, axis=1).sum(axis=1)


def build_latest_prices(
    prices: pd.DataFrame,
    tickers: list[str],
    data_as_of: date,
    price_date: pd.Timestamp,
) -> pd.DataFrame:
    row_date = price_date.date()
    rows = []
    for t in tickers:
        if t not in prices.columns:
            continue
        val = prices.loc[price_date, t] if price_date in prices.index else np.nan
        rows.append(
            {
                "ticker": t,
                "adj_close": float(val) if pd.notna(val) else np.nan,
                "price_date_used": row_date.isoformat(),
                "data_as_of": data_as_of.isoformat(),
            }
        )
    return pd.DataFrame(rows, columns=LIVE_PRICE_COLUMNS)


def metric_result_to_row(
    m: MetricResult,
    *,
    strategy_id: str,
    data_as_of: date,
    price_date_used: date,
    lookback_days: int,
) -> dict[str, Any]:
    return {
        "strategy_id": strategy_id,
        "data_as_of": data_as_of.isoformat(),
        "price_date_used": price_date_used.isoformat(),
        "lookback_days": lookback_days,
        "cumulative_return": m.cumulative_return,
        "annualized_return": m.annualized_return,
        "annualized_volatility": m.annualized_volatility,
        "sharpe_ratio": m.sharpe_ratio,
        "sortino_ratio": m.sortino_ratio,
        "max_drawdown": m.max_drawdown,
        "calmar_ratio": m.calmar_ratio,
        "var_95_daily": m.var_95,
        "cvar_95_daily": m.cvar_95,
        "beta_vs_spy": m.beta,
        "tracking_error": m.tracking_error,
        "information_ratio": m.information_ratio,
    }


def build_signal_payload(
    *,
    data_as_of: date,
    price_date_used: date,
    signal_effective_date: date | None,
    next_rebalance_date: date | None,
    rebalance_due: bool,
    weights: pd.Series,
    metrics: MetricResult,
    data_quality: dict[str, Any],
) -> dict[str, Any]:
    m_series = metrics_to_series(metrics)
    risk_metrics = {k: _json_safe(v) for k, v in m_series.items()}
    w_dict = {t: float(weights[t]) for t in weights.index}

    return {
        "schema_version": "1.0",
        "data_as_of": data_as_of.isoformat(),
        "price_date_used": price_date_used.isoformat(),
        "signal_generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "signal_effective_date": (
            signal_effective_date.isoformat() if signal_effective_date is not None else None
        ),
        "next_rebalance_date": (
            next_rebalance_date.isoformat() if next_rebalance_date is not None else None
        ),
        "rebalance_due": rebalance_due,
        "strategy_id": LIVE_STRATEGY_ID,
        "daily_signal_mode": LIVE_DAILY_SIGNAL_MODE,
        "disclaimer": LIVE_DISCLAIMER,
        "weights": w_dict,
        "risk_metrics": risk_metrics,
        "data_quality": data_quality,
    }


def _json_safe(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        return None if (isinstance(v, float) and np.isnan(v)) or (isinstance(v, np.floating) and np.isnan(v)) else float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v


def write_live_artifacts(
    result: LiveUpdateResult,
    output_dir: Path | None = None,
) -> None:
    out = output_dir or result.output_dir
    if result.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    result.prices_df.to_csv(out / "latest_prices.csv", index=False)
    result.metrics_df.to_csv(out / "latest_risk_metrics.csv", index=False)
    result.weights_df.to_csv(out / "latest_target_weights.csv", index=False)
    with open(out / "latest_signal.json", "w", encoding="utf-8") as f:
        json.dump(result.signal, f, indent=2)


def run_daily_update(
    as_of: date | str | None = None,
    *,
    prices: pd.DataFrame | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
    metrics_lookback_days: int = TRADING_DAYS,
) -> LiveUpdateResult:
    """
    Build live monitor files under output/live/.

    Parameters
    ----------
    prices : optional injected panel for tests (must include policy tickers and SPY).
    """
    panel_full = load_prices_panel(data_dir=data_dir, prices=prices)
    data_as_of = resolve_data_as_of(as_of, panel=panel_full)
    calendar_idx = panel_full.index
    panel = panel_full.loc[panel_full.index <= pd.Timestamp(data_as_of)]

    price_ts = last_trading_day_on_or_before(panel.index, data_as_of)
    price_date_used = price_ts.date()

    quality_warnings: list[str] = []
    signal_eff_ts = next_trading_day_after(calendar_idx, price_ts)
    if signal_eff_ts is None:
        quality_warnings.append("no_next_trading_day_in_cache")
        signal_effective_date = None
    else:
        signal_effective_date = signal_eff_ts.date()

    next_reb_ts = next_scheduled_rebalance_date(calendar_idx, price_ts)
    if next_reb_ts is None:
        quality_warnings.append("no_next_rebalance_date_in_cache")
        next_rebalance_date = None
    else:
        next_rebalance_date = next_reb_ts.date()

    rebalance_due = is_rebalance_due(calendar_idx, price_ts)

    tickers_for_prices = sorted(set(LIVE_POLICY_TICKERS) | {BENCHMARK})
    missing_tickers = [t for t in LIVE_POLICY_TICKERS if t not in panel.columns]

    daily = prices_to_returns(panel)
    for t in LIVE_POLICY_TICKERS:
        if t not in daily.columns:
            daily[t] = 0.0
    weights = policy_weights()
    port = portfolio_returns(daily, weights)
    bench = daily[BENCHMARK] if BENCHMARK in daily.columns else port * np.nan
    aligned = pd.concat([port, bench], axis=1, join="inner").dropna()
    lookback = min(metrics_lookback_days, len(aligned))
    if lookback < 2:
        raise ValueError("Insufficient return history for risk metrics")
    port_slice = aligned.iloc[-lookback:, 0]
    bench_slice = aligned.iloc[-lookback:, 1]
    metrics = compute_metrics(port_slice, bench_slice)

    prices_df = build_latest_prices(panel, tickers_for_prices, data_as_of, price_ts)
    metrics_df = pd.DataFrame([metric_result_to_row(
        metrics,
        strategy_id=LIVE_STRATEGY_ID,
        data_as_of=data_as_of,
        price_date_used=price_date_used,
        lookback_days=lookback,
    )])

    weights_df = pd.DataFrame(
        {
            "ticker": weights.index,
            "target_weight": weights.values,
            "strategy_id": LIVE_STRATEGY_ID,
            "next_rebalance_date": (
                next_rebalance_date.isoformat() if next_rebalance_date is not None else ""
            ),
            "data_as_of": data_as_of.isoformat(),
            "rebalance_due": rebalance_due,
        }
    )

    stale_days = int((pd.Timestamp(data_as_of) - price_ts).days)
    data_quality = {
        "missing_tickers": missing_tickers,
        "warnings": quality_warnings,
        "stale_calendar_days": stale_days,
        "vendor": "local_cache" if prices is None else "injected_panel",
        "timezone": LIVE_TIMEZONE,
        "defaulted_to_cache_end": as_of is None,
    }

    signal = build_signal_payload(
        data_as_of=data_as_of,
        price_date_used=price_date_used,
        signal_effective_date=signal_effective_date,
        next_rebalance_date=next_rebalance_date,
        rebalance_due=rebalance_due,
        weights=weights,
        metrics=metrics,
        data_quality=data_quality,
    )

    out_path = output_dir or OUTPUT_LIVE_DIR
    result = LiveUpdateResult(
        data_as_of=data_as_of,
        price_date_used=price_date_used,
        signal_effective_date=signal_effective_date,
        next_rebalance_date=next_rebalance_date,
        rebalance_due=rebalance_due,
        prices_df=prices_df,
        metrics_df=metrics_df,
        weights_df=weights_df,
        signal=signal,
        output_dir=out_path,
        dry_run=dry_run,
    )
    write_live_artifacts(result, out_path)
    return result
