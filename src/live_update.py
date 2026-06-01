"""Daily live monitor artifacts (v3 MVP): prices, risk metrics, rebalance targets.

No broker connectivity, no order execution, no intraday or real-time claims.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from src.backtest import first_trading_day_on_or_after
from src.config import (
    BENCHMARK,
    DATA_DIR,
    DEFAULT_PRICE_START,
    PRICE_CACHE_FILENAME,
    LIVE_DAILY_SIGNAL_MODE,
    LIVE_DISCLAIMER,
    LIVE_POLICY_TICKERS,
    LIVE_POLICY_WEIGHTS,
    LIVE_PRICE_TICKERS,
    LIVE_STRATEGY_ID,
    LIVE_TIMEZONE,
    OUTPUT_LIVE_DIR,
    TRADING_DAYS,
)
from src.data_loader import _price_cache_path
from src.metrics import MetricResult, compute_metrics, metrics_to_series
NY_TZ = ZoneInfo(LIVE_TIMEZONE)
US_EQUITY_CLOSE_HOUR = 16  # America/New_York; daily bars treated complete after cash close

LIVE_PRICE_COLUMNS = ["ticker", "adj_close", "price_date_used", "data_as_of"]


class PriceRefreshError(RuntimeError):
    """Raised when yfinance refresh fails and no usable local cache exists."""


LIVE_INCREMENTAL_FETCH_DAYS = 10
MIN_PRICE_OBS_FOR_METRICS = TRADING_DAYS + 1  # closes needed for ~252 return observations
SHORT_METRICS_LOOKBACK_WARN = 60

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


def _normalize_price_panel(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Parse yfinance download output to a sorted DatetimeIndex x tickers panel."""
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        col = tickers[0] if len(tickers) == 1 else "Close"
        prices = raw[[col]].copy()
        if len(tickers) == 1:
            prices.columns = tickers
    prices = prices.sort_index()
    prices.index = pd.to_datetime(prices.index)
    return prices


def fetch_adjusted_closes(
    tickers: list[str],
    start: str,
    *,
    download_fn: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Fetch adjusted daily closes via yfinance (completed daily bars only).

    Parameters
    ----------
    download_fn : optional injectable for tests (signature like yf.download).
    """
    download = download_fn or yf.download
    raw = download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    panel = _normalize_price_panel(raw, tickers)
    if panel.empty:
        raise PriceRefreshError(f"yfinance returned no rows for {tickers} from {start}")
    return panel


def valid_price_count(series: pd.Series) -> int:
    return int(series.notna().sum())


def tickers_needing_full_backfill(
    panel: pd.DataFrame | None,
    tickers: list[str],
    min_obs: int = MIN_PRICE_OBS_FOR_METRICS,
) -> list[str]:
    """Tickers missing from panel or with fewer than min_obs non-null closes."""
    if panel is None or panel.empty:
        return list(tickers)
    need: list[str] = []
    for t in tickers:
        if t not in panel.columns:
            need.append(t)
        elif valid_price_count(panel[t]) < min_obs:
            need.append(t)
    return need


def policy_daily_returns(
    panel: pd.DataFrame,
    price_ts: pd.Timestamp,
    *,
    missing_tickers: list[str],
) -> pd.DataFrame:
    """
    Daily simple returns for live metrics.

    Only requires non-null returns on active policy columns plus benchmark,
    so one ticker with sparse NaN history does not collapse the full panel.
    """
    sub = panel.loc[panel.index <= price_ts]
    active = [t for t in LIVE_POLICY_TICKERS if t not in missing_tickers and t in sub.columns]
    cols = list(dict.fromkeys(active + ([BENCHMARK] if BENCHMARK in sub.columns else [])))
    if not cols:
        return pd.DataFrame()
    rets = sub[cols].pct_change(fill_method=None)
    daily = rets.dropna(how="any").copy()
    for t in LIVE_POLICY_TICKERS:
        if t not in daily.columns:
            daily.loc[:, t] = 0.0
    return daily


def insufficient_history_tickers(
    panel: pd.DataFrame,
    price_ts: pd.Timestamp,
    *,
    min_price_obs: int = MIN_PRICE_OBS_FOR_METRICS,
) -> list[str]:
    """Policy tickers present but with too few non-null prices through price_ts."""
    sub = panel.loc[panel.index <= price_ts]
    return [
        t
        for t in LIVE_POLICY_TICKERS
        if t in sub.columns and valid_price_count(sub[t]) < min_price_obs
    ]


def merge_price_panels(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """
    Merge price panels on the union of dates and columns.

    New non-null values overwrite existing values on duplicate dates; NaN in new
    preserves existing. Single-ticker backfills do not remove unrelated columns.
    """
    if existing is None or existing.empty:
        return new.sort_index()
    if new is None or new.empty:
        return existing.sort_index()

    new = new.sort_index()
    all_index = existing.index.union(new.index).sort_values()
    all_columns = existing.columns.union(new.columns)

    merged = existing.reindex(all_index).reindex(columns=all_columns)
    new_aligned = new.reindex(all_index).reindex(columns=all_columns)

    for col in all_columns:
        if col not in new.columns:
            continue
        new_s = new_aligned[col]
        if col in existing.columns:
            merged[col] = new_s.combine_first(merged[col])
        else:
            merged[col] = new_s

    return merged.sort_index()


def _processed_cache_path(data_dir: Path | None = None) -> Path:
    base = data_dir or DATA_DIR
    return base / "processed" / PRICE_CACHE_FILENAME


def save_price_cache(panel: pd.DataFrame, data_dir: Path | None = None) -> Path:
    path = _processed_cache_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.sort_index().to_csv(path)
    return path


def last_completed_trading_date(
    index: pd.DatetimeIndex,
    *,
    now: datetime | None = None,
) -> date:
    """
    Latest completed US equity daily bar date in the panel.

    If the last row is today's calendar date in NY but before the cash close,
    drop that row (incomplete same-day bar).
    """
    if len(index) == 0:
        raise ValueError("Price panel has no trading dates")
    now = now or datetime.now(NY_TZ)
    last_ts = index.max()
    if last_ts.tzinfo is None:
        last_local = last_ts
    else:
        last_local = last_ts.tz_convert(NY_TZ)
    today_ny = now.date()
    if last_local.date() == today_ny and now.hour < US_EQUITY_CLOSE_HOUR:
        prior = index[index < last_ts.normalize()]
        if len(prior) == 0:
            raise ValueError("No completed trading day before today's incomplete bar")
        return prior[-1].date()
    return last_ts.date()


def refresh_price_cache(
    data_dir: Path | None = None,
    tickers: list[str] | None = None,
    *,
    download_fn: Callable[..., pd.DataFrame] | None = None,
    write: bool = True,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fetch latest yfinance adjusted closes and merge into the processed price cache.

    On fetch failure, fall back to existing cache when present and record a warning.
    """
    data_dir = data_dir or DATA_DIR
    tickers = tickers or LIVE_PRICE_TICKERS
    warnings: list[str] = []
    existing: pd.DataFrame | None = None
    cache_path = _processed_cache_path(data_dir)
    if cache_path.exists():
        existing = pd.read_csv(cache_path, index_col=0, parse_dates=True).sort_index()
        existing.index = pd.to_datetime(existing.index)

    if existing is not None and len(existing.index):
        recent_start = (
            existing.index.max() - pd.Timedelta(days=LIVE_INCREMENTAL_FETCH_DAYS)
        ).strftime("%Y-%m-%d")
    else:
        recent_start = DEFAULT_PRICE_START

    try:
        fetched = fetch_adjusted_closes(
            tickers,
            recent_start if existing is not None and len(existing.index) else DEFAULT_PRICE_START,
            download_fn=download_fn,
        )
        refreshed = True
        vendor = "yfinance"
    except Exception as exc:
        if existing is None or existing.empty:
            raise PriceRefreshError(
                f"yfinance price refresh failed and no local cache at {cache_path}"
            ) from exc
        warnings.append("yfinance_fetch_failed_using_cache")
        return existing, {
            "refreshed_prices": False,
            "vendor": "local_cache",
            "warnings": warnings,
        }

    missing = [t for t in tickers if t not in fetched.columns]
    if missing:
        warnings.append(f"missing_tickers_in_fetch:{','.join(missing)}")

    merged = merge_price_panels(existing, fetched)

    need_backfill = tickers_needing_full_backfill(merged, tickers, MIN_PRICE_OBS_FOR_METRICS)
    if need_backfill:
        backfill = fetch_adjusted_closes(
            need_backfill,
            DEFAULT_PRICE_START,
            download_fn=download_fn,
        )
        merged = merge_price_panels(merged, backfill)
        missing = [t for t in tickers if t not in backfill.columns]
        still_short = tickers_needing_full_backfill(merged, tickers, MIN_PRICE_OBS_FOR_METRICS)
        for t in still_short:
            warnings.append(f"insufficient_history_after_backfill:{t}")

    if write:
        save_price_cache(merged, data_dir)

    return merged, {
        "refreshed_prices": refreshed,
        "vendor": vendor,
        "warnings": warnings,
        "missing_tickers_fetch": missing,
    }


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
    refresh_prices: bool = False,
    download_fn: Callable[..., pd.DataFrame] | None = None,
    metrics_lookback_days: int = TRADING_DAYS,
    now: datetime | None = None,
) -> LiveUpdateResult:
    """
    Build live monitor files under output/live/.

    Parameters
    ----------
    prices : optional injected panel for tests (must include policy tickers and SPY).
    refresh_prices : when True, fetch yfinance and update local cache before artifacts.
    download_fn : injectable yfinance download for tests.
    """
    if refresh_prices and dry_run:
        raise ValueError(
            "--refresh-prices cannot be combined with --dry-run: refresh updates the "
            "local price cache and is not a read-only operation."
        )
    if refresh_prices and prices is not None:
        raise ValueError("refresh_prices cannot be used with an injected prices panel")

    refresh_meta: dict[str, Any] = {
        "refreshed_prices": False,
        "vendor": "local_cache",
        "warnings": [],
        "missing_tickers_fetch": [],
    }

    if refresh_prices:
        panel_full, refresh_meta = refresh_price_cache(
            data_dir=data_dir,
            download_fn=download_fn,
            write=not dry_run,
            now=now,
        )
    else:
        panel_full = load_prices_panel(data_dir=data_dir, prices=prices)

    if as_of is None and refresh_prices:
        data_as_of = last_completed_trading_date(panel_full.index, now=now)
    else:
        data_as_of = resolve_data_as_of(as_of, panel=panel_full)
    calendar_idx = panel_full.index
    panel = panel_full.loc[panel_full.index <= pd.Timestamp(data_as_of)]

    price_ts = last_trading_day_on_or_before(panel.index, data_as_of)
    price_date_used = price_ts.date()

    quality_warnings: list[str] = list(refresh_meta.get("warnings") or [])
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
    no_price_policy = [
        t
        for t in LIVE_POLICY_TICKERS
        if t not in panel.columns or panel.loc[panel.index <= price_ts, t].notna().sum() == 0
    ]
    insuff_history = insufficient_history_tickers(panel, price_ts)
    for t in insuff_history:
        quality_warnings.append(f"insufficient_history:{t}")

    missing_tickers = sorted(
        set(refresh_meta.get("missing_tickers_fetch") or [])
        | set(no_price_policy)
        | set(insuff_history)
    )
    if missing_tickers:
        quality_warnings.append(f"missing_tickers:{','.join(missing_tickers)}")

    daily = policy_daily_returns(panel, price_ts, missing_tickers=missing_tickers)
    weights = policy_weights()
    port = portfolio_returns(daily, weights)
    bench = daily[BENCHMARK] if BENCHMARK in daily.columns else port * np.nan
    aligned = pd.concat([port, bench], axis=1, join="inner").dropna()
    lookback = min(metrics_lookback_days, len(aligned))
    if lookback < SHORT_METRICS_LOOKBACK_WARN:
        quality_warnings.append(f"short_metrics_lookback:{lookback}_of_{metrics_lookback_days}")
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

    now_dt = now or datetime.now(NY_TZ)
    stale_days = int((now_dt.date() - price_date_used).days)
    if prices is not None:
        vendor = "injected_panel"
        refreshed_flag = False
    else:
        vendor = str(refresh_meta.get("vendor", "local_cache"))
        refreshed_flag = bool(refresh_meta.get("refreshed_prices", False))

    data_quality = {
        "missing_tickers": missing_tickers,
        "warnings": quality_warnings,
        "stale_days": stale_days,
        "vendor": vendor,
        "refreshed_prices": refreshed_flag,
        "metrics_lookback_requested": metrics_lookback_days,
        "metrics_lookback_used": lookback,
        "timezone": LIVE_TIMEZONE,
        "defaulted_to_cache_end": as_of is None and not refresh_prices,
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
