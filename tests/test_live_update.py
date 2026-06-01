"""Live daily update (v3 MVP) - synthetic data only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import BENCHMARK, DEFAULT_PRICE_START, LIVE_POLICY_TICKERS, LIVE_STRATEGY_ID
from src import live_update as lu
from src.live_update import (
    is_rebalance_due,
    merge_price_panels,
    next_scheduled_rebalance_date,
    run_daily_update,
)
from tests.fixtures import make_synthetic_prices

REQUIRED_SIGNAL_KEYS = {
    "schema_version",
    "data_as_of",
    "price_date_used",
    "signal_generated_at_utc",
    "signal_effective_date",
    "next_rebalance_date",
    "rebalance_due",
    "strategy_id",
    "daily_signal_mode",
    "disclaimer",
    "weights",
    "risk_metrics",
    "data_quality",
}


def _live_prices(end: str = "2022-12-31") -> pd.DataFrame:
    tickers = list(LIVE_POLICY_TICKERS) + ([BENCHMARK] if BENCHMARK not in LIVE_POLICY_TICKERS else [])
    tickers = list(dict.fromkeys(tickers))
    if "SHY" not in tickers:
        tickers.append("SHY")
    return make_synthetic_prices(start="2020-01-01", end=end, tickers=tickers)


def test_run_daily_update_writes_four_files(tmp_path: Path) -> None:
    prices = _live_prices()
    out = tmp_path / "live"
    result = run_daily_update("2022-06-15", prices=prices, output_dir=out, dry_run=False)

    for name in (
        "latest_prices.csv",
        "latest_risk_metrics.csv",
        "latest_target_weights.csv",
        "latest_signal.json",
    ):
        assert (out / name).exists(), name

    assert result.signal["strategy_id"] == LIVE_STRATEGY_ID
    assert result.signal["daily_signal_mode"] is False


def test_signal_json_schema_keys(tmp_path: Path) -> None:
    run_daily_update("2022-06-15", prices=_live_prices(), output_dir=tmp_path / "live")
    payload = json.loads((tmp_path / "live" / "latest_signal.json").read_text(encoding="utf-8"))
    assert REQUIRED_SIGNAL_KEYS <= set(payload.keys())


def test_policy_weights_sum_to_one(tmp_path: Path) -> None:
    run_daily_update("2022-06-15", prices=_live_prices(), output_dir=tmp_path / "live")
    payload = json.loads((tmp_path / "live" / "latest_signal.json").read_text(encoding="utf-8"))
    w = payload["weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["QQQ"] == pytest.approx(0.25)
    assert w["SHY"] == pytest.approx(0.05)


def test_no_future_prices_in_panel(tmp_path: Path) -> None:
    prices = _live_prices(end="2022-12-31")
    as_of = "2022-06-15"
    run_daily_update(as_of, prices=prices, output_dir=tmp_path / "live")
    written = pd.read_csv(tmp_path / "live" / "latest_prices.csv")
    assert (written["data_as_of"] == as_of).all()
    assert pd.Timestamp(written["price_date_used"].iloc[0]) <= pd.Timestamp(as_of)


def test_rebalance_due_first_trading_day_of_month(tmp_path: Path) -> None:
    prices = _live_prices(end="2022-12-31")
    idx = prices.index

    # 2022-06-01 is Wednesday - first business day of June 2022
    due_result = run_daily_update("2022-06-01", prices=prices, output_dir=tmp_path / "due")
    assert is_rebalance_due(idx, pd.Timestamp("2022-06-01"))
    assert due_result.rebalance_due is True
    assert due_result.signal["rebalance_due"] is True

    not_due = run_daily_update("2022-06-15", prices=prices, output_dir=tmp_path / "not_due")
    assert not_due.rebalance_due is False
    assert not_due.signal["rebalance_due"] is False


def test_next_rebalance_is_next_month_first_trading_day(tmp_path: Path) -> None:
    prices = _live_prices(end="2022-12-31")
    idx = prices.index
    mid = pd.Timestamp("2022-06-15")
    expected = next_scheduled_rebalance_date(idx, mid)
    result = run_daily_update("2022-06-15", prices=prices, output_dir=tmp_path / "live")
    assert result.next_rebalance_date == expected.date()


def test_dry_run_does_not_write_files(tmp_path: Path) -> None:
    out = tmp_path / "dry"
    run_daily_update("2022-06-15", prices=_live_prices(), output_dir=out, dry_run=True)
    assert not (out / "latest_signal.json").exists()


def test_target_weights_for_next_monthly_rebalance(tmp_path: Path) -> None:
    result = run_daily_update("2022-06-15", prices=_live_prices(), output_dir=tmp_path / "live")
    wdf = result.weights_df
    assert (wdf["next_rebalance_date"] == result.next_rebalance_date.isoformat()).all()
    assert result.next_rebalance_date.month == 7


def test_default_as_of_uses_cache_end_no_crash(tmp_path: Path) -> None:
    """Panel ends 2022-06-15: default as-of must not require calendar days after cache end."""
    prices = _live_prices(end="2022-06-15")
    result = run_daily_update(as_of=None, prices=prices, output_dir=tmp_path / "live", dry_run=True)
    assert result.data_as_of == pd.Timestamp("2022-06-15").date()
    assert result.signal["signal_effective_date"] is None
    assert "no_next_trading_day_in_cache" in result.signal["data_quality"]["warnings"]
    assert result.signal["data_quality"]["defaulted_to_cache_end"] is True


def test_signal_effective_date_after_data_as_of_when_available(tmp_path: Path) -> None:
    result = run_daily_update("2022-06-15", prices=_live_prices(end="2022-12-31"), output_dir=tmp_path / "live")
    assert result.signal_effective_date == pd.Timestamp("2022-06-16").date()
    assert result.signal["signal_effective_date"] == "2022-06-16"
    assert "no_next_trading_day_in_cache" not in result.signal["data_quality"]["warnings"]


def test_missing_shy_recorded_in_data_quality(tmp_path: Path) -> None:
    tickers = [t for t in LIVE_POLICY_TICKERS if t != "SHY"]
    if BENCHMARK not in tickers:
        tickers.append(BENCHMARK)
    prices = make_synthetic_prices(start="2020-01-01", end="2022-12-31", tickers=tickers)
    result = run_daily_update("2022-06-15", prices=prices, output_dir=tmp_path / "live", dry_run=True)
    assert "SHY" in result.signal["data_quality"]["missing_tickers"]


def test_refresh_prices_uses_mocked_fetch(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    fresh = _live_prices(end="2023-06-30")

    monkeypatch.setattr(lu, "fetch_adjusted_closes", lambda tickers, start, **kw: fresh)
    result = run_daily_update(
        as_of=None,
        data_dir=data_dir,
        output_dir=tmp_path / "live",
        refresh_prices=True,
    )
    dq = result.signal["data_quality"]
    assert dq["refreshed_prices"] is True
    assert dq["vendor"] == "yfinance"
    assert result.data_as_of == pd.Timestamp("2023-06-30").date()
    cache = data_dir / "processed" / "vance_etf_prices.csv"
    assert cache.exists()


def test_refresh_data_as_of_excludes_incomplete_same_day_bar(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    fresh = _live_prices(end="2024-03-15")
    ny = lu.NY_TZ
    now = __import__("datetime").datetime(2024, 3, 15, 10, 0, tzinfo=ny)

    monkeypatch.setattr(lu, "fetch_adjusted_closes", lambda tickers, start, **kw: fresh)
    result = run_daily_update(
        as_of=None,
        data_dir=data_dir,
        output_dir=tmp_path / "live",
        refresh_prices=True,
        now=now,
    )
    assert result.data_as_of == pd.Timestamp("2024-03-14").date()


def test_refresh_missing_ticker_in_fetch(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    tickers = [t for t in LIVE_POLICY_TICKERS if t != "SHY"]
    if BENCHMARK not in tickers:
        tickers.append(BENCHMARK)
    fresh = make_synthetic_prices(start="2020-01-01", end="2022-12-31", tickers=tickers)

    monkeypatch.setattr(lu, "fetch_adjusted_closes", lambda tickers, start, **kw: fresh)
    result = run_daily_update(
        "2022-06-15",
        data_dir=data_dir,
        output_dir=tmp_path / "live",
        refresh_prices=True,
    )
    assert "SHY" in result.signal["data_quality"]["missing_tickers"]
    assert any("missing_tickers" in w for w in result.signal["data_quality"]["warnings"])


def test_refresh_fetch_failure_falls_back_to_cache(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    cache = _live_prices(end="2022-06-15")
    path = data_dir / "processed" / "vance_etf_prices.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_csv(path)

    def _fail(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(lu, "fetch_adjusted_closes", _fail)
    result = run_daily_update(
        as_of=None,
        data_dir=data_dir,
        output_dir=tmp_path / "live",
        refresh_prices=True,
    )
    dq = result.signal["data_quality"]
    assert dq["refreshed_prices"] is False
    assert dq["vendor"] == "local_cache"
    assert "yfinance_fetch_failed_using_cache" in dq["warnings"]


def test_refresh_prices_with_dry_run_raises() -> None:
    with pytest.raises(ValueError, match="refresh-prices cannot be combined"):
        run_daily_update(refresh_prices=True, dry_run=True)


def test_no_refresh_path_unchanged(tmp_path: Path) -> None:
    result = run_daily_update("2022-06-15", prices=_live_prices(), output_dir=tmp_path / "live", dry_run=True)
    assert result.signal["data_quality"]["refreshed_prices"] is False
    assert result.signal["data_quality"]["vendor"] == "injected_panel"


def test_merge_retains_new_dates() -> None:
    existing = pd.DataFrame({"SPY": [100.0]}, index=pd.DatetimeIndex(["2026-05-29"]))
    new = pd.DataFrame({"SPY": [101.0]}, index=pd.DatetimeIndex(["2026-06-01"]))
    merged = merge_price_panels(existing, new)
    assert pd.Timestamp("2026-05-29") in merged.index
    assert pd.Timestamp("2026-06-01") in merged.index


def test_merge_new_non_null_overwrites_duplicate_date() -> None:
    existing = pd.DataFrame({"SPY": [50.0]}, index=pd.DatetimeIndex(["2026-05-29"]))
    new = pd.DataFrame({"SPY": [99.0]}, index=pd.DatetimeIndex(["2026-05-29"]))
    merged = merge_price_panels(existing, new)
    assert merged.loc["2026-05-29", "SPY"] == pytest.approx(99.0)


def test_merge_new_nan_preserves_existing_on_duplicate_date() -> None:
    existing = pd.DataFrame({"SPY": [50.0]}, index=pd.DatetimeIndex(["2026-05-29"]))
    new = pd.DataFrame({"SPY": [np.nan]}, index=pd.DatetimeIndex(["2026-05-29"]))
    merged = merge_price_panels(existing, new)
    assert merged.loc["2026-05-29", "SPY"] == pytest.approx(50.0)


def test_merge_single_ticker_backfill_keeps_other_columns() -> None:
    existing = pd.DataFrame(
        {"QQQ": [300.0], "SPY": [400.0]},
        index=pd.DatetimeIndex(["2026-05-29"]),
    )
    new = pd.DataFrame({"SHY": [80.0]}, index=pd.DatetimeIndex(["2026-05-29"]))
    merged = merge_price_panels(existing, new)
    assert set(merged.columns) == {"QQQ", "SPY", "SHY"}
    assert merged.loc["2026-05-29", "SPY"] == pytest.approx(400.0)
    assert merged.loc["2026-05-29", "SHY"] == pytest.approx(80.0)


def test_refresh_backfills_shy_from_default_start(tmp_path: Path, monkeypatch) -> None:
    """Incremental fetch leaves SHY sparse; backfill must request DEFAULT_PRICE_START for SHY."""
    data_dir = tmp_path / "data"
    base = _live_prices(end="2023-06-30").drop(columns=["SHY"])
    path = data_dir / "processed" / "vance_etf_prices.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(path)

    fetch_calls: list[tuple[frozenset[str], str]] = []

    def mock_fetch(tickers, start, **kw):
        tset = frozenset(tickers)
        fetch_calls.append((tset, start))
        tlist = list(tickers)
        if start == DEFAULT_PRICE_START:
            return make_synthetic_prices(start=DEFAULT_PRICE_START, end="2023-06-30", tickers=tlist)
        return make_synthetic_prices(start="2023-06-15", end="2023-06-30", tickers=tlist)

    monkeypatch.setattr(lu, "fetch_adjusted_closes", mock_fetch)
    result = run_daily_update(
        as_of="2023-06-30",
        data_dir=data_dir,
        output_dir=tmp_path / "live",
        refresh_prices=True,
    )
    assert any(start == DEFAULT_PRICE_START and "SHY" in ts for ts, start in fetch_calls)
    lookback = int(result.metrics_df["lookback_days"].iloc[0])
    assert lookback >= lu.SHORT_METRICS_LOOKBACK_WARN
    assert "SHY" not in result.signal["data_quality"]["missing_tickers"]


def test_metrics_lookback_not_collapsed_by_sparse_shy_column(tmp_path: Path) -> None:
    """Sparse SHY NaNs must not shrink portfolio metrics to a handful of days."""
    prices = _live_prices(end="2023-06-30")
    prices["SHY"] = np.nan
    prices.loc[prices.index[-7:], "SHY"] = _live_prices(end="2023-06-30").loc[prices.index[-7:], "SHY"]

    result = run_daily_update(
        "2023-06-30",
        prices=prices,
        output_dir=tmp_path / "live",
        dry_run=True,
    )
    lookback = int(result.metrics_df["lookback_days"].iloc[0])
    assert lookback >= lu.SHORT_METRICS_LOOKBACK_WARN
    assert any("insufficient_history:SHY" in w for w in result.signal["data_quality"]["warnings"])


def test_short_history_emits_short_metrics_lookback_warning(tmp_path: Path) -> None:
    prices = make_synthetic_prices(start="2023-05-01", end="2023-06-30", tickers=list(LIVE_POLICY_TICKERS))
    result = run_daily_update(
        "2023-06-30",
        prices=prices,
        output_dir=tmp_path / "live",
        dry_run=True,
    )
    warnings = result.signal["data_quality"]["warnings"]
    assert any(w.startswith("short_metrics_lookback:") for w in warnings)
    assert int(result.metrics_df["lookback_days"].iloc[0]) < lu.SHORT_METRICS_LOOKBACK_WARN


def test_no_look_ahead_after_refresh_style_panel(tmp_path: Path) -> None:
    prices = _live_prices(end="2023-06-30")
    result = run_daily_update("2023-06-15", prices=prices, output_dir=tmp_path / "live", dry_run=True)
    written = result.prices_df
    assert (written["data_as_of"] == "2023-06-15").all()
    assert pd.Timestamp(written["price_date_used"].iloc[0]) <= pd.Timestamp("2023-06-15")
