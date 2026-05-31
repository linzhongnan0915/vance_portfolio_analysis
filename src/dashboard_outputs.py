"""Build dashboard CSV artifacts (run via pipeline, not Streamlit)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import (
    BENCHMARK,
    DASHBOARD_OUTPUT_DIR,
    DATA_DIR,
    IN_SAMPLE_END,
    IN_SAMPLE_START,
    OUT_SAMPLE_START,
    TARGET_WEIGHTS,
    TICKERS,
)
from src.data_loader import load_adjusted_prices
from src.mandate_constraints import DEFENSIVE_ETF, MandateConstraints, make_mandate_weight_functions
from src.metrics import compute_metrics, metrics_to_series, subperiod_metrics_to_series
from src.portfolio import monthly_rebalanced_returns
from src.preprocessing import prices_to_returns
from src.signals import make_weight_functions
from src.backtest import run_walk_forward


def _buy_and_hold_returns(asset_returns: pd.DataFrame, target_weights: pd.Series) -> pd.Series:
    w = target_weights.values.astype(float)
    w = w / w.sum()
    tickers = list(target_weights.index)
    return (asset_returns[tickers] * w).sum(axis=1).rename("Buy-and-hold")


def build_key_return_series() -> pd.DataFrame:
    """Walk-forward return series for dashboard comparison charts."""
    prices = load_adjusted_prices(DATA_DIR)
    if DEFENSIVE_ETF not in prices.columns:
        shy = yf.download(
            DEFENSIVE_ETF,
            start=prices.index[0].strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        shy_s = shy["Close"].iloc[:, 0] if isinstance(shy.columns, pd.MultiIndex) else shy["Close"]
        prices = prices.join(shy_s.rename(DEFENSIVE_ETF), how="inner").dropna(how="any")

    daily = prices_to_returns(prices)
    bench = daily[BENCHMARK]
    tickers_ex = TICKERS + [DEFENSIVE_ETF]
    tw_shy = pd.Series(
        {"QQQ": 0.25, "SPY": 0.25, "DIA": 0.25, "GLD": 0.15, "TLT": 0.05, DEFENSIVE_ETF: 0.05}
    )

    series = {}
    wf_fix = run_walk_forward(
        daily[TICKERS], bench, tickers=TICKERS, target_weights=TARGET_WEIGHTS, strategies=["fixed_baseline"]
    )
    series["Original fixed"] = wf_fix.daily_returns["fixed_baseline"]

    mandate = MandateConstraints(shy_cap=0.05, min_equity_total=0.60)
    wfn = make_mandate_weight_functions(tickers_ex, tw_shy, mandate)
    wf_shy = run_walk_forward(
        daily[tickers_ex],
        bench,
        tickers=tickers_ex,
        target_weights=tw_shy,
        strategies=["fixed_baseline"],
        custom_weight_fn={"fixed_baseline": wfn["fixed_baseline"]},
    )
    series["Fixed + SHY (5%)"] = wf_shy.daily_returns["fixed_baseline"]

    wfn_iv = make_weight_functions(tickers_ex, tw_shy)
    wf_iv = run_walk_forward(
        daily[tickers_ex],
        bench,
        tickers=tickers_ex,
        target_weights=tw_shy,
        strategies=["inverse_volatility"],
        custom_weight_fn={"inverse_volatility": wfn_iv["inverse_volatility"]},
    )
    series["Inverse-vol (uncapped)"] = wf_iv.daily_returns["inverse_volatility"]

    mandate10 = MandateConstraints(shy_cap=0.10, min_equity_total=0.60)
    wfn_iv10 = make_mandate_weight_functions(tickers_ex, tw_shy, mandate10)
    wf_iv10 = run_walk_forward(
        daily[tickers_ex],
        bench,
        tickers=tickers_ex,
        target_weights=tw_shy,
        strategies=["inverse_volatility"],
        custom_weight_fn={"inverse_volatility": wfn_iv10["inverse_volatility"]},
    )
    series["Inverse-vol (SHY cap 10%)"] = wf_iv10.daily_returns["inverse_volatility"]

    wf_mv = run_walk_forward(
        daily[TICKERS], bench, tickers=TICKERS, target_weights=TARGET_WEIGHTS, strategies=["minimum_variance"]
    )
    series["Minimum-variance"] = wf_mv.daily_returns["minimum_variance"]
    series["SPY"] = bench
    return pd.DataFrame(series)


def build_baseline_stage1_returns() -> pd.DataFrame:
    prices = load_adjusted_prices(DATA_DIR)
    daily = prices_to_returns(prices)
    port_tickers = [t for t in TARGET_WEIGHTS.index if t in daily.columns]
    rebal = monthly_rebalanced_returns(daily[port_tickers], TARGET_WEIGHTS).rename("Monthly rebalance")
    bh = _buy_and_hold_returns(daily[port_tickers], TARGET_WEIGHTS)
    spy = daily[BENCHMARK].rename("SPY")
    return pd.DataFrame({"Monthly rebalance": rebal, "Buy-and-hold": bh, "SPY": spy}).dropna(how="any")


def _split_is_oos(rets: pd.Series) -> tuple[pd.Series, pd.Series]:
    return rets.loc[IN_SAMPLE_START:IN_SAMPLE_END], rets.loc[OUT_SAMPLE_START:]


def build_baseline_is_oos_metrics() -> pd.DataFrame:
    rets = build_baseline_stage1_returns()
    rows = []
    for col in rets.columns:
        series = rets[col]
        bench = rets["SPY"]
        for period, sl in [
            ("In-Sample (2009–2019)", _split_is_oos(series)[0]),
            ("Out-of-Sample (2020+)", _split_is_oos(series)[1]),
        ]:
            bsl = _split_is_oos(bench)[0] if "In-Sample" in period else _split_is_oos(bench)[1]
            m = compute_metrics(sl, bsl)
            ms = subperiod_metrics_to_series(m)
            rows.append({"Series": col, "Period": period, **ms.to_dict()})
    return pd.DataFrame(rows)


def build_rebalance_vs_buyhold_summary() -> pd.DataFrame:
    rets = build_baseline_stage1_returns()
    rows = []
    for period_name, slicer in [
        ("Full history", lambda s: s),
        ("In-Sample", lambda s: _split_is_oos(s)[0]),
        ("Out-of-Sample", lambda s: _split_is_oos(s)[1]),
    ]:
        rb = slicer(rets["Monthly rebalance"])
        bh = slicer(rets["Buy-and-hold"])
        if len(rb) < 2:
            continue
        cum_rb = (1 + rb).prod() - 1
        cum_bh = (1 + bh).prod() - 1
        rows.append(
            {
                "Period": period_name,
                "Monthly Rebalance": cum_rb,
                "Buy-and-Hold": cum_bh,
                "Difference (Rebal − BH)": cum_rb - cum_bh,
            }
        )
    return pd.DataFrame(rows)


def build_metrics_from_key_returns(rets: pd.DataFrame) -> pd.DataFrame:
    bench = rets["SPY"]
    rows = []
    for col in rets.columns:
        aligned = rets[[col, "SPY"]].dropna()
        if len(aligned) < 2:
            continue
        m = compute_metrics(aligned[col], aligned["SPY"])
        ms = metrics_to_series(m)
        rows.append({"Strategy": col, **ms.to_dict()})
    return pd.DataFrame(rows)


def build_all(output_dir: Path | None = None) -> None:
    out = output_dir or DASHBOARD_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    key = build_key_return_series()
    key.to_csv(out / "key_return_series.csv")

    baseline = build_baseline_stage1_returns()
    baseline.to_csv(out / "baseline_stage1_returns.csv")

    build_baseline_is_oos_metrics().to_csv(out / "baseline_is_oos_metrics.csv", index=False)
    build_rebalance_vs_buyhold_summary().to_csv(out / "rebalance_vs_buyhold_summary.csv", index=False)
    build_metrics_from_key_returns(key).to_csv(out / "strategy_metrics_from_key_returns.csv", index=False)
