"""Performance and risk metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import RISK_FREE_RATE, TRADING_DAYS


def compute_drawdown(cum_wealth: pd.Series) -> pd.Series:
    return cum_wealth / cum_wealth.cummax() - 1.0


@dataclass
class MetricResult:
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_95: float
    cvar_95: float
    beta: float
    tracking_error: float
    information_ratio: float


def compute_metrics(
    port: pd.Series,
    bench: pd.Series,
    rf: float = RISK_FREE_RATE,
    periods: int = TRADING_DAYS,
) -> MetricResult:
    aligned = pd.concat([port, bench], axis=1, join="inner").dropna()
    rp = aligned.iloc[:, 0]
    rb = aligned.iloc[:, 1]
    active = rp - rb

    n = len(rp)
    if n == 0:
        return MetricResult(*([np.nan] * 12))

    cum_ret = (1 + rp).prod() - 1
    ann_ret = (1 + cum_ret) ** (periods / n) - 1
    ann_vol = rp.std(ddof=1) * np.sqrt(periods)

    wealth = (1 + rp).cumprod()
    max_dd = compute_drawdown(wealth).min()

    downside = rp[rp < 0]
    downside_std = downside.std(ddof=1) * np.sqrt(periods) if len(downside) > 1 else np.nan

    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    sortino = (ann_ret - rf) / downside_std if downside_std and downside_std > 0 else np.nan
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

    var_95 = -np.quantile(rp, 0.05)
    cvar_95 = -rp[rp <= -var_95].mean() if (rp <= -var_95).any() else np.nan

    cov = np.cov(rp, rb, ddof=1)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan
    te = active.std(ddof=1) * np.sqrt(periods)
    ir = (active.mean() * periods) / te if te > 0 else np.nan

    return MetricResult(
        cumulative_return=cum_ret,
        annualized_return=ann_ret,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        var_95=var_95,
        cvar_95=cvar_95,
        beta=beta,
        tracking_error=te,
        information_ratio=ir,
    )


def subperiod_metrics_to_series(m: MetricResult) -> pd.Series:
    return pd.Series(
        {
            "Annualized Return": m.annualized_return,
            "Annualized Volatility": m.annualized_volatility,
            "Sharpe Ratio": m.sharpe_ratio,
            "Maximum Drawdown": m.max_drawdown,
            "Calmar Ratio": m.calmar_ratio,
            "VaR 95% (daily)": m.var_95,
            "CVaR 95% (daily)": m.cvar_95,
            "Beta vs SPY": m.beta,
            "Tracking Error vs SPY": m.tracking_error,
            "Information Ratio vs SPY": m.information_ratio,
        }
    )


def metrics_to_series(m: MetricResult) -> pd.Series:
    return pd.Series(
        {
            "Cumulative Return": m.cumulative_return,
            "Annualized Return": m.annualized_return,
            "Annualized Volatility": m.annualized_volatility,
            "Sharpe Ratio": m.sharpe_ratio,
            "Sortino Ratio": m.sortino_ratio,
            "Maximum Drawdown": m.max_drawdown,
            "Calmar Ratio": m.calmar_ratio,
            "VaR 95% (daily)": m.var_95,
            "CVaR 95% (daily)": m.cvar_95,
            "Beta vs SPY": m.beta,
            "Tracking Error vs SPY": m.tracking_error,
            "Information Ratio vs SPY": m.information_ratio,
        }
    )
