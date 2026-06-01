"""Project paths and portfolio/backtest constants."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "output"
OUTPUT_LIVE_DIR = OUTPUT_DIR / "live"
DASHBOARD_OUTPUT_DIR = OUTPUT_DIR / "dashboard"
REPORTS_DIR = ROOT / "reports"

# Live daily monitor (v3 MVP)
LIVE_STRATEGY_ID = "fixed_shy_version_a"
LIVE_POLICY_TICKERS = ["QQQ", "SPY", "DIA", "GLD", "TLT", "SHY"]
LIVE_POLICY_WEIGHTS = pd.Series(
    {"QQQ": 0.25, "SPY": 0.25, "DIA": 0.25, "GLD": 0.15, "TLT": 0.05, "SHY": 0.05},
    name="target_weight",
)
LIVE_DAILY_SIGNAL_MODE = False
LIVE_TIMEZONE = "America/New_York"
LIVE_DISCLAIMER = (
    "Educational monitoring output. Not investment advice. Not real-time. "
    "No order execution."
)

TICKERS = ["QQQ", "SPY", "DIA", "GLD", "TLT"]
BENCHMARK = "SPY"
# Universe for live price refresh (policy sleeve + benchmark for metrics)
LIVE_PRICE_TICKERS = sorted(set(LIVE_POLICY_TICKERS + [BENCHMARK]))
TARGET_WEIGHTS = pd.Series(
    {"QQQ": 0.25, "SPY": 0.25, "DIA": 0.25, "GLD": 0.15, "TLT": 0.10},
    name="target_weight",
)

IN_SAMPLE_START = "2009-01-01"
IN_SAMPLE_END = "2019-12-31"
OUT_SAMPLE_START = "2020-01-01"

TRADING_DAYS = 252
RISK_FREE_RATE = 0.0
DEFAULT_PRICE_START = "2008-01-01"
PRICE_CACHE_FILENAME = "vance_etf_prices.csv"

ANALYSIS_START = "2009-01-01"
TRAIN_MONTHS = 12
MAX_WEIGHT = 0.40
MIN_WEIGHT = 0.0
STRATEGIES = ["fixed_baseline", "inverse_volatility", "minimum_variance"]
