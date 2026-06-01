# Live Daily Update Mode (v3) - Design Document

**Status:** Design only. No implementation in this document.  
**Repo:** `vance_portfolio_analysis`  
**Policy context:** Educational Vance-style ETF risk research; monthly rebalance baseline; recommended policy Fixed + SHY Version A.

---

## 1. System Layers: Dashboard vs Daily Refresh vs Production Trading

Three distinct layers must not be conflated in interviews or operations.

| Layer | What it is | What it is not |
|-------|------------|----------------|
| **Hosted dashboard** | Read-only Streamlit UI (`dashboard/app.py`) over precomputed files under `output/` | A trading system, OMS, or risk engine of record |
| **Daily data refresh (v3)** | Scheduled or manual job that updates prices, risk metrics, and **next rebalance target weights** as of a stated `data_as_of` date | Live trading, order generation, or broker connectivity |
| **Production live trading** | Order management, compliance, settlement, intraday risk, audit trail, entitlements data | In scope for this repo **only as explicit non-goals** |

```text
  [Vendor prices] --> daily refresh job --> output/live/*
                              |
                              v
                    Streamlit "Live / Daily Monitor"
                              |
                              X (no path) --> broker / execution
```

**v3 adds the middle row:** operational **monitoring and signal preparation** for the **next** rebalance decision, while preserving the existing monthly backtest research pipeline (`scripts/run_all_stages.py`).

---

## 2. Current Strategy Assumption (Baseline)

The research pipeline and walk-forward engine (`src/backtest.py`) assume:

| Assumption | Current setting |
|------------|-----------------|
| Rebalance frequency | **Monthly**, first trading day of each month |
| Default policy weights | QQQ 25%, SPY 25%, DIA 25%, GLD 15%, TLT 10% (SHY in extension stages) |
| Dynamic strategies | Inverse-volatility and minimum-variance use **12-month** trailing windows |
| Benchmark | SPY daily returns |
| Costs (baseline) | 0 bps (`tx_cost_bps=0` in walk-forward) |

Daily refresh **does not change** the economic policy to daily rebalancing unless a future flag explicitly enables a separate **daily signal mode** (see Section 3). By default, v3 answers: "Given prices through `data_as_of`, what are today's risk metrics and what weights should we implement on the **next scheduled monthly rebalance**?"

---

## 3. Proposed v3 Feature: Daily Refresh / Next-Day Signal Prep

### 3.1 Goals

1. **Daily price refresh** - Append or rebuild adjusted closes through `data_as_of` (last available close).
2. **Daily risk metric update** - Recompute portfolio and benchmark metrics on a rolling window ending `data_as_of`.
3. **Next rebalance target weights** - Produce mandate-aware target weights for the **upcoming** rebalance date using only data available through `data_as_of`.
4. **Optional daily signal mode** - Off by default; when enabled via config, may emit daily diagnostic signals without changing the monthly rebalance calendar.

### 3.2 Configuration (proposed)

```text
# config keys (illustrative, not implemented yet)
LIVE_UPDATE_ENABLED=true
LIVE_DAILY_SIGNAL_MODE=false          # default false
LIVE_STRATEGY=fixed_shy_version_a     # or fixed_baseline, inverse_volatility, etc.
LIVE_MANDATE_PROFILE=version_a
```

| Mode | Behavior |
|------|----------|
| **Default (monthly policy)** | Update metrics daily; refresh **target weights** only when `rebalance_due=true` or within N days of `next_rebalance_date` |
| **Daily signal mode (opt-in)** | Additional `latest_signal.json` fields for diagnostic weight suggestions; **must not** auto-trade; label clearly as non-production |

### 3.3 Relationship to full pipeline

| Job | Frequency | Output location |
|-----|-----------|-----------------|
| `scripts/run_all_stages.py` | Ad hoc / research | `output/stage*`, `output/dashboard/` |
| `scripts/update_daily.py` (v3) | Daily (cron / manual) | `output/live/` |

Full pipeline remains the source of truth for historical research. Live update is a **thin incremental layer** on top of cached prices and existing `src/` logic.

---

## 4. Data Timing and Look-Ahead Discipline

### 4.1 Definitions

| Field | Meaning |
|-------|---------|
| `data_as_of` | Calendar date of the **last close** included in the price panel (America/New York convention for US ETFs) |
| `data_timestamp_utc` | When the refresh job ran (audit only; not used for signal math) |
| `next_rebalance_date` | First trading day of the next month on or after `data_as_of`, or the scheduled monthly rebalance if already in rebalance window |
| `signal_effective_date` | **Next trading day** after `data_as_of` for human-readable "what to do tomorrow" framing |

### 4.2 Timing rules (no look-ahead)

1. **Default `data_as_of`:** When `--as-of` is omitted, use the **latest trading date in the local price cache** (not today's calendar date). This avoids failures when the cache ends before the current day.
2. **Prices:** Use only rows with index date `<= data_as_of`. Never use same-day adjusted close if vendor timestamp implies future availability (document vendor lag).
3. **Returns:** Daily simple returns through `data_as_of`; no partial-day intraday bars.
4. **`signal_effective_date`:** Next trading day after `price_date_used` when present in the cache calendar; otherwise `null` with `data_quality.warnings` including `no_next_trading_day_in_cache` (no crash).
5. **`next_rebalance_date`:** Next month's first trading day when present in the cache; otherwise `null` with `no_next_rebalance_date_in_cache`.
6. **12-month estimation window:** For inverse-vol / min-var (future), trailing window ends on last trading day `<= data_as_of`.
7. **Recommendation intent:** Outputs describe weights intended for **next rebalance execution**, not daily trading unless `LIVE_DAILY_SIGNAL_MODE=true`.
8. **Holiday handling:** Non-trading `data_as_of` rolls back via `last_trading_day_on_or_before`; record `price_date_used` in JSON.

### 4.3 Example timeline

```text
  Fri close (data_as_of = Fri) --> job runs Fri evening or Sat morning
                              --> latest_target_weights for Mon (next trading day) *monitoring*
                              --> actual rebalance only on next_rebalance_date (e.g. first trading day of month)
```

Clarify in UI: **monitoring signal != automatic trade.**

---

## 5. Output Artifacts

All paths under `output/live/` (gitignored like other `output/`; optional committed **sample** in releases later).

### 5.1 `output/live/latest_prices.csv`

| Column | Description |
|--------|-------------|
| `ticker` | QQQ, SPY, DIA, GLD, TLT, SHY (per policy universe) |
| `adj_close` | Adjusted close as of `price_date_used` |
| `price_date_used` | Trading date of the quote |
| `data_as_of` | Panel as-of date for the job |

### 5.2 `output/live/latest_risk_metrics.csv`

One row per run (or per strategy if multiple). Fields aligned with `src/metrics.py` where possible:

- `annualized_return`, `annualized_volatility`, `sharpe_ratio`, `max_drawdown`
- `var_95_daily`, `cvar_95_daily`
- `beta_vs_spy`, `tracking_error`, `information_ratio`
- `lookback_days`, `data_as_of`, `strategy_id`

Computed on portfolio returns through `data_as_of` (rolling window documented in JSON).

### 5.3 `output/live/latest_target_weights.csv`

| Column | Description |
|--------|-------------|
| `ticker` | Asset |
| `target_weight` | Proposed weight for **next_rebalance_date** |
| `strategy_id` | e.g. `fixed_shy_version_a` |
| `next_rebalance_date` | Scheduled rebalance |
| `data_as_of` | Data cutoff |
| `rebalance_due` | Boolean: true if today is on or past rebalance trigger rule |

Apply `src/mandate_constraints.py` after weight function when policy requires SHY cap / equity floor.

### 5.4 `output/live/latest_signal.json`

Machine-readable summary for dashboard and external tools:

```json
{
  "schema_version": "1.0",
  "data_as_of": "2026-05-30",
  "price_date_used": "2026-05-30",
  "signal_generated_at_utc": "2026-05-31T12:00:00Z",
  "signal_effective_date": "2026-06-01",
  "next_rebalance_date": "2026-06-02",
  "rebalance_due": false,
  "strategy_id": "fixed_shy_version_a",
  "daily_signal_mode": false,
  "disclaimer": "Educational monitoring output. Not investment advice. Not real-time.",
  "weights": {"QQQ": 0.25, "SPY": 0.25, "DIA": 0.25, "GLD": 0.15, "TLT": 0.05, "SHY": 0.05},
  "risk_metrics": {"sharpe_ratio": 1.04, "max_drawdown": -0.24},
  "data_quality": {
    "missing_tickers": [],
    "warnings": [],
    "stale_days": 0,
    "vendor": "local_cache",
    "refreshed_prices": false,
    "timezone": "America/New_York",
    "defaulted_to_cache_end": false
  }
}
```

---

## 6. Dashboard Additions (Design Only)

### 6.1 New sidebar page

Add **"Live / Daily Monitor"** to `dashboard/app.py` navigation list. Page reads **only** `output/live/*` (same pattern as existing `dashboard/data_loader.py`).

### 6.2 UI elements

| Element | Source |
|---------|--------|
| `data_as_of` | `latest_signal.json` |
| `next_rebalance_date` | JSON / CSV |
| `rebalance_due` | JSON flag with color badge (due / not due) |
| Latest risk metrics | `latest_risk_metrics.csv` table + sparkline if history file added later |
| Latest target weights | Bar chart from `latest_target_weights.csv` |
| Staleness warning | If `data_as_of` older than 2 trading days |
| Disclaimer banner | Reuse educational disclaimer; add **not real-time** |

### 6.3 Empty state

If `output/live/` missing, show:

```text
Run: python scripts/update_daily.py
Requires prior price cache or network for yfinance.
```

No computation inside Streamlit.

---

## 7. Risks and Disclosures

| Risk | Mitigation (design) |
|------|---------------------|
| **yfinance reliability** | Retries, stale detection, `data_quality` block in JSON; prefer vendor upgrade path in production |
| **Transaction costs** | Display turnover estimate vs last weights; default 0 bps but show sensitivity note from Stage 5 |
| **Turnover** | Log `weight_change_l1` in JSON when rebalance_due |
| **Market holidays** | Trading calendar helper; `price_date_used` != calendar `data_as_of` documented |
| **Missing data** | Fail partial tickers with explicit `missing_tickers`; do not silently renormalize without warning |
| **Not investment advice** | Fixed disclaimer in JSON and dashboard |
| **Not real-time** | Label all timestamps; no WebSocket / intraday claims |
| **Look-ahead** | Enforced cutoff at `data_as_of`; unit tests on synthetic panels |
| **Confusion with daily rebalance** | Default monthly policy; daily signal mode behind flag |

---

## 8. Implementation Plan

### Phase 1 - Core library (`src/live_update.py`)

| Function (proposed) | Responsibility |
|---------------------|----------------|
| `resolve_data_as_of(as_of: date | None)` | Default yesterday NY; map to last trading day |
| `load_prices_through(as_of)` | Extend cache via `src/data_loader.py` or isolated fetch |
| `compute_live_risk_metrics(returns, bench)` | Wrap `src/metrics.py` |
| `compute_next_rebalance_date(as_of, calendar)` | First trading day of month rule consistent with `src/backtest.py` |
| `compute_target_weights(train, strategy, mandate)` | Reuse `src/signals.py`, `src/mandate_constraints.py` |
| `build_live_artifacts(...)` | Write four outputs under `output/live/` |
| `run_daily_update(config)` | Orchestration entry |

Dependencies: existing `src/config.py` paths; new `OUTPUT_LIVE_DIR = OUTPUT_DIR / "live"`.

### Phase 2 - CLI (`scripts/update_daily.py`)

```text
python scripts/update_daily.py
python scripts/update_daily.py --refresh-prices
python scripts/update_daily.py --as-of 2026-05-30
python scripts/update_daily.py --dry-run
```

`--refresh-prices` fetches adjusted closes via yfinance, merges into
`data/processed/vance_etf_prices.csv` (union of dates/columns; new non-null
overwrites, new NaN preserves existing), then writes `output/live/latest_*`.
Cannot be combined with `--dry-run` (refresh mutates the price cache).

- Idempotent overwrite of `latest_*` files
- Exit code non-zero on missing tickers or stale data beyond threshold
- UTF-8 stdout for Windows compatibility (match `run_all_stages.py`)

### Phase 3 - Tests (`tests/test_live_update.py`)

Use synthetic fixtures (no network, no `data/*.csv`):

- `data_as_of` cutoff excludes future rows
- `training_end < next_rebalance_date` style invariant where applicable
- weights sum to 1, long-only, SHY cap for Version A
- `rebalance_due` true on first trading day of month
- JSON schema required keys present

### Phase 4 - Dashboard page

- `dashboard/data_loader.py`: `load_live_signal()`, `load_live_metrics()`, etc.
- `dashboard/app.py`: new page block only
- Optional: ASCII-safe page icon (avoid emoji in production docs; dashboard may keep existing)

### Phase 5 - Documentation and ops

- This file (`docs/LIVE_UPDATE.md`) maintained as spec
- Link from `docs/GUIDE.md` (one paragraph, when implemented)
- Optional GitHub Action **separate** from unit tests (scheduled workflow with `continue-on-error` if network); not required for CI green

### Phase 6 - Explicit non-goals

- No trading execution
- No broker API (IBKR, Alpaca, etc.)
- No claim of real-time or low-latency data
- No portfolio accounting or tax lots

---

## 9. Module and File Map (Target State)

```text
src/live_update.py          # new: daily refresh logic
scripts/update_daily.py     # new: CLI entry
tests/test_live_update.py   # new: offline tests
tests/fixtures.py           # extend: live price panels (optional)
output/live/                # gitignored generated artifacts
  latest_prices.csv
  latest_risk_metrics.csv
  latest_target_weights.csv
  latest_signal.json
dashboard/data_loader.py    # extend: live loaders
dashboard/app.py            # extend: Live / Daily Monitor page
docs/LIVE_UPDATE.md         # this document
```

Existing files **unchanged in v3 design** unless implementing: `src/backtest.py` (monthly engine stays research-only).

---

## 10. Scheduling and Deployment Notes

| Environment | Suggested schedule |
|-------------|-------------------|
| Local dev | Manual `python scripts/update_daily.py` after market close |
| GitHub Actions | Optional `workflow_dispatch` or cron; artifacts uploaded as workflow artifacts only |
| Streamlit Cloud | Dashboard reads committed **sample** `output/live/` OR user runs job elsewhere and syncs files |

**Hosted dashboard** without a daily job shows last successful `data_as_of` and a staleness warning.

---

## 11. Acceptance Criteria (v3 Done)

1. `scripts/update_daily.py` produces all four `output/live/` files without running full stage pipeline.
2. Unit tests pass with no network and no dependency on gitignored CSV cache.
3. Dashboard page renders metrics and weights from live files only.
4. Documentation states monthly rebalance policy, `data_as_of` semantics, and non-goals.
5. `latest_signal.json` includes disclaimer and `daily_signal_mode: false` by default.

---

## v3 MVP Decisions

Locked scope for the first implementation. Items not listed here remain out of v3 MVP.

| # | Decision | MVP rule |
|---|----------|----------|
| 1 | Default live strategy | **`fixed_shy_version_a`** (QQQ/SPY/DIA 25% each, GLD 15%, TLT 5%, SHY 5%) |
| 2 | Daily refresh scope | Update **prices** and **risk metrics** every trading day through `data_as_of` |
| 3 | Target weights intent | Weights are for the **next scheduled monthly rebalance**, not daily trading |
| 4 | Daily signal mode | **`LIVE_DAILY_SIGNAL_MODE` stays `false`** and is **out of v3 MVP** |
| 5 | Timezone | **America/New_York** for `data_as_of`, trading-day calendar, and rebalance scheduling |
| 6 | `rebalance_due` | **`true` only on the first trading day of the month** (same rule family as `src/backtest.py`) |
| 7 | Output retention | Write **only `latest_*` files** under `output/live/`; **no historical archive** in v3 MVP |
| 8 | Research output isolation | **Do not mutate** existing research artifacts (`output/stage*`, `output/dashboard/`, etc.). Live outputs are **isolated** under `output/live/` |
| 9 | Execution and data claims | **No broker API**, **no order execution**, **no intraday data**, **no real-time claims** |
| 10 | Price refresh (Phase 3) | Optional `--refresh-prices` merges **yfinance** into `data/processed/vance_etf_prices.csv`; tickers with **&lt; 253** non-null closes are **backfilled from `DEFAULT_PRICE_START`**; sparse columns do not collapse metrics via `dropna(how="any")`; warnings: `insufficient_history:*`, `short_metrics_lookback:*`; **not combinable with `--dry-run`** |

**Implications for operators:**

- Run `python scripts/update_daily.py --refresh-prices` after the NY close to update the cache and live artifacts; cache-only: `python scripts/update_daily.py`.
- The dashboard **Live / Daily Monitor** page reads `output/live/` only.
- On non-rebalance days, users still see updated risk metrics and provisional target weights for the upcoming rebalance date; `rebalance_due=false` means "monitor only, not rebalance day."
- Full research pipeline (`scripts/run_all_stages.py`) remains a separate, heavier job.

---

## 12. Open Questions (Resolve Before Implementation)

1. ~~Should live price refresh append to `data/processed/vance_etf_prices.csv`?~~ **Resolved (Phase 3):** append/merge into `data/processed/vance_etf_prices.csv`.

---

## References

- [docs/GUIDE.md](GUIDE.md) - current runbook
- [docs/STRUCTURE.md](STRUCTURE.md) - layer separation
- [reports/methodology.md](../reports/methodology.md) - walk-forward and assumptions
- [reports/interview_defense.md](../reports/interview_defense.md) - policy defense narrative
- `src/backtest.py` - monthly rebalance and walk-forward timing
- `src/data_loader.py` - price cache and yfinance fetch

**Disclaimer:** Design for educational portfolio risk monitoring. Not investment advice. Not a real-time trading system.
