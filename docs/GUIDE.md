# Project Guide

This guide explains what the repository does, how to run it, and how to read the outputs.
All research logic lives in `src/`; the dashboard only displays precomputed artifacts.

---

## What this project is

A staged risk analytics workflow on **QQQ, SPY, DIA, GLD, TLT**, with optional **SHY**
in later stages. The pipeline moves from in-sample baseline checks through walk-forward
out-of-sample testing, COVID and 2022 stress windows, robustness and overfitting
diagnostics, policy comparison, mandate constraints, and dashboard-ready exports.

**End product:** A defensible recommendation for a fixed policy portfolio with a small
SHY sleeve, supported by walk-forward metrics and mandate checks.

---

## Standard workflow

```text
1. python scripts/run_all_stages.py      # Run full analysis; write under output/
2. streamlit run dashboard/app.py        # Explore results (read-only)
3. python scripts/run_tests.py           # Run unit tests (or: python -m pytest tests/ -q)
```

To rebuild dashboard CSVs only:

```text
python scripts/build_dashboard_outputs.py
```

---

## Folder roles

| Path | Purpose | Typical edits |
|------|---------|---------------|
| `src/` | Data loading, signals, backtest engine, metrics, mandate logic | When changing models or rules |
| `scripts/run_all_stages.py` | Runs stages in order | When changing pipeline sequence |
| `scripts/stages/` | One module per research stage | When adding a new stage |
| `dashboard/` | Charts and narrative UI | Presentation only |
| `output/` | Pipeline artifacts | Never edit by hand; regenerate |
| `reports/` | Human-readable memos | Before interviews or reviews |
| `tests/` | Regression checks on weights, dates, returns | When changing `src/` logic |
| `data/raw/`, `data/processed/` | Optional price caches | Data hygiene only |
| `notebooks/` | Exploratory analysis (optional) | Not the source of truth for production logic |

---

## Pipeline stages

`scripts/run_all_stages.py` executes approximately:

| Order | Stage | Focus |
|-------|-------|--------|
| 1 | Baseline | In-sample / out-of-sample fixed allocation |
| 2 | Rolling | Rolling window diagnostics |
| 3 | Walk-forward | Fixed, inverse-volatility, minimum-variance |
| 4 | Stress | COVID-19 and 2022 rate-shock windows |
| 5 | Robustness | Parameter and cost sensitivity |
| 5B | SHY extension | Add short-duration Treasury proxy |
| - | Overfitting | Stability and overfitting diagnostics |
| 6 | Policy | Compare policy variants |
| 6B | Mandate | Wealth-management constraint scoring |
| 6 audit | Reconciliation | Cross-check stage outputs |
| 7 | Mandate constrained | Final constrained weights |
| - | Backtest summary | Summary tables and dashboard inputs |

After the pipeline completes, open the dashboard to review return series, weights, and
summary metrics under `output/`.

---

## Assumptions you should be able to defend

- **Returns:** Daily simple returns on adjusted close prices.
- **Annualization:** Volatility scaled by `sqrt(252)`; cumulative returns compounded.
- **Walk-forward:** Training window ends strictly before the test month (no look-ahead).
- **Transaction costs:** 0 bps by default; sensitivity explored in Stage 5.
- **2022 narrative:** Equity and long-duration bonds fell together; TLT was a weak
  hedge. SHY is tested as a modest defensive sleeve without abandoning the core policy.

---

## Final recommendation

| Status | Policy |
|--------|--------|
| **Primary** | Fixed + SHY Version A: QQQ 25%, SPY 25%, DIA 25%, GLD 15%, TLT 5%, SHY 5% |
| **Not recommended** | Uncapped inverse-volatility + SHY (~73% SHY; cash-like, mandate-inconsistent) |

---

## Optional maintenance

Relocate legacy root-level stage scripts into `scripts/_legacy/`:

```text
python scripts/_relocate_stages.py
```

---

## Troubleshooting

**`UnicodeDecodeError` (gbk) on Windows**

The orchestrator forces UTF-8 for child processes. If errors persist, set:

```powershell
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUTF8="1"
```

**Dashboard: missing `key_return_series.csv` or similar**

Run `python scripts/build_dashboard_outputs.py` or the full pipeline first.

**Imports from old module names**

Use `from src...` in new code. Root shims exist only for backward-compatible stage scripts.

**Notebooks vs `src/`**

Use notebooks for exploration. Any logic you need in the pipeline or tests must live in `src/`.
