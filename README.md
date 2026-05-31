# Vance Portfolio Risk Analysis

[![Tests](https://github.com/linzhongnan0915/vance_portfolio_analysis/actions/workflows/tests.yml/badge.svg)](https://github.com/linzhongnan0915/vance_portfolio_analysis/actions/workflows/tests.yml)

## Repository description

This is an **educational portfolio risk analysis** project on a Vance-style ETF
allocation. It is built for **research and career portfolio demonstration** (interviews,
model validation, and wealth-management risk analytics storytelling).

- **Not investment advice.** Results are simulated historical research, not a product
  recommendation or live mandate.
- **Not a verified replication** of any disclosed portfolio; weights are approximate
  for teaching and defend-in-interview purposes.
- **License:** [MIT](LICENSE) (code and docs in this repository).

**Recommendation:** **Fixed + SHY Version A** (25/25/25/15/5/5). Do not use uncapped
inverse-volatility + SHY; that variant concentrates in cash-like SHY and is inconsistent
with a diversified equity-risk mandate.

The repository implements a staged analytics pipeline (baseline, walk-forward backtest,
stress tests, robustness, mandate selection) and a read-only Streamlit dashboard.

---

## Repository status

| Check | Status |
|-------|--------|
| Modular `src/` package | Core logic separated from scripts and dashboard |
| Streamlit dashboard | `dashboard/` reads precomputed CSVs from `output/` only |
| Automated tests | `tests/` run via `python scripts/run_tests.py` (pytest) |
| CI | GitHub Actions workflow `.github/workflows/tests.yml` runs the same command on push and pull_request (Python 3.11) |
| Generated artifacts | `output/` is pipeline-produced and listed in `.gitignore` |
| External docs | English memos under `docs/` and `reports/` |

Local check: `python scripts/run_tests.py`

---

## Project overview

| Item | Description |
|------|-------------|
| Universe | QQQ, SPY, DIA, GLD, TLT (+ SHY in extension stages) |
| Benchmark | SPY |
| Rebalance | Monthly (first trading day) |
| Walk-forward | 12-month training window, 1-month test step |
| Output | CSV summaries under `output/`; dashboard reads precomputed files only |

**Research question:** Can a publicly discussed ETF-style allocation be defended as a
**repeatable risk policy**? Which **SHY defensive extension** improves 2022-style
bond-equity correlation breakdown without violating wealth-management mandate constraints?

**Motivation:** Vol-targeted rules can improve Sharpe ratios out of sample, but mandate
analysis shows uncapped inverse-vol + SHY drifts into a cash-like portfolio. The project
separates **risk metrics** from **policy fit**.

---

## Key results (walk-forward, indicative)

| Strategy | Ann. return | Ann. vol | Sharpe | Max DD |
|----------|-------------|----------|--------|--------|
| Fixed | ~13.7% | ~13.1% | ~1.04 | ~-24% |
| Inverse-vol | ~11.7% | ~10.3% | ~1.14 | ~-23% |
| Min-variance | ~12.1% | ~10.9% | ~1.12 | ~-24% |
| SPY | ~14.4% | ~17.1% | ~0.84 | ~-34% |

Final policy choice: **Fixed + SHY Version A** after mandate and audit stages. Details:
[reports/methodology.md](reports/methodology.md).

---

## Quick start

```powershell
git clone https://github.com/linzhongnan0915/vance_portfolio_analysis.git
cd vance_portfolio_analysis
pip install -r requirements.txt

# Full pipeline (all stages + dashboard CSVs)
python scripts/run_all_stages.py

# Dashboard (read-only; requires output/ from pipeline)
streamlit run dashboard/app.py

# Automated checks
python scripts/run_tests.py
```

Rebuild dashboard inputs without re-running every stage:

```powershell
python scripts/build_dashboard_outputs.py
```

---

## Repository layout

| Path | Role |
|------|------|
| `src/` | Reusable research logic: data, backtest, metrics, mandate rules |
| `scripts/stages/` | Thin stage entry points |
| `scripts/run_all_stages.py` | Pipeline orchestration |
| `dashboard/` | Streamlit UI (no embedded backtest logic) |
| `output/` | Generated artifacts (gitignored) |
| `reports/` | Methodology and interview summary |
| `tests/` | Weights, look-ahead, returns, metrics checks |

Root-level `portfolio_core.py`, `walk_forward_engine.py`, and `mandate_constraints.py`
are **compat shims** for legacy imports. New code should use `from src...`.

Full layout: [docs/STRUCTURE.md](docs/STRUCTURE.md).

---

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/GUIDE.md](docs/GUIDE.md) | How to run the pipeline and interpret outputs |
| [docs/STRUCTURE.md](docs/STRUCTURE.md) | Directory conventions and design rules |
| [reports/methodology.md](reports/methodology.md) | Assumptions, backtest design, limitations, risk interpretation |
| [reports/PROJECT_SUMMARY.md](reports/PROJECT_SUMMARY.md) | Interview one-pager |

---

## Limitations

- Educational approximate replication; not an exact copy of any disclosed portfolio.
- Yahoo Finance (or similar) data; zero transaction-cost baseline unless sensitivity stage is run.
- Walk-forward results are historical simulation, not a forecast of future performance.

---

## Disclaimer

For research and career portfolio demonstration only. **Not investment advice.**

Market data are from public sources. Verify licensing and timeliness before any production use.

See [LICENSE](LICENSE) for software terms. Data vendors retain their own licenses.
