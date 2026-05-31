# Repository Structure

Portfolio-ready layout: notebooks explore, `src/` computes, `dashboard/` displays,
`output/` stores generated artifacts only.

---

## Top-level tree

```text
vance_portfolio_analysis/
  README.md
  requirements.txt
  app.py                      # Streamlit entry (delegates to dashboard/app.py)

  portfolio_core.py           # compat shim -> src (legacy imports)
  walk_forward_engine.py      # compat shim -> src.backtest, src.signals
  mandate_constraints.py      # compat shim -> src.mandate_constraints

  src/
    config.py                 # paths, tickers, constants
    data_loader.py
    preprocessing.py
    portfolio.py
    signals.py
    backtest.py
    metrics.py
    mandate_constraints.py
    dashboard_outputs.py      # builds output/dashboard/*.csv

  scripts/
    run_all_stages.py         # pipeline orchestration
    run_tests.py
    build_dashboard_outputs.py
    stages/                   # stage01 .. stage07, backtest_summary, etc.

  dashboard/
    app.py
    data_loader.py            # reads output/ only
    charts.py
    formatting.py

  data/
    raw/                      # optional untouched downloads
    processed/                # cleaned price cache

  output/                     # generated; gitignored
  reports/                    # methodology.md, PROJECT_SUMMARY.md
  tests/
  docs/                       # GUIDE.md, this file
  notebooks/                  # optional exploration
```

---

## Layer responsibilities

| Layer | Responsibility | Edit when |
|-------|----------------|-----------|
| `src/` | Reusable, testable finance logic | Backtest rules, metrics, mandates change |
| `scripts/stages/` | Invoke stages with stable CLI | Adding or renaming a research step |
| `scripts/run_all_stages.py` | Stage order and error handling | Pipeline workflow changes |
| `dashboard/` | Visualization and UX | Chart or layout changes only |
| `output/` | CSVs, plots, summary tables | Never manually; regenerate via scripts |
| `reports/` | External narrative for humans | Interview or methodology updates |
| `tests/` | Guardrails on weights, dates, math | Any change to `src/` behavior |

---

## Design rules

1. **No research logic in the dashboard.** Streamlit loads precomputed files from `output/`.
2. **No hand-editing of `output/`** as source code. Treat it like a build directory.
3. **Prefer `from src.module import ...`** over root shims for new modules.
4. **Stages stay thin.** Heavy computation belongs in `src/`, not in `scripts/stages/*.py`.

---

## Compat shims (repository root)

`portfolio_core.py`, `walk_forward_engine.py`, and `mandate_constraints.py` forward
to `src/` so older stage scripts keep working during migration. Reviewers should focus
on `src/` and `tests/` for implementation quality.

---

## Common commands

```powershell
python scripts/run_all_stages.py
python scripts/run_tests.py
streamlit run dashboard/app.py
```
