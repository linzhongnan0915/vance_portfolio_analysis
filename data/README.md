# Data directory

Market prices and pipeline outputs in this folder are **local and reproducible**;
they are **not committed** to the repository.

## Layout

| Path | Purpose |
|------|---------|
| `raw/` | Optional untouched vendor downloads (empty in a fresh clone) |
| `processed/` | Cached adjusted-close panel written by `src/data_loader.py` |

Ignored by git (see root `.gitignore`):

- `data/*.csv`
- `data/processed/*.csv`

## How to obtain data

1. Install dependencies: `pip install -r requirements.txt`
2. Run the full pipeline: `python scripts/run_all_stages.py`

The first run may download ETF history via **yfinance** and write
`data/processed/vance_etf_prices.csv`. Subsequent runs reuse that cache.

Unit tests use **synthetic fixtures** in `tests/` and do not require this cache or
network access.

## Outputs

Research tables and dashboard CSVs are written under `output/` (also gitignored).
Regenerate with the pipeline or `python scripts/build_dashboard_outputs.py`.

## Licensing and reproducibility

- Public vendor data (e.g. Yahoo Finance) carry their own terms; verify before any
  production or client-facing use.
- Download timestamps and vendor revisions can change series slightly; document
  your run date when comparing results across machines.
- This project is for **educational research**; not investment advice.
