#!/usr/bin/env python3
"""Run full analysis pipeline (in-process — no subprocess encoding issues on Windows)."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PIPELINE = [
    ("Stage 1 — Baseline", "scripts.stages.stage01_baseline"),
    ("Stage 2 — Rolling", "scripts.stages.stage02_rolling"),
    ("Stage 3 — Walk-forward", "scripts.stages.stage03_walkforward"),
    ("Stage 4 — Stress", "scripts.stages.stage04_stress"),
    ("Stage 5 — Robustness", "scripts.stages.stage05_robustness"),
    ("Stage 5B — SHY", "scripts.stages.stage05b_shy"),
    ("Overfitting", "scripts.stages.stage05_overfitting"),
    ("Stage 6 — Policy", "scripts.stages.stage06_policy"),
    ("Stage 6B — Mandate", "scripts.stages.stage06b_mandate"),
    ("Stage 6 — Audit", "scripts.stages.stage06_audit"),
    ("Stage 7 — Mandate", "scripts.stages.stage07_mandate"),
    ("Backtest summary", "scripts.stages.backtest_summary"),
]


def _setup() -> None:
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _run_module(module: str) -> None:
    # Same process as parent — avoids Windows GBK subprocess decode errors
    runpy.run_module(module, run_name="__main__", alter_sys=True)


def main() -> None:
    _setup()
    for label, module in PIPELINE:
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
        _run_module(module)

    print("\nBuilding dashboard CSVs...")
    _run_module("scripts.build_dashboard_outputs")
    print("\nDone -> streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
