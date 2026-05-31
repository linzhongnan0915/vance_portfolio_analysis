"""Relocate root stage scripts into scripts/stages/ (one-time, safe to re-run).

Run from repo root:
    python scripts/_relocate_stages.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DST = ROOT / "scripts" / "stages"
DST.mkdir(parents=True, exist_ok=True)

MAPPING = {
    "Vance_portfolio_analysis.py": "stage01_baseline.py",
    "Vance_portfolio_analysis_stage2.py": "stage02_rolling.py",
    "Vance_portfolio_analysis_stage3.py": "stage03_walkforward.py",
    "Vance_portfolio_analysis_stage4.py": "stage04_stress.py",
    "Vance_portfolio_analysis_stage5.py": "stage05_robustness.py",
    "Vance_portfolio_analysis_stage5b.py": "stage05b_shy.py",
    "overfitting_diagnostics.py": "stage05_overfitting.py",
    "Vance_portfolio_analysis_stage6.py": "stage06_policy.py",
    "Vance_portfolio_analysis_stage6b.py": "stage06b_mandate.py",
    "stage6_audit.py": "stage06_audit.py",
    "Vance_portfolio_analysis_stage7.py": "stage07_mandate.py",
    "backtest_summary.py": "backtest_summary.py",
}

BOOTSTRAP = """
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
"""

ROOT_LINE = "ROOT = Path(__file__).resolve().parent"
ROOT_IMPORT = "from src.config import ROOT"


def _already_migrated(text: str) -> bool:
    return "from src.config import ROOT" in text and "_REPO_ROOT" in text


def _patch(text: str) -> str:
    if _already_migrated(text):
        return text
    if ROOT_LINE in text:
        text = text.replace(ROOT_LINE, ROOT_IMPORT, 1)
    if "_REPO_ROOT" not in text:
        if "from __future__ import annotations" in text:
            text = text.replace(
                "from __future__ import annotations\n",
                "from __future__ import annotations\n" + BOOTSTRAP,
                1,
            )
        else:
            text = BOOTSTRAP + "\n" + text
    # Remove old thin-wrapper bootstrap if present
    text = text.replace("from scripts._bootstrap import ensure_repo_root\n\nensure_repo_root()\n", "")
    text = text.replace("from scripts._bootstrap import ensure_repo_root\n", "")
    return text


def main() -> None:
    moved = 0
    skipped = 0
    for src_name, dst_name in MAPPING.items():
        src = ROOT / src_name
        dst = DST / dst_name

        if not src.exists():
            if dst.exists() and _already_migrated(dst.read_text(encoding="utf-8")):
                print(f"skip (already done): {dst_name}")
                skipped += 1
                continue
            if dst.exists():
                print(f"skip (no source, dst exists): {dst_name}")
                skipped += 1
                continue
            raise FileNotFoundError(f"Missing source and destination: {src_name}")

        text = _patch(src.read_text(encoding="utf-8"))
        dst.write_text(text, encoding="utf-8")
        src.unlink()
        print(f"moved: {src_name} -> scripts/stages/{dst_name}")
        moved += 1

    print(f"\nDone. moved={moved}, skipped={skipped}")
    print("Root should now only have: app.py, portfolio_core.py, walk_forward_engine.py, mandate_constraints.py (+ README, etc.)")


if __name__ == "__main__":
    main()
