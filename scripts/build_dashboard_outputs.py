#!/usr/bin/env python3
"""Build dashboard CSV artifacts only (after stages or for refresh)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dashboard_outputs import build_all


if __name__ == "__main__":
    build_all()
    print("Wrote output/dashboard/*.csv")
