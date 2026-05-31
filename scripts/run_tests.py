#!/usr/bin/env python3
"""Run pytest via the same Python interpreter."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    return subprocess.call([sys.executable, "-m", "pytest", "tests/", "-q"] + sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
