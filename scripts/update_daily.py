#!/usr/bin/env python3
"""Generate output/live/ daily monitor artifacts (v3 MVP)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

from src.live_update import run_daily_update  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Update output/live/ monitor files.")
    parser.add_argument(
        "--as-of",
        dest="as_of",
        metavar="YYYY-MM-DD",
        help="Data as-of calendar date (America/New_York intent). Default: today in NY.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute artifacts but do not write files.",
    )
    parser.add_argument(
        "--refresh-prices",
        action="store_true",
        help="Fetch latest adjusted closes via yfinance, update local cache, then build live artifacts.",
    )
    args = parser.parse_args()

    if args.refresh_prices and args.dry_run:
        print(
            "ERROR: --refresh-prices cannot be used with --dry-run. "
            "Refresh updates data/processed price cache; use --dry-run without --refresh-prices.",
            file=sys.stderr,
        )
        return 2

    try:
        result = run_daily_update(
            as_of=args.as_of,
            dry_run=args.dry_run,
            refresh_prices=args.refresh_prices,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        from src.live_update import PriceRefreshError

        if isinstance(e, PriceRefreshError):
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        raise

    mode = "dry-run" if args.dry_run else "written"
    dq = result.signal.get("data_quality", {})
    warnings = dq.get("warnings", [])
    warn_txt = f" warnings={warnings}" if warnings else ""
    print(
        f"Live update ({mode}): data_as_of={result.data_as_of} "
        f"price_date_used={result.price_date_used} "
        f"signal_effective_date={result.signal_effective_date} "
        f"next_rebalance_date={result.next_rebalance_date} "
        f"rebalance_due={result.rebalance_due} "
        f"vendor={dq.get('vendor')} refreshed_prices={dq.get('refreshed_prices')}{warn_txt}"
    )
    if not args.dry_run:
        print(f"Output directory: {result.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
